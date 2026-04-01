"""
XAI client for AI-powered trading decisions.

Interfaces with Grok models through the xAI SDK.  All prompt construction,
response parsing, and daily-cost accounting are delegated to dedicated
utility classes so this module stays focused on network I/O and retry logic.
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from xai_sdk import AsyncClient
from xai_sdk.chat import user as xai_user
from xai_sdk.search import SearchParameters

from src.config.settings import settings
from src.utils.cost_tracker import CostTracker
from src.utils.logging_setup import TradingLoggerMixin
from src.utils.prompt_builder import PromptBuilder
from src.utils.response_parser import ResponseParser


@dataclass
class TradingDecision:
    """Represents an AI trading decision."""

    action: str           # "BUY" or "SKIP"
    side: str             # "YES" or "NO"
    confidence: float     # 0.0 – 1.0
    limit_price: Optional[int] = None   # cents (1–99)
    reasoning: str = ""


class XAIClient(TradingLoggerMixin):
    """
    xAI client for AI-powered trading decisions.

    Uses Grok models for market analysis and trading strategy.  The class
    owns a long-lived ``AsyncClient`` and delegates cost tracking, prompt
    construction, and response parsing to helper utilities.
    """

    def __init__(self, api_key: Optional[str] = None, db_manager=None):
        """
        Initialise the xAI client.

        Args:
            api_key:    xAI API key (defaults to ``settings.api.xai_api_key``).
            db_manager: Optional ``DatabaseManager`` for logging queries.
        """
        self.api_key = api_key or settings.api.xai_api_key
        self.db_manager = db_manager

        # Long-lived async HTTP client (3600 s timeout as recommended by xAI docs)
        self.client = AsyncClient(api_key=self.api_key, timeout=3600.0)

        # Model configuration
        self.primary_model = settings.trading.primary_model
        self.fallback_model = settings.trading.fallback_model
        self.temperature = settings.trading.ai_temperature
        self.max_tokens = settings.trading.ai_max_tokens

        # Session-level counters (not persisted — use CostTracker for daily totals)
        self.total_cost: float = 0.0
        self.request_count: int = 0

        # Daily cost enforcement
        self._cost_tracker = CostTracker(storage_path="logs/daily_ai_usage.pkl")
        self._daily_limit: float = getattr(
            settings.trading, "daily_ai_cost_limit", 50.0
        )

        # API credit-exhaustion state (in-memory, reset on new day)
        self.is_api_exhausted: bool = False
        self.api_exhausted_until: Optional[datetime] = None

        self.logger.info(
            "xAI client initialised",
            primary_model=self.primary_model,
            logging_enabled=bool(db_manager),
            daily_limit=self._daily_limit,
            today_cost=self._cost_tracker.get_daily_total(),
            today_requests=self._cost_tracker.get_request_count(),
        )

    # ------------------------------------------------------------------
    # Daily limit helpers
    # ------------------------------------------------------------------

    def _record_cost(self, cost: float) -> None:
        """Record *cost* in the CostTracker and session totals."""
        self._cost_tracker.reset_if_new_day()
        self._cost_tracker.record_cost(cost)
        self.total_cost += cost
        self.request_count += 1

        if self._cost_tracker.is_over_limit(self._daily_limit):
            self.logger.warning(
                "Daily AI cost limit reached — further requests will be skipped",
                daily_cost=self._cost_tracker.get_daily_total(),
                daily_limit=self._daily_limit,
            )

    async def _check_daily_limits(self) -> bool:
        """
        Return ``True`` if the bot may proceed with an API call, ``False`` if
        the daily budget is exhausted or the API credits are gone.
        """
        self._cost_tracker.reset_if_new_day()

        if self._cost_tracker.is_over_limit(self._daily_limit):
            self.logger.info(
                "Daily AI limit reached — request skipped",
                daily_cost=self._cost_tracker.get_daily_total(),
                daily_limit=self._daily_limit,
            )
            return False

        if self.is_api_exhausted:
            now = datetime.now()
            if self.api_exhausted_until and now < self.api_exhausted_until:
                self.logger.info("API credits exhausted — skipping request until reset")
                return False
            # New day — clear exhaustion
            self.is_api_exhausted = False
            self.api_exhausted_until = None

        return True

    async def _persist_cost_to_db(self, cost: float) -> None:
        """Fire-and-forget: persist a cost increment to the database."""
        if not self.db_manager or cost <= 0:
            return
        try:
            await self.db_manager.record_ai_cost(cost)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to persist xAI cost to DB", error=str(exc))

    def _handle_resource_exhausted(self, error: Exception) -> None:
        """Mark API as exhausted until the next calendar day."""
        from datetime import timedelta

        self.logger.error(
            "xAI API credits exhausted",
            error=str(error),
            daily_cost=self._cost_tracker.get_daily_total(),
        )
        self.is_api_exhausted = True
        midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.api_exhausted_until = midnight + timedelta(days=1)
        self._cost_tracker.is_exhausted = True

    @staticmethod
    def _is_resource_exhausted_error(error: Exception) -> bool:
        """Return ``True`` if *error* indicates API credit exhaustion."""
        msg = str(error).lower()
        return any(kw in msg for kw in ("resource_exhausted", "credits", "spending limit", "quota"))

    # ------------------------------------------------------------------
    # Query logging
    # ------------------------------------------------------------------

    async def _log_query(
        self,
        strategy: str,
        query_type: str,
        prompt: str,
        response: str,
        market_id: Optional[str] = None,
        tokens_used: Optional[int] = None,
        cost_usd: Optional[float] = None,
        confidence_extracted: Optional[float] = None,
        decision_extracted: Optional[str] = None,
    ) -> None:
        """Log an LLM query to the database if a manager is available."""
        if not self.db_manager:
            return
        try:
            from src.utils.database import LLMQuery

            llm_query = LLMQuery(
                timestamp=datetime.now().isoformat(),
                market_id=market_id or "",
                query_type=query_type,
                model=self.primary_model,
                cost=cost_usd or 0.0,
                response=(response or "")[:5000],
                strategy=strategy,
                prompt=prompt[:2000],
                tokens_used=tokens_used,
                confidence_extracted=confidence_extracted,
                decision_extracted=decision_extracted,
            )
            asyncio.create_task(self.db_manager.log_llm_query(llm_query))
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Failed to log LLM query", error=str(exc))

    # ------------------------------------------------------------------
    # Core completion / search
    # ------------------------------------------------------------------

    async def _make_completion_request(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
    ) -> Tuple[Optional[str], float]:
        """
        Make a completion request with retry logic, cost tracking, and
        automatic fallback to the secondary model.

        Args:
            messages:    Chat messages to send.
            model:       Override the primary model.
            temperature: Sampling temperature.
            max_tokens:  Maximum tokens in the response.
            max_retries: Number of attempts before giving up.

        Returns:
            ``(response_content, cost_usd)`` tuple.  ``response_content`` is
            ``None`` when the request could not be fulfilled.
        """
        if not await self._check_daily_limits():
            return None, 0.0

        model_to_use = model or self.primary_model
        temperature = temperature if temperature is not None else self.temperature

        if max_tokens is None:
            max_tokens = (
                settings.trading.ai_max_tokens
                if model_to_use == self.primary_model
                else self.max_tokens
            )
        original_max_tokens = max_tokens

        for attempt in range(max_retries):
            try:
                start_time = time.time()
                chat = self.client.chat.create(
                    model=model_to_use,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                for message in messages:
                    chat.append(message)

                response = await chat.sample()
                content = response.content
                processing_time = time.time() - start_time

                self.logger.debug(
                    "Raw xAI response received",
                    model=model_to_use,
                    response_length=len(content) if content else 0,
                    processing_time=processing_time,
                    attempt=attempt + 1,
                    finish_reason=getattr(response, "finish_reason", "unknown"),
                )

                if not content or not content.strip():
                    # Reasoning models may exhaust their token budget
                    reasoning_tokens = (
                        getattr(response.usage, "reasoning_tokens", 0)
                        if hasattr(response, "usage")
                        else 0
                    )
                    hit_limit = getattr(response, "finish_reason", None) == "REASON_MAX_LEN"

                    if reasoning_tokens and hit_limit and attempt < max_retries - 1:
                        # Scale token budget and retry
                        scale = 2 if attempt == 0 else 1
                        max_tokens = min(
                            max_tokens * scale, settings.trading.ai_max_tokens
                        )
                        self.logger.warning(
                            "Reasoning model hit token limit — retrying with more tokens",
                            attempt=attempt + 1,
                            max_tokens=max_tokens,
                        )
                        continue

                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue

                    # Last attempt — try fallback model
                    fallback = await self._try_fallback_model(
                        messages, temperature, original_max_tokens
                    )
                    if fallback:
                        return fallback
                    return None, 0.0

                estimated_tokens = (
                    getattr(response.usage, "total_tokens", len(content) // 4)
                    if hasattr(response, "usage")
                    else len(content) // 4
                )
                cost = estimated_tokens * 0.00001

                self._record_cost(cost)
                asyncio.create_task(self._persist_cost_to_db(cost))

                return content, cost

            except Exception as exc:  # noqa: BLE001
                if self._is_resource_exhausted_error(exc):
                    self._handle_resource_exhausted(exc)
                    return None, 0.0

                self.logger.warning(
                    f"Completion attempt {attempt + 1} failed",
                    model=model_to_use,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

                if attempt == max_retries - 1:
                    if model_to_use == self.primary_model:
                        fallback = await self._try_fallback_model(
                            messages, temperature, original_max_tokens
                        )
                        if fallback:
                            return fallback
                    return None, 0.0

                await asyncio.sleep(2 ** attempt)

        return None, 0.0

    async def _try_fallback_model(
        self,
        messages: List[Dict],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> Optional[Tuple[str, float]]:
        """
        Attempt a single request using the configured fallback model.

        Args:
            messages:    Chat messages to send.
            temperature: Sampling temperature.
            max_tokens:  Token budget (capped at 4 000 for the fallback).

        Returns:
            ``(content, cost)`` on success, ``None`` on failure.
        """
        fallback_model = settings.trading.fallback_model
        fallback_tokens = min(max_tokens or self.max_tokens, 4000)
        self.logger.info(f"Attempting fallback to {fallback_model}")
        try:
            chat = self.client.chat.create(
                model=fallback_model,
                temperature=temperature or self.temperature,
                max_tokens=fallback_tokens,
            )
            for message in messages:
                chat.append(message)
            response = await chat.sample()
            content = response.content
            if content and content.strip():
                estimated_tokens = (
                    getattr(response.usage, "total_tokens", len(content) // 4)
                    if hasattr(response, "usage")
                    else len(content) // 4
                )
                cost = estimated_tokens * 0.00001
                self._record_cost(cost)
                asyncio.create_task(self._persist_cost_to_db(cost))
                self.logger.info(
                    f"Fallback model {fallback_model} succeeded",
                    response_length=len(content),
                    cost=cost,
                )
                return content, cost
            self.logger.warning(f"Fallback model {fallback_model} returned empty response")
            return None
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Fallback model failed", error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(self, query: str, max_length: int = 300) -> str:
        """
        Perform a live web search using xAI's search-enabled chat.

        Falls back to a generic context string when the API is unavailable,
        over budget, or returns an empty result.

        Args:
            query:      Natural-language search query.
            max_length: Maximum character length of the returned summary.

        Returns:
            Search result string (never raises).
        """
        if not await self._check_daily_limits():
            return PromptBuilder.fallback_context(query, max_length)

        optimised = PromptBuilder.optimize_search_query(query)

        # Simple in-memory result cache
        if not hasattr(self, "_search_cache"):
            self._search_cache: Dict[str, str] = {}
        cache_key = f"{optimised[:50]}:{max_length}"
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]

        try:
            from xai_sdk import Client

            sync_client = Client(api_key=self.api_key)
            chat = sync_client.chat.create(
                model=self.primary_model,
                search_parameters=SearchParameters(mode="auto", return_citations=True),
                temperature=0.3,
                max_tokens=min(2000, self.max_tokens),
            )
            chat.append(xai_user(PromptBuilder.search_prompt(optimised, max_length)))

            start = time.time()
            response = chat.sample()
            if hasattr(response, "__await__"):
                response = await response
            processing_time = time.time() - start

            if not response or not getattr(response, "content", "").strip():
                return PromptBuilder.fallback_context(query, max_length)

            sources = (
                getattr(response.usage, "num_sources_used", 0)
                if hasattr(response, "usage")
                else 0
            )
            search_cost = sources * 0.025
            self._record_cost(search_cost)
            if search_cost > 0:
                asyncio.create_task(self._persist_cost_to_db(search_cost))

            self.logger.info(
                "xAI search completed",
                query=optimised[:50],
                sources=sources,
                cost=search_cost,
                processing_time=processing_time,
            )

            result = ResponseParser.truncate_to_length(response.content, max_length)
            if sources > 0:
                result += f"\n[Based on {sources} live sources]"
            elif getattr(response, "citations", None):
                result += f"\n[Based on {len(response.citations)} sources]"
            else:
                result += "\n[Based on model knowledge]"

            if len(self._search_cache) < 100:
                self._search_cache[cache_key] = result
            return result

        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "Live search failed — using fallback",
                query=query[:50],
                error=str(exc),
            )
            return PromptBuilder.fallback_context(query, max_length)

    async def get_trading_decision(
        self,
        market_data: Dict,
        portfolio_data: Dict,
        news_summary: str = "",
    ) -> Optional[TradingDecision]:
        """
        Get a trading decision from Grok for a given market.

        Tries the full prompt first; falls back to the simplified prompt if
        the first attempt fails.

        Args:
            market_data:    Dict with market price and metadata.
            portfolio_data: Dict with current portfolio balance and positions.
            news_summary:   Recent news context string.

        Returns:
            ``TradingDecision`` or ``None`` if the model could not decide.
        """
        for use_simplified in (False, True):
            decision = await self._get_decision_with_prompt(
                market_data, portfolio_data, news_summary, use_simplified
            )
            if decision:
                return decision
            if not use_simplified:
                self.logger.info("Full prompt failed — retrying with simplified prompt")
        return None

    async def _get_decision_with_prompt(
        self,
        market_data: Dict,
        portfolio_data: Dict,
        news_summary: str,
        use_simplified: bool,
    ) -> Optional[TradingDecision]:
        """
        Internal helper: build a prompt, call the model, parse the result.

        Args:
            market_data:    Market price and metadata dict.
            portfolio_data: Portfolio balance and positions dict.
            news_summary:   News context string.
            use_simplified: If ``True`` use the short prompt template.

        Returns:
            ``TradingDecision`` or ``None``.
        """
        try:
            if use_simplified:
                prompt = PromptBuilder.simplified_trading_prompt(
                    market_data, portfolio_data, news_summary
                )
            else:
                prompt = PromptBuilder.full_trading_prompt(
                    market_data, portfolio_data, news_summary
                )

            messages = [{"role": "user", "content": prompt}]
            max_tokens = 4000 if use_simplified else None

            content, cost = await self._make_completion_request(
                messages, temperature=0.1, max_tokens=max_tokens
            )
            if not content:
                return None

            parsed = ResponseParser.parse_trading_decision(content)
            if not parsed:
                return None

            return TradingDecision(
                action=parsed["action"],
                side=parsed["side"],
                confidence=parsed["confidence"],
                limit_price=parsed["limit_price"],
                reasoning=parsed["reasoning"],
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.error(
                "Error getting trading decision",
                simplified=use_simplified,
                error=str(exc),
            )
            return None

    async def get_completion(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        strategy: str = "unknown",
        query_type: str = "completion",
        market_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get a free-form completion from Grok.

        Args:
            prompt:      The prompt string.
            max_tokens:  Token budget override.
            temperature: Sampling temperature override.
            strategy:    Label for DB logging (e.g. ``"market_making"``).
            query_type:  Sub-label for DB logging (e.g. ``"analysis"``).
            market_id:   Market ticker for DB logging.

        Returns:
            Response text string, or ``None`` on failure / budget exhaustion.
        """
        try:
            messages = [xai_user(prompt)]
            content, cost = await self._make_completion_request(
                messages, max_tokens=max_tokens, temperature=temperature
            )
            if content is None:
                return None

            await self._log_query(
                strategy=strategy,
                query_type=query_type,
                prompt=prompt,
                response=content,
                market_id=market_id,
                cost_usd=cost,
            )
            return content
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Error in get_completion", error=str(exc))
            return None

    async def close(self) -> None:
        """Log session totals. The xAI SDK does not require explicit closing."""
        self.logger.info(
            "xAI client closed",
            total_estimated_cost=self.total_cost,
            total_requests=self.request_count,
        )
