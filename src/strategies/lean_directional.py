"""
Lean Directional Trading Strategy.

Single-model (Grok-3) directional bets placed only on:
  • High-volume markets (scored ≥ 45 by MarketScorer)
  • Categories with proven historical win rates (≥ 50 category score)
  • Markets where EdgeFilter confirms ≥ 6-12% probability edge

Deliberately omits:
  • 5-model ensemble          → saves ~$0.085/decision
  • Market making             → capital-intensive, complex
  • Quick-flip scalping       → too speculative for lean mode

Every Grok-3 call is enriched with real-world context from free data APIs:
  Sports   → ESPN live scores + injuries + Vegas consensus odds
  Weather  → Open-Meteo 7-day forecast
  Crypto   → CoinGecko live prices + 24h change
  Politics → Metaculus community predictions
  All      → Polymarket cross-reference probability
  High-vol → NewsAPI breaking headlines

The strategy wires together five existing utilities:
  MarketScorer         (ranking)
  MarketContextBuilder (real-world context injection)
  XAIClient            (single Grok-3 call)
  EdgeFilter           (mandatory edge check)
  Kelly criterion      (fractional position sizing)
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.clients.claude_client import ClaudeClient
from src.clients.kalshi_client import KalshiClient
from src.clients.xai_client import XAIClient
from src.config.settings import settings
from src.data.market_context_builder import MarketContextBuilder
from src.utils.database import DatabaseManager, Market, Position
from src.utils.edge_filter import EdgeFilter
from src.utils.logging_setup import get_trading_logger
from src.utils.market_scorer import MarketScore, MarketScorer
from src.utils.price_utils import cents_to_dollars
from src.utils.response_parser import ResponseParser

logger = get_trading_logger("lean_directional")

# ---------------------------------------------------------------------------
# Strategy configuration
# ---------------------------------------------------------------------------

#: Maximum number of markets to send for AI analysis per cycle.
MAX_MARKETS_PER_CYCLE = 20

#: Minimum AI confidence required to open a position.
MIN_CONFIDENCE = 0.60

#: Fractional Kelly cap (never bet more than 25% of Kelly suggested).
KELLY_FRACTION_CAP = 0.25

#: Minimum dollar position size.
MIN_POSITION_SIZE = 2.0

#: Maximum number of concurrent open positions.
MAX_POSITIONS = 10

#: Analysis deduplication window in hours.
ANALYSIS_COOLDOWN_HOURS = 2.0

#: Maximum AI cost per market decision.
MAX_COST_PER_DECISION = 0.20


@dataclass
class LeanOpportunity:
    """A fully-vetted opportunity ready for position sizing and execution."""

    market: Market
    market_score: MarketScore
    side: str                   # "YES" or "NO"
    ai_probability: float       # AI's estimated YES probability
    ai_confidence: float        # AI's confidence in that estimate
    edge: float                 # |ai_probability - market_price|
    market_price: float         # current ask price in dollars
    rationale: str = ""


@dataclass
class LeanStrategyResult:
    """Aggregated result from one lean-strategy cycle."""

    positions_created: int = 0
    positions_skipped: int = 0
    markets_analyzed: int = 0
    markets_skipped_score: int = 0
    markets_skipped_cooldown: int = 0
    markets_skipped_edge: int = 0
    total_capital_deployed: float = 0.0
    ai_cost: float = 0.0
    errors: List[str] = field(default_factory=list)


class LeanDirectionalStrategy:
    """
    High-volume, category-filtered directional trading strategy.

    Uses a dual-model consensus system:
      1. Grok-3 (xAI)         — primary analysis, full JSON decision
      2. Claude Opus 4.6      — secondary verification with adaptive thinking
    Both models must agree before a position is opened.
    If ``claude_client`` is not provided (or ANTHROPIC_API_KEY is absent),
    the strategy falls back to Grok-3-only mode automatically.

    Args:
        db_manager:    Shared :class:`DatabaseManager` instance.
        kalshi_client: Shared :class:`KalshiClient` instance.
        xai_client:    Shared :class:`XAIClient` instance (Grok-3).
        claude_client: Optional :class:`ClaudeClient` instance (Claude Opus 4.6).
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        kalshi_client: KalshiClient,
        xai_client: XAIClient,
        claude_client: Optional[ClaudeClient] = None,
    ) -> None:
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.xai_client = xai_client
        self.claude_client = claude_client
        self.scorer = MarketScorer()
        self.context_builder = MarketContextBuilder(
            odds_api_key=getattr(settings.api, "odds_api_key", ""),
            fred_api_key=getattr(settings.api, "fred_api_key", ""),
            newsapi_key=getattr(settings.api, "newsapi_key", ""),
            metaculus_api_key=getattr(settings.api, "metaculus_api_key", ""),
            bls_api_key=getattr(settings.api, "bls_api_key", ""),
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self, total_capital: float) -> LeanStrategyResult:
        """
        Execute one full lean-strategy cycle.

        1. Loads eligible markets from the database.
        2. Scores and ranks them via :class:`MarketScorer`.
        3. Sends the top-scoring markets for single-model AI analysis.
        4. Applies the edge filter to every AI verdict.
        5. Sizes positions with fractional Kelly criterion.
        6. Returns approved :class:`Position` objects for execution.

        Args:
            total_capital: Available capital in dollars (balance or phase capital).

        Returns:
            :class:`LeanStrategyResult` with counts, cost, and any errors.
        """
        result = LeanStrategyResult()

        # --- Guard: check position count ---
        occupied = await self.db_manager.get_markets_with_positions()
        if len(occupied) >= MAX_POSITIONS:
            logger.info(
                "Position cap reached — skipping cycle",
                open_positions=len(occupied),
                cap=MAX_POSITIONS,
            )
            return result

        # --- Load and score markets ---
        markets = await self.db_manager.get_eligible_markets(
            volume_min=10_000,
            max_days_to_expiry=7,
        )

        if not markets:
            logger.info("No eligible markets found (volume ≥ $10k, ≤ 7 days)")
            return result

        scored = self.scorer.rank_markets(markets, top_n=MAX_MARKETS_PER_CYCLE)
        result.markets_skipped_score = len(markets) - len(scored)

        logger.info(
            "Lean strategy cycle started",
            eligible_markets=len(markets),
            scored_markets=len(scored),
            open_positions=len(occupied),
            capital=f"${total_capital:,.2f}",
        )

        # --- Analyse each scored market ---
        for market_score in scored:
            market = market_score.market

            # Skip already-occupied markets
            if market.market_id in occupied:
                result.positions_skipped += 1
                continue

            # Skip recently-analysed markets (deduplication)
            if await self.db_manager.was_recently_analyzed(
                market.market_id, ANALYSIS_COOLDOWN_HOURS
            ):
                result.markets_skipped_cooldown += 1
                logger.debug(f"Cooldown — skipping {market.market_id}")
                continue

            # Stop if daily AI budget exhausted
            daily_cost = await self.db_manager.get_daily_ai_cost()
            if daily_cost >= getattr(settings.trading, "daily_ai_budget", 10.0):
                logger.warning(
                    "Daily AI budget exhausted — stopping analysis",
                    daily_cost=daily_cost,
                )
                break

            # --- AI analysis ---
            opportunity, cost = await self._analyse_market(market, market_score)
            result.ai_cost += cost
            result.markets_analyzed += 1

            if opportunity is None:
                result.markets_skipped_edge += 1
                continue

            # --- Size position with Kelly ---
            position = self._size_position(opportunity, total_capital)
            if position is None:
                result.positions_skipped += 1
                continue

            # --- Persist to DB ---
            position_id = await self.db_manager.add_position(position)
            if position_id is not None:
                position.id = position_id
                result.positions_created += 1
                result.total_capital_deployed += (
                    position.quantity * position.entry_price
                )
                occupied.add(market.market_id)

                logger.info(
                    "Lean position created",
                    market=market.market_id,
                    side=position.side,
                    qty=position.quantity,
                    price=f"${position.entry_price:.2f}",
                    edge=f"{opportunity.edge:.1%}",
                    confidence=f"{opportunity.ai_confidence:.1%}",
                    market_score=market_score.total_score,
                )
            else:
                result.positions_skipped += 1

        logger.info(
            "Lean strategy cycle complete",
            positions_created=result.positions_created,
            markets_analyzed=result.markets_analyzed,
            ai_cost=f"${result.ai_cost:.4f}",
            capital_deployed=f"${result.total_capital_deployed:.2f}",
        )
        return result

    # ------------------------------------------------------------------
    # AI analysis
    # ------------------------------------------------------------------

    async def _analyse_market(
        self,
        market: Market,
        market_score: MarketScore,
    ) -> Tuple[Optional[LeanOpportunity], float]:
        """
        Run single-model AI analysis on *market*.

        Fetches live prices, calls Grok-3 with a concise prompt, parses the
        response, and applies the edge filter.

        Args:
            market:       Market to analyse.
            market_score: Pre-computed score (used for logging).

        Returns:
            ``(LeanOpportunity, cost_usd)`` or ``(None, cost_usd)`` if the
            market does not pass the edge filter.
        """
        cost = 0.0
        try:
            # Fetch live market data
            market_data = await self.kalshi_client.get_market(market.market_id)
            if not market_data:
                return None, cost

            mkt = market_data.get("market", market_data)
            yes_ask = float(
                mkt.get("yes_ask_dollars", 0) or mkt.get("yes_ask", 0) or 0
            )
            no_ask = float(
                mkt.get("no_ask_dollars", 0) or mkt.get("no_ask", 0) or 0
            )
            volume = float(mkt.get("volume_fp", 0) or mkt.get("volume", 0) or 0)

            if yes_ask > 1.0:
                yes_ask = cents_to_dollars(yes_ask)
            if no_ask > 1.0:
                no_ask = cents_to_dollars(no_ask)

            if yes_ask <= 0 or no_ask <= 0:
                return None, cost

            # Build rich real-world context from free data APIs
            real_context = ""
            try:
                real_context = await self.context_builder.build_context(market)
            except Exception as ctx_exc:  # noqa: BLE001
                logger.debug(
                    "Context builder failed (non-fatal)",
                    market=market.market_id,
                    error=str(ctx_exc),
                )

            # Build lean prompt
            prompt = self._build_prompt(market, yes_ask, no_ask, volume, real_context)

            # Single Grok-3 call
            response = await self.xai_client.get_completion(
                prompt=prompt,
                max_tokens=400,
                temperature=0.1,
                strategy="lean_directional",
                query_type="directional_analysis",
                market_id=market.market_id,
            )

            if response is None:
                await self.db_manager.record_market_analysis(
                    market.market_id, "SKIPPED_BUDGET", 0.0, cost, "AI budget exhausted"
                )
                return None, cost

            # Estimate cost ($0.015 per typical single-model call)
            # Context builder uses free HTTP APIs — no additional AI cost
            cost = 0.015

            # Parse response
            parsed = ResponseParser.parse_trading_decision(response)
            if not parsed:
                await self.db_manager.record_market_analysis(
                    market.market_id, "PARSE_FAILED", 0.0, cost, "Could not parse AI response"
                )
                return None, cost

            action = parsed["action"]         # "BUY" or "SKIP"
            side = parsed["side"]             # "YES" or "NO"
            confidence = parsed["confidence"]
            limit_price_cents = parsed["limit_price"]
            rationale = parsed["reasoning"]

            if action == "SKIP" or confidence < MIN_CONFIDENCE:
                await self.db_manager.record_market_analysis(
                    market.market_id, "SKIP", confidence, cost,
                    f"AI confidence {confidence:.1%} < {MIN_CONFIDENCE:.0%} or action=SKIP"
                )
                return None, cost

            # Determine market price for the chosen side
            market_price = yes_ask if side == "YES" else no_ask
            ai_probability = limit_price_cents / 100.0  # AI's fair-value estimate

            # Edge filter
            edge_result = EdgeFilter.calculate_edge(ai_probability, market_price, confidence)
            if not edge_result.passes_filter:
                await self.db_manager.record_market_analysis(
                    market.market_id, "EDGE_FILTERED", confidence, cost,
                    f"Edge {edge_result.edge:.1%} below threshold for confidence {confidence:.1%}"
                )
                return None, cost

            # ── Dual-model consensus: Claude Opus 4.6 second opinion ─────────
            # Only run when ClaudeClient is available; fall back silently if not.
            if self.claude_client and self.claude_client.available:
                claude_prompt = (
                    f"You are verifying a prediction-market trade recommendation.\n"
                    f"Market: {market.title}\n"
                    f"Category: {market.category}\n"
                    f"Recommended side: {side}\n"
                    f"Market price: {market_price:.2f} ({market_price * 100:.0f}¢)\n"
                    f"Estimated fair value: {ai_probability:.2f} ({ai_probability * 100:.0f}¢)\n"
                    f"Edge: {edge_result.edge:.1%}  |  Grok-3 confidence: {confidence:.1%}\n"
                    + (f"Context: {real_context[:600]}\n" if real_context else "")
                    + f"\nDo you agree this trade has a genuine edge? "
                    f"Reply JSON only — no other text:\n"
                    f'{{"verdict": "CONFIRM|REJECT", "confidence": 0.0-1.0, "reason": "one sentence"}}'
                )
                claude_text, claude_cost = await self.claude_client.get_trading_decision(
                    prompt=claude_prompt,
                    market_id=market.market_id,
                )
                cost += claude_cost

                if claude_text:
                    upper = claude_text.upper()
                    if "REJECT" in upper and "CONFIRM" not in upper:
                        # Claude disagrees — skip the trade
                        logger.info(
                            "Claude REJECTED lean trade",
                            market=market.market_id,
                            side=side,
                            grok_conf=f"{confidence:.1%}",
                            claude_reason=claude_text[:120],
                        )
                        await self.db_manager.record_market_analysis(
                            market.market_id, "CLAUDE_REJECTED", confidence, cost,
                            f"Claude rejected: {claude_text[:200]}"
                        )
                        return None, cost

                    # Both agree — blend confidences (Claude 55%, Grok 45%)
                    import re as _re
                    nums = _re.findall(r"0\.\d+", claude_text)
                    claude_conf = float(nums[0]) if nums else confidence
                    blended_conf = 0.55 * claude_conf + 0.45 * confidence
                    logger.info(
                        "Claude CONFIRMED lean trade",
                        market=market.market_id,
                        side=side,
                        grok_conf=f"{confidence:.1%}",
                        claude_conf=f"{claude_conf:.1%}",
                        blended=f"{blended_conf:.1%}",
                    )
                    confidence = blended_conf   # use blended confidence going forward
                    rationale = f"[Dual-model consensus] {rationale}"
                else:
                    # Claude unavailable this call — proceed on Grok alone
                    logger.debug(
                        "Claude returned no response for %s — using Grok-only",
                        market.market_id,
                    )
            # ─────────────────────────────────────────────────────────────────

            await self.db_manager.record_market_analysis(
                market.market_id, "BUY", confidence, cost, rationale
            )

            return LeanOpportunity(
                market=market,
                market_score=market_score,
                side=side,
                ai_probability=ai_probability,
                ai_confidence=confidence,
                edge=edge_result.edge,
                market_price=market_price,
                rationale=rationale,
            ), cost

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Error analysing market",
                market=market.market_id,
                error=str(exc),
            )
            return None, cost

    @staticmethod
    def _build_prompt(
        market: Market,
        yes_ask: float,
        no_ask: float,
        volume: float,
        real_context: str,
    ) -> str:
        """
        Build a context-enriched, token-efficient prompt for a single call.

        Injects real-world data (scores, injuries, odds, weather, prices,
        Polymarket cross-reference, headlines) when available.  Stays under
        ~1000 tokens so Grok-3 costs ~$0.015 per market.

        Args:
            market:       Market being analyzed.
            yes_ask:      Current YES ask price in dollars (0-1 range).
            no_ask:       Current NO ask price in dollars (0-1 range).
            volume:       Total volume in dollars.
            real_context: Pre-built context string from MarketContextBuilder.
        """
        context_section = f"\n\n{real_context}" if real_context else ""
        return (
            f"You are a prediction market trader with access to real-world data. "
            f"Analyze this Kalshi market and decide whether to BUY YES, BUY NO, or SKIP.\n\n"
            f"Market: {market.title}\n"
            f"YES ask: {yes_ask:.2f} ({yes_ask * 100:.0f}¢)\n"
            f"NO ask:  {no_ask:.2f} ({no_ask * 100:.0f}¢)\n"
            f"Volume: ${volume:,.0f}\n"
            f"Category: {market.category}"
            f"{context_section}\n\n"
            f"Rules:\n"
            f"- Use the real-world data above to inform your probability estimate\n"
            f"- Only BUY if your estimate is at least 6% away from the market price\n"
            f"- Confidence must be 60%+ to trade\n"
            f"- Return JSON only — no other text\n\n"
            f'{{"action": "BUY|SKIP", "side": "YES|NO", '
            f'"limit_price": 1-99, "confidence": 0.0-1.0, "reasoning": "one sentence"}}'
        )

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def _size_position(
        self, opp: LeanOpportunity, total_capital: float
    ) -> Optional[Position]:
        """
        Compute a fractional-Kelly position size for *opp*.

        The Kelly fraction is: ``f = edge / (1 - market_price)``
        Capped at ``KELLY_FRACTION_CAP`` (25%) of capital.

        Args:
            opp:           Approved opportunity.
            total_capital: Available capital in dollars.

        Returns:
            :class:`Position` ready for DB insertion, or ``None`` if the
            computed size is too small.
        """
        try:
            market_price = opp.market_price

            # Kelly formula for binary market
            if market_price >= 1.0 or market_price <= 0.0:
                return None
            kelly_f = opp.edge / (1.0 - market_price)
            kelly_f = min(kelly_f, KELLY_FRACTION_CAP)

            # Scale by confidence (lower confidence → smaller position)
            confidence_scalar = max(0.5, min(1.0, opp.ai_confidence))
            dollar_size = total_capital * kelly_f * confidence_scalar

            # Apply max position pct from settings
            max_pct = getattr(settings.trading, "max_position_size_pct", 5.0)
            dollar_size = min(dollar_size, total_capital * max_pct / 100.0)
            dollar_size = max(dollar_size, MIN_POSITION_SIZE)

            quantity = max(1, int(dollar_size / market_price))

            return Position(
                market_id=opp.market.market_id,
                side=opp.side,
                entry_price=market_price,
                quantity=quantity,
                timestamp=datetime.now(),
                rationale=(
                    f"LEAN: {opp.rationale[:200]} | "
                    f"edge={opp.edge:.1%} conf={opp.ai_confidence:.1%} "
                    f"score={opp.market_score.total_score:.0f}"
                ),
                confidence=opp.ai_confidence,
                strategy="lean_directional",
                stop_loss_price=market_price * 0.90,    # 10% stop
                take_profit_price=market_price * 1.25,  # 25% target
                max_hold_hours=48,
            )
        except (ValueError, ZeroDivisionError, TypeError) as exc:
            logger.error("Error sizing position", error=str(exc))
            return None


# ---------------------------------------------------------------------------
# Module-level entry point
# ---------------------------------------------------------------------------

async def run_lean_directional(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
    xai_client: XAIClient,
    total_capital: float,
    claude_client: Optional[ClaudeClient] = None,
) -> Dict:
    """
    Run one cycle of the lean directional strategy.

    Args:
        db_manager:    Shared database manager.
        kalshi_client: Shared Kalshi REST client.
        xai_client:    Shared xAI client.
        total_capital: Capital available for deployment.
        claude_client: Optional Claude client for dual-model consensus.

    Returns:
        Result dict with positions created, AI cost, and capital deployed.
    """
    strategy = LeanDirectionalStrategy(db_manager, kalshi_client, xai_client, claude_client)
    result = await strategy.run(total_capital)
    return {
        "positions_created": result.positions_created,
        "markets_analyzed": result.markets_analyzed,
        "capital_deployed": result.total_capital_deployed,
        "ai_cost": result.ai_cost,
        "skipped_score": result.markets_skipped_score,
        "skipped_edge": result.markets_skipped_edge,
        "errors": result.errors,
    }
