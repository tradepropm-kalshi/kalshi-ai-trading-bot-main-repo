"""
Prompt construction utilities for AI trading decisions.

Centralises all LLM prompt templates and construction logic so that
XAIClient stays focused on network I/O and response handling.
"""

from typing import Dict, Any

from src.config.settings import settings


class PromptBuilder:
    """
    Builds LLM prompts for every query type used by the trading bot.

    All methods are static so callers can use the class without instantiation,
    or create an instance for convenience.
    """

    # ------------------------------------------------------------------
    # Trading decision prompts
    # ------------------------------------------------------------------

    @staticmethod
    def simplified_trading_prompt(
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        news_summary: str,
    ) -> str:
        """
        Build a short, token-efficient prompt for a trading decision.

        Used as a fallback when the full prompt exceeds the model's token
        budget or when a fast decision is required.

        Args:
            market_data:    Dict with keys: title, yes_bid, yes_ask, no_bid, no_ask, volume.
            portfolio_data: Dict with keys: balance, positions.
            news_summary:   Recent news context string (will be truncated).

        Returns:
            Formatted prompt string ready to send to the LLM.
        """
        title = market_data.get("title", "Unknown Market")
        yes_price = (market_data.get("yes_bid", 0) + market_data.get("yes_ask", 100)) / 2
        no_price = (market_data.get("no_bid", 0) + market_data.get("no_ask", 100)) / 2
        volume = market_data.get("volume", 0)

        truncated_news = (
            news_summary[:500] + "..." if len(news_summary) > 500 else news_summary
        )

        return f"""Analyze this prediction market and decide whether to trade.

Market: {title}
YES: {yes_price}¢ | NO: {no_price}¢ | Volume: ${volume:,.0f}

News: {truncated_news}

Rules:
- Only trade if you have >10% edge (your probability - market price)
- High confidence (>60%) required
- Return JSON only

Required format:
{{"action": "BUY|SKIP", "side": "YES|NO", "limit_price": 1-99, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

    @staticmethod
    def full_trading_prompt(
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        news_summary: str,
    ) -> str:
        """
        Build the comprehensive trading prompt using the multi-agent template.

        Args:
            market_data:    Dict with full market details.
            portfolio_data: Dict with balance and position information.
            news_summary:   Recent news context string.

        Returns:
            Formatted prompt string ready to send to the LLM.
        """
        from src.utils.prompts import MULTI_AGENT_PROMPT_TPL

        return MULTI_AGENT_PROMPT_TPL.format(
            title=market_data.get("title", "Unknown Market"),
            rules=market_data.get("rules", "No specific rules provided"),
            yes_price=(market_data.get("yes_bid", 0) + market_data.get("yes_ask", 100)) / 2,
            no_price=(market_data.get("no_bid", 0) + market_data.get("no_ask", 100)) / 2,
            volume=market_data.get("volume", 0),
            days_to_expiry=market_data.get("days_to_expiry", 30),
            news_summary=news_summary,
            cash=portfolio_data.get("cash", 1000),
            max_trade_value=portfolio_data.get("max_trade_value", 100),
            max_position_pct=portfolio_data.get("max_position_pct", 5),
            ev_threshold=10,
        )

    @staticmethod
    def settings_trading_prompt(
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        news_summary: str,
    ) -> str:
        """
        Build the settings-aware trading prompt using SIMPLIFIED_PROMPT_TPL.

        Parses close_time to compute days_to_expiry and pulls limits from
        the global settings object.

        Args:
            market_data:    Dict with full market details including close_time.
            portfolio_data: Dict with balance and positions.
            news_summary:   Recent news context string.

        Returns:
            Formatted prompt string ready to send to the LLM.
        """
        import datetime as dt

        from src.utils.prompts import SIMPLIFIED_PROMPT_TPL

        max_trade_value = min(
            portfolio_data.get("balance", 0)
            * settings.trading.max_position_size_pct
            / 100,
            portfolio_data.get("balance", 0) * 0.05,
        )

        close_time = market_data.get("close_time", "Unknown")
        days_to_expiry: Any = "Unknown"

        if close_time != "Unknown":
            try:
                if isinstance(close_time, str):
                    close_dt = None
                    for fmt in [
                        "%Y-%m-%dT%H:%M:%SZ",
                        "%Y-%m-%d %H:%M:%S",
                        "%Y-%m-%dT%H:%M:%S",
                    ]:
                        try:
                            close_dt = dt.datetime.strptime(close_time, fmt)
                            break
                        except ValueError:
                            continue
                    if close_dt is None:
                        close_dt = dt.datetime.now()
                elif hasattr(close_time, "timestamp"):
                    close_dt = close_time
                else:
                    close_dt = dt.datetime.now()

                now = dt.datetime.now()
                if close_dt.tzinfo is None:
                    close_dt = close_dt.replace(tzinfo=dt.timezone.utc)
                if now.tzinfo is None:
                    now = now.replace(tzinfo=dt.timezone.utc)

                days_to_expiry = max(0, (close_dt - now).days)
            except (ValueError, AttributeError, OverflowError):
                days_to_expiry = 0

        prompt_params = {
            "ticker": market_data.get("ticker", "UNKNOWN"),
            "title": market_data.get("title", "Unknown Market"),
            "yes_price": float(
                market_data.get("yes_bid_dollars", 0)
                or market_data.get("yes_bid", 0)
                or 0
            ),
            "no_price": float(
                market_data.get("no_bid_dollars", 0)
                or market_data.get("no_bid", 0)
                or 0
            ),
            "volume": int(
                float(
                    market_data.get("volume_fp", 0)
                    or market_data.get("volume", 0)
                    or 0
                )
            ),
            "close_time": close_time,
            "days_to_expiry": days_to_expiry,
            "news_summary": news_summary[:1000],
            "cash": portfolio_data.get("balance", 0),
            "balance": portfolio_data.get("balance", 0),
            "existing_positions": len(portfolio_data.get("positions", [])),
            "ev_threshold": settings.trading.min_confidence_to_trade * 100,
            "max_trade_value": max_trade_value,
            "max_position_pct": settings.trading.max_position_size_pct,
        }

        return SIMPLIFIED_PROMPT_TPL.format(**prompt_params)

    # ------------------------------------------------------------------
    # Search prompts
    # ------------------------------------------------------------------

    @staticmethod
    def search_prompt(query: str, max_length: int) -> str:
        """
        Build a focused search prompt that asks for a brief factual summary.

        Args:
            query:      The optimised search query string.
            max_length: Maximum character length of the desired response.

        Returns:
            Prompt string to send as a search-enabled chat message.
        """
        return (
            f"Find current, relevant information about: {query}\n\n"
            f"Focus on:\n"
            f"- Recent news, data, or announcements\n"
            f"- Factual information from reliable sources\n"
            f"- Current conditions or forecasts if applicable\n\n"
            f"Provide a brief, factual summary under {max_length // 2} words. "
            f"If no current information is available, clearly state that."
        )

    @staticmethod
    def optimize_search_query(query: str) -> str:
        """
        Optimise a raw market title into a better web-search query.

        Strips prediction-market boilerplate and replaces known patterns
        with more search-friendly phrasing.

        Args:
            query: Raw query, often derived from the market title.

        Returns:
            Shorter, more search-friendly query string.
        """
        query = query.replace("Will the", "").replace("**", "")
        query = query.replace("on Jul 18, 2025?", "July 2025")
        query = query.replace("before 2025-", "2025 ")

        # Pattern shortcuts for common market categories
        lower = query.lower()
        if "high temp" in lower and ("la" in lower or "los angeles" in lower):
            return "Los Angeles weather forecast July 2025 temperature"
        if "high temp" in lower and ("philadelphia" in lower or "philly" in lower):
            return "Philadelphia weather forecast July 2025 temperature"
        if "rotten tomatoes" in lower:
            movie_part = query.split("Rotten Tomatoes")[0].strip()
            return f"{movie_part} movie Rotten Tomatoes score reviews"
        if "youngboy" in lower and "release" in lower:
            return "NBA YoungBoy album release date 2025 MASA"

        return query[:150].strip()

    # ------------------------------------------------------------------
    # Fallback context
    # ------------------------------------------------------------------

    @staticmethod
    def fallback_context(query: str, max_length: int) -> str:
        """
        Return a generic fallback string when live search is unavailable.

        Args:
            query:      The original search query.
            max_length: Ignored (kept for API symmetry with search methods).

        Returns:
            Human-readable fallback context string.
        """
        lower = query.lower()

        if "temp" in lower and ("la" in query or "philadelphia" in query):
            city = "Los Angeles" if "LA" in query else "Philadelphia"
            return (
                f"Weather forecasts for {city} in July typically show temperatures "
                f"varying by day and weather patterns. Precise daily predictions "
                f"require current meteorological data. [Fallback: Search unavailable]"
            )

        if "rotten tomatoes" in lower:
            return (
                "Movie ratings and reviews vary based on critic and audience feedback. "
                "Rotten Tomatoes scores depend on the number and sentiment of reviews "
                "received. [Fallback: Search unavailable]"
            )

        if "release" in lower and "album" in lower:
            return (
                "Music release dates are typically announced by artists or labels "
                "through official channels. Release schedules can change based on "
                "various factors. [Fallback: Search unavailable]"
            )

        truncated = query[:100] + "..." if len(query) > 100 else query
        return (
            f"Current information about '{truncated}' requires live data access. "
            f"Analyzing based on available market data and general knowledge. "
            f"[Fallback: Search unavailable]"
        )
