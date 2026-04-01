"""
Market Context Builder — wires free data sources into AI-ready context strings.

For every market the lean bot is about to analyze, this module dispatches to the
appropriate free data sources based on the market's category and title keywords,
then assembles a compact text block that gets injected into the Grok-3 prompt.

Why this matters:
  Without real-world context, an LLM reasons purely from training-data priors.
  A basketball market at 55¢ with the home team's star player out with an ankle
  sprain should be worth far less — but the model won't know unless we tell it.

Dispatch table:
  Category contains sports/ncaab/nba/nfl/ncaa → ESPN scores + injuries + Vegas odds
  Category contains weather                    → Open-Meteo city forecast
  Category contains crypto                     → CoinGecko prices + 24h change
  Category contains politics/election          → Metaculus questions
  Category contains econ/fed/cpi (blocked,     → FRED series (informational only)
    but data fetched if ever re-enabled)
  All markets                                  → Polymarket cross-reference
  High-volume markets (≥ $50k)                 → NewsAPI top headlines
"""

import asyncio
import re
from typing import Dict, List, Optional

from src.data.free_data_sources import (
    CoinGeckoClient,
    ESPNClient,
    FREDClient,
    MetaculusClient,
    NewsAPIClient,
    OddsAPIClient,
    OpenMeteoClient,
    PolymarketClient,
)
from src.utils.database import Market
from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("context_builder")

# ---------------------------------------------------------------------------
# Volume threshold above which we also pull NewsAPI headlines
# ---------------------------------------------------------------------------
_NEWS_VOLUME_THRESHOLD = 50_000.0

# ---------------------------------------------------------------------------
# Sport-name → ESPN sport/league slugs
# ---------------------------------------------------------------------------
_SPORT_MAP = {
    "ncaab": ("basketball", "mens-college-basketball"),
    "ncaa":  ("basketball", "mens-college-basketball"),
    "nba":   ("basketball", "nba"),
    "nfl":   ("football",   "nfl"),
    "mlb":   ("baseball",   "mlb"),
    "nhl":   ("hockey",     "nhl"),
    "mls":   ("soccer",     "usa.1"),
    "wnba":  ("basketball", "wnba"),
}

# ---------------------------------------------------------------------------
# Crypto token → CoinGecko coin id
# ---------------------------------------------------------------------------
_CRYPTO_MAP = {
    "bitcoin":  "bitcoin",
    "btc":      "bitcoin",
    "ethereum": "ethereum",
    "eth":      "ethereum",
    "solana":   "solana",
    "sol":      "solana",
    "xrp":      "ripple",
    "bnb":      "binancecoin",
    "doge":     "dogecoin",
    "ada":      "cardano",
    "avax":     "avalanche-2",
    "matic":    "matic-network",
    "link":     "chainlink",
    "dot":      "polkadot",
}

# ---------------------------------------------------------------------------
# US city patterns for weather lookups
# ---------------------------------------------------------------------------
_CITY_PATTERNS = [
    "new york", "los angeles", "chicago", "houston", "phoenix",
    "philadelphia", "san antonio", "san diego", "dallas", "san jose",
    "austin", "jacksonville", "fort worth", "columbus", "charlotte",
    "indianapolis", "san francisco", "seattle", "denver", "nashville",
    "oklahoma city", "el paso", "washington", "las vegas", "louisville",
    "portland", "memphis", "raleigh", "boston", "miami",
    "minneapolis", "atlanta", "new orleans", "kansas city", "omaha",
    "cleveland", "tampa", "pittsburgh", "cincinnati", "salt lake city",
    "buffalo", "detroit", "milwaukee", "baltimore",
]


class MarketContextBuilder:
    """
    Assembles a rich, real-world context string for a single Kalshi market.

    The context is designed to fit within ~400 tokens so the full prompt
    stays under 1000 tokens — keeping every Grok-3 call at ~$0.015.

    Usage::

        builder = MarketContextBuilder(
            odds_api_key="...",
            fred_api_key="...",
            newsapi_key="...",
            metaculus_api_key=None,   # optional
        )
        context = await builder.build_context(market)
        # context is a plain-text string injected into the AI prompt
    """

    def __init__(
        self,
        odds_api_key: str = "",
        fred_api_key: str = "",
        newsapi_key: str = "",
        metaculus_api_key: str = "",
    ) -> None:
        self._odds_key      = odds_api_key
        self._fred_key      = fred_api_key
        self._newsapi_key   = newsapi_key
        self._metaculus_key = metaculus_api_key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def build_context(self, market: Market) -> str:
        """
        Build a compact context string for *market* using appropriate sources.

        Args:
            market: :class:`~src.utils.database.Market` to fetch context for.

        Returns:
            Multi-line context string (≤ 400 tokens), or empty string on
            total failure.  Never raises.
        """
        category = (market.category or "other").lower().strip()
        title    = (market.title    or "").lower()
        volume   = float(market.volume or 0)

        parts: List[str] = []

        # Dispatch coroutines based on category/title keywords
        tasks = {}

        sport_slug = self._detect_sport(category, title)
        if sport_slug:
            tasks["espn"]    = self._get_sports_context(market.title, sport_slug)
            tasks["odds"]    = self._get_odds_context(market.title, sport_slug[1])

        if "weather" in category or "temperature" in title or "rain" in title:
            tasks["weather"] = self._get_weather_context(market.title)

        if "crypto" in category or any(k in title for k in _CRYPTO_MAP):
            tasks["crypto"]  = self._get_crypto_context(market.title)

        if any(k in category for k in ("politics", "election")):
            tasks["meta"]    = self._get_metaculus_context(market.title)

        # Cross-reference on Polymarket regardless of category
        tasks["poly"]  = self._get_polymarket_context(market.title)

        # Headlines for high-volume markets
        if volume >= _NEWS_VOLUME_THRESHOLD and self._newsapi_key:
            tasks["news"]  = self._get_news_context(market.title)

        # Run all dispatched tasks concurrently
        if tasks:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for key, result in zip(tasks.keys(), results):
                if isinstance(result, Exception):
                    logger.debug(f"Context source '{key}' failed: {result}")
                elif result:
                    parts.append(result)

        if not parts:
            return ""

        header  = f"[Real-world context for: {market.title[:80]}]"
        context = "\n".join([header] + parts)

        # Hard cap at 1800 characters (~450 tokens)
        if len(context) > 1800:
            context = context[:1797] + "..."

        logger.debug(
            "Context built",
            market=market.market_id,
            sources=list(tasks.keys()),
            length=len(context),
        )
        return context

    # ------------------------------------------------------------------
    # Sports context
    # ------------------------------------------------------------------

    async def _get_sports_context(
        self, title: str, sport_slug: tuple
    ) -> str:
        """Fetch ESPN live scores and injury report for the relevant sport."""
        sport, league = sport_slug
        espn = ESPNClient()
        lines: List[str] = []

        scores_data = await espn.get_live_scores(sport, league)
        if scores_data:
            games = scores_data.get("events", [])[:5]
            if games:
                lines.append("ESPN live games:")
                for g in games:
                    comps = g.get("competitions", [{}])[0]
                    competitors = comps.get("competitors", [])
                    if len(competitors) >= 2:
                        home = competitors[0]
                        away = competitors[1]
                        h_name  = home.get("team", {}).get("abbreviation", "?")
                        a_name  = away.get("team", {}).get("abbreviation", "?")
                        h_score = home.get("score", "")
                        a_score = away.get("score", "")
                        status  = g.get("status", {}).get("type", {}).get("description", "")
                        lines.append(f"  {h_name} {h_score} vs {a_name} {a_score} ({status})")

        injuries = await espn.get_injury_report(sport, league)
        if injuries:
            entries = injuries.get("injuries", [])[:6]
            if entries:
                lines.append("Key injuries:")
                for inj in entries:
                    player = inj.get("athlete", {}).get("fullName", "?")
                    team   = inj.get("team", {}).get("abbreviation", "?")
                    status = inj.get("status", "?")
                    lines.append(f"  {player} ({team}) — {status}")

        return "\n".join(lines) if lines else ""

    async def _get_odds_context(self, title: str, league: str) -> str:
        """Fetch Vegas consensus odds for the market's sport."""
        if not self._odds_key:
            return ""
        odds_client = OddsAPIClient(api_key=self._odds_key)

        # Map ESPN league slug to Odds API sport key
        sport_key_map = {
            "nba":                   "basketball_nba",
            "nfl":                   "americanfootball_nfl",
            "mlb":                   "baseball_mlb",
            "nhl":                   "icehockey_nhl",
            "mens-college-basketball": "basketball_ncaab",
            "wnba":                  "basketball_wnba",
        }
        sport_key = sport_key_map.get(league)
        if not sport_key:
            return ""

        odds_data = await odds_client.get_sports_odds(sport_key)
        if not odds_data:
            return ""

        lines = ["Vegas odds (moneyline):"]
        title_lower = title.lower()
        for game in odds_data[:8]:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            # Fuzzy match: show game if either team name appears in the title
            if home.lower() in title_lower or away.lower() in title_lower:
                bookmakers = game.get("bookmakers", [])
                if bookmakers:
                    markets = bookmakers[0].get("markets", [])
                    ml = next((m for m in markets if m.get("key") == "h2h"), None)
                    if ml:
                        outcomes = ml.get("outcomes", [])
                        odds_str = ", ".join(
                            f"{o['name']} {o['price']:+d}" if isinstance(o['price'], int)
                            else f"{o['name']} {o['price']}"
                            for o in outcomes
                        )
                        lines.append(f"  {home} vs {away}: {odds_str}")
                        break

        return "\n".join(lines) if len(lines) > 1 else ""

    # ------------------------------------------------------------------
    # Weather context
    # ------------------------------------------------------------------

    async def _get_weather_context(self, title: str) -> str:
        """Fetch city weather forecast matching the market title."""
        city = self._detect_city(title)
        if not city:
            return ""
        weather = OpenMeteoClient()
        data = await weather.get_forecast_for_city(city)
        if not data:
            return ""
        daily  = data.get("daily", {})
        temps  = daily.get("temperature_2m_max", [])
        precip = daily.get("precipitation_probability_max", [])
        times  = daily.get("time", [])
        if not temps or not times:
            return ""
        lines = [f"Open-Meteo forecast for {city.title()}:"]
        for i, (t, d) in enumerate(zip(times[:3], temps[:3])):
            rain = precip[i] if i < len(precip) else "?"
            lines.append(f"  {t}: high {t}°F, precip {rain}%")
            # override the duplicated var
            lines[-1] = f"  {times[i]}: high {temps[i]}°F, precip {rain}%"
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Crypto context
    # ------------------------------------------------------------------

    async def _get_crypto_context(self, title: str) -> str:
        """Fetch live CoinGecko prices for crypto coins detected in title."""
        title_lower = title.lower()
        coin_ids = [
            cid for token, cid in _CRYPTO_MAP.items()
            if token in title_lower
        ]
        # Deduplicate preserving order
        seen: Dict[str, bool] = {}
        unique_ids = [seen.setdefault(c, c) for c in coin_ids if c not in seen]

        if not unique_ids:
            # Default to BTC + ETH as market sentiment proxies
            unique_ids = ["bitcoin", "ethereum"]

        cg = CoinGeckoClient()
        prices = await cg.get_prices(unique_ids)
        if not prices:
            return ""

        lines = ["CoinGecko live prices:"]
        for coin_id, data in prices.items():
            usd     = data.get("usd", "?")
            change  = data.get("usd_24h_change", 0.0)
            dir_sym = "▲" if change >= 0 else "▼"
            lines.append(
                f"  {coin_id}: ${usd:,.2f} {dir_sym}{abs(change):.1f}% (24h)"
                if isinstance(usd, (int, float)) else
                f"  {coin_id}: unavailable"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Politics / election context
    # ------------------------------------------------------------------

    async def _get_metaculus_context(self, title: str) -> str:
        """Fetch matching Metaculus questions for political markets."""
        meta = MetaculusClient(api_key=self._metaculus_key)
        questions = await meta.search_questions(title, limit=2)
        if not questions:
            return ""
        lines = ["Metaculus community predictions:"]
        for q in questions:
            qtitle      = q.get("title", "?")[:80]
            resolution  = q.get("resolution_criteria", "")[:60]
            community   = q.get("community_prediction", {})
            prob        = community.get("full", {}).get("q2") if isinstance(community, dict) else None
            prob_str    = f"{prob:.0%}" if prob is not None else "N/A"
            lines.append(f"  Q: {qtitle}")
            lines.append(f"     Community prob: {prob_str}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Polymarket cross-reference (all markets)
    # ------------------------------------------------------------------

    async def _get_polymarket_context(self, title: str) -> str:
        """Cross-reference with Polymarket to calibrate Kalshi prices."""
        pm = PolymarketClient()
        markets = await pm.search_market(title, limit=2)
        if not markets:
            return ""
        lines = ["Polymarket cross-reference:"]
        for m in markets:
            q          = (m.get("question") or m.get("title", "?"))[:80]
            yes_price  = m.get("yes_price") or m.get("outcomePrices", [None])[0]
            volume     = m.get("volume") or m.get("volumeNum", 0)
            try:
                yes_pct = f"{float(yes_price)*100:.0f}¢" if yes_price is not None else "N/A"
                vol_str = f"${float(volume):,.0f}"
            except (TypeError, ValueError):
                yes_pct = "N/A"
                vol_str = "N/A"
            lines.append(f"  {q}: YES {yes_pct}, vol {vol_str}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # News headlines (high-volume markets only)
    # ------------------------------------------------------------------

    async def _get_news_context(self, title: str) -> str:
        """Fetch top 3 relevant headlines from NewsAPI."""
        news = NewsAPIClient(api_key=self._newsapi_key)
        articles = await news.search_news(title, max_articles=3)
        if not articles:
            return ""
        lines = ["Recent headlines:"]
        for art in articles:
            headline = (art.get("title") or "")[:100]
            source   = art.get("source", {}).get("name", "?")
            if headline:
                lines.append(f"  [{source}] {headline}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_sport(category: str, title: str) -> Optional[tuple]:
        """Return (sport, league) ESPN slugs if the market is a sports market."""
        combined = f"{category} {title}"
        for keyword, slugs in _SPORT_MAP.items():
            if keyword in combined:
                return slugs
        if any(k in combined for k in ("game", "match", "champion", "finals", "playoff")):
            # Generic sports — default to NBA for basketball feel
            if "basket" in combined:
                return _SPORT_MAP["nba"]
            if "football" in combined or " nfl" in combined:
                return _SPORT_MAP["nfl"]
        return None

    @staticmethod
    def _detect_city(title: str) -> Optional[str]:
        """Return the first US city name found in *title*, or None."""
        title_lower = title.lower()
        for city in _CITY_PATTERNS:
            if city in title_lower:
                return city
        return None
