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
    BLSClient,
    CoinGeckoClient,
    ESPNClient,
    FREDClient,
    JolpicaF1Client,
    ManifoldClient,
    MetaculusClient,
    MLBStatsClient,
    NewsAPIClient,
    NWSClient,
    OddsAPIClient,
    OpenMeteoClient,
    PolymarketClient,
    PredictItClient,
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
    # Basketball
    "ncaab": ("basketball", "mens-college-basketball"),
    "ncaa":  ("basketball", "mens-college-basketball"),
    "nba":   ("basketball", "nba"),
    "wnba":  ("basketball", "wnba"),
    # Football
    "nfl":   ("football",   "nfl"),
    # Baseball
    "mlb":   ("baseball",   "mlb"),
    "baseball": ("baseball", "mlb"),
    # Hockey
    "nhl":   ("hockey",     "nhl"),
    # Soccer
    "mls":      ("soccer", "usa.1"),
    "soccer":   ("soccer", "usa.1"),
    "epl":      ("soccer", "eng.1"),
    "premier":  ("soccer", "eng.1"),
    "ucl":      ("soccer", "uefa.champions"),
    "champions": ("soccer", "uefa.champions"),
    # Golf
    "golf":  ("golf", "pga"),
    "pga":   ("golf", "pga"),
    # Tennis
    "tennis": ("tennis", "atp"),
    "atp":    ("tennis", "atp"),
    "wta":    ("tennis", "wta"),
    # Racing
    "nascar": ("racing", "nascar-premier"),
    "f1":     ("racing", "f1"),
    "formula1": ("racing", "f1"),
    "indycar":  ("racing", "indycar"),
    # Combat
    "ufc":    ("mma", "ufc"),
    "mma":    ("mma", "ufc"),
}

# ---------------------------------------------------------------------------
# MLB team name fragments → ballpark city (for weather context)
# Weather is the single biggest external factor in MLB prediction markets:
# wind direction affects home runs, rain causes delays, cold suppresses offense.
# ---------------------------------------------------------------------------
_MLB_TEAM_CITIES = {
    "yankees": "new york",    "mets": "new york",
    "red sox": "boston",      "cubs": "chicago",
    "white sox": "chicago",   "dodgers": "los angeles",
    "angels": "los angeles",  "giants": "san francisco",
    "athletics": "oakland",   "padres": "san diego",
    "mariners": "seattle",    "astros": "houston",
    "rangers": "dallas",      "cardinals": "st louis",
    "brewers": "milwaukee",   "twins": "minneapolis",
    "tigers": "detroit",      "indians": "cleveland",
    "guardians": "cleveland", "reds": "cincinnati",
    "pirates": "pittsburgh",  "phillies": "philadelphia",
    "braves": "atlanta",      "marlins": "miami",
    "nationals": "washington","orioles": "baltimore",
    "rays": "tampa",          "blue jays": "toronto",
    "royals": "kansas city",  "rockies": "denver",
    "diamondbacks": "phoenix","padres": "san diego",
}

# ---------------------------------------------------------------------------
# Active PGA Tour venue cities by typical month (approximate)
# Used to fetch weather at the course location for golf markets.
# ---------------------------------------------------------------------------
_PGA_VENUE_BY_KEYWORD = {
    "masters": "augusta",
    "augusta": "augusta",
    "players": "ponte vedra",
    "us open": "los angeles",     # 2025 venue
    "open championship": "royal troon",
    "pga championship": "charlotte",
    "wells fargo": "charlotte",
    "colonial": "fort worth",
    "memorial": "columbus",
    "travelers": "hartford",
    "scottish open": "north berwick",
    "genesis": "los angeles",
    "waste management": "phoenix",
    "farmers": "san diego",
    "att pebble": "monterey",
    "torrey": "san diego",
    "houston open": "houston",
    "valspar": "tampa",
    "zurich": "new orleans",
    "rbc heritage": "hilton head",
    "new orleans": "new orleans",
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
        bls_api_key: str = "",
    ) -> None:
        self._odds_key      = odds_api_key
        self._fred_key      = fred_api_key
        self._newsapi_key   = newsapi_key
        self._metaculus_key = metaculus_api_key
        self._bls_key       = bls_api_key

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
            sport_type = sport_slug[0]  # "golf", "racing", "tennis", "baseball", etc.
            if sport_type == "golf":
                tasks["golf"] = self._get_golf_context(market.title)
            elif sport_type == "racing":
                tasks["racing"] = self._get_racing_context(market.title, sport_slug[1])
            else:
                tasks["espn"] = self._get_sports_context(market.title, sport_slug)
                tasks["odds"] = self._get_odds_context(market.title, sport_slug[1])

            # For baseball, also fetch ballpark weather — it directly affects outcomes
            if sport_type == "baseball" or any(k in category for k in ("mlb", "baseball")):
                tasks["ballpark_wx"] = self._get_mlb_weather_context(market.title)

        if "weather" in category or "temperature" in title or "rain" in title:
            tasks["weather"] = self._get_weather_context(market.title)

        if "crypto" in category or any(k in title for k in _CRYPTO_MAP):
            tasks["crypto"]  = self._get_crypto_context(market.title)

        if any(k in category for k in ("politics", "election")):
            tasks["meta"]      = self._get_metaculus_context(market.title)
            tasks["predictit"] = self._get_predictit_context(market.title)

        # F1 gets dedicated Jolpica data (more detailed than ESPN)
        if sport_slug and sport_slug[1] in ("f1", "indycar"):
            tasks["f1"] = self._get_f1_context(market.title)

        # Cross-reference on Polymarket + Manifold for all markets
        tasks["poly"]    = self._get_polymarket_context(market.title)
        tasks["manifold"] = self._get_manifold_context(market.title)

        # NWS weather for outdoor US sports (more authoritative than Open-Meteo)
        if sport_slug and sport_slug[0] in ("baseball", "golf", "racing"):
            tasks["nws"] = self._get_nws_context(market.title)

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
        _sport, league = sport_slug  # sport unused — ESPN uses league slug directly
        espn = ESPNClient()
        lines: List[str] = []

        scores = await espn.get_live_scores(league)
        if scores:
            title_lower = title.lower()
            # Show all games, or filter to ones matching the market title
            relevant = [
                g for g in scores
                if g["away_team"].lower() in title_lower
                or g["home_team"].lower() in title_lower
            ] or scores[:4]
            lines.append("ESPN live games:")
            for g in relevant[:5]:
                score_str = (
                    f"{g['away_score']}-{g['home_score']}"
                    if g.get("away_score", "") != "" else "scheduled"
                )
                lines.append(
                    f"  {g['away_team']} @ {g['home_team']}: "
                    f"{score_str} ({g['status']})"
                )

        injuries = await espn.get_injury_report(league)
        if injuries:
            lines.append("Key injuries:")
            for inj in injuries[:6]:
                lines.append(
                    f"  {inj['player']} ({inj['team']}) — {inj['status']}"
                )

        return "\n".join(lines) if lines else ""

    async def _get_odds_context(self, title: str, league: str) -> str:
        """Fetch Vegas consensus odds for the market's sport."""
        if not self._odds_key:
            return ""
        odds_client = OddsAPIClient(api_key=self._odds_key)

        # Map ESPN league slug to Odds API sport key
        sport_key_map = {
            # Basketball
            "nba":                     "basketball_nba",
            "mens-college-basketball": "basketball_ncaab",
            "wnba":                    "basketball_wnba",
            # Football
            "nfl":                     "americanfootball_nfl",
            # Baseball
            "mlb":                     "baseball_mlb",
            # Hockey
            "nhl":                     "icehockey_nhl",
            # Soccer
            "usa.1":                   "soccer_usa_mls",
            "eng.1":                   "soccer_epl",
            "uefa.champions":          "soccer_uefa_champs_league",
            # Tennis
            "atp":                     "tennis_atp",
            "wta":                     "tennis_wta",
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
    # Golf context — leaderboard + course weather
    # ------------------------------------------------------------------

    async def _get_golf_context(self, title: str) -> str:
        """
        Fetch PGA Tour leaderboard from ESPN + course weather from Open-Meteo.

        Golf prediction markets ("Will X win the Masters?", "Will someone
        shoot 59?") are highly weather-sensitive and leaderboard-dependent.
        Knowing the current leader, cut line, and wind speed at the course
        is the single most predictive data point available.
        """
        lines: List[str] = []
        title_lower = title.lower()

        # --- ESPN golf leaderboard ---
        try:
            espn = ESPNClient()
            leaders = await espn.get_live_scores("pga")
            if leaders:
                lines.append("PGA Tour leaderboard (ESPN):")
                for g in leaders[:8]:
                    away = g.get("away_team", "")
                    score = g.get("away_score", "")
                    status = g.get("status", "")
                    if away:
                        lines.append(f"  {away}: {score} ({status})")
        except Exception:
            pass

        # --- Course weather ---
        venue_city = None
        for keyword, city in _PGA_VENUE_BY_KEYWORD.items():
            if keyword in title_lower:
                venue_city = city
                break
        if not venue_city:
            venue_city = self._detect_city(title)

        if venue_city:
            try:
                wx = OpenMeteoClient()
                data = await wx.get_forecast_for_city(venue_city)
                if data:
                    daily  = data.get("daily", {})
                    times  = daily.get("time", [])
                    temps  = daily.get("temperature_2m_max", [])
                    wind   = daily.get("windspeed_10m_max", [])
                    precip = daily.get("precipitation_probability_max", [])
                    if times:
                        lines.append(f"Course weather ({venue_city.title()}):")
                        for i in range(min(3, len(times))):
                            t  = temps[i]  if i < len(temps)  else "?"
                            w  = wind[i]   if i < len(wind)   else "?"
                            r  = precip[i] if i < len(precip) else "?"
                            lines.append(
                                f"  {times[i]}: high {t}°F, "
                                f"wind {w} mph, precip {r}%"
                            )
            except Exception:
                pass

        return "\n".join(lines) if lines else ""

    # ------------------------------------------------------------------
    # Racing context — standings + circuit weather
    # ------------------------------------------------------------------

    async def _get_racing_context(self, title: str, league_slug: str) -> str:
        """
        Fetch F1/NASCAR/IndyCar driver standings + circuit weather.

        Championship-position markets ("Will Verstappen win the title?")
        need current standings. Race-winner markets need track conditions.
        """
        lines: List[str] = []
        title_lower = title.lower()

        # --- ESPN standings / live race ---
        try:
            espn = ESPNClient()
            results = await espn.get_live_scores(league_slug)
            if results:
                lines.append(f"ESPN {league_slug} standings/results:")
                for g in results[:6]:
                    away  = g.get("away_team", "")
                    home  = g.get("home_team", "")
                    score = g.get("away_score", "")
                    if away or home:
                        lines.append(f"  {away or home}: {score} ({g['status']})")
        except Exception:
            pass

        # --- Circuit weather (city detection from title) ---
        city = self._detect_city(title)
        if city:
            try:
                wx = OpenMeteoClient()
                data = await wx.get_forecast_for_city(city)
                if data:
                    daily  = data.get("daily", {})
                    times  = daily.get("time", [])
                    temps  = daily.get("temperature_2m_max", [])
                    wind   = daily.get("windspeed_10m_max", [])
                    precip = daily.get("precipitation_probability_max", [])
                    if times:
                        lines.append(f"Track weather ({city.title()}):")
                        for i in range(min(2, len(times))):
                            t = temps[i]  if i < len(temps)  else "?"
                            w = wind[i]   if i < len(wind)   else "?"
                            r = precip[i] if i < len(precip) else "?"
                            lines.append(
                                f"  {times[i]}: high {t}°F, "
                                f"wind {w} mph, precip {r}%"
                            )
            except Exception:
                pass

        return "\n".join(lines) if lines else ""

    # ------------------------------------------------------------------
    # MLB context — probable pitchers + standings + ballpark weather
    # ------------------------------------------------------------------

    async def _get_mlb_weather_context(self, title: str) -> str:
        """
        Full MLB context: probable starters + team standings + ballpark weather.

        Combines three free sources that collectively give the AI the most
        predictive MLB data available:

          1. **MLB Stats API** — probable pitchers (ERA/WHIP), today's scores,
             division standings. A Cy Young starter vs a AAA callup can swing
             a Kalshi run-total market by 15–20¢.

          2. **Open-Meteo** — ballpark weather. Wind blowing out at Wrigley
             increases home-run rates measurably. Rain probability determines
             whether a game gets delayed (relevant for "played today?" markets).

        All sources are free with no auth required.
        """
        title_lower = title.lower()
        lines: List[str] = []

        # --- MLB Stats API: probable pitchers + today's games ---
        try:
            mlb = MLBStatsClient()
            games, pitchers = await asyncio.gather(
                mlb.get_todays_games(),
                mlb.get_probable_pitchers(),
                return_exceptions=True,
            )
            if isinstance(games, Exception):
                games = []
            if isinstance(pitchers, Exception):
                pitchers = []

            mlb_context = mlb.format_context(
                games, pitchers, title_filter=title
            )
            if mlb_context:
                lines.append(mlb_context)

            # Division standings for the teams in the title
            standings = await mlb.get_standings(league_id=103)  # AL
            nl_standings = await mlb.get_standings(league_id=104)  # NL
            all_standings = standings + nl_standings
            relevant = [
                s for s in all_standings
                if s["team"].lower() in title_lower
            ]
            if relevant:
                lines.append("Standings:")
                for s in relevant[:2]:
                    lines.append(
                        f"  {s['team']} ({s['division']}): "
                        f"{s['wins']}-{s['losses']} ({s['pct']}), "
                        f"{s['gb']} GB"
                    )
        except Exception:
            pass

        # --- Ballpark weather ---
        city = None
        for team, team_city in _MLB_TEAM_CITIES.items():
            if team in title_lower:
                city = team_city
                break
        if not city:
            city = self._detect_city(title)

        if city:
            try:
                wx = OpenMeteoClient()
                data = await wx.get_forecast_for_city(city)
                if data:
                    daily  = data.get("daily", {})
                    times  = daily.get("time", [])
                    temps  = daily.get("temperature_2m_max", [])
                    wind   = daily.get("windspeed_10m_max", [])
                    precip = daily.get("precipitation_probability_max", [])
                    if times:
                        lines.append(f"Ballpark weather ({city.title()}):")
                        for i in range(min(2, len(times))):
                            t = temps[i]  if i < len(temps)  else "?"
                            w = wind[i]   if i < len(wind)   else "?"
                            r = precip[i] if i < len(precip) else "?"
                            lines.append(
                                f"  {times[i]}: high {t}°F, "
                                f"wind {w} mph, precip {r}%"
                            )
            except Exception:
                pass

        return "\n".join(lines) if lines else ""

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
    # Manifold Markets cross-reference
    # ------------------------------------------------------------------

    async def _get_manifold_context(self, title: str) -> str:
        """Fetch Manifold Markets community probability as a second cross-reference."""
        mf = ManifoldClient()
        markets = await mf.search_markets(title, limit=2)
        if not markets:
            return ""
        lines = ["Manifold community odds (play-money):"]
        for m in markets:
            prob = m.get("probability")
            prob_str = f"{prob:.0%}" if prob is not None else "N/A"
            vol  = m.get("volume", 0)
            lines.append(
                f"  {m['question'][:80]}: {prob_str} YES "
                f"(M${vol:,.0f} vol)"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # PredictIt cross-reference (politics only)
    # ------------------------------------------------------------------

    async def _get_predictit_context(self, title: str) -> str:
        """Fetch PredictIt real-money political market prices."""
        pi = PredictItClient()
        # Use first 3 keywords from title as search
        keywords = " ".join(title.split()[:4])
        markets = await pi.search_markets(keywords)
        if not markets:
            return ""
        return pi.format_context(markets[:2], query=title)

    # ------------------------------------------------------------------
    # F1 context via Jolpica (more detailed than ESPN)
    # ------------------------------------------------------------------

    async def _get_f1_context(self, title: str) -> str:
        """Fetch F1 driver standings, last race result, and next race info."""
        f1 = JolpicaF1Client()
        standings, last_result, next_race = await asyncio.gather(
            f1.get_driver_standings(),
            f1.get_last_race_result(),
            f1.get_next_race(),
            return_exceptions=True,
        )
        if isinstance(standings, Exception):
            standings = []
        if isinstance(last_result, Exception):
            last_result = []
        if isinstance(next_race, Exception):
            next_race = None
        return f1.format_context(standings, last_result, next_race)

    # ------------------------------------------------------------------
    # NWS weather (US outdoor venues)
    # ------------------------------------------------------------------

    async def _get_nws_context(self, title: str) -> str:
        """
        Fetch NWS official forecast for the venue in the market title.

        Tries venue-name lookup first (fast, pre-resolved grid), then falls
        back to Open-Meteo if the venue isn't in the pre-resolved table.
        """
        nws = NWSClient()
        # Try pre-resolved venue grid first
        forecast = await nws.get_forecast_by_venue(title.lower())
        if forecast:
            return nws.format_context(forecast, venue=title[:40])
        return ""

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_sport(category: str, title: str) -> Optional[tuple]:
        """
        Return ``(sport, league_slug)`` ESPN slugs for *category* / *title*.

        Checks exact keyword matches first (most reliable), then falls back to
        semantic title patterns for ambiguous cases.
        """
        combined = f"{category} {title}"

        # Exact keyword match against _SPORT_MAP keys
        for keyword, slugs in _SPORT_MAP.items():
            if keyword in combined:
                return slugs

        # Semantic fallbacks for common title patterns
        if any(k in combined for k in ("birdie", "eagle", "par ", "bogey", "hole",
                                        "course", "round ", "green jacket", "cut line",
                                        "tour championship")):
            return _SPORT_MAP["golf"]

        if any(k in combined for k in ("grand slam", "set ", "ace ", "serve",
                                        "wimbledon", "roland garros", "us open tennis",
                                        "australian open", "french open")):
            return _SPORT_MAP["tennis"]

        if any(k in combined for k in ("lap ", "pit stop", "qualifying", "pole position",
                                        "circuit", "grand prix", "checkered")):
            return _SPORT_MAP["f1"]

        if any(k in combined for k in ("innings", "strikeout", "home run", "batting",
                                        "world series", "no-hitter", "shutout")):
            return _SPORT_MAP["mlb"]

        if any(k in combined for k in ("goal ", "penalty kick", "offsides",
                                        "clean sheet", "premier league", "champions league")):
            return _SPORT_MAP["soccer"]

        if any(k in combined for k in ("game", "match", "champion", "finals", "playoff")):
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
