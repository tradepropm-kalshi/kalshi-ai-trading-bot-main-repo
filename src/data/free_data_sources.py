"""
Free real-world data integrations for the lean trading bot.

This module gives the AI actual ground-truth information before it makes
a prediction — eliminating the LLM knowledge-cutoff problem that causes
most AI trading systems to fail.

The key insight: prediction markets move on *new information*. A model
told "the game starts in 2 hours and the home team is favored by 7.5
points with their starting QB injured" will outperform one that guesses
from memory every single time.

Data sources (all free-tier / no-auth-required where possible):

  Polymarket       — Cross-market probability calibration (no auth needed)
  Open-Meteo       — 7-day NWS/ECMWF weather forecasts (no auth needed)
  ESPN unofficial  — Live scores, injuries, standings (no auth needed)
  The Odds API     — Vegas consensus sports odds (free tier: 500 req/month)
  FRED             — Federal Reserve economic data (free API key)
  Metaculus        — Crowdsourced long-range predictions (free API key)
  CoinGecko        — Crypto prices + sentiment (free, no auth)
  NewsAPI          — Breaking news headlines (free tier: 100 req/day)
"""

import asyncio
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("free_data")

# ---------------------------------------------------------------------------
# Shared async HTTP client (reused across all sources)
# ---------------------------------------------------------------------------
_HTTP_TIMEOUT = 8.0   # seconds per request
_HEADERS = {"User-Agent": "KalshiBot/1.0 (prediction market research)"}


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=_HTTP_TIMEOUT, headers=_HEADERS, follow_redirects=True)


# ===========================================================================
# Polymarket — free probability calibration
# ===========================================================================

class PolymarketClient:
    """
    Fetch Polymarket markets matching a search term to calibrate Kalshi prices.

    No authentication required.  Rate limit: generous (public API).

    Usage::

        async with PolymarketClient() as pm:
            data = await pm.search_market("Super Bowl winner 2025")
    """

    BASE = "https://gamma-api.polymarket.com"

    async def search_market(self, query: str, limit: int = 3) -> List[Dict]:
        """
        Search for Polymarket markets matching *query*.

        Args:
            query: Natural language search string (e.g. "CPI March 2025").
            limit: Maximum number of results to return.

        Returns:
            List of market dicts with keys: ``question``, ``yes_price``,
            ``no_price``, ``volume``, ``end_date``.  Empty list on error.
        """
        try:
            async with _client() as http:
                r = await http.get(
                    f"{self.BASE}/markets",
                    params={"search": query, "limit": limit, "active": "true"},
                )
                if r.status_code != 200:
                    return []
                markets = r.json()
                if isinstance(markets, dict):
                    markets = markets.get("markets", [])
                results = []
                for m in markets[:limit]:
                    tokens = m.get("tokens", [])
                    yes_token = next((t for t in tokens if t.get("outcome") == "Yes"), None)
                    no_token  = next((t for t in tokens if t.get("outcome") == "No"),  None)
                    results.append({
                        "question":  m.get("question", ""),
                        "yes_price": float(yes_token.get("price", 0.5)) if yes_token else 0.5,
                        "no_price":  float(no_token.get("price",  0.5)) if no_token  else 0.5,
                        "volume":    float(m.get("volume", 0)),
                        "end_date":  m.get("end_date_iso", ""),
                        "source":    "polymarket",
                    })
                return results
        except Exception as exc:
            logger.debug(f"Polymarket search failed: {exc}")
            return []

    def format_context(self, markets: List[Dict]) -> str:
        """Format Polymarket results as a concise context string for the LLM."""
        if not markets:
            return ""
        lines = ["Polymarket consensus (external calibration):"]
        for m in markets:
            lines.append(
                f"  • {m['question'][:80]}: YES={m['yes_price']:.0%} "
                f"(vol ${m['volume']:,.0f})"
            )
        return "\n".join(lines)


# ===========================================================================
# Open-Meteo — free 7-day weather forecasts
# ===========================================================================

# Major US cities that appear in weather prediction markets
_CITY_COORDS: Dict[str, Dict[str, float]] = {
    "los angeles":   {"lat": 34.05,  "lon": -118.24},
    "la":            {"lat": 34.05,  "lon": -118.24},
    "new york":      {"lat": 40.71,  "lon": -74.01},
    "nyc":           {"lat": 40.71,  "lon": -74.01},
    "chicago":       {"lat": 41.88,  "lon": -87.63},
    "houston":       {"lat": 29.76,  "lon": -95.37},
    "phoenix":       {"lat": 33.45,  "lon": -112.07},
    "philadelphia":  {"lat": 39.95,  "lon": -75.17},
    "philly":        {"lat": 39.95,  "lon": -75.17},
    "san antonio":   {"lat": 29.42,  "lon": -98.49},
    "san diego":     {"lat": 32.72,  "lon": -117.16},
    "dallas":        {"lat": 32.78,  "lon": -96.80},
    "san jose":      {"lat": 37.34,  "lon": -121.89},
    "austin":        {"lat": 30.27,  "lon": -97.74},
    "jacksonville":  {"lat": 30.33,  "lon": -81.66},
    "fort worth":    {"lat": 32.75,  "lon": -97.33},
    "columbus":      {"lat": 39.96,  "lon": -82.99},
    "charlotte":     {"lat": 35.23,  "lon": -80.84},
    "indianapolis":  {"lat": 39.77,  "lon": -86.16},
    "san francisco": {"lat": 37.77,  "lon": -122.42},
    "sf":            {"lat": 37.77,  "lon": -122.42},
    "seattle":       {"lat": 47.61,  "lon": -122.33},
    "denver":        {"lat": 39.74,  "lon": -104.98},
    "washington":    {"lat": 38.91,  "lon": -77.04},
    "dc":            {"lat": 38.91,  "lon": -77.04},
    "nashville":     {"lat": 36.17,  "lon": -86.78},
    "oklahoma city": {"lat": 35.47,  "lon": -97.52},
    "el paso":       {"lat": 31.76,  "lon": -106.49},
    "boston":        {"lat": 42.36,  "lon": -71.06},
    "portland":      {"lat": 45.52,  "lon": -122.68},
    "miami":         {"lat": 25.77,  "lon": -80.19},
    "atlanta":       {"lat": 33.75,  "lon": -84.39},
    "minneapolis":   {"lat": 44.98,  "lon": -93.27},
    "tampa":         {"lat": 27.95,  "lon": -82.46},
    "new orleans":   {"lat": 29.95,  "lon": -90.07},
    "cleveland":     {"lat": 41.50,  "lon": -81.69},
    "pittsburgh":    {"lat": 40.44,  "lon": -79.99},
    "kansas city":   {"lat": 39.10,  "lon": -94.58},
    "cincinnati":    {"lat": 39.10,  "lon": -84.51},
    "salt lake city":{"lat": 40.76,  "lon": -111.89},
    "slc":           {"lat": 40.76,  "lon": -111.89},
    "memphis":       {"lat": 35.15,  "lon": -90.05},
    "richmond":      {"lat": 37.54,  "lon": -77.44},
    "louisville":    {"lat": 38.25,  "lon": -85.76},
    "buffalo":       {"lat": 42.89,  "lon": -78.88},
    "raleigh":       {"lat": 35.78,  "lon": -78.64},
    "detroit":       {"lat": 42.33,  "lon": -83.05},
}


class OpenMeteoClient:
    """
    Fetch NWS/ECMWF 7-day weather forecasts from Open-Meteo (no API key needed).

    Usage::

        wc = OpenMeteoClient()
        forecast = await wc.get_forecast_for_city("Los Angeles", target_date="2025-07-18")
    """

    BASE = "https://api.open-meteo.com/v1/forecast"

    async def get_forecast_for_city(
        self,
        city: str,
        target_date: Optional[str] = None,
        days: int = 7,
    ) -> Optional[Dict]:
        """
        Fetch temperature and precipitation forecast for *city*.

        Args:
            city:        City name (case-insensitive; matched against known coords).
            target_date: ISO date string ``YYYY-MM-DD`` for the forecast of interest.
            days:        Number of forecast days to fetch (1–16).

        Returns:
            Dict with keys: ``city``, ``date``, ``max_temp_f``, ``min_temp_f``,
            ``precip_mm``, ``weather_code``, ``source``.
            Returns ``None`` if the city is unknown or the API fails.
        """
        key = city.lower().strip()
        coords = _CITY_COORDS.get(key)
        if not coords:
            # Fuzzy match
            for k, v in _CITY_COORDS.items():
                if k in key or key in k:
                    coords = v
                    break
        if not coords:
            logger.debug(f"Open-Meteo: unknown city '{city}'")
            return None

        try:
            async with _client() as http:
                r = await http.get(self.BASE, params={
                    "latitude":  coords["lat"],
                    "longitude": coords["lon"],
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode",
                    "temperature_unit": "fahrenheit",
                    "forecast_days": min(days, 16),
                    "timezone": "America/New_York",
                })
                if r.status_code != 200:
                    return None
                data = r.json().get("daily", {})
                dates   = data.get("time", [])
                max_t   = data.get("temperature_2m_max", [])
                min_t   = data.get("temperature_2m_min", [])
                precip  = data.get("precipitation_sum", [])
                codes   = data.get("weathercode", [])

                if not dates:
                    return None

                # Find the target date (or return today's forecast)
                idx = 0
                if target_date and target_date in dates:
                    idx = dates.index(target_date)

                return {
                    "city":         city,
                    "date":         dates[idx] if idx < len(dates) else "",
                    "max_temp_f":   max_t[idx]  if idx < len(max_t)  else None,
                    "min_temp_f":   min_t[idx]  if idx < len(min_t)  else None,
                    "precip_mm":    precip[idx] if idx < len(precip) else None,
                    "weather_code": codes[idx]  if idx < len(codes)  else None,
                    "all_dates":    dates,
                    "all_max_temps":max_t,
                    "source":       "open-meteo (NWS/ECMWF)",
                }
        except Exception as exc:
            logger.debug(f"Open-Meteo failed for {city}: {exc}")
            return None

    def format_context(self, forecast: Optional[Dict]) -> str:
        """Format a forecast dict as a concise context string for the LLM."""
        if not forecast:
            return ""
        hi  = forecast.get("max_temp_f")
        lo  = forecast.get("min_temp_f")
        pre = forecast.get("precip_mm", 0)
        return (
            f"Weather forecast ({forecast['source']}): "
            f"{forecast['city']} on {forecast['date']}: "
            f"High {hi:.0f}°F / Low {lo:.0f}°F"
            + (f", Precipitation {pre:.1f}mm" if pre else "")
        )


# ===========================================================================
# ESPN unofficial API — sports scores, injuries, standings
# ===========================================================================

_ESPN_LEAGUE_MAP = {
    "nba":   "basketball/nba",
    "nfl":   "football/nfl",
    "ncaab": "basketball/mens-college-basketball",
    "ncaa":  "basketball/mens-college-basketball",
    "mlb":   "baseball/mlb",
    "nhl":   "hockey/nhl",
    "mls":   "soccer/usa.1",
    "nascar":"racing/nascar-cup",
    "pga":   "golf/pga",
    "ufc":   "mma/ufc",
}


class ESPNClient:
    """
    Fetch live scores, injuries, and standings from ESPN's unofficial API.

    No authentication required.  Rate limit: ~100 req/min (unofficial).

    Usage::

        ec = ESPNClient()
        scores = await ec.get_live_scores("nba")
        injuries = await ec.get_injury_report("nfl")
    """

    BASE = "https://site.api.espn.com/apis/site/v2/sports"

    async def get_live_scores(self, league: str) -> List[Dict]:
        """
        Fetch currently live or today's scheduled scores for *league*.

        Args:
            league: Sport league key (e.g. ``"nba"``, ``"nfl"``, ``"ncaab"``).

        Returns:
            List of game dicts with home/away team names, scores, and status.
        """
        path = _ESPN_LEAGUE_MAP.get(league.lower(), "")
        if not path:
            return []
        try:
            async with _client() as http:
                r = await http.get(f"{self.BASE}/{path}/scoreboard")
                if r.status_code != 200:
                    return []
                events = r.json().get("events", [])
                results = []
                for ev in events[:10]:
                    comps = ev.get("competitions", [{}])[0]
                    competitors = comps.get("competitors", [])
                    home = next((c for c in competitors if c.get("homeAway") == "home"), {})
                    away = next((c for c in competitors if c.get("homeAway") == "away"), {})
                    status = ev.get("status", {}).get("type", {})
                    results.append({
                        "home_team":  home.get("team", {}).get("displayName", "?"),
                        "away_team":  away.get("team", {}).get("displayName", "?"),
                        "home_score": home.get("score", ""),
                        "away_score": away.get("score", ""),
                        "status":     status.get("description", ""),
                        "completed":  status.get("completed", False),
                        "source":     "espn",
                    })
                return results
        except Exception as exc:
            logger.debug(f"ESPN scores failed for {league}: {exc}")
            return []

    async def get_injury_report(self, league: str, team_name: Optional[str] = None) -> List[Dict]:
        """
        Fetch injury report for *league*, optionally filtered by *team_name*.

        Args:
            league:    League key.
            team_name: If set, only return injuries for this team (partial match).

        Returns:
            List of injury dicts with player name, type, and status.
        """
        path = _ESPN_LEAGUE_MAP.get(league.lower(), "")
        if not path:
            return []
        try:
            async with _client() as http:
                r = await http.get(f"{self.BASE}/{path}/injuries")
                if r.status_code != 200:
                    return []
                data = r.json()
                injuries = []
                for team_data in data.get("injuries", []):
                    team = team_data.get("team", {}).get("displayName", "")
                    if team_name and team_name.lower() not in team.lower():
                        continue
                    for player in team_data.get("injuries", [])[:5]:
                        injuries.append({
                            "team":   team,
                            "player": player.get("athlete", {}).get("displayName", "?"),
                            "type":   player.get("type", ""),
                            "status": player.get("status", ""),
                            "source": "espn",
                        })
                return injuries[:15]
        except Exception as exc:
            logger.debug(f"ESPN injuries failed for {league}: {exc}")
            return []

    def format_scores_context(self, scores: List[Dict]) -> str:
        """Format score list as context string."""
        if not scores:
            return ""
        lines = ["Live/recent scores (ESPN):"]
        for g in scores[:5]:
            if g["home_score"] or g["away_score"]:
                lines.append(
                    f"  • {g['away_team']} {g['away_score']} @ "
                    f"{g['home_team']} {g['home_score']} — {g['status']}"
                )
            else:
                lines.append(f"  • {g['away_team']} @ {g['home_team']} ({g['status']})")
        return "\n".join(lines)

    def format_injury_context(self, injuries: List[Dict]) -> str:
        """Format injury list as context string."""
        if not injuries:
            return ""
        lines = ["Key injuries (ESPN):"]
        for inj in injuries[:5]:
            lines.append(
                f"  • [{inj['team']}] {inj['player']} — {inj['type']} ({inj['status']})"
            )
        return "\n".join(lines)


# ===========================================================================
# The Odds API — Vegas consensus sports betting lines
# ===========================================================================

class OddsAPIClient:
    """
    Fetch Vegas consensus sports odds via The Odds API (free tier: 500 req/month).

    Requires env var ``ODDS_API_KEY``.  Free tier is sufficient for the lean
    bot's scanning frequency (~200 calls/month at 2-hour cycles).

    Usage::

        oc = OddsAPIClient(api_key="...")
        odds = await oc.get_event_odds("basketball_nba", "Golden State Warriors")
    """

    BASE = "https://api.the-odds-api.com/v4"
    _SPORT_MAP = {
        "nba":   "basketball_nba",
        "nfl":   "americanfootball_nfl",
        "ncaab": "basketball_ncaab",
        "mlb":   "baseball_mlb",
        "nhl":   "icehockey_nhl",
    }

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key or ""

    async def get_event_odds(
        self,
        league: str,
        team_name: Optional[str] = None,
    ) -> List[Dict]:
        """
        Fetch moneyline/spread odds for *league*, filtered by *team_name*.

        Args:
            league:    League key (``"nba"``, ``"nfl"``, etc.).
            team_name: Partial team name to filter on (e.g. ``"Warriors"``).

        Returns:
            List of game odds dicts, or empty list if API key missing/failed.
        """
        if not self.api_key:
            return []
        sport = self._SPORT_MAP.get(league.lower(), "")
        if not sport:
            return []
        try:
            async with _client() as http:
                r = await http.get(
                    f"{self.BASE}/sports/{sport}/odds",
                    params={
                        "apiKey": self.api_key,
                        "regions": "us",
                        "markets": "h2h,spreads",
                        "oddsFormat": "decimal",
                    },
                )
                if r.status_code != 200:
                    return []
                events = r.json()
                results = []
                for ev in events:
                    home = ev.get("home_team", "")
                    away = ev.get("away_team", "")
                    if team_name and (
                        team_name.lower() not in home.lower()
                        and team_name.lower() not in away.lower()
                    ):
                        continue
                    # Grab first bookmaker's odds
                    bookmakers = ev.get("bookmakers", [])
                    if not bookmakers:
                        continue
                    bk = bookmakers[0]
                    markets = {m["key"]: m for m in bk.get("markets", [])}
                    h2h = markets.get("h2h", {})
                    outcomes = {o["name"]: o["price"] for o in h2h.get("outcomes", [])}
                    results.append({
                        "home_team":      home,
                        "away_team":      away,
                        "home_odds":      outcomes.get(home),
                        "away_odds":      outcomes.get(away),
                        "commence_time":  ev.get("commence_time", ""),
                        "bookmaker":      bk.get("title", ""),
                        "source":         "the-odds-api",
                    })
                return results[:3]
        except Exception as exc:
            logger.debug(f"OddsAPI failed for {league}: {exc}")
            return []

    def format_context(self, odds: List[Dict]) -> str:
        """Format odds as an implied-probability context string."""
        if not odds:
            return ""
        lines = ["Vegas implied probabilities (The Odds API):"]
        for ev in odds:
            ho = ev.get("home_odds")
            ao = ev.get("away_odds")
            if ho and ao:
                home_prob = 1 / ho
                away_prob = 1 / ao
                lines.append(
                    f"  • {ev['away_team']} ({away_prob:.1%}) "
                    f"@ {ev['home_team']} ({home_prob:.1%})"
                )
        return "\n".join(lines)


# ===========================================================================
# FRED — Federal Reserve economic data
# ===========================================================================

class FREDClient:
    """
    Fetch Federal Reserve economic data via the FRED API (free API key required).

    Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html

    Usage::

        fc = FREDClient(api_key="...")
        cpi = await fc.get_series_latest("CPIAUCSL")  # CPI
        ffr = await fc.get_series_latest("FEDFUNDS")  # Fed funds rate
    """

    BASE = "https://api.stlouisfed.org/fred"

    # Common series IDs for Kalshi economic markets
    COMMON_SERIES = {
        "cpi":          "CPIAUCSL",    # Consumer Price Index
        "core_cpi":     "CPILFESL",    # Core CPI (ex food/energy)
        "pce":          "PCEPI",       # PCE Price Index
        "fed_funds":    "FEDFUNDS",    # Federal Funds Rate
        "unemployment": "UNRATE",      # Unemployment Rate
        "nfp":          "PAYEMS",      # Nonfarm Payrolls
        "gdp":          "GDP",         # Gross Domestic Product
        "sp500":        "SP500",       # S&P 500 (daily)
    }

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key or ""

    async def get_series_latest(
        self, series_id: str, observations: int = 3
    ) -> Optional[Dict]:
        """
        Fetch the most recent *observations* data points for *series_id*.

        Args:
            series_id:    FRED series ID (e.g. ``"CPIAUCSL"``).
            observations: Number of recent data points to return.

        Returns:
            Dict with ``series_id``, ``values``, ``dates``, ``units``.
            Returns ``None`` if the API key is missing or the call fails.
        """
        if not self.api_key:
            return None
        try:
            async with _client() as http:
                r = await http.get(
                    f"{self.BASE}/series/observations",
                    params={
                        "series_id":    series_id,
                        "api_key":      self.api_key,
                        "file_type":    "json",
                        "sort_order":   "desc",
                        "limit":        observations,
                    },
                )
                if r.status_code != 200:
                    return None
                obs = r.json().get("observations", [])
                valid = [(o["date"], o["value"]) for o in obs if o["value"] != "."]
                if not valid:
                    return None
                dates, values = zip(*valid)
                return {
                    "series_id": series_id,
                    "dates":     list(dates),
                    "values":    [float(v) for v in values],
                    "source":    "FRED (Federal Reserve)",
                }
        except Exception as exc:
            logger.debug(f"FRED failed for {series_id}: {exc}")
            return None

    async def get_multiple_series(self, keys: List[str]) -> Dict[str, Optional[Dict]]:
        """
        Fetch multiple named series concurrently.

        Args:
            keys: List of short-name keys from ``COMMON_SERIES``
                  (e.g. ``["cpi", "fed_funds"]``).

        Returns:
            Dict mapping each key to its series result or ``None``.
        """
        series_ids = {k: self.COMMON_SERIES[k] for k in keys if k in self.COMMON_SERIES}
        tasks = {k: self.get_series_latest(sid) for k, sid in series_ids.items()}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        return {
            k: (r if not isinstance(r, Exception) else None)
            for k, r in zip(tasks.keys(), results)
        }

    def format_context(self, series_data: Dict[str, Optional[Dict]]) -> str:
        """Format FRED series data as a brief economic context string."""
        lines = []
        for key, data in series_data.items():
            if data and data.get("values"):
                latest = data["values"][0]
                date   = data["dates"][0]
                lines.append(f"  {key.upper()}: {latest:.2f} (as of {date})")
        if not lines:
            return ""
        return "FRED economic data (latest readings):\n" + "\n".join(lines)


# ===========================================================================
# Metaculus — crowdsourced long-range predictions
# ===========================================================================

class MetaculusClient:
    """
    Search Metaculus for questions related to a Kalshi market topic.

    Provides calibration probabilities from a large forecasting community.
    API key optional (higher rate limits with key).

    Get a free key at: https://www.metaculus.com/api2/

    Usage::

        mc = MetaculusClient(api_key="...")
        questions = await mc.search_questions("Federal Reserve rate hike 2025")
    """

    BASE = "https://www.metaculus.com/api2"

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key or ""
        self._headers = {**_HEADERS}
        if self.api_key:
            self._headers["Authorization"] = f"Token {self.api_key}"

    async def search_questions(
        self, query: str, limit: int = 3
    ) -> List[Dict]:
        """
        Search for Metaculus questions matching *query*.

        Args:
            query: Search term.
            limit: Maximum results.

        Returns:
            List of question dicts with ``title``, ``community_prediction``,
            ``resolution_criteria``, ``close_time``.
        """
        try:
            async with httpx.AsyncClient(
                timeout=_HTTP_TIMEOUT,
                headers=self._headers,
                follow_redirects=True,
            ) as http:
                r = await http.get(
                    f"{self.BASE}/questions/",
                    params={
                        "search":  query,
                        "limit":   limit,
                        "status":  "open",
                        "order_by": "-activity",
                    },
                )
                if r.status_code != 200:
                    return []
                results_raw = r.json().get("results", [])
                results = []
                for q in results_raw[:limit]:
                    cp = q.get("community_prediction", {})
                    full = cp.get("full", {}) if isinstance(cp, dict) else {}
                    results.append({
                        "title":       q.get("title", ""),
                        "probability": full.get("q2"),   # median community probability
                        "close_time":  q.get("close_time", ""),
                        "forecasters": q.get("number_of_forecasters", 0),
                        "source":      "metaculus",
                    })
                return results
        except Exception as exc:
            logger.debug(f"Metaculus search failed: {exc}")
            return []

    def format_context(self, questions: List[Dict]) -> str:
        """Format Metaculus results as a calibration context string."""
        if not questions:
            return ""
        lines = ["Metaculus community forecasts (calibration):"]
        for q in questions:
            prob = q.get("probability")
            prob_str = f"{prob:.1%}" if prob is not None else "N/A"
            lines.append(
                f"  • {q['title'][:80]}: {prob_str} "
                f"({q['forecasters']} forecasters)"
            )
        return "\n".join(lines)


# ===========================================================================
# CoinGecko — free crypto prices and sentiment
# ===========================================================================

_COINGECKO_IDS = {
    "btc":      "bitcoin",
    "bitcoin":  "bitcoin",
    "eth":      "ethereum",
    "ethereum": "ethereum",
    "sol":      "solana",
    "solana":   "solana",
    "bnb":      "binancecoin",
    "xrp":      "ripple",
    "doge":     "dogecoin",
    "ada":      "cardano",
}


class CoinGeckoClient:
    """
    Fetch crypto prices and 24-hour sentiment from CoinGecko (free, no auth).

    Usage::

        cg = CoinGeckoClient()
        data = await cg.get_price("bitcoin")
    """

    BASE = "https://api.coingecko.com/api/v3"

    async def get_prices(self, coins: List[str]) -> Dict[str, Dict]:
        """
        Fetch current prices for a list of coin tickers.

        Args:
            coins: List of coin tickers (e.g. ``["btc", "eth"]``).

        Returns:
            Dict mapping ticker to price data.
        """
        ids = [_COINGECKO_IDS.get(c.lower(), c.lower()) for c in coins]
        ids_str = ",".join(set(ids))
        try:
            async with _client() as http:
                r = await http.get(
                    f"{self.BASE}/simple/price",
                    params={
                        "ids": ids_str,
                        "vs_currencies": "usd",
                        "include_24hr_change": "true",
                        "include_market_cap": "true",
                    },
                )
                if r.status_code != 200:
                    return {}
                return r.json()
        except Exception as exc:
            logger.debug(f"CoinGecko failed: {exc}")
            return {}

    def format_context(self, data: Dict, coins: List[str]) -> str:
        """Format crypto price data as a context string."""
        if not data:
            return ""
        lines = ["Crypto prices (CoinGecko):"]
        for coin in coins:
            cg_id = _COINGECKO_IDS.get(coin.lower(), coin.lower())
            price_data = data.get(cg_id, {})
            price  = price_data.get("usd", 0)
            change = price_data.get("usd_24h_change", 0)
            if price:
                lines.append(
                    f"  {coin.upper()}: ${price:,.2f} "
                    f"({'▲' if change >= 0 else '▼'}{abs(change):.1f}% 24h)"
                )
        return "\n".join(lines) if len(lines) > 1 else ""


# ===========================================================================
# NewsAPI — breaking news headlines
# ===========================================================================

class NewsAPIClient:
    """
    Fetch relevant breaking news headlines from NewsAPI (free tier: 100 req/day).

    Get a free key at: https://newsapi.org/register

    Usage::

        nc = NewsAPIClient(api_key="...")
        articles = await nc.search_news("Federal Reserve interest rates")
    """

    BASE = "https://newsapi.org/v2"

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key or ""

    async def search_news(
        self, query: str, limit: int = 5
    ) -> List[Dict]:
        """
        Search for news articles matching *query*.

        Args:
            query: Search terms.
            limit: Maximum articles to return.

        Returns:
            List of article dicts with ``title``, ``description``, ``published_at``.
        """
        if not self.api_key:
            return []
        try:
            async with _client() as http:
                r = await http.get(
                    f"{self.BASE}/everything",
                    params={
                        "q":        query,
                        "apiKey":   self.api_key,
                        "language": "en",
                        "sortBy":   "publishedAt",
                        "pageSize": limit,
                    },
                )
                if r.status_code != 200:
                    return []
                articles = r.json().get("articles", [])
                return [
                    {
                        "title":        a.get("title", ""),
                        "description":  a.get("description", ""),
                        "published_at": a.get("publishedAt", ""),
                        "source":       a.get("source", {}).get("name", ""),
                    }
                    for a in articles[:limit]
                ]
        except Exception as exc:
            logger.debug(f"NewsAPI failed: {exc}")
            return []

    def format_context(self, articles: List[Dict]) -> str:
        """Format news articles as a brief context string."""
        if not articles:
            return ""
        lines = ["Recent news:"]
        for a in articles[:4]:
            ts = a["published_at"][:10] if a.get("published_at") else ""
            lines.append(f"  [{ts}] {a['title'][:100]}")
        return "\n".join(lines)
