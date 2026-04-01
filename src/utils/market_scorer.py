"""
Market scoring and ranking for the lean high-volume bot.

Scores every eligible market on a 0–100 scale across four dimensions so
the bot invests its AI budget only on the highest-quality setups:

  Volume   40 pts  Liquidity proxy — $100k+ gets a perfect score
  Category 30 pts  Historical win-rate from CategoryScorer seeded data
  Timing   20 pts  Hours-to-expiry sweet spot (1–24 h preferred)
  Spread   10 pts  Bid-ask quality — tight spreads score higher

Markets scoring below MIN_SCORE_TO_TRADE (default 45) are skipped before
any AI call is made, saving both API cost and latency.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from src.utils.database import Market
from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("market_scorer")

# ---------------------------------------------------------------------------
# Tuneable thresholds
# ---------------------------------------------------------------------------

#: Minimum volume (in dollars) for a market to be considered at all.
MIN_VOLUME = 10_000.0

#: Volume at which the full 40-point volume score is awarded.
FULL_VOLUME_SCORE_AT = 100_000.0

#: Markets below this composite score are not sent for AI analysis.
MIN_SCORE_TO_TRADE = 45.0

#: Hard-blocked categories (never trade regardless of score).
BLOCKED_CATEGORIES = frozenset({
    "econ", "economics", "cpi", "fed", "federal_reserve",
    "econ_macro", "macro", "inflation", "gdp", "jobs", "nfp",
    "employment", "unemployment", "pce",
    "entertainment", "celebrity",
})

#: Category override scores for categories with strong historical data.
#  Derived from CategoryScorer seeded win-rate data.
CATEGORY_SCORE_OVERRIDES = {
    "ncaab":    74,
    "ncaa":     65,
    "nba":      55,
    "nfl":      52,
    "sports":   50,
    "politics": 60,
    "election": 62,
    "crypto":   45,
    "weather":  40,
    "markets":  40,
    "other":    35,
}


@dataclass
class MarketScore:
    """Composite score and breakdown for a single market."""

    market: Market
    total_score: float = 0.0
    volume_score: float = 0.0
    category_score: float = 0.0
    timing_score: float = 0.0
    spread_score: float = 0.0
    hours_to_expiry: float = 0.0
    category: str = "other"
    blocked: bool = False
    block_reason: str = ""

    def __post_init__(self) -> None:
        self.total_score = round(
            self.volume_score
            + self.category_score
            + self.timing_score
            + self.spread_score,
            2,
        )

    @property
    def passes(self) -> bool:
        """True when this market should be sent for AI analysis."""
        return not self.blocked and self.total_score >= MIN_SCORE_TO_TRADE


class MarketScorer:
    """
    Scores and ranks a list of :class:`~src.utils.database.Market` objects.

    The scorer is stateless — instantiate once and call :meth:`rank_markets`
    as many times as needed.

    Args:
        min_volume:          Hard volume floor below which markets are blocked.
        full_volume_at:      Volume level that yields a perfect volume score.
        min_score_to_trade:  Composite score threshold; markets below this are
                             excluded from AI analysis.
    """

    def __init__(
        self,
        min_volume: float = MIN_VOLUME,
        full_volume_at: float = FULL_VOLUME_SCORE_AT,
        min_score_to_trade: float = MIN_SCORE_TO_TRADE,
    ) -> None:
        self.min_volume = min_volume
        self.full_volume_at = full_volume_at
        self.min_score_to_trade = min_score_to_trade

        # Try to load live category scores from CategoryScorer
        self._category_scores = dict(CATEGORY_SCORE_OVERRIDES)
        try:
            from src.strategies.category_scorer import CategoryScorer
            scorer = CategoryScorer()
            for cat, raw in scorer.category_scores.items():
                # raw is a dict with keys: win_count, total_count, total_pnl, trend_pnl
                if isinstance(raw, dict) and raw.get("total_count", 0) >= 5:
                    wins = raw.get("win_count", 0)
                    total = raw.get("total_count", 1)
                    win_rate = wins / total
                    roi = raw.get("total_pnl", 0) / max(1, total)
                    # Blend win-rate and ROI into a 0-100 score
                    computed = min(100, max(0, win_rate * 60 + roi * 40))
                    self._category_scores[cat.lower()] = computed
        except Exception:
            pass  # Fall back to static overrides

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank_markets(
        self,
        markets: List[Market],
        top_n: Optional[int] = None,
    ) -> List[MarketScore]:
        """
        Score and rank *markets*, returning only those that pass the threshold.

        Args:
            markets: List of :class:`Market` objects from the database.
            top_n:   If set, return at most this many top-scoring markets.

        Returns:
            List of :class:`MarketScore` objects sorted by descending total
            score, filtered to ``passes == True``.
        """
        scores = [self.score_market(m) for m in markets]
        eligible = [s for s in scores if s.passes]
        eligible.sort(key=lambda s: s.total_score, reverse=True)

        logger.info(
            "Market scoring complete",
            total=len(markets),
            eligible=len(eligible),
            blocked=len(scores) - len(eligible),
            top_score=round(eligible[0].total_score, 1) if eligible else 0,
        )

        return eligible[:top_n] if top_n else eligible

    def score_market(self, market: Market) -> MarketScore:
        """
        Compute the composite score for a single *market*.

        Args:
            market: :class:`Market` dataclass from the database.

        Returns:
            :class:`MarketScore` with per-dimension breakdown.
        """
        category = (market.category or "other").lower().strip()

        # --- Hard blocks ---
        if market.volume < self.min_volume:
            return MarketScore(
                market=market,
                category=category,
                blocked=True,
                block_reason=f"volume ${market.volume:,.0f} < min ${self.min_volume:,.0f}",
            )

        if category in BLOCKED_CATEGORIES:
            return MarketScore(
                market=market,
                category=category,
                blocked=True,
                block_reason=f"category '{category}' is hard-blocked",
            )

        cat_score_raw = self._category_scores.get(category, 35)
        if cat_score_raw < 30:
            return MarketScore(
                market=market,
                category=category,
                blocked=True,
                block_reason=f"category '{category}' score {cat_score_raw:.0f} < 30 threshold",
            )

        # --- Score dimensions ---
        volume_score   = self._score_volume(market.volume)
        category_score = self._score_category(cat_score_raw)
        hours           = self._hours_to_expiry(market)
        timing_score   = self._score_timing(hours)
        spread_score   = self._score_spread(market)

        return MarketScore(
            market=market,
            volume_score=volume_score,
            category_score=category_score,
            timing_score=timing_score,
            spread_score=spread_score,
            hours_to_expiry=hours,
            category=category,
        )

    # ------------------------------------------------------------------
    # Dimension scorers (0–max_pts each)
    # ------------------------------------------------------------------

    def _score_volume(self, volume: float) -> float:
        """
        Award 0–40 points based on trading volume.

        Uses a logarithmic scale so a $10k market scores ~13 pts and a
        $1M market scores the full 40 pts.
        """
        if volume <= 0:
            return 0.0
        log_ratio = math.log10(volume / self.min_volume) / math.log10(
            self.full_volume_at / self.min_volume
        )
        return round(min(40.0, max(0.0, log_ratio * 40.0)), 2)

    def _score_category(self, raw_score: float) -> float:
        """Award 0–30 points proportional to the category's historical win score."""
        return round(min(30.0, max(0.0, (raw_score / 100.0) * 30.0)), 2)

    def _score_timing(self, hours_to_expiry: float) -> float:
        """
        Award 0–20 points based on hours remaining before expiry.

        Sweet spot is 1–24 hours: enough time for price movement but close
        enough for information to be actionable.

        Points:
          < 0.5 h  →  2  (illiquid, skip)
          0.5–1 h  →  8
          1–4 h    → 20  (high-confidence near-expiry zone)
          4–24 h   → 15
          24–72 h  →  8
          72–168 h →  4
          > 168 h  →  0
        """
        if hours_to_expiry < 0.5:
            return 2.0
        if hours_to_expiry < 1.0:
            return 8.0
        if hours_to_expiry < 4.0:
            return 20.0
        if hours_to_expiry < 24.0:
            return 15.0
        if hours_to_expiry < 72.0:
            return 8.0
        if hours_to_expiry < 168.0:
            return 4.0
        return 0.0

    def _score_spread(self, market: Market) -> float:
        """
        Award 0–10 points based on bid-ask spread quality.

        A tight spread indicates a liquid, efficiently priced market.
        """
        try:
            yes_spread = abs(market.yes_price - (1.0 - market.no_price))
            if yes_spread < 0.0:
                yes_spread = 0.0
            if yes_spread < 0.02:
                return 10.0
            if yes_spread < 0.04:
                return 7.0
            if yes_spread < 0.07:
                return 4.0
            if yes_spread < 0.12:
                return 2.0
            return 0.0
        except (TypeError, ValueError, AttributeError):
            return 3.0  # Unknown spread → neutral score

    @staticmethod
    def _hours_to_expiry(market: Market) -> float:
        """Return hours remaining before *market* expires (0 if already expired)."""
        try:
            if market.expiration_ts and market.expiration_ts > 0:
                delta = market.expiration_ts - datetime.now().timestamp()
                return max(0.0, delta / 3600.0)
        except (TypeError, ValueError, OSError):
            pass
        return 48.0  # Unknown expiry → assume 48 h
