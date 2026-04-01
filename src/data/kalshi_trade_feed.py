"""
Kalshi Real-Time Order Flow Scanner

Polls the public GET /trades endpoint every 3 seconds to detect smart-money
signals without needing any user identity (Kalshi is centralised, not blockchain).

Signal types detected
---------------------
block       — Single trade with notional value > BLOCK_THRESHOLD ($200).
              Indicates institutional / whale entry.
momentum    — 3+ trades in the same direction within 30 seconds.
              Indicates directional conviction.
imbalance   — >70 % of trade volume on one side over a 60-second window.
              Indicates lopsided order flow.
velocity    — Current trades-per-minute > 3× the 5-minute baseline.
              Indicates sudden market interest spike.

Each detected signal is emitted as a ``BlockSignal`` dataclass and stored in
an internal deque.  Consumers (e.g. flow_copy_trade.py) call
``get_active_signals()`` to drain pending signals.

Signal learning
---------------
The scanner also maintains a per-signal-type outcome register.  After a trade
is closed the caller (flow_copy_trade.py) should call ``record_outcome()`` to
feed back the realised P&L.  The scanner uses an exponential moving average to
compute a ``signal_strength`` multiplier for each type, so successively
profitable signal types receive higher strength scores.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger("kalshi_trade_feed")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POLL_INTERVAL_S: float = 3.0          # REST poll cadence (< 20 req/s Basic limit)
BLOCK_THRESHOLD_USD: float = 200.0    # Notional $ for a "block trade"
MOMENTUM_WINDOW_S: float = 30.0       # Seconds to look back for momentum
MOMENTUM_MIN_TRADES: int = 3          # Same-direction trades to trigger momentum
IMBALANCE_WINDOW_S: float = 60.0      # Window for side-imbalance calculation
IMBALANCE_THRESHOLD: float = 0.70     # Fraction of volume on dominant side
VELOCITY_WINDOW_S: float = 60.0       # Current activity window
VELOCITY_BASELINE_S: float = 300.0    # Baseline (5 min) for rate comparison
VELOCITY_MULTIPLIER: float = 3.0      # Current rate must be > baseline × this
SIGNAL_TTL_S: float = 120.0           # Active signals expire after 2 minutes
MAX_ACTIVE_SIGNALS: int = 20          # Cap on pending signal queue

# Exponential moving average alpha for signal-strength learning
EMA_ALPHA: float = 0.2

# Kalshi public REST base (no auth needed for /trades)
KALSHI_API_BASE = "https://trading.kalshi.com/trade-api/rest/v2"
KALSHI_DEMO_BASE = "https://demo-api.kalshi.com/trade-api/rest/v2"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RawTrade:
    """One row from GET /trades."""
    trade_id: str
    market_id: str
    yes_price: float      # cents (0-99) → we normalise to 0-1
    count: int            # number of contracts
    taker_side: str       # "yes" | "no"
    created_ts: float     # unix timestamp


@dataclass
class BlockSignal:
    """
    A detected order-flow signal ready for the copy-trade executor.

    ``signal_strength`` is a 0-100 score blending raw size/velocity with the
    historical win-rate EMA for this signal type.
    """
    signal_type: str          # "block" | "momentum" | "imbalance" | "velocity"
    market_id: str
    direction: str            # "yes" | "no"
    total_size: int           # contracts in the signal window
    avg_price: float          # avg fill price 0-1
    signal_strength: float    # 0-100
    detected_at: float        # unix timestamp
    raw_trades: List[RawTrade] = field(default_factory=list)

    def age_seconds(self) -> float:
        return time.time() - self.detected_at

    def is_expired(self) -> bool:
        return self.age_seconds() > SIGNAL_TTL_S


@dataclass
class _MarketBuffer:
    """Rolling trade history for one market."""
    trades: Deque[RawTrade] = field(default_factory=lambda: deque(maxlen=500))
    baseline_counts: Deque[Tuple[float, int]] = field(
        default_factory=lambda: deque(maxlen=1000)
    )  # (timestamp, count) tuples for velocity baseline

    def prune(self, cutoff_ts: float) -> None:
        """Remove trades older than *cutoff_ts*."""
        while self.trades and self.trades[0].created_ts < cutoff_ts:
            self.trades.popleft()


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class KalshiTradeFeed:
    """
    Real-time order flow scanner over the Kalshi public trade feed.

    Usage::

        feed = KalshiTradeFeed(live=False)
        asyncio.create_task(feed.run())

        # in trading loop:
        signals = feed.get_active_signals()
    """

    def __init__(self, live: bool = False) -> None:
        self.live = live
        self._base = KALSHI_API_BASE if live else KALSHI_DEMO_BASE
        self._client = httpx.AsyncClient(base_url=self._base, timeout=10.0)

        # Per-market rolling buffers  {market_id → _MarketBuffer}
        self._buffers: Dict[str, _MarketBuffer] = defaultdict(_MarketBuffer)

        # Signals waiting to be consumed  (deque acts as a bounded FIFO queue)
        self._pending: Deque[BlockSignal] = deque(maxlen=MAX_ACTIVE_SIGNALS)

        # Signal-type learning: EMA of P&L outcomes  {signal_type → ema_pnl}
        self._signal_ema: Dict[str, float] = {
            "block": 0.0,
            "momentum": 0.0,
            "imbalance": 0.0,
            "velocity": 0.0,
        }
        # Outcome counts for confidence weighting
        self._signal_counts: Dict[str, int] = defaultdict(int)

        # Deduplication: market_ids with a recently-fired signal of each type
        self._fired_recently: Dict[str, float] = {}  # "{type}:{market}" → fired_ts
        _DEDUP_TTL_S = 60.0
        self._dedup_ttl = _DEDUP_TTL_S

        # Cursor for incremental polling
        self._cursor: Optional[str] = None

        self._running = False
        logger.info("KalshiTradeFeed initialised (live=%s)", live)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_active_signals(self) -> List[BlockSignal]:
        """
        Drain and return all non-expired pending signals.

        Called by the copy-trade strategy every cycle to pick up new signals.
        """
        now = time.time()
        active = [s for s in self._pending if not s.is_expired()]
        self._pending.clear()
        # Re-queue still-active signals (race-free because single-threaded asyncio)
        for s in active:
            self._pending.append(s)
        return list(self._pending)

    def drain_signals(self) -> List[BlockSignal]:
        """
        Consume and clear all pending signals (destructive).

        Preferred over ``get_active_signals`` inside the executor loop so each
        signal fires exactly once.
        """
        signals = [s for s in self._pending if not s.is_expired()]
        self._pending.clear()
        return signals

    def record_outcome(self, signal_type: str, pnl_pct: float) -> None:
        """
        Feed back the realised P&L percentage for a completed trade.

        Updates the EMA for the signal type so future ``signal_strength``
        scores reflect historical profitability.

        Args:
            signal_type: One of "block", "momentum", "imbalance", "velocity".
            pnl_pct: Realised return as a decimal (e.g. 0.12 for 12 %).
        """
        if signal_type not in self._signal_ema:
            return
        self._signal_counts[signal_type] += 1
        alpha = EMA_ALPHA
        self._signal_ema[signal_type] = (
            alpha * pnl_pct + (1 - alpha) * self._signal_ema[signal_type]
        )
        logger.debug(
            "Signal EMA updated type=%s pnl=%.3f new_ema=%.4f",
            signal_type, pnl_pct, self._signal_ema[signal_type],
        )

    def get_signal_performance(self) -> Dict[str, Dict]:
        """Return current EMA stats per signal type (for dashboard display)."""
        return {
            stype: {
                "ema_pnl": round(self._signal_ema[stype], 4),
                "trade_count": self._signal_counts[stype],
                "strength_bonus": self._ema_to_bonus(stype),
            }
            for stype in ("block", "momentum", "imbalance", "velocity")
        }

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Main poll loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Continuously poll GET /trades, detect signals, and populate the queue.

        Designed to run as a long-lived asyncio Task.  Errors are caught and
        logged; the loop resumes after a short backoff.
        """
        self._running = True
        logger.info("KalshiTradeFeed started polling every %.1fs", POLL_INTERVAL_S)
        consecutive_errors = 0

        while self._running:
            try:
                trades = await self._fetch_recent_trades()
                if trades:
                    self._ingest(trades)
                    self._detect_signals()
                consecutive_errors = 0
            except httpx.RequestError as exc:
                consecutive_errors += 1
                backoff = min(30.0, POLL_INTERVAL_S * consecutive_errors)
                logger.warning("Trade feed request error (%s); backoff %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                continue
            except Exception as exc:  # noqa: BLE001
                consecutive_errors += 1
                logger.error("Trade feed unexpected error: %s", exc, exc_info=True)
                await asyncio.sleep(min(30.0, POLL_INTERVAL_S * consecutive_errors))
                continue

            await asyncio.sleep(POLL_INTERVAL_S)

    # ------------------------------------------------------------------
    # REST polling
    # ------------------------------------------------------------------

    async def _fetch_recent_trades(self) -> List[RawTrade]:
        """
        Call GET /trades with an optional cursor for incremental fetching.

        Returns a list of new RawTrade objects sorted oldest-first.
        """
        params: Dict = {"limit": 200}
        if self._cursor:
            params["cursor"] = self._cursor

        try:
            resp = await self._client.get("/trades", params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning("GET /trades HTTP %s", exc.response.status_code)
            return []

        raw_list = data.get("trades", [])
        if not raw_list:
            return []

        # Advance cursor for next poll
        new_cursor = data.get("cursor")
        if new_cursor:
            self._cursor = new_cursor

        trades: List[RawTrade] = []
        for t in raw_list:
            try:
                yes_price_raw = t.get("yes_price") or t.get("yes_price_dollars", 0)
                # Normalise: values > 1.0 are in cents
                yes_price = yes_price_raw / 100.0 if yes_price_raw > 1.0 else float(yes_price_raw)

                trades.append(RawTrade(
                    trade_id=t.get("trade_id", "") or t.get("id", ""),
                    market_id=t.get("ticker", "") or t.get("market_id", ""),
                    yes_price=yes_price,
                    count=int(t.get("count", 0) or t.get("contracts", 0)),
                    taker_side=str(t.get("taker_side", "yes")).lower(),
                    created_ts=_parse_ts(t.get("created_time", "")),
                ))
            except (KeyError, TypeError, ValueError):
                continue

        # Sort oldest-first so buffers receive trades in chronological order
        trades.sort(key=lambda x: x.created_ts)
        return trades

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def _ingest(self, trades: List[RawTrade]) -> None:
        """Push new trades into per-market rolling buffers."""
        for trade in trades:
            if not trade.market_id:
                continue
            self._buffers[trade.market_id].trades.append(trade)

    # ------------------------------------------------------------------
    # Signal detection
    # ------------------------------------------------------------------

    def _detect_signals(self) -> None:
        """Scan all active market buffers and emit new signals."""
        now = time.time()
        self._clean_dedup(now)

        for market_id, buf in list(self._buffers.items()):
            # Prune old trades beyond longest window
            buf.prune(now - max(IMBALANCE_WINDOW_S, VELOCITY_BASELINE_S))

            if not buf.trades:
                continue

            recent = list(buf.trades)

            self._check_block(market_id, recent, now)
            self._check_momentum(market_id, recent, now)
            self._check_imbalance(market_id, recent, now)
            self._check_velocity(market_id, recent, now)

    def _check_block(
        self, market_id: str, trades: List[RawTrade], now: float
    ) -> None:
        """Detect any single trade exceeding BLOCK_THRESHOLD_USD notional."""
        for trade in trades:
            if trade.created_ts < now - MOMENTUM_WINDOW_S:
                continue
            notional = trade.count * trade.yes_price  # approximate $ value
            if notional >= BLOCK_THRESHOLD_USD:
                self._emit(
                    signal_type="block",
                    market_id=market_id,
                    direction=trade.taker_side,
                    trades=[trade],
                    now=now,
                )

    def _check_momentum(
        self, market_id: str, trades: List[RawTrade], now: float
    ) -> None:
        """Detect 3+ same-direction trades in the momentum window."""
        window = [t for t in trades if t.created_ts >= now - MOMENTUM_WINDOW_S]
        if len(window) < MOMENTUM_MIN_TRADES:
            return

        yes_streak = _longest_streak(window, "yes")
        no_streak = _longest_streak(window, "no")

        if yes_streak >= MOMENTUM_MIN_TRADES:
            streak_trades = _streak_trades(window, "yes", yes_streak)
            self._emit("momentum", market_id, "yes", streak_trades, now)
        elif no_streak >= MOMENTUM_MIN_TRADES:
            streak_trades = _streak_trades(window, "no", no_streak)
            self._emit("momentum", market_id, "no", streak_trades, now)

    def _check_imbalance(
        self, market_id: str, trades: List[RawTrade], now: float
    ) -> None:
        """Detect lopsided volume (>70%) in the imbalance window."""
        window = [t for t in trades if t.created_ts >= now - IMBALANCE_WINDOW_S]
        if len(window) < 5:  # need meaningful sample
            return

        yes_vol = sum(t.count for t in window if t.taker_side == "yes")
        no_vol = sum(t.count for t in window if t.taker_side == "no")
        total = yes_vol + no_vol
        if total == 0:
            return

        yes_frac = yes_vol / total
        no_frac = no_vol / total

        if yes_frac >= IMBALANCE_THRESHOLD:
            self._emit("imbalance", market_id, "yes", window, now)
        elif no_frac >= IMBALANCE_THRESHOLD:
            self._emit("imbalance", market_id, "no", window, now)

    def _check_velocity(
        self, market_id: str, trades: List[RawTrade], now: float
    ) -> None:
        """Detect a sudden spike in trade rate vs. the 5-minute baseline."""
        current = [t for t in trades if t.created_ts >= now - VELOCITY_WINDOW_S]
        baseline = [t for t in trades if t.created_ts >= now - VELOCITY_BASELINE_S]

        if not baseline:
            return

        current_rate = len(current) / VELOCITY_WINDOW_S * 60  # trades per minute
        baseline_rate = len(baseline) / VELOCITY_BASELINE_S * 60

        if baseline_rate < 1.0:
            return  # Not enough baseline activity to compare meaningfully

        if current_rate >= baseline_rate * VELOCITY_MULTIPLIER:
            # Dominant direction in current window
            yes_count = sum(1 for t in current if t.taker_side == "yes")
            direction = "yes" if yes_count >= len(current) / 2 else "no"
            self._emit("velocity", market_id, direction, current, now)

    # ------------------------------------------------------------------
    # Signal emission
    # ------------------------------------------------------------------

    def _emit(
        self,
        signal_type: str,
        market_id: str,
        direction: str,
        trades: List[RawTrade],
        now: float,
    ) -> None:
        """Create a BlockSignal and add it to the pending queue (dedup-aware)."""
        dedup_key = f"{signal_type}:{market_id}"
        if dedup_key in self._fired_recently:
            return  # Suppress duplicate within dedup TTL

        total_size = sum(t.count for t in trades)
        if total_size == 0:
            return

        prices = [t.yes_price for t in trades if t.yes_price > 0]
        avg_price = sum(prices) / len(prices) if prices else 0.0

        strength = self._compute_strength(signal_type, total_size, trades)

        signal = BlockSignal(
            signal_type=signal_type,
            market_id=market_id,
            direction=direction,
            total_size=total_size,
            avg_price=round(avg_price, 4),
            signal_strength=round(strength, 1),
            detected_at=now,
            raw_trades=trades,
        )

        self._pending.append(signal)
        self._fired_recently[dedup_key] = now

        logger.info(
            "SIGNAL %s %s dir=%s size=%d strength=%.1f",
            signal_type, market_id, direction, total_size, strength,
        )

    def _compute_strength(
        self, signal_type: str, total_size: int, trades: List[RawTrade]
    ) -> float:
        """
        Compute a 0-100 signal strength score.

        Components:
          - Base score from size/count
          - EMA bonus from historical profitability of this signal type
        """
        # Raw size component (sigmoid-like, saturates at ~500 contracts)
        size_score = min(60.0, total_size / 5.0)

        # Count component
        count_score = min(20.0, len(trades) * 4.0)

        # Historical profitability bonus (+/-20)
        ema_bonus = self._ema_to_bonus(signal_type)

        return max(0.0, min(100.0, size_score + count_score + ema_bonus))

    def _ema_to_bonus(self, signal_type: str) -> float:
        """
        Convert EMA P&L to a -20 … +20 bonus.

        0.10 (10% avg return) → +20 bonus.
        Negative EMA → penalty down to -20.
        """
        ema = self._signal_ema.get(signal_type, 0.0)
        # Scale: ±10% EMA → ±20 points
        return max(-20.0, min(20.0, ema * 200.0))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _clean_dedup(self, now: float) -> None:
        """Remove expired deduplication entries."""
        expired = [k for k, ts in self._fired_recently.items()
                   if now - ts > self._dedup_ttl]
        for k in expired:
            del self._fired_recently[k]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parse_ts(ts_str: str) -> float:
    """Parse ISO 8601 or unix float to a unix timestamp float."""
    if not ts_str:
        return time.time()
    if isinstance(ts_str, (int, float)):
        return float(ts_str)
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except (ValueError, AttributeError):
        return time.time()


def _longest_streak(trades: List[RawTrade], side: str) -> int:
    """Return the length of the longest consecutive streak of *side* in *trades*."""
    max_streak = 0
    streak = 0
    for t in trades:
        if t.taker_side == side:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def _streak_trades(trades: List[RawTrade], side: str, length: int) -> List[RawTrade]:
    """Return the last *length* consecutive *side* trades from the tail of *trades*."""
    result: List[RawTrade] = []
    for t in reversed(trades):
        if t.taker_side == side:
            result.append(t)
            if len(result) == length:
                break
        else:
            result.clear()
    result.reverse()
    return result
