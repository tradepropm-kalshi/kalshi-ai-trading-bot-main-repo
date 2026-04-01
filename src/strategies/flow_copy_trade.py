"""
Flow Copy Trade Strategy — Beast Mode primary engine

Reads order-flow signals from KalshiTradeFeed, performs a fast AI confirmation
(< 200 tokens), sizes the position using Kelly criterion, and executes
immediately.

Design goals
------------
* Sub-second path from signal to order (IO-bound await chain only).
* 5–15 % profit target per trade, tight stop at −5 %.
* Signal-type learning: outcomes fed back to KalshiTradeFeed.record_outcome()
  so the strength scores improve over time.
* Hard caps: max BEAST_MAX_POSITIONS concurrent positions, max 12 % of balance
  per trade.

Usage (from bot orchestrator)::

    flow = FlowCopyTradeStrategy(db, kalshi, xai, feed, live=False)
    await flow.run_cycle()   # call every 60 s from beast loop
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import aiosqlite

from src.data.kalshi_trade_feed import BlockSignal, KalshiTradeFeed
from src.utils.database import DatabaseManager
from src.clients.kalshi_client import KalshiClient
from src.clients.xai_client import XAIClient

logger = logging.getLogger("flow_copy_trade")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BEAST_MAX_POSITIONS: int = 7          # absolute cap on open positions
MAX_POSITION_PCT: float = 0.12        # max 12 % of available balance per trade
MIN_SIGNAL_STRENGTH: float = 25.0     # ignore weak signals
TAKE_PROFIT_PCT: float = 0.10         # exit at +10 % (hits the 5-15% range)
STOP_LOSS_PCT: float = 0.05           # exit at −5 %
AI_CONFIRM_MIN_CONFIDENCE: float = 0.55  # skip if AI is less sure
MIN_CONTRACTS: int = 1
KELLY_FRACTION: float = 0.25          # fractional Kelly cap


# ---------------------------------------------------------------------------
# Internal state for one open flow trade
# ---------------------------------------------------------------------------

@dataclass
class FlowPosition:
    signal_id: int            # flow_signals row id
    signal_type: str
    market_id: str
    direction: str            # "yes" | "no"
    entry_price: float        # 0-1
    quantity: int
    take_profit: float        # 0-1
    stop_loss: float          # 0-1
    position_id: int          # positions table row id
    opened_at: float          # unix ts
    live: bool


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class FlowCopyTradeStrategy:
    """
    Beast Mode order-flow copy engine.

    Parameters
    ----------
    db : DatabaseManager
    kalshi : KalshiClient
    xai : XAIClient
    feed : KalshiTradeFeed  — the running trade-feed scanner
    live : bool             — True = real orders, False = paper
    position_cap : int      — override BEAST_MAX_POSITIONS
    """

    def __init__(
        self,
        db: DatabaseManager,
        kalshi: KalshiClient,
        xai: XAIClient,
        feed: KalshiTradeFeed,
        live: bool = False,
        position_cap: int = BEAST_MAX_POSITIONS,
    ) -> None:
        self.db = db
        self.kalshi = kalshi
        self.xai = xai
        self.feed = feed
        self.live = live
        self.position_cap = position_cap

        # Track open flow positions in-memory (fast exit checks)
        self._open: Dict[str, FlowPosition] = {}   # market_id → FlowPosition

    # ------------------------------------------------------------------
    # Main cycle (called every 60 s from beast loop)
    # ------------------------------------------------------------------

    async def run_cycle(self) -> Dict:
        """
        One cycle:
          1. Exit any positions that hit TP/SL.
          2. Drain new signals from the feed.
          3. AI-confirm and enter top signals (up to position cap).

        Returns a summary dict for logging / dashboard telemetry.
        """
        exits = await self._check_exits()
        entries = 0
        skipped = 0

        signals = self.feed.drain_signals()
        if signals and len(self._open) < self.position_cap:
            # Sort by signal_strength descending — take the strongest first
            signals.sort(key=lambda s: s.signal_strength, reverse=True)

            for sig in signals:
                if len(self._open) >= self.position_cap:
                    break
                if sig.market_id in self._open:
                    continue  # already have a position on this market
                if sig.signal_strength < MIN_SIGNAL_STRENGTH:
                    skipped += 1
                    continue

                entered = await self._try_enter(sig)
                if entered:
                    entries += 1
                else:
                    skipped += 1

        return {
            "exits": exits,
            "entries": entries,
            "skipped": skipped,
            "open_positions": len(self._open),
        }

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    async def _try_enter(self, sig: BlockSignal) -> bool:
        """
        AI-confirm the signal then size and place the order.

        Returns True if a position was successfully opened.
        """
        # Fast AI confirmation — tiny prompt, < 200 tokens
        confidence, rationale = await self._ai_confirm(sig)
        if confidence < AI_CONFIRM_MIN_CONFIDENCE:
            logger.info(
                "SKIP %s %s — AI confidence %.2f < threshold",
                sig.signal_type, sig.market_id, confidence,
            )
            return False

        # Size the position
        balance, quantity = await self._size_position(sig, confidence)
        if quantity < MIN_CONTRACTS:
            return False

        # Place order
        side = sig.direction  # "yes" or "no"
        price = sig.avg_price  # 0-1

        try:
            if self.live:
                order_result = await self.kalshi.place_order(
                    ticker=sig.market_id,
                    side=side,
                    type="limit",
                    count=quantity,
                    yes_price=int(price * 100),  # Kalshi uses cent-integers
                )
            else:
                # Paper mode — simulate fill
                order_result = {
                    "order": {
                        "order_id": f"paper_{sig.market_id}_{int(time.time())}",
                        "status": "resting",
                    }
                }
        except Exception as exc:
            logger.error("Order placement failed %s: %s", sig.market_id, exc)
            return False

        # Persist to positions table
        now = datetime.now().isoformat()
        take_profit = min(0.99, price * (1 + TAKE_PROFIT_PCT))
        stop_loss = max(0.01, price * (1 - STOP_LOSS_PCT))

        try:
            async with aiosqlite.connect(self.db.db_path) as db_conn:
                cur = await db_conn.execute(
                    """
                    INSERT OR IGNORE INTO positions
                        (market_id, side, entry_price, quantity, timestamp, rationale,
                         confidence, live, strategy, stop_loss_price, take_profit_price,
                         max_hold_hours, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'flow_copy_trade', ?, ?, 4, 'open')
                    """,
                    (
                        sig.market_id, side, price, quantity, now,
                        rationale[:500], confidence,
                        1 if self.live else 0,
                        stop_loss, take_profit,
                    ),
                )
                await db_conn.commit()
                position_id = cur.lastrowid
        except Exception as exc:
            logger.error("DB position insert failed: %s", exc)
            return False

        # Log signal to flow_signals table
        signal_id = await self.db.log_flow_signal(
            signal_type=sig.signal_type,
            market_id=sig.market_id,
            direction=sig.direction,
            signal_strength=sig.signal_strength,
            total_size=sig.total_size,
            avg_price=sig.avg_price,
            position_id=position_id,
        )

        # Track in-memory
        self._open[sig.market_id] = FlowPosition(
            signal_id=signal_id or 0,
            signal_type=sig.signal_type,
            market_id=sig.market_id,
            direction=side,
            entry_price=price,
            quantity=quantity,
            take_profit=take_profit,
            stop_loss=stop_loss,
            position_id=position_id,
            opened_at=time.time(),
            live=self.live,
        )

        logger.info(
            "ENTER %s %s dir=%s qty=%d @ %.3f  TP=%.3f SL=%.3f  strength=%.1f conf=%.2f",
            sig.signal_type, sig.market_id, side, quantity, price,
            take_profit, stop_loss, sig.signal_strength, confidence,
        )
        return True

    # ------------------------------------------------------------------
    # Exit management
    # ------------------------------------------------------------------

    async def _check_exits(self) -> int:
        """
        Check all open flow positions for TP/SL hits.  Exits and records
        outcome for signal learning.

        Returns the number of positions closed.
        """
        if not self._open:
            return 0

        closed = 0
        for market_id in list(self._open.keys()):
            pos = self._open[market_id]
            try:
                market = await self.kalshi.get_market(market_id)
                current_price = _extract_price(market, pos.direction)
            except Exception:
                continue

            hit_tp = current_price >= pos.take_profit
            hit_sl = current_price <= pos.stop_loss
            max_hold = (time.time() - pos.opened_at) > 4 * 3600  # 4-hr hard stop

            if hit_tp or hit_sl or max_hold:
                await self._close_position(pos, current_price, hit_tp, hit_sl)
                del self._open[market_id]
                closed += 1

        return closed

    async def _close_position(
        self,
        pos: FlowPosition,
        current_price: float,
        hit_tp: bool,
        hit_sl: bool,
    ) -> None:
        """Close a position, log the trade, and feed outcome back to scanner."""
        pnl_pct = (current_price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0.0
        outcome = "win" if pnl_pct > 0 else ("scratch" if abs(pnl_pct) < 0.01 else "loss")

        reason = "take_profit" if hit_tp else ("stop_loss" if hit_sl else "max_hold")
        logger.info(
            "EXIT %s %s reason=%s pnl=%.2f%%",
            pos.market_id, pos.direction, reason, pnl_pct * 100,
        )

        # Update positions table
        try:
            async with aiosqlite.connect(self.db.db_path) as db_conn:
                await db_conn.execute(
                    "UPDATE positions SET status = 'closed', fill_price = ? WHERE id = ?",
                    (current_price, pos.position_id),
                )
                await db_conn.commit()
        except Exception as exc:
            logger.warning("DB close update failed: %s", exc)

        # Record signal outcome (closes the loop for learning)
        if pos.signal_id:
            await self.db.close_flow_signal(pos.signal_id, outcome, pnl_pct)

        # Feed back to scanner EMA
        self.feed.record_outcome(pos.signal_type, pnl_pct)

    # ------------------------------------------------------------------
    # AI confirmation
    # ------------------------------------------------------------------

    async def _ai_confirm(self, sig: BlockSignal) -> Tuple[float, str]:
        """
        Ask the AI for a quick yes/no confirmation on the signal.

        Prompt is intentionally tiny (< 200 tokens) so this adds
        < $0.002 per signal check.
        """
        direction_label = "YES" if sig.direction == "yes" else "NO"
        prompt = (
            f"Kalshi market: {sig.market_id}\n"
            f"Signal: {sig.signal_type.upper()} — {sig.total_size} contracts "
            f"on {direction_label} side, avg price {sig.avg_price:.2f}, "
            f"strength {sig.signal_strength:.0f}/100.\n"
            f"Should I copy this trade? Reply: CONFIRM or SKIP, then one sentence why."
        )

        try:
            text = await self.xai.get_completion(
                prompt=prompt,
                max_tokens=80,
                strategy="flow_copy_trade",
                query_type="signal_confirm",
                market_id=sig.market_id,
            )
            if not text:
                return 0.6, "AI returned empty response — proceeding on signal strength"

            upper = text.upper()
            # Parse decision and extract any numeric confidence
            if "SKIP" in upper and "CONFIRM" not in upper:
                confidence = 0.40
            elif "CONFIRM" in upper:
                confidence = 0.72
            else:
                confidence = 0.55  # neutral

            # Boost confidence based on signal strength
            strength_bonus = (sig.signal_strength - 50) / 500  # ±0.10
            confidence = max(0.0, min(1.0, confidence + strength_bonus))

            return confidence, text[:300]
        except Exception as exc:
            logger.warning("AI confirm failed for %s: %s — proceeding with 0.6", sig.market_id, exc)
            return 0.6, "AI unavailable — proceeding on signal strength"

    # ------------------------------------------------------------------
    # Position sizing (Kelly)
    # ------------------------------------------------------------------

    async def _size_position(
        self, sig: BlockSignal, confidence: float
    ) -> Tuple[float, int]:
        """
        Kelly-based sizing capped at MAX_POSITION_PCT of balance.

        Returns (balance, quantity).
        """
        try:
            bal_resp = await self.kalshi.get_balance()
            balance = float(
                bal_resp.get("balance", 0) or bal_resp.get("available_balance", 0)
            )
            # Kalshi balance is in cents
            if balance > 10000:
                balance /= 100.0
        except Exception:
            balance = 500.0  # conservative fallback

        if balance <= 0:
            return 0.0, 0

        # Kelly: f* = (p - q) / b  where b = payoff ratio
        p = confidence
        q = 1 - p
        price = max(0.01, min(0.99, sig.avg_price))
        b = (1 - price) / price  # payoff: profit per dollar risked

        kelly = max(0.0, (p - q / b)) * KELLY_FRACTION

        # Cap at MAX_POSITION_PCT
        fraction = min(kelly, MAX_POSITION_PCT)
        dollar_amount = balance * fraction

        # Each contract costs sig.avg_price dollars (approx)
        quantity = max(MIN_CONTRACTS, int(dollar_amount / max(0.01, price)))

        return balance, quantity
