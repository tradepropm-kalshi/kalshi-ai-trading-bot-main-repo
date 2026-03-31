"""
Portfolio Enforcer — Runs before every trade scan.

Now fully phase-aware: uses effective_phase_capital = $100 base + current_phase_profit
for ALL risk/position checks while still respecting the original Kalshi balance for logging.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import aiosqlite

from src.config.settings import settings
from src.strategies.category_scorer import CategoryScorer, infer_category, BLOCK_THRESHOLD, get_allocation_pct
from src.utils.database import DatabaseManager

logger = logging.getLogger(__name__)


class BlockedTradeError(Exception):
    pass


class PortfolioEnforcer:
    def __init__(
        self,
        db_path: str = "trading_system.db",
        portfolio_value: float = 0.0,
        max_drawdown_pct: float = 0.15,
        max_position_pct: float = 0.03,
        max_sector_pct: float = 0.30,
    ):
        self.db_path = db_path
        self._db_manager = DatabaseManager(db_path)
        self.max_drawdown_pct = max_drawdown_pct
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.scorer = CategoryScorer(db_path)
        self._blocked_count = 0
        self._allowed_count = 0
        self._effective_portfolio_value = portfolio_value

    async def initialize(self) -> None:
        await self.scorer.initialize()
        await self._db_manager.initialize()

        if settings.trading.phase_mode_enabled:
            phase = await self._db_manager.get_phase_state()
            self._effective_portfolio_value = (
                settings.trading.phase_base_capital + phase.get("current_phase_profit", 0.0)
            )
            logger.info(f"PHASE MODE ACTIVE → effective capital = ${self._effective_portfolio_value:.2f} "
                        f"(base ${settings.trading.phase_base_capital:.2f} + phase profit ${phase.get('current_phase_profit', 0.0):.2f})")
        else:
            self._effective_portfolio_value = self._effective_portfolio_value or 1000.0

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS blocked_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    category TEXT NOT NULL,
                    side TEXT NOT NULL,
                    amount REAL NOT NULL,
                    reason TEXT NOT NULL,
                    score REAL,
                    blocked_at TEXT NOT NULL
                )
            """)
            await db.commit()

    async def check_trade(
        self,
        ticker: str,
        side: str,
        amount: float,
        title: str = "",
        category: Optional[str] = None,
        current_positions: Optional[Dict[str, float]] = None,
    ) -> Tuple[bool, str]:
        cat = category or infer_category(ticker, title)
        score = await self.scorer.get_score(cat)
        max_alloc = get_allocation_pct(score)

        effective_capital = self._effective_portfolio_value

        if score < BLOCK_THRESHOLD:
            reason = f"Category '{cat}' score {score:.1f} < {BLOCK_THRESHOLD} (blocked)"
            await self._log_blocked(ticker, cat, side, amount, reason, score)
            self._blocked_count += 1
            return False, reason

        if max_alloc == 0.0:
            reason = f"Category '{cat}' score {score:.1f} → 0% allocation"
            await self._log_blocked(ticker, cat, side, amount, reason, score)
            self._blocked_count += 1
            return False, reason

        if effective_capital > 0:
            max_allowed = effective_capital * max_alloc
            if amount > max_allowed:
                reason = (f"Trade ${amount:.2f} exceeds category max ${max_allowed:.2f} "
                          f"({max_alloc*100:.0f}% of phase capital ${effective_capital:.2f})")
                await self._log_blocked(ticker, cat, side, amount, reason, score)
                self._blocked_count += 1
                return False, reason

        if effective_capital > 0:
            max_single = effective_capital * self.max_position_pct
            if amount > max_single:
                reason = (f"Trade ${amount:.2f} exceeds max position size ${max_single:.2f} "
                          f"({self.max_position_pct*100:.0f}% of phase capital)")
                await self._log_blocked(ticker, cat, side, amount, reason, score)
                self._blocked_count += 1
                return False, reason

        if current_positions and effective_capital > 0:
            sector_exposure = sum(v for k, v in current_positions.items() if infer_category(k) == cat)
            if (sector_exposure + amount) / effective_capital > self.max_sector_pct:
                reason = f"Adding ${amount:.2f} would exceed {self.max_sector_pct*100:.0f}% sector limit"
                await self._log_blocked(ticker, cat, side, amount, reason, score)
                self._blocked_count += 1
                return False, reason

        self._allowed_count += 1
        return True, f"Trade allowed (phase capital=${effective_capital:.2f})"

    async def enforce(self, *args, **kwargs) -> None:
        allowed, reason = await self.check_trade(*args, **kwargs)
        if not allowed:
            raise BlockedTradeError(reason)

    async def get_blocked_trades(self, limit: int = 50) -> List[Dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM blocked_trades ORDER BY blocked_at DESC LIMIT ?", (limit,))
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_blocked_summary(self) -> Dict:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT category, COUNT(*) as count, SUM(amount) as total_amount
                FROM blocked_trades GROUP BY category ORDER BY count DESC
            """)
            rows = await cursor.fetchall()
        return {
            "by_category": [dict(r) for r in rows],
            "session_blocked": self._blocked_count,
            "session_allowed": self._allowed_count,
            "session_block_rate": self._blocked_count / max(1, self._blocked_count + self._allowed_count),
        }

    def reset_session_counts(self) -> None:
        self._blocked_count = 0
        self._allowed_count = 0

    async def _log_blocked(self, ticker: str, category: str, side: str, amount: float, reason: str, score: float):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO blocked_trades (ticker, category, side, amount, reason, score, blocked_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (ticker, category, side, amount, reason, score, datetime.now().isoformat()))
            await db.commit()