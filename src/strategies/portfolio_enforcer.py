"""
Portfolio Enforcer — Original RyanFrigo category scoring + risk guardrails
Extended with Bible Phase effective capital ($100 base + current_phase_profit)
"""

import asyncio
import logging
from datetime import datetime
from typing import Tuple

import aiosqlite

from src.config.settings import settings
from src.strategies.category_scorer import CategoryScorer, infer_category, BLOCK_THRESHOLD, get_allocation_pct
from src.utils.database import DatabaseManager

logger = logging.getLogger(__name__)

class BlockedTradeError(Exception):
    pass

class PortfolioEnforcer:
    def __init__(self, db_path: str = "trading_system.db"):
        self.db_path = db_path
        self.db_manager = DatabaseManager(db_path)
        self.scorer = CategoryScorer(db_path)
        self._effective_capital = 100.0
        self._blocked_count = 0
        self._allowed_count = 0

    async def initialize(self) -> None:
        await self.db_manager.initialize()
        await self.scorer.initialize()
        await self._update_effective_capital()

    async def _update_effective_capital(self) -> None:
        if not settings.trading.phase_mode_enabled:
            self._effective_capital = 1000.0
            return

        phase = await self.db_manager.get_phase_state()
        base = getattr(settings.trading, 'phase_base_capital', 100.0)
        current_profit = phase.get('current_phase_profit', 0.0)
        self._effective_capital = base + current_profit
        logger.info(f"PHASE ENFORCER → effective capital updated to ${self._effective_capital:.2f}")

    async def check_trade(self, ticker: str, side: str, amount: float, title: str = "", category: str = None) -> Tuple[bool, str]:
        cat = category or infer_category(ticker, title)
        score = await self.scorer.get_score(cat)
        max_alloc_pct = get_allocation_pct(score)

        if score < BLOCK_THRESHOLD or max_alloc_pct == 0.0:
            reason = f"Category '{cat}' blocked (score {score:.1f})"
            await self._log_blocked(ticker, cat, side, amount, reason, score)
            self._blocked_count += 1
            return False, reason

        max_allowed = self._effective_capital * max_alloc_pct
        if amount > max_allowed:
            reason = f"Amount ${amount:.2f} exceeds phase max allowed ${max_allowed:.2f}"
            await self._log_blocked(ticker, cat, side, amount, reason, score)
            self._blocked_count += 1
            return False, reason

        self._allowed_count += 1
        return True, f"Trade allowed under phase capital ${self._effective_capital:.2f}"

    async def enforce(self, ticker: str, side: str, amount: float, title: str = "", category: str = None):
        allowed, reason = await self.check_trade(ticker, side, amount, title, category)
        if not allowed:
            raise BlockedTradeError(reason)

    async def _log_blocked(self, ticker: str, category: str, side: str, amount: float, reason: str, score: float):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO blocked_trades (ticker, category, side, amount, reason, score, blocked_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (ticker, category, side, amount, reason, score, datetime.now().isoformat()))
            await db.commit()