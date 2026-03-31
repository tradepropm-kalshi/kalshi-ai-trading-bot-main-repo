"""
Database manager for the Kalshi trading system.
NOW INCLUDES PHASE STATE TRACKING + all missing methods required by the bot.
"""

import aiosqlite
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any

from src.utils.logging_setup import TradingLoggerMixin
from src.config.settings import settings


@dataclass
class Market:
    market_id: str
    title: str
    yes_price: float
    no_price: float
    volume: int
    expiration_ts: int
    category: str
    status: str
    last_updated: datetime
    has_position: bool = False


@dataclass
class Position:
    market_id: str
    side: str
    entry_price: float
    quantity: int
    timestamp: datetime
    rationale: Optional[str] = None
    confidence: Optional[float] = None
    live: bool = False
    status: str = "open"
    id: Optional[int] = None
    strategy: Optional[str] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    max_hold_hours: Optional[int] = None
    target_confidence_change: Optional[float] = None


@dataclass
class TradeLog:
    market_id: str
    side: str
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    entry_timestamp: datetime
    exit_timestamp: datetime
    rationale: str
    strategy: Optional[str] = None
    id: Optional[int] = None


@dataclass
class LLMQuery:
    timestamp: datetime
    strategy: str
    query_type: str
    prompt: str
    response: str
    market_id: Optional[str] = None
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    confidence_extracted: Optional[float] = None
    decision_extracted: Optional[str] = None
    id: Optional[int] = None


@dataclass
class PhaseState:
    current_phase_profit: float = 0.0
    total_secured_profit: float = 0.0
    last_reset: Optional[datetime] = None


class DatabaseManager(TradingLoggerMixin):
    def __init__(self, db_path: str = "trading_system.db"):
        self.db_path = db_path

    async def initialize(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await self._create_tables(db)
            await self._run_migrations(db)
            await db.commit()
        self.logger.info("Database initialized (with phase_state table)")

    async def _create_tables(self, db: aiosqlite.Connection) -> None:
        # markets table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS markets (
                market_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                yes_price REAL NOT NULL,
                no_price REAL NOT NULL,
                volume INTEGER NOT NULL,
                expiration_ts INTEGER NOT NULL,
                category TEXT NOT NULL,
                status TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                has_position BOOLEAN NOT NULL DEFAULT 0
            )
        """)

        # positions table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                rationale TEXT,
                confidence REAL,
                live BOOLEAN NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'open',
                strategy TEXT,
                stop_loss_price REAL,
                take_profit_price REAL,
                max_hold_hours INTEGER,
                target_confidence_change REAL,
                UNIQUE(market_id, side)
            )
        """)

        # trade_logs table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS trade_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                pnl REAL NOT NULL,
                entry_timestamp TEXT NOT NULL,
                exit_timestamp TEXT NOT NULL,
                rationale TEXT,
                strategy TEXT
            )
        """)

        # llm_queries table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS llm_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy TEXT NOT NULL,
                query_type TEXT NOT NULL,
                market_id TEXT,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                tokens_used INTEGER,
                cost_usd REAL,
                confidence_extracted REAL,
                decision_extracted TEXT
            )
        """)

        # daily_cost_tracking table (needed by evaluate.py)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS daily_cost_tracking (
                date TEXT PRIMARY KEY,
                total_cost REAL NOT NULL DEFAULT 0.0,
                request_count INTEGER NOT NULL DEFAULT 0
            )
        """)

        # PHASE STATE TABLE - core of your Bible phasing
        await db.execute("""
            CREATE TABLE IF NOT EXISTS phase_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                current_phase_profit REAL NOT NULL DEFAULT 0.0,
                total_secured_profit REAL NOT NULL DEFAULT 0.0,
                last_reset TEXT,
                phase_start_balance REAL NOT NULL DEFAULT 100.0
            )
        """)
        await db.execute("""
            INSERT OR IGNORE INTO phase_state (id, current_phase_profit, total_secured_profit, last_reset, phase_start_balance)
            VALUES (1, 0.0, 0.0, NULL, 100.0)
        """)

    async def _run_migrations(self, db: aiosqlite.Connection) -> None:
        pass

    # ====================== PHASE METHODS ======================
    async def get_phase_state(self) -> Dict:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM phase_state WHERE id = 1")
            row = await cursor.fetchone()
            return dict(row) if row else {"current_phase_profit": 0.0, "total_secured_profit": 0.0, "phase_start_balance": 100.0}

    async def update_phase_profit(self, pnl: float) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("UPDATE phase_state SET current_phase_profit = current_phase_profit + ? WHERE id = 1", (pnl,))
            await db.commit()

    async def secure_phase_profit(self, secure_amount: float) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE phase_state 
                SET total_secured_profit = total_secured_profit + ?,
                    current_phase_profit = 0.0,
                    last_reset = ?
                WHERE id = 1
            """, (secure_amount, datetime.now().isoformat()))
            await db.commit()

    # ====================== MISSING METHODS ADDED ======================
    async def get_markets_with_positions(self) -> set:
        """Returns set of market_ids that have open positions."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT DISTINCT market_id FROM positions WHERE status = 'open'")
            rows = await cursor.fetchall()
            return {row['market_id'] for row in rows}

    async def upsert_markets(self, markets: List[Market]) -> None:
        """Upsert markets (used by ingest.py)."""
        async with aiosqlite.connect(self.db_path) as db:
            for m in markets:
                await db.execute("""
                    INSERT OR REPLACE INTO markets 
                    (market_id, title, yes_price, no_price, volume, expiration_ts, category, status, last_updated, has_position)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (m.market_id, m.title, m.yes_price, m.no_price, m.volume, m.expiration_ts,
                      m.category, m.status, m.last_updated.isoformat(), m.has_position))
            await db.commit()

    async def update_position_to_live(self, position_id: int, fill_price: float) -> None:
        """Mark position as live (used by execute.py)."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE positions 
                SET live = 1, entry_price = ?, status = 'open'
                WHERE id = ?
            """, (fill_price, position_id))
            await db.commit()

    async def log_llm_query(self, llm_query: LLMQuery) -> None:
        """Log LLM query (used by xai_client and openrouter_client)."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO llm_queries 
                (timestamp, strategy, query_type, market_id, prompt, response, tokens_used, cost_usd, confidence_extracted, decision_extracted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (llm_query.timestamp.isoformat(), llm_query.strategy, llm_query.query_type,
                  llm_query.market_id, llm_query.prompt, llm_query.response,
                  llm_query.tokens_used, llm_query.cost_usd,
                  llm_query.confidence_extracted, llm_query.decision_extracted))
            await db.commit()

    async def upsert_daily_cost(self, cost: float) -> None:
        """Upsert daily AI cost (used by xai_client)."""
        today = datetime.now().strftime("%Y-%m-%d")
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO daily_cost_tracking (date, total_cost, request_count)
                VALUES (?, ?, 1)
                ON CONFLICT(date) DO UPDATE SET 
                    total_cost = total_cost + ?,
                    request_count = request_count + 1
            """, (today, cost, cost))
            await db.commit()

    # ====================== ORIGINAL METHODS ======================
    async def get_open_live_positions(self) -> List[Position]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM positions WHERE status = 'open'")
            rows = await cursor.fetchall()
            return [Position(**dict(row)) for row in rows]

    async def add_trade_log(self, trade_log: TradeLog) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO trade_logs (market_id, side, entry_price, exit_price, quantity, pnl, entry_timestamp, exit_timestamp, rationale, strategy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (trade_log.market_id, trade_log.side, trade_log.entry_price, trade_log.exit_price,
                  trade_log.quantity, trade_log.pnl, trade_log.entry_timestamp.isoformat(),
                  trade_log.exit_timestamp.isoformat(), trade_log.rationale, trade_log.strategy))
            await db.commit()

    async def get_all_trade_logs(self) -> List[TradeLog]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM trade_logs ORDER BY exit_timestamp")
            rows = await cursor.fetchall()
            return [TradeLog(**dict(row)) for row in rows]

    async def get_eligible_markets(self, volume_min: int = 20000, max_days_to_expiry: int = 365) -> List[Market]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT * FROM markets 
                WHERE volume >= ? AND status = 'open'
                ORDER BY volume DESC LIMIT 50
            """, (volume_min,))
            rows = await cursor.fetchall()
            return [Market(**dict(row)) for row in rows]

    async def close(self):
        pass