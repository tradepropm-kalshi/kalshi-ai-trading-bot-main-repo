"""
Database Manager for Kalshi AI Trading Bot

Full schema including:
  - Market, Position, TradeLog, LLMQuery dataclasses
  - Phase-profit state
  - Quick-flip persistent tracking table
  - Market-analysis deduplication
  - Daily AI-cost tracking
  - Daily P&L circuit-breaker helper
"""

import aiosqlite
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class Market:
    market_id: str
    title: str
    yes_price: float
    no_price: float
    volume: int
    expiration_ts: int
    category: str = "unknown"
    status: str = "active"
    last_updated: Optional[datetime] = None
    has_position: bool = False


@dataclass
class Position:
    market_id: str
    side: str
    entry_price: float
    quantity: int
    timestamp: datetime
    rationale: str
    confidence: float = 0.0
    live: bool = False
    strategy: str = "directional_trading"
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    max_hold_hours: Optional[int] = None
    target_confidence_change: Optional[float] = None
    id: Optional[int] = None
    status: str = "open"
    fill_price: Optional[float] = None


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
    rationale: str = ""
    id: Optional[int] = None


@dataclass
class LLMQuery:
    timestamp: str
    market_id: str
    query_type: str
    model: str
    cost: float
    response: str = ""
    id: Optional[int] = None
    # Extended fields used by xai_client._log_query — stored but not required
    strategy: Optional[str] = None
    prompt: Optional[str] = None
    tokens_used: Optional[int] = None
    confidence_extracted: Optional[float] = None
    decision_extracted: Optional[str] = None


@dataclass
class PhaseState:
    current_phase_profit: float = 0.0
    total_secured_profit: float = 0.0
    phase_start_time: str = ""
    last_reset_time: str = ""


# ── DatabaseManager ───────────────────────────────────────────────────────────

class DatabaseManager:
    def __init__(self, db_path: str = "trading_system.db"):
        self.db_path = db_path
        self.logger = get_trading_logger("database")

    # ── Schema init ───────────────────────────────────────────────────────────

    async def initialize(self):
        """Create all tables and seed the single phase_state row."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS phase_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    current_phase_profit REAL DEFAULT 0.0,
                    total_secured_profit REAL DEFAULT 0.0,
                    phase_start_time TEXT,
                    last_reset_time TEXT
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS markets (
                    market_id TEXT PRIMARY KEY,
                    title TEXT,
                    yes_price REAL,
                    no_price REAL,
                    volume INTEGER,
                    expiration_ts INTEGER,
                    category TEXT DEFAULT 'unknown',
                    status TEXT DEFAULT 'active',
                    last_updated TEXT,
                    has_position INTEGER DEFAULT 0
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id TEXT UNIQUE,
                    side TEXT,
                    entry_price REAL,
                    quantity INTEGER,
                    timestamp TEXT,
                    rationale TEXT,
                    confidence REAL DEFAULT 0.0,
                    live INTEGER DEFAULT 0,
                    strategy TEXT DEFAULT 'directional_trading',
                    stop_loss_price REAL,
                    take_profit_price REAL,
                    max_hold_hours INTEGER,
                    target_confidence_change REAL,
                    status TEXT DEFAULT 'open',
                    fill_price REAL
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS trade_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id TEXT,
                    side TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    quantity INTEGER,
                    pnl REAL,
                    entry_timestamp TEXT,
                    exit_timestamp TEXT,
                    rationale TEXT DEFAULT '',
                    phase_profit_at_time REAL DEFAULT 0.0
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS daily_cost_tracking (
                    date TEXT PRIMARY KEY,
                    total_cost REAL DEFAULT 0.0,
                    request_count INTEGER DEFAULT 0
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS llm_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    market_id TEXT,
                    query_type TEXT,
                    model TEXT,
                    cost REAL DEFAULT 0.0,
                    response TEXT DEFAULT '',
                    strategy TEXT,
                    prompt TEXT,
                    tokens_used INTEGER,
                    confidence_extracted REAL,
                    decision_extracted TEXT
                )
            """)

            # Records every analysis decision for deduplication + cost accounting
            await db.execute("""
                CREATE TABLE IF NOT EXISTS market_analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id TEXT,
                    timestamp TEXT,
                    action TEXT,
                    confidence REAL DEFAULT 0.0,
                    cost REAL DEFAULT 0.0,
                    reason TEXT DEFAULT ''
                )
            """)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_ma_market ON market_analyses(market_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_ma_ts ON market_analyses(timestamp)"
            )

            # Persists quick-flip positions across 60-second trading cycles
            await db.execute("""
                CREATE TABLE IF NOT EXISTS quick_flip_tracking (
                    market_id TEXT PRIMARY KEY,
                    side TEXT,
                    quantity INTEGER,
                    entry_price REAL,
                    target_price REAL,
                    placed_at TEXT,
                    max_hold_until TEXT,
                    position_id INTEGER
                )
            """)

            # Seed the single phase_state row if absent
            await db.execute("""
                INSERT OR IGNORE INTO phase_state
                    (id, current_phase_profit, total_secured_profit, phase_start_time, last_reset_time)
                VALUES (1, 0.0, 0.0, ?, ?)
            """, (datetime.now().isoformat(), datetime.now().isoformat()))

            await db.commit()
        self.logger.info("Database initialized successfully")

    # ── Phase state ───────────────────────────────────────────────────────────

    async def get_phase_state(self) -> Dict:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT current_phase_profit, total_secured_profit, phase_start_time, last_reset_time "
                "FROM phase_state WHERE id = 1"
            ) as cur:
                row = await cur.fetchone()
                if row:
                    return {
                        "current_phase_profit": row[0],
                        "total_secured_profit": row[1],
                        "phase_start_time": row[2],
                        "last_reset_time": row[3],
                    }
                return {"current_phase_profit": 0.0, "total_secured_profit": 0.0}

    async def update_phase_profit(self, realized_pnl: float):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE phase_state "
                "SET current_phase_profit = current_phase_profit + ? "
                "WHERE id = 1",
                (realized_pnl,),
            )
            await db.commit()

    async def secure_phase_profit(self, amount: Optional[float] = None):
        """Secure one completed-phase chunk and reset current_phase_profit to 0."""
        secure_amount = amount if amount is not None else getattr(
            settings.trading, "secure_profit_per_chunk", 2400.0
        )
        target = getattr(settings.trading, "phase_profit_target", 2500.0)

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT current_phase_profit, total_secured_profit FROM phase_state WHERE id = 1"
            ) as cur:
                row = await cur.fetchone()
            current = row[0] if row else 0.0
            secured = row[1] if row else 0.0

            if current >= target:
                new_secured = secured + secure_amount
                await db.execute(
                    "UPDATE phase_state "
                    "SET current_phase_profit = 0.0, total_secured_profit = ?, last_reset_time = ? "
                    "WHERE id = 1",
                    (new_secured, datetime.now().isoformat()),
                )
                await db.commit()
                self.logger.info(
                    f"PHASE COMPLETE — Secured ${secure_amount:,.2f} | "
                    f"Total secured: ${new_secured:,.2f} | Reset to base"
                )
            else:
                self.logger.info(
                    f"Phase profit ${current:.2f} not yet at target ${target:.2f}"
                )

    # ── Markets ───────────────────────────────────────────────────────────────

    async def upsert_markets(self, markets: List[Market]):
        async with aiosqlite.connect(self.db_path) as db:
            for m in markets:
                lu = m.last_updated.isoformat() if m.last_updated else datetime.now().isoformat()
                await db.execute(
                    """
                    INSERT INTO markets
                        (market_id, title, yes_price, no_price, volume, expiration_ts,
                         category, status, last_updated, has_position)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(market_id) DO UPDATE SET
                        title=excluded.title, yes_price=excluded.yes_price,
                        no_price=excluded.no_price, volume=excluded.volume,
                        expiration_ts=excluded.expiration_ts, category=excluded.category,
                        status=excluded.status, last_updated=excluded.last_updated,
                        has_position=excluded.has_position
                    """,
                    (
                        m.market_id, m.title, m.yes_price, m.no_price,
                        m.volume, m.expiration_ts, m.category, m.status,
                        lu, int(m.has_position),
                    ),
                )
            await db.commit()

    async def get_eligible_markets(
        self, volume_min: float = 500, max_days_to_expiry: int = 365
    ) -> List[Market]:
        cutoff_ts = int((datetime.now() + timedelta(days=max_days_to_expiry)).timestamp())
        now_ts = int(datetime.now().timestamp())
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT market_id, title, yes_price, no_price, volume, expiration_ts,
                       category, status, last_updated, has_position
                FROM markets
                WHERE volume >= ? AND expiration_ts <= ? AND expiration_ts > ?
                  AND status = 'active' AND has_position = 0
                """,
                (volume_min, cutoff_ts, now_ts),
            ) as cur:
                rows = await cur.fetchall()
        return [
            Market(
                market_id=r[0], title=r[1], yes_price=r[2], no_price=r[3],
                volume=r[4], expiration_ts=r[5], category=r[6], status=r[7],
                last_updated=datetime.fromisoformat(r[8]) if r[8] else None,
                has_position=bool(r[9]),
            )
            for r in rows
        ]

    async def get_markets_with_positions(self) -> set:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT DISTINCT market_id FROM positions WHERE status = 'open'"
            ) as cur:
                rows = await cur.fetchall()
        return {r[0] for r in rows}

    # ── Positions ─────────────────────────────────────────────────────────────

    async def add_position(self, position: Position) -> Optional[int]:
        """Insert a new position.  Returns the row-id, or None if duplicate."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cur = await db.execute(
                    """
                    INSERT OR IGNORE INTO positions
                        (market_id, side, entry_price, quantity, timestamp, rationale,
                         confidence, live, strategy, stop_loss_price, take_profit_price,
                         max_hold_hours, target_confidence_change, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
                    """,
                    (
                        position.market_id, position.side, position.entry_price,
                        position.quantity,
                        position.timestamp.isoformat() if position.timestamp else datetime.now().isoformat(),
                        position.rationale, position.confidence, int(position.live),
                        position.strategy, position.stop_loss_price,
                        position.take_profit_price, position.max_hold_hours,
                        position.target_confidence_change,
                    ),
                )
                await db.commit()
                return cur.lastrowid if cur.rowcount > 0 else None
        except Exception as e:
            self.logger.error(f"Error adding position for {position.market_id}: {e}")
            return None

    def _row_to_position(self, row) -> Position:
        return Position(
            id=row[0], market_id=row[1], side=row[2], entry_price=row[3],
            quantity=row[4],
            timestamp=datetime.fromisoformat(row[5]) if row[5] else datetime.now(),
            rationale=row[6] or "", confidence=float(row[7] or 0.0),
            live=bool(row[8]), strategy=row[9] or "directional_trading",
            stop_loss_price=row[10], take_profit_price=row[11],
            max_hold_hours=row[12], target_confidence_change=row[13],
            status=row[14] or "open", fill_price=row[15],
        )

    async def get_position_by_market_id(self, market_id: str) -> Optional[Position]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT id, market_id, side, entry_price, quantity, timestamp, rationale,
                       confidence, live, strategy, stop_loss_price, take_profit_price,
                       max_hold_hours, target_confidence_change, status, fill_price
                FROM positions WHERE market_id = ? AND status = 'open'
                """,
                (market_id,),
            ) as cur:
                row = await cur.fetchone()
        return self._row_to_position(row) if row else None

    async def get_open_positions(self) -> List[Position]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT id, market_id, side, entry_price, quantity, timestamp, rationale,
                       confidence, live, strategy, stop_loss_price, take_profit_price,
                       max_hold_hours, target_confidence_change, status, fill_price
                FROM positions WHERE status = 'open'
                """
            ) as cur:
                rows = await cur.fetchall()
        return [self._row_to_position(r) for r in rows]

    async def get_open_live_positions(self) -> List[Position]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT id, market_id, side, entry_price, quantity, timestamp, rationale,
                       confidence, live, strategy, stop_loss_price, take_profit_price,
                       max_hold_hours, target_confidence_change, status, fill_price
                FROM positions WHERE status = 'open' AND live = 1
                """
            ) as cur:
                rows = await cur.fetchall()
        return [self._row_to_position(r) for r in rows]

    async def update_position_to_live(self, position_id: int, fill_price: float):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE positions SET live = 1, fill_price = ? WHERE id = ?",
                (fill_price, position_id),
            )
            await db.commit()

    async def update_position_status(self, position_id: int, status: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE positions SET status = ? WHERE id = ?",
                (status, position_id),
            )
            await db.commit()

    # ── Trade logs ────────────────────────────────────────────────────────────

    async def add_trade_log(self, trade_log: TradeLog):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO trade_logs
                    (market_id, side, entry_price, exit_price, quantity, pnl,
                     entry_timestamp, exit_timestamp, rationale)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade_log.market_id, trade_log.side,
                    trade_log.entry_price, trade_log.exit_price,
                    trade_log.quantity, trade_log.pnl,
                    trade_log.entry_timestamp.isoformat() if trade_log.entry_timestamp else datetime.now().isoformat(),
                    trade_log.exit_timestamp.isoformat() if trade_log.exit_timestamp else datetime.now().isoformat(),
                    trade_log.rationale,
                ),
            )
            await db.commit()

    async def log_trade(self, market_id: str, side: str, size: float, pnl: Optional[float] = None):
        """Lightweight helper kept for backward compatibility."""
        now = datetime.now().isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO trade_logs
                    (market_id, side, entry_price, exit_price, quantity, pnl,
                     entry_timestamp, exit_timestamp, rationale)
                VALUES (?, ?, ?, ?, 1, ?, ?, ?, '')
                """,
                (market_id, side, size, size, pnl, now, now),
            )
            await db.commit()

    # ── AI cost tracking ──────────────────────────────────────────────────────

    async def get_daily_ai_cost(self) -> float:
        today = datetime.now().strftime("%Y-%m-%d")
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT total_cost FROM daily_cost_tracking WHERE date = ?", (today,)
            ) as cur:
                row = await cur.fetchone()
        return float(row[0]) if row else 0.0

    async def record_ai_cost(self, cost: float):
        today = datetime.now().strftime("%Y-%m-%d")
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO daily_cost_tracking (date, total_cost, request_count)
                VALUES (?, ?, 1)
                ON CONFLICT(date) DO UPDATE SET
                    total_cost = total_cost + excluded.total_cost,
                    request_count = request_count + 1
                """,
                (today, cost),
            )
            await db.commit()

    async def log_llm_query(self, llm_query: "LLMQuery"):
        """Persist an LLM query record.  Fire-and-forget — errors are swallowed."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT INTO llm_queries
                        (timestamp, market_id, query_type, model, cost, response,
                         strategy, prompt, tokens_used, confidence_extracted, decision_extracted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        llm_query.timestamp,
                        llm_query.market_id,
                        llm_query.query_type,
                        llm_query.model,
                        llm_query.cost,
                        llm_query.response,
                        llm_query.strategy,
                        llm_query.prompt,
                        llm_query.tokens_used,
                        llm_query.confidence_extracted,
                        llm_query.decision_extracted,
                    ),
                )
                await db.commit()
        except Exception as e:
            self.logger.warning(f"Failed to log LLM query: {e}")

    async def add_llm_query(self, llm_query: "LLMQuery") -> Optional[int]:
        """Insert an LLM query and return the row id."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cur = await db.execute(
                    """
                    INSERT INTO llm_queries
                        (timestamp, market_id, query_type, model, cost, response,
                         strategy, prompt, tokens_used, confidence_extracted, decision_extracted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        llm_query.timestamp,
                        llm_query.market_id,
                        llm_query.query_type,
                        llm_query.model,
                        llm_query.cost,
                        llm_query.response,
                        llm_query.strategy,
                        llm_query.prompt,
                        llm_query.tokens_used,
                        llm_query.confidence_extracted,
                        llm_query.decision_extracted,
                    ),
                )
                await db.commit()
                return cur.lastrowid
        except Exception as e:
            self.logger.error(f"Failed to add LLM query: {e}")
            return None

    # ── Market-analysis deduplication ────────────────────────────────────────

    async def was_recently_analyzed(self, market_id: str, cooldown_hours: float) -> bool:
        cutoff = (datetime.now() - timedelta(hours=cooldown_hours)).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT 1 FROM market_analyses
                WHERE market_id = ? AND timestamp > ?
                  AND action NOT IN ('ERROR', 'SKIP', 'COST_LIMITED')
                LIMIT 1
                """,
                (market_id, cutoff),
            ) as cur:
                return await cur.fetchone() is not None

    async def get_market_analysis_count_today(self, market_id: str) -> int:
        today = datetime.now().strftime("%Y-%m-%d")
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT COUNT(*) FROM market_analyses WHERE market_id = ? AND timestamp LIKE ?",
                (market_id, f"{today}%"),
            ) as cur:
                row = await cur.fetchone()
        return row[0] if row else 0

    async def record_market_analysis(
        self,
        market_id: str,
        action: str,
        confidence: float,
        cost: float,
        reason: str = "",
    ):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO market_analyses (market_id, timestamp, action, confidence, cost, reason)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (market_id, datetime.now().isoformat(), action, confidence, cost, reason),
            )
            await db.commit()
        if cost > 0:
            await self.record_ai_cost(cost)

    # ── Quick-flip persistent tracking ───────────────────────────────────────

    async def save_quick_flip_position(
        self,
        market_id: str,
        side: str,
        quantity: int,
        entry_price: float,
        target_price: float,
        max_hold_until: datetime,
        position_id: int,
    ):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO quick_flip_tracking
                    (market_id, side, quantity, entry_price, target_price,
                     placed_at, max_hold_until, position_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    market_id, side, quantity, entry_price, target_price,
                    datetime.now().isoformat(), max_hold_until.isoformat(),
                    position_id,
                ),
            )
            await db.commit()

    async def get_quick_flip_pending(self) -> List[Dict]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT * FROM quick_flip_tracking") as cur:
                cols = [d[0] for d in cur.description]
                rows = await cur.fetchall()
        return [dict(zip(cols, r)) for r in rows]

    async def remove_quick_flip_position(self, market_id: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM quick_flip_tracking WHERE market_id = ?", (market_id,)
            )
            await db.commit()

    # ── Daily P&L (circuit-breaker) ───────────────────────────────────────────

    async def get_daily_pnl(self) -> float:
        today = datetime.now().strftime("%Y-%m-%d")
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT COALESCE(SUM(pnl), 0.0) FROM trade_logs WHERE exit_timestamp LIKE ?",
                (f"{today}%",),
            ) as cur:
                row = await cur.fetchone()
        return float(row[0]) if row else 0.0


# ── Top-level helpers ─────────────────────────────────────────────────────────

async def get_phase_summary() -> Dict:
    db = DatabaseManager()
    return await db.get_phase_state()


async def secure_profit_if_needed():
    db = DatabaseManager()
    await db.secure_phase_profit()
