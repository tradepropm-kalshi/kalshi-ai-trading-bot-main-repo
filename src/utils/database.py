"""
Database Manager for Kalshi AI Trading Bot
Original RyanFrigo design preserved exactly
WITH BIBLE PHASE PROFIT STATE SUPPORT
"""

import aiosqlite
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger


@dataclass
class PhaseState:
    current_phase_profit: float = 0.0
    total_secured_profit: float = 0.0
    phase_start_time: str = ""
    last_reset_time: str = ""


class DatabaseManager:
    def __init__(self):
        self.db_path = "trading_system.db"
        self.logger = get_trading_logger("database")

    async def initialize(self):
        """Initialize database with all tables including phase_state"""
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
                CREATE TABLE IF NOT EXISTS trade_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    market_id TEXT,
                    side TEXT,
                    size REAL,
                    pnl REAL,
                    phase_profit_at_time REAL
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS daily_cost_tracking (
                    date TEXT PRIMARY KEY,
                    total_cost REAL DEFAULT 0.0,
                    request_count INTEGER DEFAULT 0
                )
            """)

            # Ensure single phase_state row exists
            await db.execute("""
                INSERT OR IGNORE INTO phase_state (id, current_phase_profit, total_secured_profit, phase_start_time, last_reset_time)
                VALUES (1, 0.0, 0.0, ?, ?)
            """, (datetime.now().isoformat(), datetime.now().isoformat()))

            await db.commit()
        self.logger.info("Database initialized with phase_state table")

    async def get_phase_state(self) -> Dict:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT current_phase_profit, total_secured_profit, phase_start_time, last_reset_time FROM phase_state WHERE id = 1") as cursor:
                row = await cursor.fetchone()
                if row:
                    return {
                        "current_phase_profit": row[0],
                        "total_secured_profit": row[1],
                        "phase_start_time": row[2],
                        "last_reset_time": row[3]
                    }
                return {"current_phase_profit": 0.0, "total_secured_profit": 0.0}

    async def update_phase_profit(self, realized_pnl: float):
        """Add realized PnL to current phase profit"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE phase_state 
                SET current_phase_profit = current_phase_profit + ?,
                    phase_start_time = COALESCE(phase_start_time, ?)
                WHERE id = 1
            """, (realized_pnl, datetime.now().isoformat()))
            await db.commit()

    async def secure_phase_profit(self):
        """Secure $2,400 and reset phase to new $100 base (Bible rule)"""
        async with aiosqlite.connect(self.db_path) as db:
            # Get current values
            async with db.execute("SELECT current_phase_profit, total_secured_profit FROM phase_state WHERE id = 1") as cursor:
                row = await cursor.fetchone()
                current = row[0] if row else 0.0
                secured = row[1] if row else 0.0

            if current >= getattr(settings.trading, 'phase_profit_target', 2500.0):
                new_secured = secured + 2400.0
                await db.execute("""
                    UPDATE phase_state 
                    SET current_phase_profit = 0.0,
                        total_secured_profit = ?,
                        last_reset_time = ?
                    WHERE id = 1
                """, (new_secured, datetime.now().isoformat()))
                await db.commit()
                self.logger.info(f"PHASE COMPLETE — Secured $2,400 | New total secured: ${new_secured:,.2f} | Reset to $100 base")
            else:
                self.logger.info(f"Phase profit ${current:.2f} not yet at target")

    async def get_open_positions(self) -> List[Dict]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT * FROM trade_logs WHERE pnl IS NULL") as cursor:
                rows = await cursor.fetchall()
                return [dict(zip([col[0] for col in cursor.description], row)) for row in rows]

    async def log_trade(self, market_id: str, side: str, size: float, pnl: Optional[float] = None):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO trade_logs (timestamp, market_id, side, size, pnl, phase_profit_at_time)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (datetime.now().isoformat(), market_id, side, size, pnl, 0.0))
            await db.commit()


# Top-level helpers
async def get_phase_summary() -> Dict:
    db = DatabaseManager()
    return await db.get_phase_state()


async def secure_profit_if_needed():
    db = DatabaseManager()
    await db.secure_phase_profit()