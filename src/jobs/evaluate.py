"""
Enhanced evaluation system with cost monitoring and trading performance analysis.
"""

import asyncio
import aiosqlite
from datetime import datetime
from typing import Dict

from src.utils.database import DatabaseManager
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger


async def analyze_ai_costs(db_manager: DatabaseManager) -> Dict:
    """Simple AI cost analysis using existing daily_cost_tracking table."""
    logger = get_trading_logger("cost_analysis")
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    async with aiosqlite.connect(db_manager.db_path) as db:
        cursor = await db.execute("""
            SELECT total_cost, request_count 
            FROM daily_cost_tracking 
            WHERE date = ?
        """, (today,))
        row = await cursor.fetchone()
        
        today_cost = float(row[0]) if row else 0.0
        today_requests = int(row[1]) if row else 0
    
    actual_limit = getattr(settings.trading, 'daily_ai_cost_limit', 50.0)
    
    logger.info(
        "AI Cost Analysis",
        today_cost=round(today_cost, 4),
        today_requests=today_requests,
        daily_limit=actual_limit,
        utilization=round(today_cost / actual_limit * 100, 1) if actual_limit else 0
    )
    
    return {
        'today_cost': today_cost,
        'today_requests': today_requests,
        'daily_limit': actual_limit,
        'utilization': today_cost / actual_limit if actual_limit else 0
    }


async def analyze_trading_performance(db_manager: DatabaseManager) -> Dict:
    """Simple trading performance summary."""
    logger = get_trading_logger("trading_performance")
    
    async with aiosqlite.connect(db_manager.db_path) as db:
        cursor = await db.execute("SELECT COUNT(*), SUM(pnl) FROM trade_logs")
        row = await cursor.fetchone()
        total_trades = row[0] if row else 0
        total_pnl = float(row[1]) if row and row[1] else 0.0
    
    logger.info("Trading Performance", total_trades=total_trades, total_pnl=round(total_pnl, 2))
    
    return {
        'total_trades': total_trades,
        'total_pnl': total_pnl
    }


async def run_evaluation():
    logger = get_trading_logger("evaluation")
    logger.info("Starting enhanced evaluation job.")
    
    db_manager = DatabaseManager()
    await db_manager.initialize()
    
    try:
        cost_analysis = await analyze_ai_costs(db_manager)
        performance_analysis = await analyze_trading_performance(db_manager)
        
        logger.info(
            "Evaluation Complete",
            today_cost=round(cost_analysis['today_cost'], 4),
            total_trades=performance_analysis['total_trades'],
            total_pnl=round(performance_analysis['total_pnl'], 2)
        )
        
    except Exception as e:
        logger.error(f"Error in evaluation job: {e}")


if __name__ == "__main__":
    asyncio.run(run_evaluation())