"""
Trade Job — Entry point for executing the unified trading strategy.

Fix: The previous implementation created a fresh DatabaseManager, KalshiClient,
and XAIClient on every 60-second trading cycle.  This leaked connections and
wasted initialisation overhead.

run_trading_job() now accepts optional pre-built instances.  When called from
BeastModeBot (which holds long-lived shared instances), those are reused.
When called standalone (e.g. scripts / tests), fresh instances are created.
"""

import asyncio
from typing import Dict, Optional

from src.utils.database import DatabaseManager
from src.clients.kalshi_client import KalshiClient
from src.clients.xai_client import XAIClient
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger
from src.strategies.unified_trading_system import run_unified_trading_system, TradingSystemConfig


async def run_trading_job(
    db_manager: Optional[DatabaseManager] = None,
    kalshi_client: Optional[KalshiClient] = None,
    xai_client: Optional[XAIClient] = None,
) -> Dict:
    """
    Execute one unified trading cycle.

    Pass shared instances from BeastModeBot to avoid creating new DB
    connections and HTTP clients on every 60-second cycle.
    If instances are not provided, they are created locally and the
    KalshiClient is closed when the cycle finishes.
    """
    logger = get_trading_logger("trade_job")
    logger.info("Starting trade job")

    # Track whether we own the client so we know to close it on exit
    _owns_kalshi_client = kalshi_client is None

    try:
        if db_manager is None:
            db_manager = DatabaseManager()
            await db_manager.initialize()

        if kalshi_client is None:
            kalshi_client = KalshiClient()

        if xai_client is None:
            xai_client = XAIClient(db_manager=db_manager)

        config = TradingSystemConfig()

        results = await run_unified_trading_system(
            db_manager=db_manager,
            kalshi_client=kalshi_client,
            xai_client=xai_client,
            config=config,
        )

        logger.info(
            f"Trade job completed — {results.total_positions} positions, "
            f"${results.total_capital_used:.2f} capital used"
        )
        return {
            "total_positions": results.total_positions,
            "total_capital_used": results.total_capital_used,
            "capital_efficiency": results.capital_efficiency,
        }

    except Exception as e:
        logger.error(f"Critical error in trade job: {e}")
        return {"total_positions": 0, "total_capital_used": 0.0}

    finally:
        # Only close the client if we created it locally
        if _owns_kalshi_client and kalshi_client is not None:
            try:
                await kalshi_client.close()
            except Exception:
                pass


async def run_trading_job_async():
    """Backward-compatible async wrapper."""
    return await run_trading_job()
