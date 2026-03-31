"""
Trade Job — Entry point for executing unified trading
"""

import asyncio
from typing import Dict

from src.utils.database import DatabaseManager
from src.clients.kalshi_client import KalshiClient
from src.clients.xai_client import XAIClient
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger
from src.strategies.unified_trading_system import run_unified_trading_system, TradingSystemConfig


async def _initialize_components():
    db_manager = DatabaseManager()
    await db_manager.initialize()

    kalshi_client = KalshiClient()
    xai_client = XAIClient(db_manager=db_manager)

    config = TradingSystemConfig()

    return db_manager, kalshi_client, xai_client, config


async def _run_unified_trade_cycle(db_manager: DatabaseManager, kalshi_client: KalshiClient, xai_client: XAIClient, config: TradingSystemConfig) -> Dict:
    logger = get_trading_logger("trade_cycle")
    try:
        results = await run_unified_trading_system(
            db_manager=db_manager,
            kalshi_client=kalshi_client,
            xai_client=xai_client,
            config=config
        )
        logger.info(f"Unified trade cycle completed — {results.total_positions} positions, ${results.total_capital_used:.2f} capital used")
        return results
    except Exception as e:
        logger.error(f"Error in unified trade cycle: {e}")
        return {"total_positions": 0, "total_capital_used": 0.0}


async def run_trading_job() -> Dict:
    logger = get_trading_logger("trade_job")
    logger.info("Starting trade job")

    try:
        db_manager, kalshi_client, xai_client, config = await _initialize_components()

        results = await _run_unified_trade_cycle(db_manager, kalshi_client, xai_client, config)

        logger.info("Trade job completed successfully")
        return results

    except Exception as e:
        logger.error(f"Critical error in trade job: {e}")
        return {"total_positions": 0, "total_capital_used": 0.0}

    finally:
        try:
            if 'kalshi_client' in locals():
                await kalshi_client.close()
        except Exception:
            pass


async def run_trading_job_async():
    return await run_trading_job()