"""
Trade Job — Central entry point for trading cycles (original RyanFrigo pipeline preserved)
"""

import asyncio
from src.utils.logging_setup import get_trading_logger
from src.strategies.unified_trading_system import run_unified_trading_system
from src.config.settings import settings

async def run_trading_job():
    logger = get_trading_logger("trading_job")
    logger.info("Starting Enhanced Trading Job - Beast Mode Activated")

    if settings.trading.phase_mode_enabled:
        logger.info("PHASE MODE ACTIVE → sizing against $100 base + current phase profit")

    results = await run_unified_trading_system()

    logger.info("Unified trading strategy completed successfully")
    return results