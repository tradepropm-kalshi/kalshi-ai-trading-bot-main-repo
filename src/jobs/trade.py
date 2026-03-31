"""
Enhanced Trading Job - Beast Mode 🚀 — NOW WITH PHASE PROFIT MODE
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from src.clients.kalshi_client import KalshiClient
from src.clients.xai_client import XAIClient
from src.utils.database import DatabaseManager
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger

from src.strategies.unified_trading_system import run_unified_trading_system, TradingSystemConfig, TradingSystemResults

from src.jobs.decide import make_decision_for_market
from src.jobs.execute import execute_position


async def run_trading_job() -> Optional[TradingSystemResults]:
    logger = get_trading_logger("trading_job")
    try:
        logger.info("🚀 Starting Enhanced Trading Job - Beast Mode Activated!")
        if settings.trading.phase_mode_enabled:
            logger.info("🎯 PHASE MODE ACTIVE → sizing against $100 base + current phase profit")
        
        db_manager = DatabaseManager()
        kalshi_client = KalshiClient()
        xai_client = XAIClient(db_manager=db_manager)
        
        config = TradingSystemConfig()
        
        logger.info("🎯 Executing Unified Advanced Trading System")
        results = await run_unified_trading_system(
            db_manager, kalshi_client, xai_client, config
        )
        
        return results
    except Exception as e:
        logger.error(f"Error in enhanced trading job: {e}")
        logger.warning("🔄 Falling back to legacy decision-making system")
        return await _fallback_legacy_trading()


async def _fallback_legacy_trading() -> Optional[TradingSystemResults]:
    logger = get_trading_logger("trading_job_fallback")
    try:
        logger.info("🔄 Executing fallback legacy trading system")
        db_manager = DatabaseManager()
        kalshi_client = KalshiClient()
        xai_client = XAIClient()
        
        markets = await db_manager.get_eligible_markets(volume_min=20000, max_days_to_expiry=365)
        if not markets:
            return TradingSystemResults()
        
        positions_created = 0
        total_exposure = 0.0
        for market in markets[:5]:
            try:
                position = await make_decision_for_market(market, db_manager, xai_client, kalshi_client)
                if position:
                    success = await execute_position(position, kalshi_client, db_manager)
                    if success:
                        positions_created += 1
                        total_exposure += position.entry_price * position.quantity
            except Exception as e:
                logger.error(f"Error processing market {market.market_id}: {e}")
                continue
        
        return TradingSystemResults(
            directional_positions=positions_created,
            directional_exposure=total_exposure,
            total_capital_used=total_exposure,
            total_positions=positions_created,
            capital_efficiency=total_exposure / 10000 if total_exposure > 0 else 0.0
        )
    except Exception as e:
        logger.error(f"Error in fallback trading system: {e}")
        return TradingSystemResults()


async def run_legacy_trading():
    logger = get_trading_logger("legacy_redirect")
    logger.info("🔄 Legacy trading call redirected to enhanced system")
    return await run_trading_job()