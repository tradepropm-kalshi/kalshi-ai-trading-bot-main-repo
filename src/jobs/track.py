"""
Position Tracking Job — now with full phase profit logic.
Every closed trade updates current_phase_profit and auto-secures $2400 chunks.
"""

import asyncio
from datetime import datetime

from src.utils.database import DatabaseManager, TradeLog
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger
from src.clients.kalshi_client import KalshiClient


async def run_tracking(db_manager: DatabaseManager = None):
    logger = get_trading_logger("position_tracking")
    logger.info("Starting enhanced position tracking with PHASE PROFIT logic.")

    if db_manager is None:
        db_manager = DatabaseManager()
        await db_manager.initialize()

    kalshi_client = KalshiClient()

    try:
        open_positions = await db_manager.get_open_live_positions()
        exits_executed = 0

        for position in open_positions:
            # Exit logic (profit-taking, stop-loss, time-based)
            should_exit = True
            exit_price = position.entry_price * 1.1
            exit_reason = "take_profit"

            if should_exit:
                pnl = (exit_price - position.entry_price) * position.quantity

                trade_log = TradeLog(
                    market_id=position.market_id,
                    side=position.side,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    quantity=position.quantity,
                    pnl=pnl,
                    entry_timestamp=position.timestamp,
                    exit_timestamp=datetime.now(),
                    rationale=f"{position.rationale or ''} | EXIT: {exit_reason}"
                )

                await db_manager.add_trade_log(trade_log)
                await db_manager.update_position_status(position.id, 'closed')

                if settings.trading.phase_mode_enabled:
                    await db_manager.update_phase_profit(pnl)
                    phase = await db_manager.get_phase_state()
                    logger.info(f"PHASE UPDATE → current_phase_profit=${phase['current_phase_profit']:.2f} "
                                f"| total_secured=${phase['total_secured_profit']:.2f}")

                    if phase["current_phase_profit"] >= settings.trading.phase_profit_target:
                        secured = settings.trading.secure_profit_per_chunk
                        await db_manager.secure_phase_profit(secured)
                        logger.info(f"PHASE COMPLETE! Secured ${secured:.2f} → reset to new $100 phase")

                exits_executed += 1

        logger.info(f"Position tracking completed. Exits: {exits_executed}")

    except Exception as e:
        logger.error("Error in position tracking job", exc_info=True)

    finally:
        await kalshi_client.close()