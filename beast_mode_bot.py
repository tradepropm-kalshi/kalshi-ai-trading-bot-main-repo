#!/usr/bin/env python3
"""
Beast Mode Trading Bot — WITH BIBLE PHASE PROFIT MODE
"""

import asyncio
import argparse
import signal
import sys
from datetime import datetime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from src.jobs.trade import run_trading_job
from src.jobs.ingest import run_ingestion
from src.jobs.track import run_tracking
from src.jobs.evaluate import run_evaluation
from src.utils.logging_setup import setup_logging, get_trading_logger
from src.utils.database import DatabaseManager
from src.clients.kalshi_client import KalshiClient
from src.clients.xai_client import XAIClient
from src.config.settings import settings
from beast_mode_dashboard import BeastModeDashboard


class BeastModeBot:
    def __init__(self, live_mode: bool = False, dashboard_mode: bool = False, phase_mode: bool = False):
        self.live_mode = live_mode
        self.dashboard_mode = dashboard_mode
        self.phase_mode = phase_mode
        self.logger = get_trading_logger("beast_mode_bot")
        self.shutdown_event = asyncio.Event()

        settings.trading.live_trading_enabled = live_mode
        settings.trading.paper_trading_mode = not live_mode
        settings.trading.phase_mode_enabled = phase_mode

        self.logger.info(f"Beast Mode Bot initialized - Mode: {'LIVE' if live_mode else 'PAPER'} | Phase Mode: {'ENABLED' if phase_mode else 'DISABLED'}")
        if phase_mode:
            self.logger.info("PHASE MODE ACTIVE → $100 base capital | $2,500 profit target | $2,400 secure per chunk")
        if live_mode:
            self.logger.warning("LIVE TRADING MODE ENABLED - REAL MONEY WILL BE USED")
        else:
            self.logger.info("Paper trading mode - orders will be simulated")

    async def run_dashboard_mode(self):
        self.logger.info("Starting Beast Mode Dashboard Mode")
        dashboard = BeastModeDashboard()
        await dashboard.show_live_dashboard()

    async def run_trading_mode(self):
        self.logger.info("BEAST MODE TRADING BOT STARTED")
        self.logger.info(f"Trading Mode: {'LIVE' if self.live_mode else 'PAPER'}")
        if self.phase_mode:
            self.logger.info("PHASE PROFIT MODE: $100 base → +$2,500 → secure $2,400 + reset")

        db_manager = DatabaseManager()
        await self._ensure_database_ready(db_manager)

        kalshi_client = KalshiClient()
        xai_client = XAIClient(db_manager=db_manager)

        ingestion_task = asyncio.create_task(self._run_market_ingestion(db_manager, kalshi_client))
        tasks = [
            ingestion_task,
            asyncio.create_task(self._run_trading_cycles(db_manager, kalshi_client, xai_client)),
            asyncio.create_task(self._run_position_tracking(db_manager, kalshi_client)),
            asyncio.create_task(self._run_performance_evaluation(db_manager))
        ]

        def signal_handler():
            self.logger.info("Shutdown signal received")
            self.shutdown_event.set()
            for task in tasks:
                if not task.done():
                    task.cancel()

        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, lambda s, f: signal_handler())

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            try:
                await kalshi_client.close()
            except Exception:
                pass
            self.logger.info("Beast Mode Bot shut down gracefully")

    async def _ensure_database_ready(self, db_manager: DatabaseManager):
        await db_manager.initialize()
        self.logger.info("Database tables verified and ready")

    async def _run_market_ingestion(self, db_manager: DatabaseManager, kalshi_client: KalshiClient):
        while not self.shutdown_event.is_set():
            try:
                await run_ingestion(db_manager)
                await asyncio.sleep(300)
            except Exception as e:
                self.logger.error(f"Error in market ingestion: {e}")
                await asyncio.sleep(60)

    async def _run_trading_cycles(self, db_manager: DatabaseManager, kalshi_client: KalshiClient, xai_client: XAIClient):
        cycle_count = 0
        while not self.shutdown_event.is_set():
            try:
                cycle_count += 1
                self.logger.info(f"Starting Beast Mode Trading Cycle #{cycle_count}")
                await run_trading_job()
                await asyncio.sleep(60)
            except Exception as e:
                self.logger.error(f"Error in trading cycle #{cycle_count}: {e}")
                await asyncio.sleep(60)

    async def _run_position_tracking(self, db_manager: DatabaseManager, kalshi_client: KalshiClient):
        while not self.shutdown_event.is_set():
            try:
                await run_tracking(db_manager)
                await asyncio.sleep(120)
            except Exception as e:
                self.logger.error(f"Error in position tracking: {e}")
                await asyncio.sleep(30)

    async def _run_performance_evaluation(self, db_manager: DatabaseManager):
        while not self.shutdown_event.is_set():
            try:
                await run_evaluation()
                await asyncio.sleep(300)
            except Exception as e:
                self.logger.error(f"Error in performance evaluation: {e}")
                await asyncio.sleep(300)

    async def run(self):
        if self.dashboard_mode:
            await self.run_dashboard_mode()
        else:
            await self.run_trading_mode()


async def main():
    parser = argparse.ArgumentParser(
        description="Beast Mode Trading Bot — with PHASE PROFIT MODE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--live", action="store_true", help="Run in LIVE trading mode")
    parser.add_argument("--dashboard", action="store_true", help="Run in live dashboard mode")
    parser.add_argument("--phase", action="store_true", help="Enable PHASE PROFIT MODE ($100 → +$2,500 → secure $2,400 + reset)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()
    setup_logging(log_level=args.log_level)

    bot = BeastModeBot(
        live_mode=args.live,
        dashboard_mode=args.dashboard,
        phase_mode=args.phase
    )
    await bot.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBeast Mode Bot stopped by user")
    except Exception as e:
        print(f"Beast Mode Bot error: {e}")
        raise