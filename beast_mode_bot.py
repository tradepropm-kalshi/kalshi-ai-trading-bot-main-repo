#!/usr/bin/env python3
"""
Beast Mode Trading Bot — WITH BIBLE PHASE PROFIT MODE

Fixes applied
─────────────
1. SIGTERM on Windows: signal.SIGTERM is not supported on Win32.
   The bot now uses a platform-safe handler that calls asyncio's
   loop.add_signal_handler on POSIX and falls back to a KeyboardInterrupt
   catcher on Windows.

2. Daily loss circuit-breaker: at the start of every trading cycle the bot
   reads today's realised P&L from the database.  If it has fallen below
   max_daily_loss_pct of the starting capital the trading loop is halted and
   all tasks are cancelled.

3. Shared resource reuse: DatabaseManager, KalshiClient and XAIClient are
   created once in run_trading_mode and passed into every task.
   run_trading_job() now accepts optional injected instances so no new
   connections are opened on each 60-second cycle.
"""

import asyncio
import argparse
import sys
import signal
from datetime import datetime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.jobs.trade import run_trading_job
from src.jobs.ingest import run_ingestion
from src.jobs.track import run_tracking
from src.jobs.evaluate import run_evaluation
from src.utils.logging_setup import setup_logging, get_trading_logger
from src.utils.database import DatabaseManager
from src.clients.kalshi_client import KalshiClient
from src.clients.xai_client import XAIClient
from src.config.settings import settings


class BeastModeBot:
    def __init__(
        self,
        live_mode: bool = False,
        phase_mode: bool = False,
    ):
        self.live_mode = live_mode
        self.phase_mode = phase_mode
        self.logger = get_trading_logger("beast_mode_bot")
        self.shutdown_event = asyncio.Event()
        self._start_capital: float = settings.trading.phase_base_capital

        settings.trading.live_trading_enabled = live_mode
        settings.trading.paper_trading_mode = not live_mode
        settings.trading.phase_mode_enabled = phase_mode

        self.logger.info(
            f"Beast Mode Bot initialised — Mode: {'LIVE' if live_mode else 'PAPER'} | "
            f"Phase Mode: {'ENABLED' if phase_mode else 'DISABLED'}"
        )
        if phase_mode:
            self.logger.info(
                "PHASE MODE ACTIVE → $100 base capital | "
                "$2,500 profit target | $2,400 secure per chunk"
            )
        if live_mode:
            self.logger.warning("LIVE TRADING MODE ENABLED — REAL MONEY WILL BE USED")
        else:
            self.logger.info("Paper trading mode — orders will be simulated")

    # ── Trading mode ──────────────────────────────────────────────────────────

    async def run_trading_mode(self):
        self.logger.info("BEAST MODE TRADING BOT STARTED")
        self.logger.info(f"Trading Mode: {'LIVE' if self.live_mode else 'PAPER'}")
        if self.phase_mode:
            self.logger.info("PHASE PROFIT MODE: $100 base → +$2,500 → secure $2,400 + reset")

        # ── Create shared long-lived resources once ───────────────────────────
        db_manager = DatabaseManager()
        await self._ensure_database_ready(db_manager)

        kalshi_client = KalshiClient()
        xai_client = XAIClient(db_manager=db_manager)

        # Capture starting capital for circuit-breaker reference
        try:
            bal = await kalshi_client.get_balance()
            self._start_capital = (
                bal.get("balance", 0) / 100 + bal.get("portfolio_value", 0) / 100
            ) or settings.trading.phase_base_capital
        except Exception:
            self._start_capital = settings.trading.phase_base_capital

        tasks = [
            asyncio.create_task(
                self._run_market_ingestion(db_manager, kalshi_client)
            ),
            asyncio.create_task(
                self._run_trading_cycles(db_manager, kalshi_client, xai_client)
            ),
            asyncio.create_task(
                self._run_position_tracking(db_manager, kalshi_client)
            ),
            asyncio.create_task(
                self._run_performance_evaluation(db_manager)
            ),
        ]

        self._register_signal_handlers(tasks)

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            try:
                await kalshi_client.close()
            except Exception:
                pass
            self.logger.info("Beast Mode Bot shut down gracefully")

    # ── Signal handling (Windows-safe) ────────────────────────────────────────

    def _register_signal_handlers(self, tasks):
        """
        Register shutdown signal handlers in a platform-safe way.

        On POSIX (Linux/macOS):  loop.add_signal_handler() works for both
                                  SIGINT and SIGTERM.
        On Windows:               SIGTERM is not supported; only SIGINT
                                  (Ctrl-C) is handled — the rest falls
                                  through to the KeyboardInterrupt catcher
                                  in __main__.
        """
        def _shutdown(signame: str):
            self.logger.info(f"Shutdown signal received ({signame})")
            self.shutdown_event.set()
            for task in tasks:
                if not task.done():
                    task.cancel()

        if sys.platform != "win32":
            loop = asyncio.get_event_loop()
            loop.add_signal_handler(signal.SIGINT,  lambda: _shutdown("SIGINT"))
            loop.add_signal_handler(signal.SIGTERM, lambda: _shutdown("SIGTERM"))
        else:
            # On Windows only SIGINT (Ctrl-C) raises KeyboardInterrupt reliably;
            # SIGTERM is not available.  We still bind signal.SIGINT via the
            # legacy API so that subprocesses can send it.
            signal.signal(signal.SIGINT, lambda s, f: _shutdown("SIGINT"))

    # ── Daily loss circuit-breaker ────────────────────────────────────────────

    async def _check_daily_loss_limit(self, db_manager: DatabaseManager) -> bool:
        """
        Return True (and set shutdown_event) if today's realised P&L has
        exceeded the configured daily loss limit.
        """
        try:
            daily_pnl = await db_manager.get_daily_pnl()
            loss_limit = -(self._start_capital * settings.trading.max_daily_loss_pct / 100)

            if daily_pnl < loss_limit:
                self.logger.warning(
                    f"DAILY LOSS LIMIT HIT: today P&L = ${daily_pnl:.2f} "
                    f"(limit ${loss_limit:.2f}). Halting all trading."
                )
                self.shutdown_event.set()
                return True
        except Exception as e:
            self.logger.error(f"Error checking daily loss limit: {e}")
        return False

    # ── Background loops ──────────────────────────────────────────────────────

    async def _ensure_database_ready(self, db_manager: DatabaseManager):
        await db_manager.initialize()
        self.logger.info("Database tables verified and ready")

    async def _run_market_ingestion(
        self, db_manager: DatabaseManager, kalshi_client: KalshiClient
    ):
        while not self.shutdown_event.is_set():
            try:
                # queue=None — ingestion just populates the DB, no pipeline queue
                await run_ingestion(db_manager)
                await asyncio.sleep(300)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in market ingestion: {e}")
                await asyncio.sleep(60)

    async def _run_trading_cycles(
        self,
        db_manager: DatabaseManager,
        kalshi_client: KalshiClient,
        xai_client: XAIClient,
    ):
        cycle_count = 0
        while not self.shutdown_event.is_set():
            try:
                # Circuit-breaker: abort if daily loss limit is breached
                if await self._check_daily_loss_limit(db_manager):
                    break

                cycle_count += 1
                self.logger.info(f"Starting Beast Mode Trading Cycle #{cycle_count}")

                # Pass shared instances — no new connections created per cycle
                await run_trading_job(
                    db_manager=db_manager,
                    kalshi_client=kalshi_client,
                    xai_client=xai_client,
                )
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in trading cycle #{cycle_count}: {e}")
                await asyncio.sleep(60)

    async def _run_position_tracking(
        self, db_manager: DatabaseManager, kalshi_client: KalshiClient
    ):
        while not self.shutdown_event.is_set():
            try:
                await run_tracking(db_manager)
                await asyncio.sleep(120)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in position tracking: {e}")
                await asyncio.sleep(30)

    async def _run_performance_evaluation(self, db_manager: DatabaseManager):
        while not self.shutdown_event.is_set():
            try:
                await run_evaluation()
                await asyncio.sleep(300)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in performance evaluation: {e}")
                await asyncio.sleep(300)

    # ── Entry point ───────────────────────────────────────────────────────────

    async def run(self):
        await self.run_trading_mode()


async def main():
    parser = argparse.ArgumentParser(
        description="Beast Mode Trading Bot — with PHASE PROFIT MODE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--live",      action="store_true", help="Run in LIVE trading mode")
    parser.add_argument(
        "--phase",
        action="store_true",
        help="Enable PHASE PROFIT MODE ($100 → +$2,500 → secure $2,400 + reset)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()
    setup_logging(log_level=args.log_level)

    bot = BeastModeBot(
        live_mode=args.live,
        phase_mode=args.phase,
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
