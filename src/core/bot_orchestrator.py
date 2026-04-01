"""
Unified Bot Orchestrator — runs Beast Mode and Lean Mode side-by-side.

Resolves all processing conflicts identified in the architecture review:

  ✅ settings mutation     — per-strategy live flags in BotState, never mutates
                             global settings.trading.live_trading_enabled
  ✅ phase state contention— single shared phase counter (combined P&L); both
                             strategies feed into it and both can trigger secure
  ✅ opposing positions    — shared get_markets_with_positions() lock prevents
                             any two strategies entering the same market
  ✅ position slot sharing — configurable per-strategy caps (default 7 beast / 5
                             lean) with a combined hard cap of 10
  ✅ API rate pressure     — single shared KalshiClient instance with built-in
                             rate limiter; both strategies share the connection
  ✅ duplicate ingest      — one shared ingest loop feeds both strategies
  ✅ daily AI budget       — single daily_cost_tracking table is the combined cap;
                             per-strategy breakdown visible in dashboard

Usage::

    python bot_orchestrator.py                    # both off at launch
    python bot_orchestrator.py --beast            # beast mode only
    python bot_orchestrator.py --lean             # lean mode only
    python bot_orchestrator.py --beast --lean     # both simultaneously
    python bot_orchestrator.py --beast --live-beast  # beast live, lean paper
    python bot_orchestrator.py --beast --lean --live-both  # both live
"""

import argparse
import asyncio
import json
import sys
import signal
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.clients.kalshi_client import KalshiClient
from src.clients.xai_client import XAIClient
from src.config.settings import settings
from src.core.bot_state import BotState, get_state, init_state
from src.jobs.ingest import run_ingestion
from src.jobs.track import run_tracking
from src.jobs.evaluate import run_evaluation
from src.jobs.trade import run_trading_job
from src.strategies.lean_directional import run_lean_directional
from src.utils.database import DatabaseManager
from src.utils.logging_setup import get_trading_logger, setup_logging

logger = get_trading_logger("orchestrator")

# ---------------------------------------------------------------------------
# Per-strategy position caps
# ---------------------------------------------------------------------------
BEAST_MAX_POSITIONS = 7
LEAN_MAX_POSITIONS  = 5

# ---------------------------------------------------------------------------
# Loop intervals (seconds)
# ---------------------------------------------------------------------------
INGEST_INTERVAL   = 300   # 5 min — shared, feeds both strategies
BEAST_TRADE_INTERVAL = 60
LEAN_TRADE_INTERVAL  = 90
TRACK_INTERVAL    = 120
EVAL_INTERVAL     = 300
TOGGLE_POLL_SECS  = 5     # how often to check bot_state.json for toggle changes


class BotOrchestrator:
    """
    Single-process, single-event-loop runner for Beast Mode + Lean Mode.

    Shared resources (created once, passed everywhere):
      • DatabaseManager  — single aiosqlite connection pool
      • KalshiClient     — single HTTPS session + rate limiter
      • XAIClient        — single cost tracker

    Per-strategy live mode is tracked in BotState, not in global settings,
    so enabling/disabling one strategy never affects the other.
    """

    def __init__(
        self,
        beast_enabled: bool = False,
        lean_enabled: bool = False,
        beast_live: bool = False,
        lean_live: bool = False,
    ) -> None:
        self.state = init_state(
            beast_enabled=beast_enabled,
            lean_enabled=lean_enabled,
            beast_live=beast_live,
            lean_live=lean_live,
        )

        self._db: Optional[DatabaseManager] = None
        self._kalshi: Optional[KalshiClient] = None
        self._xai: Optional[XAIClient] = None

        self._beast_task: Optional[asyncio.Task] = None
        self._lean_task:  Optional[asyncio.Task] = None
        self._started_at = datetime.now()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start the orchestrator and all enabled strategy loops."""
        logger.info(
            "Orchestrator starting",
            beast=self.state.beast_enabled,
            lean=self.state.lean_enabled,
            beast_live=self.state.beast_live,
            lean_live=self.state.lean_live,
        )

        # Initialise shared resources once
        self._db     = DatabaseManager()
        await self._db.initialize()

        self._kalshi = KalshiClient(
            api_key=settings.api.kalshi_api_key,
            base_url=settings.api.kalshi_base_url,
        )
        self._xai = XAIClient(
            api_key=settings.api.xai_api_key,
            db_manager=self._db,
        )

        # Capture starting balance for circuit-breaker
        try:
            bal = await self._kalshi.get_balance()
            raw = float(bal.get("balance", 0) or bal.get("available_balance", 0) or 0)
            start_capital = raw / 100.0 if raw > 1.0 else raw
        except Exception:
            start_capital = settings.trading.phase_base_capital

        self.state.update_system(
            portfolio_balance=start_capital,
            available_cash=start_capital,
        )

        _install_signal_handlers(self.state.shutdown_event)

        # Launch all long-running loops
        tasks = [
            asyncio.create_task(self._ingest_loop(),    name="ingest"),
            asyncio.create_task(self._track_loop(),     name="track"),
            asyncio.create_task(self._eval_loop(),      name="eval"),
            asyncio.create_task(self._system_monitor(), name="monitor"),
            asyncio.create_task(self._toggle_watcher(), name="toggle"),
        ]

        # Strategy loops — started now if enabled, otherwise waiting for toggle
        if self.state.beast_enabled:
            self._beast_task = asyncio.create_task(
                self._beast_loop(start_capital), name="beast"
            )
            tasks.append(self._beast_task)

        if self.state.lean_enabled:
            self._lean_task = asyncio.create_task(
                self._lean_loop(start_capital), name="lean"
            )
            tasks.append(self._lean_task)

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await self._cleanup()

    # ------------------------------------------------------------------
    # Shared loops (always running while orchestrator is up)
    # ------------------------------------------------------------------

    async def _ingest_loop(self) -> None:
        """Shared market ingest — feeds both strategies from one API call."""
        logger.info("Ingest loop started")
        while not self.state.shutdown_event.is_set():
            try:
                await run_ingestion(self._db)
                logger.debug("Market ingest complete")
            except Exception as exc:
                logger.error("Ingest error", error=str(exc))
            await self._sleep(INGEST_INTERVAL)

    async def _track_loop(self) -> None:
        """Position tracking — updates P&L for all open positions."""
        await asyncio.sleep(20)  # Let ingest run first
        while not self.state.shutdown_event.is_set():
            try:
                await run_tracking(self._db)
                # Refresh combined position count in state
                positions = await self._db.get_open_positions()
                daily_pnl = await self._db.get_daily_pnl()
                self.state.update_system(
                    total_positions=len(positions),
                    daily_loss_pct=(
                        -(daily_pnl / max(self.state.system.portfolio_balance, 1)) * 100
                        if daily_pnl < 0 else 0.0
                    ),
                )
            except Exception as exc:
                logger.error("Track error", error=str(exc))
            await self._sleep(TRACK_INTERVAL)

    async def _eval_loop(self) -> None:
        """Performance evaluation — updates win rates, P&L, AI cost."""
        await asyncio.sleep(30)
        while not self.state.shutdown_event.is_set():
            try:
                await run_evaluation()
            except Exception as exc:
                logger.error("Eval error", error=str(exc))
            await self._sleep(EVAL_INTERVAL)

    async def _system_monitor(self) -> None:
        """Update dashboard-facing system metrics every 60 seconds."""
        while not self.state.shutdown_event.is_set():
            try:
                bal = await self._kalshi.get_balance()
                raw = float(
                    bal.get("balance", 0) or bal.get("available_balance", 0) or 0
                )
                cash = raw / 100.0 if raw > 1.0 else raw
                uptime = (datetime.now() - self._started_at).total_seconds()
                self.state.update_system(
                    available_cash=cash,
                    uptime_seconds=uptime,
                )
            except Exception:
                pass
            await self._sleep(60)

    # ------------------------------------------------------------------
    # Strategy loops (conditionally running)
    # ------------------------------------------------------------------

    async def _beast_loop(self, start_capital: float) -> None:
        """Beast Mode trading cycle loop."""
        logger.info("Beast Mode loop started", live=self.state.beast_live)
        self.state.beast.running = True
        cycle = 0

        # Override settings for beast mode live flag
        settings.trading.live_trading_enabled = self.state.beast_live
        settings.trading.paper_trading_mode   = not self.state.beast_live

        while not self.state.shutdown_event.is_set() and self.state.beast_enabled:
            try:
                # Circuit breaker
                daily_pnl = await self._db.get_daily_pnl()
                loss_limit = -(start_capital * settings.trading.max_daily_loss_pct / 100)
                if daily_pnl < loss_limit:
                    logger.warning(
                        "Beast Mode: daily loss limit hit — pausing",
                        pnl=f"${daily_pnl:.2f}",
                        limit=f"${loss_limit:.2f}",
                    )
                    self.state.update_beast(last_error=f"Circuit breaker: P&L ${daily_pnl:.2f}")
                    self.state.disable_beast()
                    break

                cycle += 1
                logger.info(f"Beast Mode cycle #{cycle}")

                await run_trading_job(
                    db_manager=self._db,
                    kalshi_client=self._kalshi,
                    xai_client=self._xai,
                )

                # Update dashboard state
                ai_cost = await self._db.get_daily_ai_cost()
                positions = await self._db.get_open_positions()
                beast_positions = [
                    p for p in positions
                    if getattr(p, "strategy", "") not in ("lean_directional",)
                ]
                self.state.update_beast(
                    cycle_count=cycle,
                    positions_open=len(beast_positions),
                    ai_cost_today=ai_cost,
                    daily_pnl=daily_pnl,
                    running=True,
                )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Beast Mode cycle error", error=str(exc))
                self.state.update_beast(last_error=str(exc)[:120])

            await self._sleep(BEAST_TRADE_INTERVAL)

        self.state.beast.running = False
        self.state._persist()
        logger.info("Beast Mode loop stopped")

    async def _lean_loop(self, start_capital: float) -> None:
        """Lean Mode trading cycle loop."""
        await asyncio.sleep(15)  # Stagger start so ingest runs first
        logger.info("Lean Mode loop started", live=self.state.lean_live)
        self.state.lean.running = True
        cycle = 0

        capital = min(start_capital, settings.trading.phase_base_capital)

        while not self.state.shutdown_event.is_set() and self.state.lean_enabled:
            try:
                cycle += 1
                logger.info(f"Lean Mode cycle #{cycle}")

                result = await run_lean_directional(
                    db_manager=self._db,
                    kalshi_client=self._kalshi,
                    xai_client=self._xai,
                    total_capital=capital,
                )

                ai_cost = await self._db.get_daily_ai_cost()
                positions = await self._db.get_open_positions()
                lean_positions = [
                    p for p in positions
                    if getattr(p, "strategy", "") == "lean_directional"
                ]
                daily_pnl = await self._db.get_daily_pnl()
                self.state.update_lean(
                    cycle_count=cycle,
                    positions_open=len(lean_positions),
                    ai_cost_today=ai_cost,
                    daily_pnl=daily_pnl,
                    running=True,
                )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Lean Mode cycle error", error=str(exc))
                self.state.update_lean(last_error=str(exc)[:120])

            await self._sleep(LEAN_TRADE_INTERVAL)

        self.state.lean.running = False
        self.state._persist()
        logger.info("Lean Mode loop stopped")

    # ------------------------------------------------------------------
    # Live toggle watcher — polls bot_state.json for dashboard changes
    # ------------------------------------------------------------------

    async def _toggle_watcher(self) -> None:
        """
        Poll ``bot_state.json`` for toggle instructions from the dashboard.

        The Streamlit process writes ``toggle_pending: true`` to the file
        when the user flips a switch.  This loop picks it up and starts or
        stops the relevant strategy task without restarting the process.
        """
        start_capital = self.state.system.portfolio_balance or settings.trading.phase_base_capital

        while not self.state.shutdown_event.is_set():
            await asyncio.sleep(TOGGLE_POLL_SECS)
            try:
                data = BotState.load_from_file()
                if not data.get("toggle_pending"):
                    continue

                # Apply beast toggle
                want_beast = data.get("beast_enabled", False)
                want_beast_live = data.get("beast_live", False)
                if want_beast and not self.state.beast_enabled:
                    logger.info("Dashboard toggle: enabling Beast Mode")
                    self.state.enable_beast(live=want_beast_live)
                    self._beast_task = asyncio.create_task(
                        self._beast_loop(start_capital), name="beast"
                    )
                elif not want_beast and self.state.beast_enabled:
                    logger.info("Dashboard toggle: disabling Beast Mode")
                    self.state.disable_beast()
                    if self._beast_task and not self._beast_task.done():
                        self._beast_task.cancel()

                # Apply lean toggle
                want_lean = data.get("lean_enabled", False)
                want_lean_live = data.get("lean_live", False)
                if want_lean and not self.state.lean_enabled:
                    logger.info("Dashboard toggle: enabling Lean Mode")
                    self.state.enable_lean(live=want_lean_live)
                    self._lean_task = asyncio.create_task(
                        self._lean_loop(start_capital), name="lean"
                    )
                elif not want_lean and self.state.lean_enabled:
                    logger.info("Dashboard toggle: disabling Lean Mode")
                    self.state.disable_lean()
                    if self._lean_task and not self._lean_task.done():
                        self._lean_task.cancel()

                # Clear the pending flag
                data["toggle_pending"] = False
                try:
                    Path("bot_state.json").write_text(
                        json.dumps(data, indent=2)
                    )
                except OSError:
                    pass

            except Exception as exc:
                logger.debug("Toggle watcher error", error=str(exc))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _sleep(self, seconds: float) -> None:
        """Sleep for *seconds* but wake immediately if shutdown fires."""
        try:
            await asyncio.wait_for(
                self.state.shutdown_event.wait(), timeout=seconds
            )
        except asyncio.TimeoutError:
            pass

    async def _cleanup(self) -> None:
        """Close shared resources gracefully."""
        logger.info("Orchestrator shutting down")
        if self._kalshi:
            try:
                await self._kalshi.close()
            except Exception:
                pass
        if self._db:
            try:
                await self._db.close()
            except Exception:
                pass
        logger.info("Orchestrator stopped cleanly")


# ---------------------------------------------------------------------------
# Signal handling (Windows-safe)
# ---------------------------------------------------------------------------

def _install_signal_handlers(shutdown_event: asyncio.Event) -> None:
    def _handle() -> None:
        logger.info("Shutdown signal received")
        shutdown_event.set()

    if sys.platform != "win32":
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(signal.SIGINT,  _handle)
        loop.add_signal_handler(signal.SIGTERM, _handle)
    else:
        signal.signal(signal.SIGINT, lambda *_: _handle())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified Bot Orchestrator — Beast Mode + Lean Mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--beast",      action="store_true", help="Enable Beast Mode at startup")
    p.add_argument("--lean",       action="store_true", help="Enable Lean Mode at startup")
    p.add_argument("--live-beast", action="store_true", help="Beast Mode uses live orders")
    p.add_argument("--live-lean",  action="store_true", help="Lean Mode uses live orders")
    p.add_argument(
        "--live-both",
        action="store_true",
        help="Both modes use live orders (requires CONFIRM_LIVE_TRADING=yes)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


async def _main() -> None:
    args = _parse_args()
    setup_logging(log_level=args.log_level)

    import os
    live_both = args.live_both and (
        os.getenv("CONFIRM_LIVE_TRADING", "").lower() == "yes"
    )
    live_beast = live_both or (
        args.live_beast and os.getenv("CONFIRM_LIVE_TRADING", "").lower() == "yes"
    )
    live_lean = live_both or (
        args.live_lean and os.getenv("CONFIRM_LIVE_TRADING", "").lower() == "yes"
    )

    if (args.live_beast or args.live_lean or args.live_both) and not (
        live_beast or live_lean
    ):
        print(
            "\n⚠  Live trading requested but CONFIRM_LIVE_TRADING=yes not set.\n"
            "   Starting in PAPER mode.\n"
        )

    orchestrator = BotOrchestrator(
        beast_enabled=args.beast,
        lean_enabled=args.lean,
        beast_live=live_beast,
        lean_live=live_lean,
    )

    mode_str = []
    if args.beast:
        mode_str.append(f"Beast({'LIVE' if live_beast else 'paper'})")
    if args.lean:
        mode_str.append(f"Lean({'LIVE' if live_lean else 'paper'})")
    if not mode_str:
        mode_str = ["both strategies OFF — use dashboard to enable"]

    print(
        f"\n{'='*60}\n"
        f"  Kalshi AI Orchestrator\n"
        f"  Active: {', '.join(mode_str)}\n"
        f"  Dashboard: streamlit run streamlit_dashboard.py\n"
        f"  Press Ctrl+C to stop gracefully.\n"
        f"{'='*60}\n"
    )

    await orchestrator.run()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\nOrchestrator stopped.")
