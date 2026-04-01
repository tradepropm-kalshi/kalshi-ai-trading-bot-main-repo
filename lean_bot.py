"""
Lean High-Volume Bot — entry point.

A focused, cost-efficient Kalshi trading bot that:
  1. Targets only high-volume, high-win-rate markets (NCAAB, NBA, NFL, politics)
  2. Enriches every AI call with live real-world data (ESPN, Vegas odds,
     Polymarket, Open-Meteo, CoinGecko, NewsAPI, Metaculus)
  3. Runs a single Grok-3 call per market (~$0.015) instead of a 5-model ensemble
  4. Applies Kelly criterion position sizing with a 25% fraction cap
  5. Enforces daily AI budget, max positions, and a circuit-breaker loss limit

Three async loops:
  ┌──────────────────────────────────────┐
  │  ingest   (every 5 min)             │  Pull active markets from Kalshi API
  │  trade    (every 90 sec)            │  Run lean directional strategy cycle
  │  track    (every 2 min)             │  Update P&L, check phase profit target
  └──────────────────────────────────────┘

Usage:
    python lean_bot.py                  # paper trading (default)
    python lean_bot.py --live           # live trading (requires env confirmation)
    python lean_bot.py --once           # single cycle then exit (useful for testing)
    python lean_bot.py --budget 5.0     # override daily AI budget
"""

import argparse
import asyncio
import os
import signal
import sys
from datetime import datetime
from typing import Optional

from src.clients.kalshi_client import KalshiClient
from src.clients.xai_client import XAIClient
from src.config.settings import settings
from src.strategies.lean_directional import run_lean_directional
from src.utils.database import DatabaseManager
from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("lean_bot")

# ---------------------------------------------------------------------------
# Loop intervals
# ---------------------------------------------------------------------------
INGEST_INTERVAL_SECS  = 300   # 5 minutes
TRADE_INTERVAL_SECS   = 90    # 90 seconds
TRACK_INTERVAL_SECS   = 120   # 2 minutes

# ---------------------------------------------------------------------------
# Safety limits
# ---------------------------------------------------------------------------
MAX_DAILY_LOSS_PCT    = 10.0   # Halt if portfolio drops > 10% in a day
PHASE_BASE_CAPITAL    = 100.0  # Capital base for phase-profit mode


class LeanBot:
    """
    Lean high-volume trading bot with three coordinated async loops.

    Args:
        db_manager:    Shared :class:`DatabaseManager` instance.
        kalshi_client: Authenticated :class:`KalshiClient`.
        xai_client:    Configured :class:`XAIClient` (Grok-3).
        live:          When False, positions are paper-only (no real orders).
        daily_budget:  Override for the daily AI cost cap in USD.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        kalshi_client: KalshiClient,
        xai_client: XAIClient,
        live: bool = False,
        daily_budget: float = 10.0,
    ) -> None:
        self.db_manager    = db_manager
        self.kalshi_client = kalshi_client
        self.xai_client    = xai_client
        self.live          = live
        self.daily_budget  = daily_budget

        self._running   = False
        self._stop_event = asyncio.Event()

        # Track start-of-day balance for circuit-breaker
        self._start_balance: Optional[float] = None

        # Override budget in settings for downstream checks
        settings.trading.daily_ai_budget = daily_budget

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, once: bool = False) -> None:
        """
        Start all three loops.

        Args:
            once: If True, run one trade cycle then stop.
        """
        self._running = True
        mode_str = "LIVE" if self.live else "PAPER"
        logger.info(
            "Lean bot starting",
            mode=mode_str,
            daily_budget=f"${self.daily_budget:.2f}",
            trade_interval_secs=TRADE_INTERVAL_SECS,
        )

        # Initialise database schema
        await self.db_manager.initialize()

        # Record start balance for loss circuit breaker
        try:
            bal = await self.kalshi_client.get_balance()
            self._start_balance = float(
                bal.get("balance", 0) or bal.get("available_balance", 0) or 0
            )
            if self._start_balance > 1.0:
                self._start_balance /= 100.0  # cents → dollars
        except Exception:
            self._start_balance = None

        if once:
            await self._trade_cycle()
            return

        await asyncio.gather(
            self._ingest_loop(),
            self._trade_loop(),
            self._track_loop(),
        )

    async def stop(self) -> None:
        """Signal all loops to exit gracefully."""
        self._running = False
        self._stop_event.set()
        logger.info("Lean bot stopping")

    # ------------------------------------------------------------------
    # Ingest loop — pulls active markets into the DB
    # ------------------------------------------------------------------

    async def _ingest_loop(self) -> None:
        """Periodically pull active Kalshi markets into the local database."""
        while self._running:
            try:
                await self._ingest_markets()
            except Exception as exc:  # noqa: BLE001
                logger.error("Ingest error", error=str(exc))

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=INGEST_INTERVAL_SECS
                )
                break  # Stop event fired
            except asyncio.TimeoutError:
                pass  # Normal — loop again

    async def _ingest_markets(self) -> None:
        """Fetch active markets from Kalshi API and upsert into the database."""
        try:
            response = await self.kalshi_client.get_markets(
                status="open",
                limit=200,
            )
            markets_raw = response.get("markets", [])
            if not markets_raw:
                logger.debug("Ingest: no markets returned from Kalshi")
                return

            await self.db_manager.upsert_markets(markets_raw)
            logger.info("Ingest complete", markets_fetched=len(markets_raw))
        except Exception as exc:  # noqa: BLE001
            logger.error("Market ingest failed", error=str(exc))

    # ------------------------------------------------------------------
    # Trade loop — runs the lean directional strategy
    # ------------------------------------------------------------------

    async def _trade_loop(self) -> None:
        """Run the lean directional strategy on a fixed interval."""
        # Small startup delay so ingest has time to populate the DB first
        await asyncio.sleep(15)

        while self._running:
            try:
                await self._trade_cycle()
            except Exception as exc:  # noqa: BLE001
                logger.error("Trade cycle error", error=str(exc))

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=TRADE_INTERVAL_SECS
                )
                break
            except asyncio.TimeoutError:
                pass

    async def _trade_cycle(self) -> None:
        """One trade cycle: check circuit breaker → run lean strategy."""
        # --- Circuit breaker ---
        if self._start_balance and self._start_balance > 0:
            try:
                bal_raw = await self.kalshi_client.get_balance()
                current = float(
                    bal_raw.get("balance", 0)
                    or bal_raw.get("available_balance", 0) or 0
                )
                if current > 1.0:
                    current /= 100.0
                loss_pct = (self._start_balance - current) / self._start_balance * 100
                if loss_pct >= MAX_DAILY_LOSS_PCT:
                    logger.warning(
                        "Daily loss circuit breaker triggered — halting trading",
                        loss_pct=f"{loss_pct:.1f}%",
                        limit_pct=f"{MAX_DAILY_LOSS_PCT:.0f}%",
                    )
                    await self.stop()
                    return
                total_capital = current
            except Exception:
                total_capital = PHASE_BASE_CAPITAL
        else:
            total_capital = PHASE_BASE_CAPITAL

        # --- Phase-profit mode: size against base capital, not total ---
        if getattr(settings.trading, "phase_mode_enabled", True):
            total_capital = min(total_capital, PHASE_BASE_CAPITAL)

        result = await run_lean_directional(
            db_manager=self.db_manager,
            kalshi_client=self.kalshi_client,
            xai_client=self.xai_client,
            total_capital=total_capital,
        )

        logger.info(
            "Trade cycle complete",
            positions_created=result.get("positions_created", 0),
            markets_analyzed=result.get("markets_analyzed", 0),
            capital_deployed=f"${result.get('capital_deployed', 0):.2f}",
            ai_cost=f"${result.get('ai_cost', 0):.4f}",
        )

    # ------------------------------------------------------------------
    # Track loop — P&L monitoring and phase profit check
    # ------------------------------------------------------------------

    async def _track_loop(self) -> None:
        """Periodically update P&L and check the phase profit target."""
        while self._running:
            try:
                await self._track_positions()
            except Exception as exc:  # noqa: BLE001
                logger.error("Track error", error=str(exc))

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=TRACK_INTERVAL_SECS
                )
                break
            except asyncio.TimeoutError:
                pass

    async def _track_positions(self) -> None:
        """Check open positions, log P&L summary, and handle phase profit."""
        try:
            positions = await self.db_manager.get_open_positions()
            if not positions:
                return

            total_pnl = sum(
                getattr(p, "unrealized_pnl", 0.0) or 0.0 for p in positions
            )
            logger.info(
                "Position tracker",
                open_positions=len(positions),
                unrealized_pnl=f"${total_pnl:.2f}",
            )

            # Phase profit check
            phase = await self.db_manager.get_phase_state()
            if phase:
                profit = getattr(phase, "current_phase_profit", 0.0) or 0.0
                target = getattr(
                    settings.trading, "phase_profit_target", 2500.0
                )
                if profit >= target:
                    secured = await self.db_manager.secure_phase_profit(
                        profit, target
                    )
                    if secured:
                        logger.info(
                            "Phase profit target reached — secured",
                            profit=f"${profit:.2f}",
                            target=f"${target:.2f}",
                        )
        except Exception as exc:  # noqa: BLE001
            logger.error("Position tracking failed", error=str(exc))


# ---------------------------------------------------------------------------
# Signal handling — Windows-safe
# ---------------------------------------------------------------------------

def _install_signal_handlers(bot: LeanBot) -> None:
    """
    Install SIGINT/SIGTERM handlers to stop the bot gracefully.

    On Windows, asyncio only supports SIGINT via KeyboardInterrupt, so we
    skip SIGTERM on that platform rather than crashing.
    """
    loop = asyncio.get_event_loop()

    def _handle_signal() -> None:
        logger.info("Signal received — requesting graceful shutdown")
        asyncio.create_task(bot.stop())

    if sys.platform != "win32":
        loop.add_signal_handler(signal.SIGINT,  _handle_signal)
        loop.add_signal_handler(signal.SIGTERM, _handle_signal)
    else:
        # Windows: KeyboardInterrupt is the only reliable mechanism
        signal.signal(signal.SIGINT, lambda *_: asyncio.create_task(bot.stop()))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lean High-Volume Kalshi Trading Bot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live order execution (default: paper trading)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single trade cycle then exit",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=10.0,
        metavar="USD",
        help="Daily AI API cost budget in dollars",
    )
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> None:
    """Async main: initialise clients and start the bot."""
    live = args.live or os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true"

    if live:
        confirm = os.getenv("CONFIRM_LIVE_TRADING", "").lower()
        if confirm != "yes":
            print(
                "\n⚠  LIVE TRADING MODE requested.\n"
                "   Set CONFIRM_LIVE_TRADING=yes to proceed.\n"
                "   Starting in PAPER mode instead.\n"
            )
            live = False

    db_manager    = DatabaseManager()
    kalshi_client = KalshiClient(
        api_key=settings.api.kalshi_api_key,
        base_url=settings.api.kalshi_base_url,
    )
    xai_client    = XAIClient(api_key=settings.api.xai_api_key)

    bot = LeanBot(
        db_manager=db_manager,
        kalshi_client=kalshi_client,
        xai_client=xai_client,
        live=live,
        daily_budget=args.budget,
    )

    _install_signal_handlers(bot)

    def _status(key: str) -> str:
        return "✓ active" if key else "○ add key"

    api = settings.api
    print(
        f"\n{'='*62}\n"
        f"  Lean High-Volume Kalshi Bot\n"
        f"  Mode:           {'LIVE 🔴' if live else 'PAPER 📄'}\n"
        f"  Daily AI budget: ${args.budget:.2f}\n"
        f"  Trade interval:  {TRADE_INTERVAL_SECS}s\n"
        f"  Started:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'='*62}\n"
        f"  DATA SOURCES (no-auth / free):\n"
        f"    ✓ Polymarket cross-reference      (no auth)\n"
        f"    ✓ Manifold Markets odds           (no auth)\n"
        f"    ✓ PredictIt political prices      (no auth)\n"
        f"    ✓ ESPN live scores + injuries     (no auth)\n"
        f"    ✓ MLB Stats API (pitchers/scores) (no auth)\n"
        f"    ✓ Jolpica F1 standings + results  (no auth)\n"
        f"    ✓ Open-Meteo weather forecasts    (no auth)\n"
        f"    ✓ NWS official US venue weather   (no auth)\n"
        f"    ✓ CoinGecko crypto prices         (no auth)\n"
        f"\n"
        f"  DATA SOURCES (free API key required):\n"
        f"    {_status(api.odds_api_key)} Vegas odds (The Odds API) "
        f"{'             ' if api.odds_api_key else '  → set ODDS_API_KEY'}\n"
        f"    {_status(api.fred_api_key)} FRED economic data        "
        f"{'             ' if api.fred_api_key else '  → set FRED_API_KEY'}\n"
        f"    {_status(api.bls_api_key)} BLS CPI/Jobs data         "
        f"{'             ' if api.bls_api_key else '  → set BLS_API_KEY'}\n"
        f"    {_status(api.newsapi_key)} NewsAPI headlines          "
        f"{'             ' if api.newsapi_key else '  → set NEWSAPI_KEY'}\n"
        f"    {_status(api.metaculus_api_key)} Metaculus predictions     "
        f"{'             ' if api.metaculus_api_key else '  → set METACULUS_API_KEY'}\n"
        f"{'='*62}\n"
        "  Press Ctrl+C to stop gracefully.\n"
    )

    try:
        await bot.start(once=args.once)
    finally:
        await kalshi_client.close()
        await db_manager.close()
        logger.info("Lean bot exited cleanly")


if __name__ == "__main__":
    args = _parse_args()
    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        print("\nInterrupted — bye!")
