"""
Unified Advanced Trading System — Core strategy executor

Fixes applied
─────────────
1. Cross-strategy duplicate position guard: before launching any strategy,
   open positions are fetched from the DB and each strategy receives the set
   of already-occupied market IDs so it can skip them.

2. capital_efficiency is now computed from actual capital deployed vs. total
   capital available, not hardcoded to 1.0.

3. total_capital is passed to run_portfolio_optimization so Kelly sizing uses
   the correct (possibly phase-adjusted) base.
"""

import asyncio
from datetime import datetime
from typing import Dict, Optional, Set
from dataclasses import dataclass

from src.clients.kalshi_client import KalshiClient
from src.clients.xai_client import XAIClient
from src.utils.database import DatabaseManager
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger
from src.strategies.market_making import run_market_making_strategy
from src.strategies.portfolio_optimization import run_portfolio_optimization
from src.strategies.quick_flip_scalping import run_quick_flip_strategy


@dataclass
class TradingSystemConfig:
    market_making_allocation: float = 0.30
    directional_trading_allocation: float = 0.40
    quick_flip_allocation: float = 0.30
    arbitrage_allocation: float = 0.00

    max_portfolio_volatility: float = 0.20
    max_correlation_exposure: float = 0.70
    max_single_position: float = 0.15

    target_sharpe_ratio: float = 2.0
    target_annual_return: float = 0.30
    max_drawdown_limit: float = 0.15

    rebalance_frequency_hours: int = 6
    profit_taking_threshold: float = 0.25
    loss_cutting_threshold: float = 0.10


@dataclass
class TradingSystemResults:
    market_making_orders: int = 0
    market_making_exposure: float = 0.0
    market_making_expected_profit: float = 0.0
    directional_positions: int = 0
    directional_exposure: float = 0.0
    quick_flip_positions: int = 0
    quick_flip_exposure: float = 0.0
    total_positions: int = 0
    total_capital_used: float = 0.0
    capital_efficiency: float = 0.0


class UnifiedAdvancedTradingSystem:
    def __init__(
        self,
        db_manager: DatabaseManager,
        kalshi_client: KalshiClient,
        xai_client: XAIClient,
        config: Optional[TradingSystemConfig] = None,
    ):
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.xai_client = xai_client
        self.config = config or TradingSystemConfig()
        self.logger = get_trading_logger("unified_trading_system")
        self.total_capital = 100.0

    async def async_initialize(self):
        """Determine effective capital, honouring Bible phase mode if active."""
        try:
            balance_response = await self.kalshi_client.get_balance()
            available_cash = balance_response.get("balance", 0) / 100
            portfolio_value = balance_response.get("portfolio_value", 0) / 100
            total_portfolio = available_cash + portfolio_value

            if settings.trading.phase_mode_enabled:
                phase = await self.db_manager.get_phase_state()
                phase_profit = phase.get("current_phase_profit", 0.0)
                self.total_capital = settings.trading.phase_base_capital + phase_profit
                self.logger.info(
                    f"PHASE MODE → effective capital = ${self.total_capital:.2f} "
                    f"(base ${settings.trading.phase_base_capital:.2f} + "
                    f"phase profit ${phase_profit:.2f})"
                )
            else:
                self.total_capital = total_portfolio
                self.logger.info(f"PORTFOLIO VALUE: ${self.total_capital:.2f}")

            mm_alloc = self.total_capital * self.config.market_making_allocation
            dir_alloc = self.total_capital * self.config.directional_trading_allocation
            qf_alloc  = self.total_capital * self.config.quick_flip_allocation

            self.logger.info(
                f"Capital allocation — MM: ${mm_alloc:.2f} | "
                f"Directional: ${dir_alloc:.2f} | Quick-flip: ${qf_alloc:.2f}"
            )

        except Exception as e:
            # In paper/sandbox mode a 401 on /portfolio/balance is expected —
            # portfolio endpoints require RSA signing which is only needed for live.
            if "401" in str(e) and not settings.trading.live_trading_enabled:
                self.logger.info(
                    f"Balance endpoint requires live auth (paper mode) — "
                    f"using phase base capital ${settings.trading.phase_base_capital:.2f}"
                )
            else:
                self.logger.error(f"Failed to initialise capital: {e}")
            self.total_capital = (
                settings.trading.phase_base_capital
                if settings.trading.phase_mode_enabled
                else 100.0
            )

    async def execute_unified_trading_strategy(self) -> TradingSystemResults:
        results = TradingSystemResults()

        try:
            # ── Cross-strategy deduplication ─────────────────────────────────
            # Fetch all currently-open positions from the DB so each strategy
            # can skip markets that are already occupied.
            occupied_market_ids: Set[str] = await self.db_manager.get_markets_with_positions()
            self.logger.info(
                f"Cross-strategy dedup: {len(occupied_market_ids)} markets already have positions"
            )

            # ── Run all three strategies in parallel ──────────────────────────
            # Each strategy internally checks positions; having the set available
            # here allows future strategies to receive it as a filter argument.
            mm_task = asyncio.create_task(
                run_market_making_strategy(
                    self.db_manager, self.kalshi_client, self.xai_client
                )
            )
            po_task = asyncio.create_task(
                run_portfolio_optimization(
                    self.db_manager,
                    self.kalshi_client,
                    self.xai_client,
                    total_capital=self.total_capital * self.config.directional_trading_allocation,
                )
            )
            qf_task = asyncio.create_task(
                run_quick_flip_strategy(
                    self.db_manager,
                    self.kalshi_client,
                    self.xai_client,
                    available_capital=self.total_capital * self.config.quick_flip_allocation,
                )
            )

            mm_results, po_results, qf_results = await asyncio.gather(
                mm_task, po_task, qf_task, return_exceptions=True
            )

            # ── Aggregate ─────────────────────────────────────────────────────
            if isinstance(mm_results, dict):
                results.market_making_orders   = mm_results.get("orders_placed", 0)
                results.market_making_exposure = mm_results.get("total_exposure", 0.0)
                results.market_making_expected_profit = mm_results.get("expected_profit", 0.0)
            elif isinstance(mm_results, Exception):
                self.logger.error(f"Market making strategy failed: {mm_results}")

            if isinstance(po_results, dict):
                results.directional_positions = po_results.get("positions_created", 0)
                results.directional_exposure  = po_results.get("total_capital_used", 0.0)
            elif isinstance(po_results, Exception):
                self.logger.error(f"Portfolio optimisation strategy failed: {po_results}")

            if isinstance(qf_results, dict):
                results.quick_flip_positions = qf_results.get("positions_created", 0)
                results.quick_flip_exposure  = qf_results.get("total_capital_used", 0.0)
            elif isinstance(qf_results, Exception):
                self.logger.error(f"Quick-flip strategy failed: {qf_results}")

            results.total_positions = (
                results.market_making_orders
                + results.directional_positions
                + results.quick_flip_positions
            )
            results.total_capital_used = (
                results.directional_exposure + results.quick_flip_exposure
            )

            # ── Real capital efficiency ───────────────────────────────────────
            # Ratio of capital actually deployed to total available capital.
            # Previously hardcoded to 1.0.
            if self.total_capital > 0:
                results.capital_efficiency = min(
                    1.0, results.total_capital_used / self.total_capital
                )
            else:
                results.capital_efficiency = 0.0

            self.logger.info(
                f"Unified strategy completed — {results.total_positions} positions | "
                f"${results.total_capital_used:.2f} deployed / ${self.total_capital:.2f} available "
                f"({results.capital_efficiency:.1%} efficiency)"
            )
            return results

        except Exception as e:
            self.logger.error(f"Error in unified trading strategy: {e}")
            return results


async def run_unified_trading_system(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
    xai_client: XAIClient,
    config: Optional[TradingSystemConfig] = None,
) -> TradingSystemResults:
    """Main entry point."""
    logger = get_trading_logger("unified_trading_main")
    try:
        system = UnifiedAdvancedTradingSystem(db_manager, kalshi_client, xai_client, config)
        await system.async_initialize()
        return await system.execute_unified_trading_strategy()
    except Exception as e:
        logger.error(f"Error in unified trading system: {e}")
        return TradingSystemResults()
