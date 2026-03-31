"""
Unified Advanced Trading System - The Beast Mode 🚀 — NOW WITH PHASE PROFIT MODE
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

from src.clients.kalshi_client import KalshiClient
from src.clients.xai_client import XAIClient
from src.utils.database import DatabaseManager, Market, Position
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger

from src.strategies.market_making import AdvancedMarketMaker, run_market_making_strategy
from src.strategies.portfolio_optimization import AdvancedPortfolioOptimizer, run_portfolio_optimization
from src.strategies.quick_flip_scalping import run_quick_flip_strategy, QuickFlipConfig


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
    directional_expected_return: float = 0.0
    total_capital_used: float = 0.0
    portfolio_expected_return: float = 0.0
    portfolio_sharpe_ratio: float = 0.0
    portfolio_volatility: float = 0.0
    max_portfolio_drawdown: float = 0.0
    correlation_score: float = 0.0
    diversification_ratio: float = 0.0
    total_positions: int = 0
    capital_efficiency: float = 0.0
    expected_annual_return: float = 0.0


class UnifiedAdvancedTradingSystem:
    def __init__(
        self,
        db_manager: DatabaseManager,
        kalshi_client: KalshiClient,
        xai_client: XAIClient,
        config: Optional[TradingSystemConfig] = None
    ):
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.xai_client = xai_client
        self.config = config or TradingSystemConfig()
        self.logger = get_trading_logger("unified_trading_system")
        
        self.total_capital = 100.0
        self.last_rebalance = datetime.now()
        self.market_maker = None
        self.portfolio_optimizer = None

    async def async_initialize(self):
        try:
            balance_response = await self.kalshi_client.get_balance()
            available_cash = balance_response.get('balance', 0) / 100
            total_position_value = balance_response.get('portfolio_value', 0) / 100
            total_portfolio_value = available_cash + total_position_value

            if settings.trading.phase_mode_enabled:
                phase = await self.db_manager.get_phase_state()
                self.total_capital = settings.trading.phase_base_capital + phase.get('current_phase_profit', 0.0)
                self.logger.info(f"🎯 PHASE MODE → effective capital = ${self.total_capital:.2f} "
                                 f"(base ${settings.trading.phase_base_capital:.2f} + phase profit ${phase['current_phase_profit']:.2f})")
            else:
                self.total_capital = total_portfolio_value
                self.logger.info(f"💰 PORTFOLIO VALUE (non-phase): ${self.total_capital:.2f}")

            self.market_making_capital = self.total_capital * self.config.market_making_allocation
            self.directional_capital = self.total_capital * self.config.directional_trading_allocation
            self.quick_flip_capital = self.total_capital * self.config.quick_flip_allocation
            self.arbitrage_capital = self.total_capital * self.config.arbitrage_allocation

            self.market_maker = AdvancedMarketMaker(self.db_manager, self.kalshi_client, self.xai_client)
            self.portfolio_optimizer = AdvancedPortfolioOptimizer(self.db_manager, self.kalshi_client, self.xai_client)

            self.logger.info(f"🎯 CAPITAL ALLOCATION: Market Making=${self.market_making_capital:.2f}, "
                             f"Directional=${self.directional_capital:.2f}, Quick Flip=${self.quick_flip_capital:.2f}")

        except Exception as e:
            self.logger.error(f"Failed to initialize capital: {e}")
            self.total_capital = settings.trading.phase_base_capital if settings.trading.phase_mode_enabled else 100.0

    async def execute_unified_trading_strategy(self) -> TradingSystemResults:
        results = TradingSystemResults()
        try:
            # Run the three strategies in parallel
            mm_task = asyncio.create_task(run_market_making_strategy(
                self.db_manager, self.kalshi_client, self.xai_client))
            po_task = asyncio.create_task(run_portfolio_optimization(
                self.db_manager, self.kalshi_client, self.xai_client))
            qf_task = asyncio.create_task(run_quick_flip_strategy(
                self.db_manager, self.kalshi_client, self.xai_client, self.total_capital))
            
            mm_results, po_results, qf_results = await asyncio.gather(mm_task, po_task, qf_task)
            
            # Aggregate results
            results.market_making_orders = mm_results.get('orders_placed', 0)
            results.market_making_exposure = mm_results.get('total_exposure', 0.0)
            results.market_making_expected_profit = mm_results.get('expected_profit', 0.0)
            results.directional_positions = po_results.get('positions_created', 0)
            results.directional_exposure = po_results.get('total_capital_used', 0.0)
            results.total_positions = (results.market_making_orders + results.directional_positions)
            results.total_capital_used = self.total_capital
            results.capital_efficiency = 1.0 if self.total_capital > 0 else 0.0
            
            self.logger.info("✅ Unified trading strategy completed successfully")
            return results
        except Exception as e:
            self.logger.error(f"Error in unified trading strategy: {e}")
            return results


async def run_unified_trading_system(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
    xai_client: XAIClient,
    config: Optional[TradingSystemConfig] = None
) -> TradingSystemResults:
    logger = get_trading_logger("unified_trading_main")
    try:
        trading_system = UnifiedAdvancedTradingSystem(db_manager, kalshi_client, xai_client, config)
        await trading_system.async_initialize()
        results = await trading_system.execute_unified_trading_strategy()
        return results
    except Exception as e:
        logger.error(f"Error in unified trading system: {e}")
        return TradingSystemResults()