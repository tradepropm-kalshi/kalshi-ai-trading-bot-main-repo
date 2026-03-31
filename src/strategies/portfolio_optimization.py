"""
Advanced Portfolio Optimization — Original RyanFrigo Kelly + Mean-Variance
Extended with Bible Phase effective capital ($100 base + current_phase_profit)
"""

import asyncio
import numpy as np
from dataclasses import dataclass
from typing import List

from src.utils.database import DatabaseManager
from src.clients.kalshi_client import KalshiClient
from src.clients.xai_client import XAIClient
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger

@dataclass
class MarketOpportunity:
    market_id: str
    title: str
    yes_price: float
    no_price: float
    volume: int
    days_to_expiry: int
    ai_probability: float
    ai_confidence: float
    edge: float
    kelly_fraction: float = 0.0

@dataclass
class PortfolioAllocation:
    positions: List[MarketOpportunity]
    total_capital_used: float = 0.0
    expected_return: float = 0.0
    sharpe_ratio: float = 0.0
    phase_capital_used: float = 0.0

class AdvancedPortfolioOptimizer:
    def __init__(self, db_manager: DatabaseManager, kalshi_client: KalshiClient, xai_client: XAIClient):
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.xai_client = xai_client
        self.logger = get_trading_logger("portfolio_optimizer")
        self.max_position_fraction = getattr(settings.trading, 'max_single_position', 0.25)
        self.min_position_size = getattr(settings.trading, 'min_position_size', 5)
        self.kelly_fraction_multiplier = getattr(settings.trading, 'kelly_fraction', 0.25)

    async def get_effective_capital(self) -> float:
        if not settings.trading.phase_mode_enabled:
            balance = await self.kalshi_client.get_balance()
            return (balance.get('balance', 0) + balance.get('portfolio_value', 0)) / 100.0

        phase = await self.db_manager.get_phase_state()
        base = getattr(settings.trading, 'phase_base_capital', 100.0)
        current_profit = phase.get('current_phase_profit', 0.0)
        effective = base + current_profit
        self.logger.info(f"PHASE OPTIMIZER → effective capital = ${effective:.2f} (base ${base} + profit ${current_profit:.2f})")
        return effective

    async def optimize_portfolio(self, opportunities: List[MarketOpportunity]) -> PortfolioAllocation:
        effective_capital = await self.get_effective_capital()
        allocation = PortfolioAllocation(positions=[], phase_capital_used=effective_capital)

        for opp in opportunities:
            if opp.edge <= 0:
                continue

            kelly_size = effective_capital * opp.kelly_fraction * self.kelly_fraction_multiplier
            position_size = min(kelly_size, effective_capital * self.max_position_fraction)

            if position_size >= self.min_position_size:
                opp.kelly_fraction = position_size / effective_capital
                allocation.positions.append(opp)
                allocation.total_capital_used += position_size
                allocation.expected_return += opp.edge * position_size

        if allocation.total_capital_used > 0:
            allocation.sharpe_ratio = (allocation.expected_return / allocation.total_capital_used) * 10

        self.logger.info(f"Portfolio optimized: {len(allocation.positions)} positions, ${allocation.total_capital_used:.2f} used")
        return allocation

async def run_portfolio_optimization(db_manager: DatabaseManager, kalshi_client: KalshiClient, xai_client: XAIClient):
    logger = get_trading_logger("portfolio_optimization")
    logger.info("Running portfolio optimization with Bible phase-aware capital")
    optimizer = AdvancedPortfolioOptimizer(db_manager, kalshi_client, xai_client)
    opportunities = []
    result = await optimizer.optimize_portfolio(opportunities)
    return result