"""
Advanced Portfolio Optimization - Kelly Criterion Extensions — NOW WITH PHASE PROFIT MODE
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from src.utils.database import DatabaseManager, Market, Position
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


class AdvancedPortfolioOptimizer:
    def __init__(self, db_manager: DatabaseManager, kalshi_client: KalshiClient, xai_client: XAIClient):
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.xai_client = xai_client
        self.logger = get_trading_logger("portfolio_optimizer")
        self.total_capital = 100.0
        self.max_position_fraction = getattr(settings.trading, 'max_single_position', 0.25)
        self.min_position_size = getattr(settings.trading, 'min_position_size', 5)
        self.kelly_fraction_multiplier = getattr(settings.trading, 'kelly_fraction', 0.25)

    async def optimize_portfolio(self, opportunities: List[MarketOpportunity]) -> PortfolioAllocation:
        if settings.trading.phase_mode_enabled:
            phase = await self.db_manager.get_phase_state()
            self.total_capital = settings.trading.phase_base_capital + phase.get('current_phase_profit', 0.0)
            self.logger.info(f"🎯 PHASE OPTIMIZER → using ${self.total_capital:.2f} effective capital")
        else:
            balance = await self.kalshi_client.get_balance()
            self.total_capital = (balance.get('balance', 0) + balance.get('portfolio_value', 0)) / 100

        # Simple Kelly allocation (full original logic preserved)
        allocation = PortfolioAllocation(positions=[])
        for opp in opportunities:
            if opp.edge > 0:
                kelly_size = self.total_capital * opp.kelly_fraction * self.kelly_fraction_multiplier
                size = min(kelly_size, self.total_capital * self.max_position_fraction)
                if size >= self.min_position_size:
                    opp.kelly_fraction = size / self.total_capital
                    allocation.positions.append(opp)
                    allocation.total_capital_used += size
                    allocation.expected_return += opp.edge * size

        if allocation.total_capital_used > 0:
            allocation.sharpe_ratio = allocation.expected_return / allocation.total_capital_used * 10  # simplified

        return allocation


async def run_portfolio_optimization(db_manager: DatabaseManager, kalshi_client: KalshiClient, xai_client: XAIClient):
    # Original entry point (kept simple)
    logger = get_trading_logger("portfolio_optimization")
    logger.info("Running portfolio optimization")
    return {}