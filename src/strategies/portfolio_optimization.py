"""
Advanced Portfolio Optimizer
Original RyanFrigo design preserved exactly
NOW WITH BIBLE PHASE EFFECTIVE CAPITAL FOR KELLY SIZING
"""

from typing import Dict, List, Optional
import numpy as np

from src.clients.kalshi_client import KalshiClient
from src.clients.xai_client import XAIClient
from src.utils.database import DatabaseManager
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger


class AdvancedPortfolioOptimizer:
    def __init__(
        self,
        db_manager: DatabaseManager,
        kalshi_client: KalshiClient,
        xai_client: XAIClient
    ):
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.xai_client = xai_client
        self.logger = get_trading_logger("portfolio_optimizer")

    async def optimize_portfolio(self, opportunities: List[Dict], total_capital: float) -> Dict:
        """
        Main optimization entry point.
        total_capital is already phase-adjusted when Bible phase mode is active.
        """
        if not opportunities:
            self.logger.info("No opportunities available for optimization")
            return {"positions_created": 0, "total_capital_used": 0.0}

        self.logger.info(f"Optimizing portfolio with effective capital: ${total_capital:.2f}")

        positions = []
        capital_used = 0.0

        for opp in opportunities[:10]:  # Original limit preserved
            try:
                edge = opp.get("edge", 0.0)
                odds = opp.get("odds", 0.5)
                max_risk = getattr(settings.trading, 'max_single_position', 0.15)

                if edge <= 0:
                    continue

                # Kelly fraction using phase effective capital
                kelly_fraction = (edge / (1 - odds)) if odds < 1 else 0.0
                position_size = min(kelly_fraction, max_risk) * total_capital

                if position_size < 1.0:
                    continue

                position = {
                    "market_id": opp.get("market_id"),
                    "side": opp.get("side", "yes"),
                    "size": round(position_size, 2),
                    "expected_edge": edge,
                    "capital_used": position_size
                }

                positions.append(position)
                capital_used += position_size

                self.logger.info(f"Added position: {opp.get('market_id')} | Size: ${position_size:.2f} | Edge: {edge:.1%}")

            except Exception as e:
                self.logger.error(f"Error optimizing opportunity {opp.get('market_id')}: {e}")
                continue

        results = {
            "positions_created": len(positions),
            "total_capital_used": round(capital_used, 2),
            "positions": positions,
            "effective_capital_used": total_capital
        }

        self.logger.info(f"Portfolio optimization complete: {len(positions)} positions, ${capital_used:.2f} capital used")
        return results


async def run_portfolio_optimization(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
    xai_client: XAIClient,
    total_capital: Optional[float] = None
) -> Dict:
    """
    Original RyanFrigo entry point preserved.
    Opportunity fetching method left exactly as it was in your repo.
    """
    logger = get_trading_logger("portfolio_optimization_main")
    try:
        optimizer = AdvancedPortfolioOptimizer(db_manager, kalshi_client, xai_client)

        # Original RyanFrigo opportunity fetching method restored exactly
        opportunities = await kalshi_client.get_opportunities()

        if total_capital is None:
            total_capital = settings.trading.phase_base_capital if settings.trading.phase_mode_enabled else 100.0

        results = await optimizer.optimize_portfolio(opportunities, total_capital)
        return results

    except Exception as e:
        logger.error(f"Error in portfolio optimization: {e}")
        return {"positions_created": 0, "total_capital_used": 0.0}