"""
Portfolio Enforcer — Risk & Allocation Guardrails
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

from src.utils.database import DatabaseManager
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger


@dataclass
class CategoryScore:
    category: str
    score: float
    max_allocation: float
    current_exposure: float


class PortfolioEnforcer:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = get_trading_logger("portfolio_enforcer")

    async def get_effective_capital(self) -> float:
        if not getattr(settings.trading, 'phase_mode_enabled', False):
            return 100.0

        try:
            phase = await self.db_manager.get_phase_state()
            effective = getattr(settings.trading, 'phase_base_capital', 100.0) + phase.get('current_phase_profit', 0.0)
            self.logger.info(f"Effective capital for enforcement: ${effective:.2f} "
                             f"(base ${getattr(settings.trading, 'phase_base_capital', 100.0):.2f} + phase profit ${phase.get('current_phase_profit', 0.0):.2f})")
            return effective
        except Exception as e:
            self.logger.error(f"Failed to get phase state: {e}")
            return getattr(settings.trading, 'phase_base_capital', 100.0)

    def calculate_max_position_size(self, effective_capital: float, category_score: Optional[CategoryScore] = None) -> float:
        base_max = getattr(settings.trading, 'max_single_position', 0.15)
        if category_score:
            base_max = min(base_max, category_score.max_allocation)
        return effective_capital * base_max

    async def check_trade(self, trade: Dict) -> Dict:
        effective_capital = await self.get_effective_capital()
        market_id = trade.get("market_id")
        size = trade.get("size", 0.0)
        category = trade.get("category", "default")

        max_allowed = self.calculate_max_position_size(effective_capital)

        if size > max_allowed:
            self.logger.warning(f"BLOCKED: Trade size ${size:.2f} exceeds max allowed ${max_allowed:.2f} "
                                f"for effective capital ${effective_capital:.2f}")
            return {
                "allowed": False,
                "reason": "exceeds_max_position_size",
                "max_allowed": max_allowed,
                "effective_capital": effective_capital
            }

        category_score = await self._get_category_score(category)
        if category_score and size > category_score.max_allocation * effective_capital:
            self.logger.warning(f"BLOCKED: Category {category} exposure limit hit")
            return {
                "allowed": False,
                "reason": "category_exposure_limit",
                "effective_capital": effective_capital
            }

        current_exposure = await self._get_current_exposure()
        if current_exposure + size > effective_capital * 0.70:
            self.logger.warning(f"BLOCKED: Portfolio correlation exposure would exceed limit")
            return {
                "allowed": False,
                "reason": "correlation_exposure_limit",
                "effective_capital": effective_capital
            }

        self.logger.info(f"Trade approved: ${size:.2f} on {market_id} | Effective capital ${effective_capital:.2f}")
        return {
            "allowed": True,
            "effective_capital": effective_capital,
            "max_allowed": max_allowed
        }

    async def _get_category_score(self, category: str) -> Optional[CategoryScore]:
        scores = {
            "high_confidence": CategoryScore("high_confidence", 0.85, 0.20, 0.0),
            "medium": CategoryScore("medium", 0.60, 0.15, 0.0),
            "low": CategoryScore("low", 0.40, 0.10, 0.0),
            "default": CategoryScore("default", 0.50, 0.12, 0.0)
        }
        return scores.get(category.lower(), scores["default"])

    async def _get_current_exposure(self) -> float:
        try:
            positions = await self.db_manager.get_open_positions()
            return sum(p.get("size", 0) for p in positions)
        except Exception:
            return 0.0

    async def enforce_portfolio(self, proposed_positions: List[Dict]) -> List[Dict]:
        effective_capital = await self.get_effective_capital()
        approved = []

        for pos in proposed_positions:
            check = await self.check_trade(pos)
            if check["allowed"]:
                approved.append(pos)
            else:
                self.logger.info(f"Rejected position: {pos.get('market_id')} - {check['reason']}")

        self.logger.info(f"Portfolio enforcement complete: {len(approved)}/{len(proposed_positions)} positions approved "
                         f"with effective capital ${effective_capital:.2f}")
        return approved


async def run_portfolio_enforcement(
    db_manager: DatabaseManager,
    proposed_positions: List[Dict]
) -> List[Dict]:
    enforcer = PortfolioEnforcer(db_manager)
    return await enforcer.enforce_portfolio(proposed_positions)