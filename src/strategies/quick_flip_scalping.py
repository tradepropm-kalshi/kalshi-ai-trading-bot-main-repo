"""
Quick Flip Scalping Strategy

Buys contracts at low prices (1¢–20¢) and immediately places a sell limit
order at a higher target, aiming to capture short-term price movement.

Fixes applied
─────────────
1. Price-unit bugs (÷100 applied to already-dollar values):
   • expected_profit calculation no longer divides by 100 a second time.
   • total_capital_used no longer divides by 100 a second time.

2. In-memory position state was wiped every 60-second cycle because
   QuickFlipScalpingStrategy is instantiated fresh each time.
   Positions and pending-sell metadata are now persisted to the
   quick_flip_tracking DB table and loaded at the start of each cycle.

3. place_order calls now pass the params dict directly (not **-unpacked)
   to match KalshiClient.place_order(order_params).
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

from src.clients.kalshi_client import KalshiClient
from src.clients.xai_client import XAIClient
from src.utils.database import DatabaseManager, Market, Position
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger
from src.utils.price_utils import cents_to_dollars, contract_cost_dollars, expected_pnl_dollars
from src.jobs.execute import place_sell_limit_order


@dataclass
class QuickFlipOpportunity:
    market_id: str
    market_title: str
    side: str           # "YES" or "NO"
    entry_price: float  # dollars (0.01 – 0.20)
    exit_price: float   # dollars (target sell)
    quantity: int
    expected_profit: float
    confidence_score: float
    movement_indicator: str
    max_hold_time: int  # minutes


@dataclass
class QuickFlipConfig:
    min_entry_price: float = 0.01
    max_entry_price: float = 0.20
    min_profit_margin: float = 1.0      # 100 % — must at least double
    max_position_size: int = 100
    max_concurrent_positions: int = 50
    capital_per_trade: float = 50.0
    confidence_threshold: float = 0.6
    max_hold_minutes: int = 30


class QuickFlipScalpingStrategy:
    """Rapid scalping strategy with persistent cross-cycle position tracking."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        kalshi_client: KalshiClient,
        xai_client: XAIClient,
        config: Optional[QuickFlipConfig] = None,
    ):
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.xai_client = xai_client
        self.config = config or QuickFlipConfig()
        self.logger = get_trading_logger("quick_flip_scalping")

    # ── Opportunity identification ────────────────────────────────────────────

    async def identify_quick_flip_opportunities(
        self,
        markets: List[Market],
        available_capital: float,
    ) -> List[QuickFlipOpportunity]:
        """
        Screen *markets* for quick-flip opportunities.

        A quick-flip opportunity is a contract trading at 1–20 cents where
        the AI predicts a price increase of at least 100 % within 30 minutes.

        Args:
            markets:           Markets to evaluate (should already be filtered for
                               volume and days-to-expiry by the caller).
            available_capital: Total capital available to allocate across positions.

        Returns:
            Opportunities sorted by descending expected value, capped at the
            number of positions the available capital can fund.
        """
        opportunities = []
        self.logger.info(f"Analysing {len(markets)} markets for quick-flip opportunities")

        for market in markets:
            try:
                market_data = await self.kalshi_client.get_market(market.market_id)
                if not market_data:
                    continue

                market_info = market_data.get("market", {})
                yes_price = float(
                    market_info.get("yes_ask_dollars", 0)
                    or market_info.get("yes_ask", 0)
                    or 0
                )
                no_price = float(
                    market_info.get("no_ask_dollars", 0)
                    or market_info.get("no_ask", 0)
                    or 0
                )

                # Normalise to dollars if the API returned cents
                if yes_price > 1.0:
                    yes_price = cents_to_dollars(yes_price)
                if no_price > 1.0:
                    no_price = cents_to_dollars(no_price)

                for side, price in [("YES", yes_price), ("NO", no_price)]:
                    opp = await self._evaluate_price_opportunity(market, side, price, market_info)
                    if opp:
                        opportunities.append(opp)

            except Exception as e:
                self.logger.error(f"Error analysing market {market.market_id}: {e}")
                continue

        opportunities.sort(
            key=lambda x: x.expected_profit * x.confidence_score,
            reverse=True,
        )

        max_positions = min(
            self.config.max_concurrent_positions,
            int(available_capital / self.config.capital_per_trade) if self.config.capital_per_trade > 0 else 0,
        )
        filtered = opportunities[:max_positions]
        self.logger.info(
            f"Found {len(filtered)} quick-flip opportunities (from {len(opportunities)} analysed)"
        )
        return filtered

    async def _evaluate_price_opportunity(
        self,
        market: Market,
        side: str,
        current_price: float,
        market_info: dict,
    ) -> Optional[QuickFlipOpportunity]:
        """
        Evaluate a single side (YES or NO) of a market for a quick-flip entry.

        Args:
            market:        Market dataclass.
            side:          ``"YES"`` or ``"NO"``.
            current_price: Current ask price in dollars.
            market_info:   Raw market data dict from the Kalshi API.

        Returns:
            ``QuickFlipOpportunity`` if the trade passes all filters, else ``None``.
        """
        if not current_price or current_price <= 0:
            return None
        if current_price < self.config.min_entry_price or current_price > self.config.max_entry_price:
            return None

        min_exit_price = current_price * (1 + self.config.min_profit_margin)
        if min_exit_price > 0.95:
            return None

        movement_analysis = await self._analyze_market_movement(market, side, current_price)

        if movement_analysis["confidence"] < self.config.confidence_threshold:
            return None

        quantity = min(
            self.config.max_position_size,
            int(self.config.capital_per_trade / current_price) if current_price > 0 else 0,
        )
        if quantity < 1:
            return None

        target_price = movement_analysis["target_price"]   # dollars

        # FIX: both current_price and target_price are in dollars — no ÷100
        expected_profit = quantity * (target_price - current_price)

        return QuickFlipOpportunity(
            market_id=market.market_id,
            market_title=market.title,
            side=side,
            entry_price=current_price,
            exit_price=target_price,
            quantity=quantity,
            expected_profit=expected_profit,
            confidence_score=movement_analysis["confidence"],
            movement_indicator=movement_analysis["reason"],
            max_hold_time=self.config.max_hold_minutes,
        )

    async def _analyze_market_movement(
        self,
        market: Market,
        side: str,
        current_price: float,
    ) -> dict:
        try:
            prompt = f"""
QUICK SCALP ANALYSIS for {market.title}

Current {side} price: ${current_price:.2f}
Market closes: {datetime.fromtimestamp(market.expiration_ts).strftime('%Y-%m-%d %H:%M')}

Analyse for IMMEDIATE (next 30 minutes) price movement potential:
1. Likely catalysts/news that could move price UP in the next 30 min?
2. Current momentum/volatility indicators
3. Realistic price {side} could reach in 30 min
4. Confidence (0–1) for upward movement

Respond with:
TARGET_PRICE: [realistic price in dollars, e.g. 0.15]
CONFIDENCE: [0.0-1.0]
REASON: [brief explanation]
"""
            response = await self.xai_client.get_completion(
                prompt=prompt,
                max_tokens=3000,
                strategy="quick_flip_scalping",
                query_type="movement_prediction",
                market_id=market.market_id,
            )

            if response is None:
                self.logger.info(
                    f"AI analysis unavailable for {market.market_id}, using conservative defaults"
                )
                return {
                    "target_price": current_price + 0.02,
                    "confidence": 0.2,
                    "reason": "AI analysis unavailable",
                }

            target_price = current_price + 0.05
            confidence = 0.5
            reason = "Default analysis"

            for line in response.strip().split("\n"):
                if "TARGET_PRICE:" in line:
                    try:
                        target_price = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif "CONFIDENCE:" in line:
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif "REASON:" in line:
                    reason = line.split(":", 1)[1].strip()

            # Clamp: must be at least $0.01 above entry, at most $0.95
            target_price = max(current_price + 0.01, min(target_price, 0.95))

            return {"target_price": target_price, "confidence": confidence, "reason": reason}

        except Exception as e:
            self.logger.error(f"Error in movement analysis: {e}")
            return {
                "target_price": current_price + 0.05,
                "confidence": 0.3,
                "reason": f"Analysis failed: {e}",
            }

    # ── Execution ─────────────────────────────────────────────────────────────

    async def execute_quick_flip_opportunities(
        self, opportunities: List[QuickFlipOpportunity]
    ) -> Dict:
        """
        Execute a list of approved quick-flip opportunities.

        For each opportunity: inserts the position into the DB, places the buy
        order, and immediately places a sell-limit order at the target price.
        The pending sell is persisted to ``quick_flip_tracking`` so it survives
        the 60-second cycle boundary.

        Returns:
            Aggregated result dict with counts and totals.
        """
        results = {
            "positions_created": 0,
            "sell_orders_placed": 0,
            "total_capital_used": 0.0,
            "expected_profit": 0.0,
            "failed_executions": 0,
        }

        self.logger.info(f"Executing {len(opportunities)} quick-flip opportunities")

        for opportunity in opportunities:
            try:
                position_id = await self._execute_single_quick_flip(opportunity)

                if position_id is not None:
                    results["positions_created"] += 1
                    results["total_capital_used"] += contract_cost_dollars(
                        opportunity.quantity, opportunity.entry_price
                    )
                    results["expected_profit"] += opportunity.expected_profit

                    sell_ok = await self._place_immediate_sell_order(opportunity, position_id)
                    if sell_ok:
                        results["sell_orders_placed"] += 1
                else:
                    results["failed_executions"] += 1

            except Exception as e:
                self.logger.error(f"Error executing quick flip for {opportunity.market_id}: {e}")
                results["failed_executions"] += 1

        self.logger.info(
            f"Quick-flip execution: {results['positions_created']} positions, "
            f"{results['sell_orders_placed']} sell orders, "
            f"${results['total_capital_used']:.2f} capital used"
        )
        return results

    async def _execute_single_quick_flip(
        self, opportunity: QuickFlipOpportunity
    ) -> Optional[int]:
        """
        Insert the position into the DB and execute the buy order.
        Returns the DB position id on success, None on failure.
        """
        try:
            position = Position(
                market_id=opportunity.market_id,
                side=opportunity.side,
                quantity=opportunity.quantity,
                entry_price=opportunity.entry_price,
                live=False,
                timestamp=datetime.now(),
                rationale=(
                    f"QUICK FLIP: {opportunity.movement_indicator} | "
                    f"Target: ${opportunity.entry_price:.2f}→${opportunity.exit_price:.2f}"
                ),
                strategy="quick_flip_scalping",
            )

            position_id = await self.db_manager.add_position(position)
            if position_id is None:
                self.logger.warning(f"Position already exists for {opportunity.market_id}")
                return None

            position.id = position_id

            from src.jobs.execute import execute_position

            live_mode = settings.trading.live_trading_enabled
            success = await execute_position(
                position=position,
                live_mode=live_mode,
                db_manager=self.db_manager,
                kalshi_client=self.kalshi_client,
            )

            if success:
                self.logger.info(
                    f"Quick-flip entry: {opportunity.side} x{opportunity.quantity} "
                    f"@ ${opportunity.entry_price:.2f} for {opportunity.market_id}"
                )
                return position_id
            else:
                self.logger.error(f"Failed to execute quick flip for {opportunity.market_id}")
                return None

        except Exception as e:
            self.logger.error(f"Error executing single quick flip: {e}")
            return None

    async def _place_immediate_sell_order(
        self, opportunity: QuickFlipOpportunity, position_id: int
    ) -> bool:
        """
        Place a sell-limit order and persist the pending-sell record to the DB
        so it survives across 60-second cycle boundaries.
        """
        try:
            position = await self.db_manager.get_position_by_market_id(opportunity.market_id)
            if not position:
                self.logger.error(f"No active position found for {opportunity.market_id}")
                return False

            sell_price = opportunity.exit_price  # already in dollars

            success = await place_sell_limit_order(
                position=position,
                limit_price=sell_price,
                db_manager=self.db_manager,
                kalshi_client=self.kalshi_client,
            )

            if success:
                max_hold_until = datetime.now() + timedelta(minutes=opportunity.max_hold_time)
                await self.db_manager.save_quick_flip_position(
                    market_id=opportunity.market_id,
                    side=opportunity.side,
                    quantity=opportunity.quantity,
                    entry_price=opportunity.entry_price,
                    target_price=opportunity.exit_price,
                    max_hold_until=max_hold_until,
                    position_id=position_id,
                )
                self.logger.info(
                    f"Sell order placed + tracked in DB: {opportunity.side} x{opportunity.quantity} "
                    f"@ ${opportunity.exit_price:.2f} for {opportunity.market_id}"
                )
                return True
            else:
                self.logger.error(f"Failed to place sell order for {opportunity.market_id}")
                return False

        except Exception as e:
            self.logger.error(f"Error placing immediate sell order: {e}")
            return False

    # ── Position management (loaded from DB each cycle) ───────────────────────

    async def manage_active_positions(self) -> Dict:
        """
        Load persisted pending sells from the DB and cut losses on positions
        that have been held beyond their max hold time.
        """
        results = {
            "positions_closed": 0,
            "losses_cut": 0,
            "total_pnl": 0.0,
        }

        pending = await self.db_manager.get_quick_flip_pending()
        current_time = datetime.now()

        for sell_info in pending:
            market_id = sell_info["market_id"]
            try:
                max_hold_until = datetime.fromisoformat(sell_info["max_hold_until"])

                if current_time > max_hold_until:
                    self.logger.warning(f"Quick flip held too long: {market_id}, cutting losses")

                    position = await self.db_manager.get_position_by_market_id(market_id)
                    if position:
                        cut_ok = await self._cut_losses_market_order(position)
                        if cut_ok:
                            results["losses_cut"] += 1
                            results["positions_closed"] += 1
                            await self.db_manager.remove_quick_flip_position(market_id)
                            await self.db_manager.update_position_status(position.id, "closed")

            except Exception as e:
                self.logger.error(f"Error managing position {market_id}: {e}")

        return results

    async def _cut_losses_market_order(self, position: Position) -> bool:
        """Place a market sell order to immediately exit a timed-out position."""
        try:
            order_params = {
                "ticker": position.market_id,
                "client_order_id": str(uuid.uuid4()),
                "side": position.side.lower(),
                "action": "sell",
                "count": position.quantity,
                "type_": "market",
            }

            live_mode = settings.trading.live_trading_enabled

            if live_mode:
                # FIX: pass dict directly, not **-unpacked
                response = await self.kalshi_client.place_order(order_params)
                if response and "order" in response:
                    self.logger.info(
                        f"Loss-cut order placed: {position.side} x{position.quantity} "
                        f"MARKET SELL for {position.market_id}"
                    )
                    return True
                else:
                    self.logger.error(f"Failed to place loss-cut order: {response}")
                    return False
            else:
                self.logger.info(
                    f"[SIMULATED] Loss-cut: {position.side} x{position.quantity} "
                    f"MARKET SELL for {position.market_id}"
                )
                return True

        except Exception as e:
            self.logger.error(f"Error cutting losses: {e}")
            return False


# ── Module entry point ────────────────────────────────────────────────────────

async def run_quick_flip_strategy(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
    xai_client: XAIClient,
    available_capital: float,
    config: Optional[QuickFlipConfig] = None,
) -> Dict:
    logger = get_trading_logger("quick_flip_main")

    try:
        logger.info("Starting Quick Flip Scalping Strategy")

        strategy = QuickFlipScalpingStrategy(db_manager, kalshi_client, xai_client, config)

        markets = await db_manager.get_eligible_markets(
            volume_min=100,
            max_days_to_expiry=365,
        )

        if not markets:
            logger.warning("No markets available for quick-flip analysis")
            return {"error": "No markets available"}

        # Step 1: Manage positions that carried over from the previous cycle
        management_results = await strategy.manage_active_positions()

        # Step 2: Identify new opportunities
        opportunities = await strategy.identify_quick_flip_opportunities(
            markets, available_capital
        )

        if not opportunities:
            logger.info("No quick-flip opportunities found")
            return {
                "opportunities_found": 0,
                **management_results,
            }

        # Step 3: Execute
        execution_results = await strategy.execute_quick_flip_opportunities(opportunities)

        total_results = {
            **execution_results,
            **management_results,
            "opportunities_analysed": len(opportunities),
            "strategy": "quick_flip_scalping",
        }

        logger.info(
            f"Quick-flip complete: {execution_results['positions_created']} new positions, "
            f"${execution_results['total_capital_used']:.2f} capital, "
            f"${execution_results['expected_profit']:.2f} expected profit"
        )
        return total_results

    except Exception as e:
        logger.error(f"Error in quick-flip strategy: {e}")
        return {"error": str(e)}
