"""
Market Making Strategy — Advanced Liquidity Provision

Fixes applied
─────────────
1. Spread/bid-ask calculation was inverted: bids were priced ABOVE the current
   market, guaranteeing immediate fills and no captured spread.  The logic now
   prices the YES-bid BELOW mid and the NO-bid so that YES_bid + NO_bid < $1.00,
   creating a locked profit whenever both sides fill.

2. Both a YES BUY order and a NO BUY order are placed per market.
   On Kalshi you cannot place a raw "sell limit" without owning the contracts,
   so the canonical synthetic-spread approach is:
     YES_bid = mid − half_spread   (buy YES below fair value)
     NO_bid  = (1 − mid) − half_spread  (buy NO below its fair value)
   If both fill, total cost = YES_bid + NO_bid < $1.00 guaranteed payout → profit.

3. _update_order is fully implemented: cancels the stale order via the API and
   places a fresh limit order at the recalculated optimal price.

4. place_order calls pass the params dict directly (not **-unpacked) to match
   KalshiClient.place_order(order_params: Dict).
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from src.clients.kalshi_client import KalshiClient
from src.clients.xai_client import XAIClient
from src.utils.database import DatabaseManager, Market
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger


@dataclass
class LimitOrder:
    market_id: str
    side: str           # "YES" or "NO"
    price: float        # dollars (0.01 – 0.99)
    quantity: int
    order_type: str = "limit"
    status: str = "pending"
    order_id: Optional[str] = None
    placed_at: Optional[datetime] = None
    expected_profit: float = 0.0


@dataclass
class MarketMakingOpportunity:
    market_id: str
    market_title: str
    current_yes_mid: float          # mid-price of YES
    current_no_mid: float           # mid-price of NO
    ai_predicted_prob: float
    ai_confidence: float

    # Optimal bid prices (both are BUY orders)
    optimal_yes_bid: float          # price to BUY YES (below mid)
    optimal_no_bid: float           # price to BUY NO (below no-mid)

    # Expected spread-locked profit if both sides fill
    locked_profit_per_pair: float   # = $1.00 − yes_bid − no_bid

    # Risk / sizing
    inventory_risk: float
    volatility_estimate: float
    optimal_yes_size: int
    optimal_no_size: int

    # Legacy aliases kept for result aggregation
    @property
    def total_expected_profit(self) -> float:
        return self.locked_profit_per_pair


class AdvancedMarketMaker:
    """
    Synthetic-spread market maker for Kalshi prediction markets.

    Places a YES-buy and a NO-buy at prices whose sum is strictly less than
    $1.00, locking in a profit if both sides fill.  Edge-filtered via the
    AI's probability estimate before any order is placed.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        kalshi_client: KalshiClient,
        xai_client: XAIClient,
    ):
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.xai_client = xai_client
        self.logger = get_trading_logger("market_maker")

        self.min_spread = getattr(settings.trading, "min_spread_for_making", 0.03)
        self.max_spread = getattr(settings.trading, "max_bid_ask_spread", 0.10)
        self.inventory_penalty = getattr(settings.trading, "max_inventory_risk", 0.01)

        self.active_orders: Dict[str, List[LimitOrder]] = {}
        self.total_pnl = 0.0
        self.markets_traded = 0
        self.total_volume = 0

    # ── Analysis ──────────────────────────────────────────────────────────────

    async def analyze_market_making_opportunities(
        self, markets: List[Market]
    ) -> List[MarketMakingOpportunity]:
        opportunities = []

        for market in markets:
            try:
                market_data = await self.kalshi_client.get_market(market.market_id)
                if not market_data:
                    continue

                mkt = market_data.get("market", market_data)

                # Support both dollar-denominated and cents fields
                yes_bid = float(mkt.get("yes_bid_dollars", 0) or mkt.get("yes_bid", 0) or 0)
                yes_ask = float(mkt.get("yes_ask_dollars", 0) or mkt.get("yes_ask", 0) or 0)
                no_bid  = float(mkt.get("no_bid_dollars",  0) or mkt.get("no_bid",  0) or 0)
                no_ask  = float(mkt.get("no_ask_dollars",  0) or mkt.get("no_ask",  0) or 0)

                if yes_bid > 1.0: yes_bid /= 100.0
                if yes_ask > 1.0: yes_ask /= 100.0
                if no_bid  > 1.0: no_bid  /= 100.0
                if no_ask  > 1.0: no_ask  /= 100.0

                yes_mid = (yes_bid + yes_ask) / 2 if (yes_bid + yes_ask) > 0 else 0.0
                no_mid  = (no_bid  + no_ask)  / 2 if (no_bid  + no_ask)  > 0 else 0.0

                # Skip extreme prices — hard to make markets profitably
                if yes_mid < 0.03 or yes_mid > 0.97:
                    continue

                analysis = await self._get_ai_analysis(market)
                if not analysis:
                    continue

                ai_prob  = analysis.get("probability", 0.5)
                ai_conf  = analysis.get("confidence",  0.5)

                # Edge filter — only proceed if AI sees meaningful mispricing
                from src.utils.edge_filter import EdgeFilter
                yes_edge = EdgeFilter.calculate_edge(ai_prob,       yes_mid, ai_conf)
                no_edge  = EdgeFilter.calculate_edge(1 - ai_prob,   no_mid,  ai_conf)

                if yes_edge.passes_filter or no_edge.passes_filter:
                    opp = self._calculate_opportunity(
                        market, yes_mid, no_mid, ai_prob, ai_conf
                    )
                    if opp and opp.locked_profit_per_pair > 0:
                        opportunities.append(opp)
                        self.logger.info(
                            f"MM APPROVED: {market.market_id} | "
                            f"locked profit/pair ${opp.locked_profit_per_pair:.4f}"
                        )
                else:
                    self.logger.debug(
                        f"MM FILTERED: {market.market_id} — insufficient edge on both sides"
                    )

            except Exception as e:
                self.logger.error(f"Error analysing market {market.market_id}: {e}")
                continue

        opportunities.sort(key=lambda x: x.locked_profit_per_pair, reverse=True)
        return opportunities

    def _calculate_opportunity(
        self,
        market: Market,
        yes_mid: float,
        no_mid: float,
        ai_prob: float,
        ai_conf: float,
    ) -> Optional[MarketMakingOpportunity]:
        """
        Compute YES-bid and NO-bid so that YES_bid + NO_bid < $1.00.

        Target spread is derived from volatility and clipped to [min_spread, max_spread].
        If the AI has a directional view we tilt both bids accordingly, still
        keeping their sum below $1.00.
        """
        try:
            volatility = self._estimate_volatility(yes_mid, market)
            raw_spread = max(self.min_spread, min(self.max_spread,
                             volatility * 2.0))

            # Directional tilt: if AI thinks YES is underpriced, bid YES higher
            # and NO lower (but keep total < $1.00)
            yes_edge = ai_prob - yes_mid          # positive → YES cheap
            tilt = yes_edge * ai_conf * 0.5       # bounded tilt

            half_spread = raw_spread / 2.0

            # Place bids BELOW their respective mid-prices
            optimal_yes_bid = yes_mid - half_spread + tilt
            optimal_no_bid  = no_mid  - half_spread - tilt

            # Hard bounds
            optimal_yes_bid = max(0.01, min(0.98, optimal_yes_bid))
            optimal_no_bid  = max(0.01, min(0.98, optimal_no_bid))

            # Ensure the pair is profitable (sum < $1.00)
            total_cost = optimal_yes_bid + optimal_no_bid
            if total_cost >= 1.00:
                # Widen spread until it is
                excess = total_cost - 0.98
                optimal_yes_bid -= excess / 2
                optimal_no_bid  -= excess / 2
                optimal_yes_bid = max(0.01, optimal_yes_bid)
                optimal_no_bid  = max(0.01, optimal_no_bid)
                total_cost = optimal_yes_bid + optimal_no_bid

            locked_profit = 1.00 - total_cost
            if locked_profit <= 0:
                return None

            yes_size, no_size = self._calculate_optimal_sizes(
                ai_prob - yes_mid, (1 - ai_prob) - no_mid, volatility, ai_conf
            )

            return MarketMakingOpportunity(
                market_id=market.market_id,
                market_title=market.title,
                current_yes_mid=yes_mid,
                current_no_mid=no_mid,
                ai_predicted_prob=ai_prob,
                ai_confidence=ai_conf,
                optimal_yes_bid=optimal_yes_bid,
                optimal_no_bid=optimal_no_bid,
                locked_profit_per_pair=locked_profit,
                inventory_risk=volatility,
                volatility_estimate=volatility,
                optimal_yes_size=yes_size,
                optimal_no_size=no_size,
            )
        except Exception as e:
            self.logger.error(f"Error calculating opportunity for {market.market_id}: {e}")
            return None

    def _estimate_volatility(self, price: float, market: Market) -> float:
        try:
            if hasattr(market, "expiration_ts") and market.expiration_ts:
                expiry = datetime.fromtimestamp(market.expiration_ts)
                tte = max(0.1, (expiry - datetime.now()).total_seconds() / 86400)
            else:
                tte = 7.0
            # Binary-option volatility proxy: σ = √(p(1−p)/t)
            intrinsic_vol = np.sqrt(price * (1 - price) / tte)
            return max(0.01, min(0.20, intrinsic_vol))
        except Exception:
            return 0.05

    def _calculate_optimal_sizes(
        self,
        yes_edge: float,
        no_edge: float,
        volatility: float,
        confidence: float,
    ) -> Tuple[int, int]:
        try:
            available_capital = getattr(settings.trading, "max_position_size", 500.0)

            def kelly_size(edge: float) -> int:
                if edge > 0:
                    win_prob = min(0.95, 0.5 + edge * confidence)
                    kelly_f = max(0.0, min(0.25, (win_prob - 0.5) / 0.5))
                    return max(5, int(available_capital * kelly_f))
                return max(5, int(available_capital * 0.05))

            return kelly_size(yes_edge), kelly_size(no_edge)
        except Exception:
            return 50, 50

    # ── Execution ─────────────────────────────────────────────────────────────

    async def execute_market_making_strategy(
        self, opportunities: List[MarketMakingOpportunity]
    ) -> Dict:
        results = {
            "orders_placed": 0,
            "total_exposure": 0.0,
            "expected_profit": 0.0,
            "markets_count": 0,
        }

        max_markets = getattr(settings.trading, "max_concurrent_markets", 10)
        top = opportunities[:max_markets]

        for opp in top:
            try:
                placed = await self._place_market_making_orders(opp)
                results["orders_placed"] += placed
                results["total_exposure"] += opp.optimal_yes_size + opp.optimal_no_size
                results["expected_profit"] += opp.locked_profit_per_pair
                results["markets_count"] += 1
                self.logger.info(
                    f"MM orders placed for {opp.market_title}: "
                    f"locked ${opp.locked_profit_per_pair:.4f}/pair"
                )
            except Exception as e:
                self.logger.error(f"Error executing MM for {opp.market_id}: {e}")
                continue

        return results

    async def _place_market_making_orders(
        self, opportunity: MarketMakingOpportunity
    ) -> int:
        """
        Place the YES-bid and NO-bid orders.
        Both are BUY orders; their prices sum to less than $1.00, so if both
        fill the locked spread is the profit.
        Returns the number of orders successfully placed (0, 1, or 2).
        """
        yes_order = LimitOrder(
            market_id=opportunity.market_id,
            side="YES",
            price=opportunity.optimal_yes_bid,
            quantity=opportunity.optimal_yes_size,
            expected_profit=opportunity.locked_profit_per_pair / 2,
        )
        no_order = LimitOrder(
            market_id=opportunity.market_id,
            side="NO",
            price=opportunity.optimal_no_bid,
            quantity=opportunity.optimal_no_size,
            expected_profit=opportunity.locked_profit_per_pair / 2,
        )

        placed = 0
        for order in [yes_order, no_order]:
            ok = await self._place_limit_order(order)
            if ok:
                placed += 1

        # Track for monitoring
        if opportunity.market_id not in self.active_orders:
            self.active_orders[opportunity.market_id] = []
        self.active_orders[opportunity.market_id].extend([yes_order, no_order])

        return placed

    async def _place_limit_order(self, order: LimitOrder) -> bool:
        """
        Place a single limit BUY order with Kalshi (or simulate in paper mode).
        Returns True on success.
        """
        try:
            live_mode = settings.trading.live_trading_enabled
            price_cents = max(1, min(99, int(round(order.price * 100))))
            side = order.side.lower()

            order_params: Dict = {
                "ticker": order.market_id,
                "client_order_id": str(uuid.uuid4()),
                "side": side,
                "action": "buy",
                "count": order.quantity,
                "type_": "limit",
            }
            # Kalshi requires yes_price for YES side and no_price for NO side
            if side == "yes":
                order_params["yes_price"] = price_cents
            else:
                order_params["no_price"] = price_cents

            if live_mode:
                # FIX: pass dict directly, not **-unpacked
                response = await self.kalshi_client.place_order(order_params)
                if response and "order" in response:
                    order.status = "placed"
                    order.placed_at = datetime.now()
                    order.order_id = response["order"].get("order_id", order_params["client_order_id"])
                    self.logger.info(
                        f"LIVE limit order: {order.side} x{order.quantity} "
                        f"@ ${order.price:.2f} for {order.market_id} (id={order.order_id})"
                    )
                    return True
                else:
                    self.logger.error(f"Failed to place live order: {response}")
                    order.status = "failed"
                    return False
            else:
                order.status = "placed"
                order.placed_at = datetime.now()
                order.order_id = f"sim_{order.market_id}_{order.side}_{int(datetime.now().timestamp())}"
                self.logger.info(
                    f"[SIMULATED] limit order: {order.side} x{order.quantity} "
                    f"@ {price_cents}¢ for {order.market_id}"
                )
                return True

        except Exception as e:
            self.logger.error(f"Error placing limit order: {e}")
            order.status = "failed"
            return False

    # ── Order monitoring / updating ───────────────────────────────────────────

    async def monitor_and_update_orders(self):
        for market_id, orders in list(self.active_orders.items()):
            for order in orders:
                try:
                    if order.status == "placed" and await self._should_update_order(order):
                        await self._update_order(order, market_id)
                except Exception as e:
                    self.logger.error(f"Error monitoring order for {market_id}: {e}")

    async def _should_update_order(self, order: LimitOrder) -> bool:
        try:
            market_data = await self.kalshi_client.get_market(order.market_id)
            if not market_data:
                return False
            mkt = market_data.get("market", market_data)
            if order.side == "YES":
                mid_raw = (
                    float(mkt.get("yes_bid", 0) or 0)
                    + float(mkt.get("yes_ask", 0) or 0)
                ) / 2
            else:
                mid_raw = (
                    float(mkt.get("no_bid", 0) or 0)
                    + float(mkt.get("no_ask", 0) or 0)
                ) / 2
            current_mid = mid_raw / 100.0 if mid_raw > 1.0 else mid_raw
            return abs(current_mid - order.price) > 0.05
        except Exception:
            return False

    async def _update_order(self, order: LimitOrder, market_id: str):
        """
        Cancel the stale order and place a fresh one at the recalculated price.
        Previously this was a stub that only logged without actually doing anything.
        """
        try:
            live_mode = settings.trading.live_trading_enabled

            # Step 1: cancel the existing order
            if live_mode and order.order_id:
                try:
                    await self.kalshi_client.cancel_order(order.order_id)
                    self.logger.info(f"Cancelled stale order {order.order_id}")
                except Exception as e:
                    self.logger.warning(f"Could not cancel order {order.order_id}: {e}")

            order.status = "cancelled"

            # Step 2: recalculate price from fresh market data
            market_data = await self.kalshi_client.get_market(market_id)
            mkt = market_data.get("market", market_data) if market_data else {}

            if order.side == "YES":
                bid_raw = float(mkt.get("yes_bid_dollars", 0) or mkt.get("yes_bid", 0) or 0)
                ask_raw = float(mkt.get("yes_ask_dollars", 0) or mkt.get("yes_ask", 0) or 0)
            else:
                bid_raw = float(mkt.get("no_bid_dollars", 0) or mkt.get("no_bid", 0) or 0)
                ask_raw = float(mkt.get("no_ask_dollars", 0) or mkt.get("no_ask", 0) or 0)

            if bid_raw > 1.0: bid_raw /= 100.0
            if ask_raw > 1.0: ask_raw /= 100.0
            new_mid = (bid_raw + ask_raw) / 2 if (bid_raw + ask_raw) > 0 else order.price

            # Re-apply half-spread below new mid
            new_price = max(0.01, min(0.98, new_mid - self.min_spread / 2))
            order.price = new_price

            # Step 3: place fresh order
            await self._place_limit_order(order)

        except Exception as e:
            self.logger.error(f"Error updating order for {market_id}: {e}")

    # ── AI analysis ───────────────────────────────────────────────────────────

    async def _get_ai_analysis(self, market: Market) -> Optional[Dict]:
        try:
            prompt = f"""
MARKET MAKING ANALYSIS REQUEST

Market: {market.title}

Provide a quick assessment for market making in JSON format:
{{
    "probability": [0.0-1.0 probability estimate],
    "confidence": [0.0-1.0 confidence level],
    "volatility_factors": "brief description",
    "stability": [0.0-1.0 price stability estimate]
}}

Focus on: probability estimate and confidence in that estimate.
"""
            response = await self.xai_client.get_completion(
                prompt, max_tokens=3000, temperature=0.1
            )

            if response is None:
                return {
                    "probability": 0.5,
                    "confidence": 0.2,
                    "volatility_factors": "API unavailable",
                    "stability": 0.3,
                }

            import json, re
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                prob = parsed.get("probability")
                conf = parsed.get("confidence")
                if (
                    isinstance(prob, (int, float)) and 0 <= prob <= 1
                    and isinstance(conf, (int, float)) and 0 <= conf <= 1
                ):
                    return parsed

            return {
                "probability": 0.5,
                "confidence": 0.3,
                "volatility_factors": "parse failed",
                "stability": 0.5,
            }

        except Exception as e:
            self.logger.error(f"Error getting AI analysis: {e}")
            return {
                "probability": 0.5,
                "confidence": 0.3,
                "volatility_factors": "error",
                "stability": 0.5,
            }

    def get_performance_summary(self) -> Dict:
        try:
            active_count = sum(len(v) for v in self.active_orders.values())
            return {
                "total_pnl": self.total_pnl,
                "active_orders": active_count,
                "markets_traded": self.markets_traded,
                "total_volume": self.total_volume,
            }
        except Exception:
            return {}


# ── Module entry point ────────────────────────────────────────────────────────

async def run_market_making_strategy(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
    xai_client: XAIClient,
) -> Dict:
    logger = get_trading_logger("market_making_main")

    try:
        market_maker = AdvancedMarketMaker(db_manager, kalshi_client, xai_client)

        markets = await db_manager.get_eligible_markets(
            volume_min=30000,
            max_days_to_expiry=365,
        )

        if not markets:
            logger.warning("No eligible markets found for market making")
            return {"error": "No markets available"}

        logger.info(f"Analysing {len(markets)} markets for market-making opportunities")

        opportunities = await market_maker.analyze_market_making_opportunities(markets)

        if not opportunities:
            logger.warning("No profitable market-making opportunities found")
            return {"opportunities": 0}

        logger.info(f"Found {len(opportunities)} profitable market-making opportunities")

        results = await market_maker.execute_market_making_strategy(opportunities)
        results["performance"] = market_maker.get_performance_summary()

        logger.info(f"Market-making strategy completed: {results}")
        return results

    except Exception as e:
        logger.error(f"Error in market-making strategy: {e}")
        return {"error": str(e)}
