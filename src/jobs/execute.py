"""
Trade Execution Job

This job takes a position and executes it as a trade.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Optional, Dict

from src.utils.database import DatabaseManager, Position
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger
from src.clients.kalshi_client import KalshiClient, KalshiAPIError
from src.utils.market_prices import get_market_prices


async def execute_position(
    position: Position, 
    live_mode: bool, 
    db_manager: DatabaseManager, 
    kalshi_client: KalshiClient
) -> bool:
    """
    Executes a single trade position.
    
    Args:
        position: The position to execute.
        live_mode: Whether to execute a live or simulated trade.
        db_manager: The database manager instance.
        kalshi_client: The Kalshi client instance.
        
    Returns:
        True if execution was successful, False otherwise.
    """
    logger = get_trading_logger("trade_execution")
    logger.info(f"🎯 Executing position for market: {position.market_id}")
    logger.info(f"🎛️ Live mode: {live_mode}")
    
    if live_mode:
        logger.warning(f"💰 PLACING LIVE ORDER - Real money will be used for {position.market_id}")
        try:
            # Get current market prices
            market_data = await kalshi_client.get_market(position.market_id)
            market = market_data.get('market', {})
            
            side_lower = position.side.lower()
            client_order_id = str(uuid.uuid4())
            
            order_params = {
                "ticker": position.market_id,
                "client_order_id": client_order_id,
                "side": side_lower,
                "action": "buy",
                "count": position.quantity,
                "type_": "market"
            }
            
            # Add the appropriate price field
            _yes_bid, yes_ask_dollars, _no_bid, no_ask_dollars = get_market_prices(market)
            if side_lower == "yes":
                yes_ask_cents = int(round(yes_ask_dollars * 100))
                if yes_ask_cents > 0:
                    order_params["yes_price"] = yes_ask_cents
                else:
                    logger.error(f"No valid yes_ask price for {position.market_id}")
                    return False
            else:
                no_ask_cents = int(round(no_ask_dollars * 100))
                if no_ask_cents > 0:
                    order_params["no_price"] = no_ask_cents
                else:
                    logger.error(f"No valid no_ask price for {position.market_id}")
                    return False
            
            logger.info(f"Placing order with params: {order_params}")
            order_response = await kalshi_client.place_order(order_params)
            
            fill_price = position.entry_price
            await db_manager.update_position_to_live(position.id, fill_price)
            
            logger.info(f"✅ LIVE ORDER PLACED for {position.market_id}. Order ID: {order_response.get('order', {}).get('order_id')}")
            logger.info(f"💰 Real money used: ${position.quantity * fill_price:.2f}")
            return True

        except KalshiAPIError as e:
            logger.error(f"❌ FAILED to place LIVE order for {position.market_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error placing LIVE order: {e}")
            return False
    else:
        # PAPER MODE - Simulate
        await db_manager.update_position_to_live(position.id, position.entry_price)
        logger.info(f"📝 PAPER TRADE SIMULATED for {position.market_id} - No real money used")
        logger.info(f"📊 Would have used: ${position.quantity * position.entry_price:.2f}")
        return True


async def place_sell_limit_order(
    position: Position,
    limit_price: float,
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient
) -> bool:
    """Place a sell limit order to close an existing position."""
    logger = get_trading_logger("sell_limit_order")
    
    try:
        import uuid
        client_order_id = str(uuid.uuid4())
        limit_price_cents = int(limit_price * 100)
        side = position.side.lower()
        
        order_params = {
            "ticker": position.market_id,
            "client_order_id": client_order_id,
            "side": side,
            "action": "sell",
            "count": position.quantity,
            "type_": "limit"
        }
        
        if side == "yes":
            order_params["yes_price"] = limit_price_cents
        else:
            order_params["no_price"] = limit_price_cents
        
        logger.info(f"🎯 Placing SELL LIMIT order: {position.quantity} {side.upper()} at {limit_price_cents}¢ for {position.market_id}")
        
        response = await kalshi_client.place_order(order_params)
        
        if response and 'order' in response:
            order_id = response['order'].get('order_id', client_order_id)
            logger.info(f"✅ SELL LIMIT ORDER placed successfully! Order ID: {order_id}")
            return True
        else:
            logger.error(f"❌ Failed to place sell limit order: {response}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error placing sell limit order for {position.market_id}: {e}")
        return False


async def place_profit_taking_orders(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
    profit_threshold: float = 0.25
) -> Dict[str, int]:
    """Place sell limit orders for positions that have reached profit targets."""
    logger = get_trading_logger("profit_taking")
    results = {'orders_placed': 0, 'positions_processed': 0}
    
    try:
        positions = await db_manager.get_open_live_positions()
        if not positions:
            logger.info("No open positions to process for profit taking")
            return results
        
        logger.info(f"📊 Checking {len(positions)} positions for profit-taking opportunities")
        
        for position in positions:
            try:
                results['positions_processed'] += 1
                market_response = await kalshi_client.get_market(position.market_id)
                market_data = market_response.get('market', {})
                
                if not market_data:
                    continue
                
                if position.side == "YES":
                    current_price = market_data.get('yes_price', 0) / 100
                else:
                    current_price = market_data.get('no_price', 0) / 100
                
                if current_price > 0:
                    profit_pct = (current_price - position.entry_price) / position.entry_price
                    
                    if profit_pct >= profit_threshold:
                        sell_price = current_price * 0.98
                        success = await place_sell_limit_order(
                            position=position,
                            limit_price=sell_price,
                            db_manager=db_manager,
                            kalshi_client=kalshi_client
                        )
                        if success:
                            results['orders_placed'] += 1
            except Exception as e:
                logger.error(f"Error processing position {position.market_id} for profit taking: {e}")
                continue
        
        logger.info(f"🎯 Profit-taking summary: {results['orders_placed']} orders placed from {results['positions_processed']} positions")
        return results
        
    except Exception as e:
        logger.error(f"Error in profit-taking order placement: {e}")
        return results


async def place_stop_loss_orders(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
    stop_loss_threshold: float = -0.10
) -> Dict[str, int]:
    """Place sell limit orders for positions that need stop-loss protection."""
    logger = get_trading_logger("stop_loss_orders")
    results = {'orders_placed': 0, 'positions_processed': 0}
    
    try:
        positions = await db_manager.get_open_live_positions()
        if not positions:
            logger.info("No open positions to process for stop-loss orders")
            return results
        
        logger.info(f"🛡️ Checking {len(positions)} positions for stop-loss protection")
        
        for position in positions:
            try:
                results['positions_processed'] += 1
                market_response = await kalshi_client.get_market(position.market_id)
                market_data = market_response.get('market', {})
                
                if not market_data:
                    continue
                
                if position.side == "YES":
                    current_price = market_data.get('yes_price', 0) / 100
                else:
                    current_price = market_data.get('no_price', 0) / 100
                
                if current_price > 0:
                    loss_pct = (current_price - position.entry_price) / position.entry_price
                    
                    if loss_pct <= stop_loss_threshold:
                        stop_price = position.entry_price * (1 + stop_loss_threshold * 1.1)
                        stop_price = max(0.01, stop_price)
                        
                        success = await place_sell_limit_order(
                            position=position,
                            limit_price=stop_price,
                            db_manager=db_manager,
                            kalshi_client=kalshi_client
                        )
                        if success:
                            results['orders_placed'] += 1
            except Exception as e:
                logger.error(f"Error processing position {position.market_id} for stop loss: {e}")
                continue
        
        logger.info(f"🛡️ Stop-loss summary: {results['orders_placed']} orders placed from {results['positions_processed']} positions")
        return results
        
    except Exception as e:
        logger.error(f"Error in stop-loss order placement: {e}")
        return results