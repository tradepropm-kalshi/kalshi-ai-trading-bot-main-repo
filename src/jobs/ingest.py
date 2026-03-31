"""
Market Ingestion Job
"""

import asyncio
import time
from datetime import datetime
from typing import Optional, List

from src.clients.kalshi_client import KalshiClient
from src.utils.database import DatabaseManager, Market
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger


async def process_and_queue_markets(
    markets_data: List[dict],
    db_manager: DatabaseManager,
    queue: asyncio.Queue,
    existing_position_market_ids: set,
    logger,
):
    markets_to_upsert = []
    for market_data in markets_data:
        if "yes_bid_dollars" in market_data:
            yes_price = (float(market_data.get("yes_bid_dollars", 0)) + float(market_data.get("yes_ask_dollars", 0))) / 2
            no_price = (float(market_data.get("no_bid_dollars", 0)) + float(market_data.get("no_ask_dollars", 0))) / 2
        else:
            yes_price = (market_data.get("yes_bid", 0) + market_data.get("yes_ask", 0)) / 2 / 100
            no_price = (market_data.get("no_bid", 0) + market_data.get("no_ask", 0)) / 2 / 100

        volume = int(float(market_data.get("volume_fp", 0) or market_data.get("volume", 0) or 0))

        market = Market(
            market_id=market_data["ticker"],
            title=market_data["title"],
            yes_price=yes_price,
            no_price=no_price,
            volume=volume,
            expiration_ts=int(datetime.fromisoformat(market_data["expiration_time"].replace("Z", "+00:00")).timestamp()),
            category=market_data.get("category", "unknown"),
            status=market_data["status"],
            last_updated=datetime.now(),
            has_position=market_data["ticker"] in existing_position_market_ids,
        )
        markets_to_upsert.append(market)

    if markets_to_upsert:
        await db_manager.upsert_markets(markets_to_upsert)
        logger.info(f"Successfully upserted {len(markets_to_upsert)} markets.")

        eligible_markets = [m for m in markets_to_upsert if m.volume >= 100.0]
        logger.info(f"Found {len(eligible_markets)} eligible markets.")
        for market in eligible_markets:
            await queue.put(market)


async def run_ingestion(
    db_manager: DatabaseManager,
    queue: asyncio.Queue,
    market_ticker: Optional[str] = None,
):
    logger = get_trading_logger("market_ingestion")
    logger.info("Starting market ingestion job.")

    kalshi_client = KalshiClient()

    try:
        existing_position_market_ids = await db_manager.get_markets_with_positions()

        if market_ticker:
            market_response = await kalshi_client.get_market(market_ticker)
            if market_response and "market" in market_response:
                await process_and_queue_markets([market_response["market"]], db_manager, queue, existing_position_market_ids, logger)
        else:
            cursor = None
            while True:
                response = await kalshi_client.get_markets(limit=100, cursor=cursor)
                markets_page = response.get("markets", [])
                active_markets = [m for m in markets_page if m["status"] == "active"]
                if active_markets:
                    await process_and_queue_markets(active_markets, db_manager, queue, existing_position_market_ids, logger)
                cursor = response.get("cursor")
                if not cursor:
                    break

    except Exception as e:
        logger.error(f"An error occurred during market ingestion: {e}")
    finally:
        await kalshi_client.close()
        logger.info("Market ingestion job finished.")