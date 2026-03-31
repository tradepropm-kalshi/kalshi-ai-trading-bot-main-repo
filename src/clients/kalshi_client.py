"""
Kalshi Client for the trading system.
Supports both LIVE and PAPER (sandbox) modes.
Private key is ONLY required for LIVE trading.
"""

import asyncio
import os
from typing import Dict, List, Optional, Any
import httpx

from src.config.settings import settings
from src.utils.logging_setup import TradingLoggerMixin


class KalshiAPIError(Exception):
    """Custom exception for Kalshi API errors."""
    pass


class KalshiClient(TradingLoggerMixin):
    def __init__(self):
        self.api_key = settings.api.kalshi_api_key
        self.base_url = settings.api.kalshi_base_url
        self.is_live = settings.trading.live_trading_enabled
        self.private_key = None
        self.private_key_path = None

        if not self.is_live:
            self.logger.info("🚀 PAPER / SANDBOX MODE - Using API key only (no private key required)")
            self.base_url = "https://demo-api.kalshi.com"
            self.client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
            return

        # LIVE MODE (unchanged)
        self.private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        if not self.private_key_path and os.getenv("KALSHI_PRIVATE_KEY"):
            self.private_key = os.getenv("KALSHI_PRIVATE_KEY")
            self.logger.info("Using inline private key from .env")
        elif self.private_key_path and os.path.exists(self.private_key_path):
            with open(self.private_key_path, "r") as f:
                self.private_key = f.read().strip()
            self.logger.info(f"Loaded private key from file: {self.private_key_path}")
        else:
            raise ValueError("LIVE trading requires KALSHI_PRIVATE_KEY or KALSHI_PRIVATE_KEY_PATH in .env.")

        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)

    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}{endpoint}"
        async with self.client as client:
            response = await client.request(method, url, headers=headers, **kwargs)
            response.raise_for_status()
            return response.json()

    # Updated to support cursor (used by ingest.py)
    async def get_markets(self, limit: int = 100, status: str = "open", cursor: Optional[str] = None) -> Dict:
        params = {"limit": limit, "status": status}
        if cursor:
            params["cursor"] = cursor
        return await self._make_request("GET", "/markets", params=params)

    async def get_market(self, market_id: str) -> Dict:
        return await self._make_request("GET", f"/markets/{market_id}")

    async def get_orderbook(self, market_id: str) -> Dict:
        return await self._make_request("GET", f"/markets/{market_id}/orderbook")

    async def get_balance(self) -> Dict:
        return await self._make_request("GET", "/portfolio/balance")

    async def place_order(self, order_params: Dict) -> Dict:
        return await self._make_request("POST", "/orders", json=order_params)

    async def get_positions(self) -> List[Dict]:
        return await self._make_request("GET", "/portfolio/positions")

    async def get_fills(self) -> List[Dict]:
        return await self._make_request("GET", "/fills")

    async def close(self):
        await self.client.aclose()
        self.logger.info("KalshiClient closed")


# Backward compatibility
KalshiAPIClient = KalshiClient