"""
Kalshi Client for the trading system.
Supports both LIVE and PAPER (sandbox) modes.
Private key is ONLY required for LIVE trading.

Fixes applied:
  - _make_request no longer uses 'async with self.client' which closed the
    underlying httpx.AsyncClient after every single request.
  - place_order accepts either a positional dict OR keyword-args so that
    callers (execute.py, market_making.py, quick_flip_scalping.py) all work
    regardless of whether they spread the params with ** or pass the dict.
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
            self.logger.info("PAPER / SANDBOX MODE - Using API key only (no private key required)")
            self.base_url = "https://demo-api.kalshi.co/trade-api/v2"
            # Keep the client alive across all requests — do NOT use it as a
            # context manager on individual calls or it gets closed after the first.
            self.client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
            return

        # LIVE MODE
        self.private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        if not self.private_key_path and os.getenv("KALSHI_PRIVATE_KEY"):
            self.private_key = os.getenv("KALSHI_PRIVATE_KEY")
            self.logger.info("Using inline private key from .env")
        elif self.private_key_path and os.path.exists(self.private_key_path):
            with open(self.private_key_path, "r") as f:
                self.private_key = f.read().strip()
            self.logger.info(f"Loaded private key from file: {self.private_key_path}")
        else:
            raise ValueError(
                "LIVE trading requires KALSHI_PRIVATE_KEY or KALSHI_PRIVATE_KEY_PATH in .env."
            )

        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)

    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """
        Send a single HTTP request using the long-lived client.

        Previously this used 'async with self.client as client:' which called
        aclose() on every request, making every subsequent call fail with a
        'client already closed' error.  The fix is to use self.client directly.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = await self.client.request(
                method, endpoint, headers=headers, **kwargs
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise KalshiAPIError(
                f"HTTP {e.response.status_code} for {method} {endpoint}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise KalshiAPIError(f"Request error for {method} {endpoint}: {e}") from e

    # ── Market data ───────────────────────────────────────────────────────────

    async def get_markets(
        self,
        limit: int = 100,
        status: str = "open",
        cursor: Optional[str] = None,
    ) -> Dict:
        params = {"limit": limit, "status": status}
        if cursor:
            params["cursor"] = cursor
        return await self._make_request("GET", "/markets", params=params)

    async def get_market(self, market_id: str) -> Dict:
        return await self._make_request("GET", f"/markets/{market_id}")

    async def get_orderbook(self, market_id: str) -> Dict:
        return await self._make_request("GET", f"/markets/{market_id}/orderbook")

    # ── Portfolio ─────────────────────────────────────────────────────────────

    async def get_balance(self) -> Dict:
        return await self._make_request("GET", "/portfolio/balance")

    async def get_positions(self) -> List[Dict]:
        return await self._make_request("GET", "/portfolio/positions")

    async def get_fills(self) -> List[Dict]:
        return await self._make_request("GET", "/fills")

    # ── Orders ────────────────────────────────────────────────────────────────

    async def place_order(self, order_params: Optional[Dict] = None, **kwargs) -> Dict:
        """
        Place an order.

        Accepts either:
          place_order({"ticker": ..., "side": ..., ...})   — positional dict
          place_order(ticker=..., side=..., ...)            — keyword args
          place_order(**order_params_dict)                  — spread kwargs

        This makes all three calling conventions used across the codebase work
        without requiring callers to be updated.
        """
        params = order_params if order_params is not None else kwargs
        return await self._make_request("POST", "/orders", json=params)

    async def cancel_order(self, order_id: str) -> Dict:
        return await self._make_request("DELETE", f"/orders/{order_id}")

    async def get_order(self, order_id: str) -> Dict:
        return await self._make_request("GET", f"/orders/{order_id}")

    # ── Opportunities (portfolio strategy helper) ─────────────────────────────

    async def get_opportunities(self) -> List[Dict]:
        """
        Return a lightweight list of open markets formatted as opportunity dicts.

        portfolio_optimization.py calls this to seed the Kelly-sizing loop.
        Each dict carries the fields that AdvancedPortfolioOptimizer.optimize_portfolio
        expects: market_id, edge, odds, side.

        Edge and side are left at neutral defaults here; the portfolio optimizer
        layers on its own AI-derived edge after receiving the list.
        """
        try:
            response = await self.get_markets(limit=100, status="open")
            markets = response.get("markets", [])
            opportunities = []
            for m in markets:
                ticker = m.get("ticker", "")
                if not ticker:
                    continue
                yes_ask = m.get("yes_ask", 0) or m.get("yes_ask_dollars", 0) or 0
                # Normalise to 0-1 scale (guard against string values from API)
                try:
                    yes_ask = float(yes_ask)
                except (TypeError, ValueError):
                    yes_ask = 0.5
                if yes_ask > 1.0:
                    yes_ask = yes_ask / 100.0
                yes_ask = max(0.01, min(0.99, yes_ask))
                opportunities.append({
                    "market_id": ticker,
                    "title": m.get("title", ""),
                    "edge": 0.0,        # will be updated by portfolio optimizer
                    "odds": yes_ask,
                    "side": "yes",
                })
            return opportunities
        except Exception as e:
            self.logger.error(f"Error fetching opportunities: {e}")
            return []

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def close(self):
        await self.client.aclose()
        self.logger.info("KalshiClient closed")


# Backward compatibility alias
KalshiAPIClient = KalshiClient
