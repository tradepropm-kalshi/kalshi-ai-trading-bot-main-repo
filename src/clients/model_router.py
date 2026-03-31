"""
Unified model routing layer - now gracefully handles missing OpenRouter.
Preserves full RyanFrigo architecture and your Bible phase mode.
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from src.clients.xai_client import TradingDecision, XAIClient
from src.config.settings import settings
from src.utils.logging_setup import TradingLoggerMixin

# Import OpenRouterClient safely (it handles its own missing dependencies)
from src.clients.openrouter_client import OpenRouterClient, MODEL_PRICING


CAPABILITY_MAP: Dict[str, List[Tuple[str, str]]] = {
    "fast": [("grok-4-1-fast-reasoning", "xai")],
    "cheap": [("grok-4-1-fast-reasoning", "xai")],
    "reasoning": [("grok-4-1-fast-reasoning", "xai")],
    "balanced": [("grok-4-1-fast-reasoning", "xai")],
}

FULL_FLEET: List[Tuple[str, str]] = [("grok-4-1-fast-reasoning", "xai")]


@dataclass
class ModelHealth:
    model: str
    provider: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    total_latency: float = 0.0

    @property
    def success_rate(self) -> float:
        return 1.0 if self.total_requests == 0 else self.successful_requests / self.total_requests

    @property
    def is_healthy(self) -> bool:
        return self.consecutive_failures < 5


class ModelRouter(TradingLoggerMixin):
    def __init__(self, xai_client=None, openrouter_client=None, db_manager=None):
        self.db_manager = db_manager
        self.xai_client = xai_client
        self.openrouter_client = openrouter_client

        self.model_health: Dict[str, ModelHealth] = {}
        for model_name, provider in FULL_FLEET:
            key = f"{provider}::{model_name}"
            self.model_health[key] = ModelHealth(model=model_name, provider=provider)

        self.logger.info("ModelRouter initialized - XAI primary, OpenRouter optional")

    def _ensure_xai(self) -> XAIClient:
        if self.xai_client is None:
            self.xai_client = XAIClient(db_manager=self.db_manager)
        return self.xai_client

    def _ensure_openrouter(self) -> Optional[OpenRouterClient]:
        if self.openrouter_client is None:
            try:
                self.openrouter_client = OpenRouterClient(db_manager=self.db_manager)
            except Exception:
                self.openrouter_client = None
        return self.openrouter_client

    async def get_completion(self, prompt: str, **kwargs) -> Optional[str]:
        # Always prefer XAI first for your phase bot
        try:
            client = self._ensure_xai()
            return await client.get_completion(prompt=prompt, **kwargs)
        except Exception as e:
            self.logger.warning(f"XAI failed, trying OpenRouter fallback: {e}")
            or_client = self._ensure_openrouter()
            if or_client and or_client.enabled:
                return await or_client.get_completion(prompt=prompt, **kwargs)
            return None

    async def get_trading_decision(self, market_data: Dict, portfolio_data: Dict, news_summary: str = "", **kwargs) -> Optional[TradingDecision]:
        try:
            client = self._ensure_xai()
            return await client.get_trading_decision(
                market_data=market_data,
                portfolio_data=portfolio_data,
                news_summary=news_summary,
                **kwargs
            )
        except Exception as e:
            self.logger.warning(f"XAI decision failed, trying OpenRouter: {e}")
            or_client = self._ensure_openrouter()
            if or_client and or_client.enabled:
                return await or_client.get_trading_decision(
                    market_data=market_data,
                    portfolio_data=portfolio_data,
                    news_summary=news_summary,
                    **kwargs
                )
            return None

    def get_total_cost(self) -> float:
        total = self.xai_client.total_cost if self.xai_client else 0.0
        if self.openrouter_client and hasattr(self.openrouter_client, 'total_cost'):
            total += self.openrouter_client.total_cost
        return total

    def get_total_requests(self) -> int:
        total = self.xai_client.request_count if self.xai_client else 0
        if self.openrouter_client and hasattr(self.openrouter_client, 'request_count'):
            total += self.openrouter_client.request_count
        return total

    async def close(self) -> None:
        tasks = []
        if self.xai_client:
            tasks.append(self.xai_client.close())
        if self.openrouter_client and hasattr(self.openrouter_client, 'close'):
            tasks.append(self.openrouter_client.close())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)