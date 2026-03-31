"""
OpenRouter client for multi-model AI-powered trading decisions.
Made optional - will only be used if OPENROUTER_API_KEY is present.
"""

import asyncio
import json
import os
import pickle
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from json_repair import repair_json

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None  # type: ignore

from src.clients.xai_client import TradingDecision, DailyUsageTracker
from src.config.settings import settings
from src.utils.logging_setup import TradingLoggerMixin


# Model registry: pricing per 1K tokens (USD) - kept for dashboard compatibility
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "anthropic/claude-sonnet-4.5": {"input_per_1k": 0.003, "output_per_1k": 0.015},
    "openai/o3": {"input_per_1k": 0.002, "output_per_1k": 0.008},
    "google/gemini-3-pro-preview": {"input_per_1k": 0.002, "output_per_1k": 0.012},
    "google/gemini-3-flash-preview": {"input_per_1k": 0.0005, "output_per_1k": 0.003},
    "deepseek/deepseek-v3.2": {"input_per_1k": 0.00025, "output_per_1k": 0.00038},
}

DEFAULT_FALLBACK_ORDER: List[str] = [
    "anthropic/claude-sonnet-4.5",
    "openai/o3",
    "google/gemini-3-pro-preview",
    "deepseek/deepseek-v3.2",
]


@dataclass
class ModelCostTracker:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0
    request_count: int = 0
    error_count: int = 0
    last_used: Optional[datetime] = None


class OpenRouterClient(TradingLoggerMixin):
    """
    OpenRouter client - now safe to import even if openai package is missing or key is absent.
    Will gracefully disable itself if no API key or package.
    """

    MAX_RETRIES_PER_MODEL: int = 3
    BASE_BACKOFF: float = 1.0
    MAX_BACKOFF: float = 30.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "anthropic/claude-sonnet-4",
        db_manager: Any = None,
    ):
        self.api_key = api_key or settings.api.openrouter_api_key
        self.base_url = settings.api.openrouter_base_url
        self.default_model = default_model
        self.db_manager = db_manager

        # Graceful disable if no key or openai not installed
        if not self.api_key or not OPENAI_AVAILABLE:
            self.enabled = False
            self.client = None
            self.logger.info("OpenRouterClient disabled (no API key or openai package)")
            return

        self.enabled = True
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=120.0,
            max_retries=0,
        )

        self.temperature = settings.trading.ai_temperature
        self.max_tokens = settings.trading.ai_max_tokens

        self.model_costs: Dict[str, ModelCostTracker] = {
            m: ModelCostTracker(model=m) for m in MODEL_PRICING
        }

        self.total_cost: float = 0.0
        self.request_count: int = 0
        self.usage_file = "logs/daily_openrouter_usage.pkl"
        self.daily_tracker: DailyUsageTracker = self._load_daily_tracker()

        self.logger.info(
            "OpenRouter client enabled",
            default_model=self.default_model,
            daily_limit=self.daily_tracker.daily_limit,
        )

    def _load_daily_tracker(self) -> DailyUsageTracker:
        today = datetime.now().strftime("%Y-%m-%d")
        daily_limit = getattr(settings.trading, "daily_ai_cost_limit", 50.0)
        os.makedirs("logs", exist_ok=True)

        try:
            if os.path.exists(self.usage_file):
                with open(self.usage_file, "rb") as fh:
                    tracker: DailyUsageTracker = pickle.load(fh)
                if tracker.date != today:
                    tracker = DailyUsageTracker(date=today, daily_limit=daily_limit)
                else:
                    if tracker.daily_limit != daily_limit:
                        tracker.daily_limit = daily_limit
                        if tracker.is_exhausted and tracker.total_cost < daily_limit:
                            tracker.is_exhausted = False
                return tracker
        except Exception as exc:
            self.logger.warning(f"Failed to load daily tracker: {exc}")

        return DailyUsageTracker(date=today, daily_limit=daily_limit)

    def _save_daily_tracker(self) -> None:
        try:
            os.makedirs("logs", exist_ok=True)
            with open(self.usage_file, "wb") as fh:
                pickle.dump(self.daily_tracker, fh)
        except Exception as exc:
            self.logger.error(f"Failed to save daily tracker: {exc}")

    def _update_daily_cost(self, cost: float) -> None:
        self.daily_tracker.total_cost += cost
        self.daily_tracker.request_count += 1
        self._save_daily_tracker()

        if self.daily_tracker.total_cost >= self.daily_tracker.daily_limit:
            self.daily_tracker.is_exhausted = True
            self.daily_tracker.last_exhausted_time = datetime.now()
            self._save_daily_tracker()
            self.logger.warning("Daily OpenRouter cost limit reached")

    async def _check_daily_limits(self) -> bool:
        if not self.enabled:
            return False
        self.daily_tracker = self._load_daily_tracker()
        if self.daily_tracker.is_exhausted:
            now = datetime.now()
            if self.daily_tracker.date != now.strftime("%Y-%m-%d"):
                self.daily_tracker = DailyUsageTracker(date=now.strftime("%Y-%m-%d"), daily_limit=self.daily_tracker.daily_limit)
                self._save_daily_tracker()
                return True
            return False
        return True

    # All other methods remain but return early if not enabled
    async def get_completion(self, *args, **kwargs) -> Optional[str]:
        if not self.enabled or not await self._check_daily_limits():
            return None
        self.logger.warning("OpenRouter get_completion called but disabled - returning None")
        return None

    async def get_trading_decision(self, *args, **kwargs) -> Optional[TradingDecision]:
        if not self.enabled:
            return None
        self.logger.warning("OpenRouter get_trading_decision called but disabled - returning None")
        return None

    def get_cost_summary(self) -> Dict[str, Any]:
        return {
            "total_cost": round(self.total_cost, 6),
            "total_requests": self.request_count,
            "daily_cost": round(self.daily_tracker.total_cost, 6) if hasattr(self, 'daily_tracker') else 0.0,
            "enabled": self.enabled,
        }

    async def close(self) -> None:
        if self.enabled and self.client:
            try:
                await self.client.close()
            except Exception:
                pass
        self.logger.info("OpenRouter client closed")