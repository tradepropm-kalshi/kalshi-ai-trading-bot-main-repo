"""
Configuration settings for the Kalshi AI trading system.
"""

import os
from typing import Dict, List
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class APIConfig:
    """API keys and base URLs."""
    kalshi_api_key: str = field(default_factory=lambda: os.getenv("KALSHI_API_KEY", ""))
    kalshi_base_url: str = "https://api.elections.kalshi.com"
    xai_api_key: str = field(default_factory=lambda: os.getenv("XAI_API_KEY", ""))

    # ── Free real-world data API keys (lean bot enrichment) ───────────────────
    odds_api_key: str = field(default_factory=lambda: os.getenv("ODDS_API_KEY", ""))
    fred_api_key: str = field(default_factory=lambda: os.getenv("FRED_API_KEY", ""))
    metaculus_api_key: str = field(default_factory=lambda: os.getenv("METACULUS_API_KEY", ""))
    newsapi_key: str = field(default_factory=lambda: os.getenv("NEWSAPI_KEY", ""))
    bls_api_key: str = field(default_factory=lambda: os.getenv("BLS_API_KEY", ""))
    # Jolpica F1, Manifold, PredictIt, NWS — no key required


@dataclass
class TradingConfig:
    """Trading strategy and risk parameters."""

    # ── Position sizing ───────────────────────────────────────────────────────
    max_position_size_pct: float = 3.0          # % of portfolio per position
    max_daily_loss_pct: float = 10.0            # Daily drawdown halt threshold (%)
    max_positions: int = 10
    min_balance: float = 100.0

    # ── Market filter ─────────────────────────────────────────────────────────
    min_volume: float = 500.0
    max_time_to_expiry_days: int = 14

    # ── AI decision ──────────────────────────────────────────────────────────
    min_confidence_to_trade: float = 0.60
    category_confidence_adjustments: Dict[str, float] = field(default_factory=lambda: {
        "sports": 0.90,
        "economics": 1.15,
        "politics": 1.05,
        "default": 1.0,
    })
    scan_interval_seconds: int = 60
    primary_model: str = "grok-3"
    fallback_model: str = "grok-3"
    ai_temperature: float = 0
    ai_max_tokens: int = 8000

    # ── Kelly criterion ───────────────────────────────────────────────────────
    use_kelly_criterion: bool = True
    kelly_fraction: float = 0.25
    max_single_position: float = 0.03           # 3% of portfolio per position

    # ── Live/paper toggle ─────────────────────────────────────────────────────
    live_trading_enabled: bool = field(
        default_factory=lambda: os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true"
    )
    paper_trading_mode: bool = field(
        default_factory=lambda: os.getenv("LIVE_TRADING_ENABLED", "false").lower() != "true"
    )

    # ── Phase Profit Mode ─────────────────────────────────────────────────────
    phase_mode_enabled: bool = field(
        default_factory=lambda: os.getenv("PHASE_MODE_ENABLED", "true").lower() == "true"
    )
    phase_base_capital: float = 100.0
    phase_profit_target: float = 2500.0
    secure_profit_per_chunk: float = 2400.0

    # ── Dynamic position sizing ───────────────────────────────────────────────
    default_position_size: float = 2.0
    position_size_multiplier: float = 1.0

    # ── AI cost controls ──────────────────────────────────────────────────────
    daily_ai_budget: float = 10.0
    daily_ai_cost_limit: float = 50.0
    max_ai_cost_per_decision: float = 0.20
    max_ensemble_cost: float = 0.50

    # ── Analysis deduplication ────────────────────────────────────────────────
    analysis_cooldown_hours: float = 2.0
    max_analyses_per_market_per_day: int = 3
    min_volume_for_ai_analysis: float = 1000.0
    skip_news_for_low_volume: bool = True
    news_search_volume_threshold: float = 5000.0

    # ── Category exclusions ───────────────────────────────────────────────────
    exclude_low_liquidity_categories: List[str] = field(
        default_factory=lambda: ["entertainment", "celebrity"]
    )

    # ── High-confidence near-expiry strategy ──────────────────────────────────
    enable_high_confidence_strategy: bool = True
    high_confidence_expiry_hours: float = 4.0
    high_confidence_market_odds: float = 0.85
    high_confidence_threshold: float = 0.80

    # ── Market-making parameters ──────────────────────────────────────────────
    max_concurrent_markets: int = 10
    min_spread_for_making: float = 0.03
    max_bid_ask_spread: float = 0.10
    max_inventory_risk: float = 0.01
    max_position_size: float = 500.0


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_level: str = "DEBUG"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "logs/trading_system.log"
    enable_file_logging: bool = True
    enable_console_logging: bool = True
    max_log_file_size: int = 10 * 1024 * 1024
    backup_count: int = 5


@dataclass
class Settings:
    """Root settings — single import point for the whole system."""
    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> bool:
        if not self.api.kalshi_api_key:
            raise ValueError("KALSHI_API_KEY environment variable is required")
        if not self.api.xai_api_key:
            raise ValueError("XAI_API_KEY environment variable is required")
        return True


# Global singleton — imported everywhere as `from src.config.settings import settings`
settings = Settings()

try:
    settings.validate()
except ValueError as e:
    print(f"Configuration error: {e}")
    print("Please set KALSHI_API_KEY and XAI_API_KEY in your .env file.")
