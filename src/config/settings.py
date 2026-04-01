"""
Configuration settings for the Kalshi trading system.
Manages trading parameters, API configurations, and risk management settings.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class APIConfig:
    """API configuration settings."""
    kalshi_api_key: str = field(default_factory=lambda: os.getenv("KALSHI_API_KEY", ""))
    kalshi_base_url: str = "https://api.elections.kalshi.com"
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    xai_api_key: str = field(default_factory=lambda: os.getenv("XAI_API_KEY", ""))
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openai_base_url: str = "https://api.openai.com/v1"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # ── Free real-world data API keys (lean bot enrichment) ───────────────────
    # The Odds API  — https://the-odds-api.com  (500 req/month free)
    odds_api_key: str = field(default_factory=lambda: os.getenv("ODDS_API_KEY", ""))
    # FRED (Federal Reserve)  — https://fred.stlouisfed.org/docs/api/api_key.html
    fred_api_key: str = field(default_factory=lambda: os.getenv("FRED_API_KEY", ""))
    # Metaculus  — https://www.metaculus.com/accounts/profile/#apikey  (free)
    metaculus_api_key: str = field(default_factory=lambda: os.getenv("METACULUS_API_KEY", ""))
    # NewsAPI  — https://newsapi.org/register  (100 req/day free)
    newsapi_key: str = field(default_factory=lambda: os.getenv("NEWSAPI_KEY", ""))


@dataclass
class EnsembleConfig:
    """Multi-model ensemble configuration."""
    enabled: bool = True
    models: Dict[str, Dict] = field(default_factory=lambda: {
        "grok-3": {"provider": "xai", "role": "forecaster", "weight": 0.30},
        # claude-3-5-sonnet-20241022 is the correct OpenRouter model ID
        "anthropic/claude-3-5-sonnet-20241022": {"provider": "openrouter", "role": "news_analyst", "weight": 0.20},
        "openai/gpt-4o": {"provider": "openrouter", "role": "bull_researcher", "weight": 0.20},
        # gemini-flash-1.5 is the validated model name on OpenRouter
        "google/gemini-flash-1.5": {"provider": "openrouter", "role": "bear_researcher", "weight": 0.15},
        "deepseek/deepseek-r1": {"provider": "openrouter", "role": "risk_manager", "weight": 0.15},
    })
    min_models_for_consensus: int = 3
    disagreement_threshold: float = 0.25
    parallel_requests: bool = True
    debate_enabled: bool = True
    calibration_tracking: bool = True
    max_ensemble_cost: float = 0.50


@dataclass
class SentimentConfig:
    """News and sentiment analysis configuration."""
    enabled: bool = True
    rss_feeds: List[str] = field(default_factory=lambda: [
        "https://feeds.reuters.com/reuters/topNews",
        "https://feeds.reuters.com/reuters/businessNews",
        "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
        "https://feeds.bbci.co.uk/news/business/rss.xml",
    ])
    # google/gemini-flash-1.5 is a confirmed valid OpenRouter model
    sentiment_model: str = "google/gemini-flash-1.5"
    cache_ttl_minutes: int = 30
    max_articles_per_source: int = 10
    relevance_threshold: float = 0.3


@dataclass
class TradingConfig:
    """Trading strategy configuration."""

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
    # grok-3 is a confirmed valid xAI model
    primary_model: str = "grok-3"
    # Fall back to grok-3 if primary is unavailable (was a hallucinated model ID)
    fallback_model: str = "grok-3"
    ai_temperature: float = 0
    ai_max_tokens: int = 8000

    # ── Kelly criterion ───────────────────────────────────────────────────────
    use_kelly_criterion: bool = True
    kelly_fraction: float = 0.25            # Quarter-Kelly cap
    max_single_position: float = 0.03       # 3 % of portfolio per position (as fraction)

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
    phase_base_capital: float = 100.0       # Trades sized against this $100 base
    phase_profit_target: float = 2500.0     # Trigger secure + reset at this profit
    secure_profit_per_chunk: float = 2400.0 # Amount to secure per completed phase

    # ── Dynamic position sizing (used by decide.py) ───────────────────────────
    default_position_size: float = 2.0      # Base investment as % of balance
    position_size_multiplier: float = 1.0   # Scales investment by confidence delta

    # ── AI cost controls ──────────────────────────────────────────────────────
    daily_ai_budget: float = 10.0           # Hard daily spend cap in USD
    daily_ai_cost_limit: float = 50.0       # Soft daily cap used in evaluate.py
    max_ai_cost_per_decision: float = 0.20  # Per-market analysis ceiling
    max_ensemble_cost: float = 0.50         # Per-ensemble call ceiling

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
    min_spread_for_making: float = 0.03     # $0.03 minimum bid-ask spread
    max_bid_ask_spread: float = 0.10        # $0.10 maximum spread
    max_inventory_risk: float = 0.01        # Inventory penalty coefficient
    # Dollar cap per side per market for market-making sizing
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
    """Main settings class combining all configuration."""
    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)

    def validate(self) -> bool:
        if not self.api.kalshi_api_key:
            raise ValueError("KALSHI_API_KEY environment variable is required")
        if not self.api.xai_api_key:
            raise ValueError("XAI_API_KEY environment variable is required")
        return True


# Global settings instance
settings = Settings()

try:
    settings.validate()
except ValueError as e:
    print(f"Configuration validation error: {e}")
    print("Please check your environment variables and configuration.")
