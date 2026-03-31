"""
Configuration settings for the Kalshi trading system.
Manages trading parameters, API configurations, and risk management settings.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
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


@dataclass
class EnsembleConfig:
    """Multi-model ensemble configuration."""
    enabled: bool = True
    models: Dict[str, Dict] = field(default_factory=lambda: {
        "grok-3": {"provider": "xai", "role": "forecaster", "weight": 0.30},
        "anthropic/claude-3.5-sonnet": {"provider": "openrouter", "role": "news_analyst", "weight": 0.20},
        "openai/gpt-4o": {"provider": "openrouter", "role": "bull_researcher", "weight": 0.20},
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
    sentiment_model: str = "google/gemini-3.1-flash-lite-preview"
    cache_ttl_minutes: int = 30
    max_articles_per_source: int = 10
    relevance_threshold: float = 0.3


@dataclass
class TradingConfig:
    """Trading strategy configuration."""
    max_position_size_pct: float = 3.0
    max_daily_loss_pct: float = 10.0
    max_positions: int = 10
    min_balance: float = 100.0
    
    min_volume: float = 500.0
    max_time_to_expiry_days: int = 14
    
    min_confidence_to_trade: float = 0.60
    category_confidence_adjustments: Dict[str, float] = field(default_factory=lambda: {
        "sports": 0.90,
        "economics": 1.15,
        "politics": 1.05,
        "default": 1.0
    })
    
    scan_interval_seconds: int = 60
    primary_model: str = "grok-3"
    fallback_model: str = "grok-4-1-fast-non-reasoning"
    ai_temperature: float = 0
    ai_max_tokens: int = 8000
    
    use_kelly_criterion: bool = True
    kelly_fraction: float = 0.25
    max_single_position: float = 0.03
    
    live_trading_enabled: bool = field(default_factory=lambda: os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true")
    paper_trading_mode: bool = field(default_factory=lambda: os.getenv("LIVE_TRADING_ENABLED", "false").lower() != "true")

    # === PHASE PROFIT MODE — YOUR EXACT CRITERIA ===
    phase_mode_enabled: bool = field(default_factory=lambda: os.getenv("PHASE_MODE_ENABLED", "true").lower() == "true")
    phase_base_capital: float = 100.0          # Trades sized against this $100 base
    phase_profit_target: float = 2500.0        # Realize this profit → trigger secure + reset
    secure_profit_per_chunk: float = 2400.0    # Secure exactly $2,400 per completed phase


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

# Validate settings on import
try:
    settings.validate()
except ValueError as e:
    print(f"Configuration validation error: {e}")
    print("Please check your environment variables and configuration.")