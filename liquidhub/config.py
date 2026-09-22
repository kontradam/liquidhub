from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    env: str = "dev"

    database_url: str = "postgresql+psycopg://liquidhub:liquidhub@localhost:5432/liquidhub"
    redis_url: str = "redis://localhost:6379/0"

    exchange: str = "bitget"
    exchange_api_key: str = ""
    exchange_api_secret: str = ""
    exchange_api_passphrase: str = ""

    timeframes: list[str] = Field(default_factory=lambda: ["5m", "15m", "1h", "4h"])

    vote_min_confidence: float = 0.55
    vote_min_consensus: float = 0.10
    worker_agreement_min: int = 4

    enable_futures_short: bool = True
    max_leverage: float = 5.0

    paper_capital: float = 10_000.0
    tick_interval_seconds: int = 900
    rank_top_n: int = 10

    models_dir: Path = Path("../models")
    log_level: str = "INFO"

    shadow_horizon_candles: int = 4
    shadow_reward_pct: float = 0.05
    shadow_penalty_pct: float = 0.03
    shadow_min_weight: float = 0.1
    shadow_max_weight: float = 5.0
    shadow_score_timeframe: str = "15m"
    shadow_streak_growth: float = 1.5

    signal_top_n_champions: int = 5
    signal_min_champion_agreement: int = 3
    signal_use_directional_winrate: bool = True
    signal_min_winrate_samples: int = 10

    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8000

    @field_validator("timeframes", mode="before")
    @classmethod
    def _split_csv(cls, v):
        if isinstance(v, str):
            return [p.strip() for p in v.split(",") if p.strip()]
        return v


settings = Settings()
