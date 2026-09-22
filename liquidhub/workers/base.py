from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import pandas as pd

Direction = Literal["LONG", "SHORT"]


@dataclass(slots=True)
class MarketContext:
    """Inputs passed to every worker on each tick."""

    symbol: str
    ts: datetime
    candles: dict[str, pd.DataFrame]  # timeframe -> OHLCV DataFrame
    features: dict[str, pd.Series] = field(default_factory=dict)  # timeframe -> latest features
    orderbook: dict[str, Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Signal:
    """A single worker's vote for one symbol at one tick."""

    worker: str
    direction: Direction
    confidence: float  # 0.0 .. 1.0
    reason: dict[str, Any] = field(default_factory=dict)
    veto: bool = False  # risk_manager can set this to block the trade regardless of consensus


class Worker(ABC):
    name: str = "unnamed_worker"
    kind: str = "technical"  # ml | technical | sentiment | risk
    default_weight: float = 1.0
    required_timeframes: tuple[str, ...] = ("15m",)

    @abstractmethod
    def vote(self, ctx: MarketContext) -> Signal: ...

    def __repr__(self) -> str:
        return f"<Worker {self.name} ({self.kind})>"
