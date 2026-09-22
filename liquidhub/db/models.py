from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class MarketPair(Base):
    __tablename__ = "market_pairs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    exchange: Mapped[str] = mapped_column(String(50))
    symbol: Mapped[str] = mapped_column(String(50))
    base: Mapped[str | None] = mapped_column(String(20), nullable=True)
    quote: Mapped[str | None] = mapped_column(String(20), nullable=True)
    active: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class OHLCV(Base):
    __tablename__ = "ohlcv_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pair_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("market_pairs.id"), nullable=True, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    timeframe: Mapped[str] = mapped_column(String(10), index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)

    __table_args__ = (
        UniqueConstraint("pair_id", "timeframe", "timestamp", name="idx_ohlcv_unique"),
    )


class WorkerRegistry(Base):
    __tablename__ = "worker_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    kind: Mapped[str] = mapped_column(String(32))  # ml, technical, sentiment, risk
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    meta: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    signals: Mapped[list["WorkerSignal"]] = relationship(back_populates="worker")


class WorkerSignal(Base):
    __tablename__ = "worker_signals"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    worker_id: Mapped[int] = mapped_column(ForeignKey("worker_registry.id"), index=True)
    direction: Mapped[str] = mapped_column(String(8))  # LONG / SHORT / HOLD
    confidence: Mapped[float] = mapped_column(Float)
    reason: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    worker: Mapped[WorkerRegistry] = relationship(back_populates="signals")


class WeightHistory(Base):
    __tablename__ = "weight_history"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    worker_id: Mapped[int] = mapped_column(ForeignKey("worker_registry.id"), index=True)
    old_weight: Mapped[float] = mapped_column(Float)
    new_weight: Mapped[float] = mapped_column(Float)
    reason: Mapped[str | None] = mapped_column(String(255), nullable=True)


class PaperTrade(Base):
    __tablename__ = "paper_trades"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(8))  # LONG / SHORT
    entry_price: Mapped[float] = mapped_column(Numeric(20, 8))
    exit_price: Mapped[float | None] = mapped_column(Numeric(20, 8), nullable=True)
    size: Mapped[float] = mapped_column(Numeric(24, 8))
    leverage: Mapped[float] = mapped_column(Float, default=1.0)
    pnl: Mapped[float | None] = mapped_column(Numeric(20, 8), nullable=True)
    commission: Mapped[float | None] = mapped_column(Numeric(20, 8), nullable=True)
    entry_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    exit_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="OPEN")  # OPEN / CLOSED / CANCELLED
    strategy: Mapped[str] = mapped_column(String(32), default="directional")
    reason: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class HoldDecision(Base):
    __tablename__ = "hold_decisions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    consensus: Mapped[float] = mapped_column(Float)
    agreement_count: Mapped[int] = mapped_column(Integer)
    outcome: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # Profit-based scoring (see IMMEDIATE_NEXT_STEPS.md Fix 1): stores the
    # forward-looking price move so HoldTracker can reward PROFIT, not HOLD.
    forward_return: Mapped[float | None] = mapped_column(Float, nullable=True)


class SystemEvent(Base):
    __tablename__ = "system_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    level: Mapped[str] = mapped_column(String(16))
    source: Mapped[str] = mapped_column(String(64))
    message: Mapped[str] = mapped_column(String(1024))
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class MarketOpportunity(Base):
    __tablename__ = "market_opportunities"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    consensus: Mapped[float] = mapped_column(Float)
    direction: Mapped[str] = mapped_column(String(8))
    acted: Mapped[bool] = mapped_column(Boolean, default=False)
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class ShadowTrade(Base):
    __tablename__ = "shadow_trades"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    worker_id: Mapped[int] = mapped_column(ForeignKey("worker_registry.id"), index=True)
    direction: Mapped[str] = mapped_column(String(8))
    entry_price: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    timeframe: Mapped[str] = mapped_column(String(10))
    status: Mapped[str] = mapped_column(String(16), default="OPEN")
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    pnl_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    scored: Mapped[bool] = mapped_column(Boolean, default=False)
    feedback_applied: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    worker: Mapped[WorkerRegistry] = relationship()
