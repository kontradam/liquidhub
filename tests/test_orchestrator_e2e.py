"""End-to-end test for the paper-trading orchestrator.

Spins up an in-memory SQLite DB, wires the real workers via
`_build_registry()` (minus the ML XGBoost which needs trained models on
disk), feeds the orchestrator synthetic OHLCV candles, and asserts that a
single `tick()` produces worker signals + a system event row.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from liquidhub.db.models import Base, SystemEvent, WorkerSignal
from liquidhub.db.models import WorkerRegistry as WorkerRow
from liquidhub.paper import orchestrator as orch
from liquidhub.voting.engine import WorkerVotingEngine
from liquidhub.workers.bb import BollingerBandsWorker
from liquidhub.workers.ma import MAWorker
from liquidhub.workers.macd import MACDWorker
from liquidhub.workers.momentum import MomentumWorker
from liquidhub.workers.registry import WorkerRegistry
from liquidhub.workers.rsi import RSIWorker
from liquidhub.workers.volume import VolumeWorker


# ---- fixtures -------------------------------------------------------------

@pytest.fixture()
def sqlite_session_scope(monkeypatch):
    """Replace the module-level `session_scope` with one bound to an
    in-memory SQLite DB, so orchestrator + registry DB writes are isolated.
    """
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    @contextmanager
    def _scope() -> Iterator[Session]:
        s = Local()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # Orchestrator, registry, and weights module all import `session_scope`
    # from `liquidhub.db.engine` — patch each binding.
    monkeypatch.setattr("liquidhub.paper.orchestrator.session_scope", _scope)
    monkeypatch.setattr("liquidhub.workers.registry.session_scope", _scope)
    return _scope


def _trend_candles(n: int = 120, up: bool = True) -> pd.DataFrame:
    """Synthetic 15m OHLCV: clean trend so technical workers vote directionally."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2026-04-01", periods=n, freq="15min", tz="UTC")
    drift = 0.6 if up else -0.6
    close = 100 + np.cumsum(drift + rng.normal(0, 0.2, size=n))
    high = close + rng.uniform(0.05, 0.4, size=n)
    low = close - rng.uniform(0.05, 0.4, size=n)
    open_ = close + rng.normal(0, 0.1, size=n)
    volume = rng.uniform(800, 1200, size=n)
    # spike on the last bar so volume worker fires
    volume[-1] = volume[:-1].mean() * 2.5
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ---- helpers --------------------------------------------------------------

def _technical_registry() -> WorkerRegistry:
    reg = WorkerRegistry()
    for W in (RSIWorker, MACDWorker, MAWorker, BollingerBandsWorker, VolumeWorker, MomentumWorker):
        reg.register(W(timeframe="15m"))
    reg.load_weights_from_db()
    return reg


# ---- tests ----------------------------------------------------------------

def test_build_registry_registers_technical_workers(monkeypatch, sqlite_session_scope):
    """`_build_registry` wires all six technical workers without needing models."""
    # Force "no trained models on disk" so we skip the XGBoost arm in the test.
    monkeypatch.setattr(
        "liquidhub.ml.models_registry.available_timeframes", lambda kind="xgboost": []
    )
    reg = orch._build_registry()
    names = {w.name for w in reg.active()}
    assert names == {
        "rsi_worker",
        "macd_worker",
        "ma_worker",
        "bb_worker",
        "volume_worker",
        "momentum_worker",
    }


def test_tick_writes_signals_and_event(monkeypatch, sqlite_session_scope):
    """tick(): synthetic up-trend candles → signals + a SystemEvent row."""
    candles = _trend_candles(up=True)

    monkeypatch.setattr(orch, "_load_candles", lambda symbol, timeframe, limit=500: candles)
    monkeypatch.setattr(orch.settings, "symbols", ["BTC/USDT:USDT"])
    monkeypatch.setattr(orch.settings, "timeframes", ["15m"])
    monkeypatch.setattr(orch.settings, "enable_hour_filter", False)

    reg = _technical_registry()
    voting = WorkerVotingEngine(reg)
    orch.tick(reg, voting)

    # Verify the side-effects landed in the in-memory DB. Read attributes
    # inside the session — SQLAlchemy detaches instances on commit/close.
    scope = sqlite_session_scope
    with scope() as s:
        worker_ids = {r.id for r in s.execute(select(WorkerRow)).scalars()}
        sig_worker_ids = {r.worker_id for r in s.execute(select(WorkerSignal)).scalars()}
        events = [
            (r.source, r.message, dict(r.payload or {}))
            for r in s.execute(select(SystemEvent)).scalars()
        ]

    assert len(worker_ids) == 6
    # Every active worker should have written a row (HOLD or otherwise).
    assert sig_worker_ids == worker_ids
    assert len(events) == 1
    source, message, payload = events[0]
    assert source == "orchestrator"
    assert message.startswith("BTC/USDT:USDT:")
    assert "consensus" in payload
    assert payload["total"] == 6


def test_tick_skipped_when_hour_blacklisted(monkeypatch, sqlite_session_scope):
    """Blacklisted hours → no signals, no events."""
    now = datetime.now(UTC)
    monkeypatch.setattr(orch.settings, "enable_hour_filter", True)
    monkeypatch.setattr(orch.settings, "blacklist_hours_utc", [now.hour])
    monkeypatch.setattr(orch.settings, "symbols", ["BTC/USDT:USDT"])
    monkeypatch.setattr(orch.settings, "timeframes", ["15m"])
    called = {"n": 0}

    def _loader(symbol, timeframe, limit=500):
        called["n"] += 1
        return _trend_candles()

    monkeypatch.setattr(orch, "_load_candles", _loader)
    reg = _technical_registry()
    orch.tick(reg, WorkerVotingEngine(reg))
    assert called["n"] == 0  # short-circuited before candle load


def test_tick_noop_when_no_candles(monkeypatch, sqlite_session_scope):
    """Empty candle frames → no DB writes, no crash."""
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    monkeypatch.setattr(orch, "_load_candles", lambda symbol, timeframe, limit=500: empty)
    monkeypatch.setattr(orch.settings, "symbols", ["BTC/USDT:USDT"])
    monkeypatch.setattr(orch.settings, "timeframes", ["15m"])
    monkeypatch.setattr(orch.settings, "enable_hour_filter", False)

    reg = _technical_registry()
    orch.tick(reg, WorkerVotingEngine(reg))

    scope = sqlite_session_scope
    with scope() as s:
        assert s.execute(select(WorkerSignal)).first() is None
        assert s.execute(select(SystemEvent)).first() is None


def test_tick_down_trend_direction(monkeypatch, sqlite_session_scope):
    """Down-trend candles produce a bearish aggregate per-direction weight."""
    candles = _trend_candles(up=False)
    monkeypatch.setattr(orch, "_load_candles", lambda symbol, timeframe, limit=500: candles)
    monkeypatch.setattr(orch.settings, "symbols", ["ETH/USDT:USDT"])
    monkeypatch.setattr(orch.settings, "timeframes", ["15m"])
    monkeypatch.setattr(orch.settings, "enable_hour_filter", False)

    reg = _technical_registry()
    # Verify directly via the voting engine snapshot — the orchestrator
    # persists the tally as part of `tick`, but we also want to assert the
    # direction is bearish before persistence.
    ts = datetime.now(UTC) - timedelta(minutes=15)
    ctx = orch.MarketContext(
        symbol="ETH/USDT:USDT",
        ts=ts,
        candles={"15m": candles},
        features={},
    )
    signals = [w.vote(ctx) for w in reg.active()]
    tally = WorkerVotingEngine(reg).tally(signals)
    assert tally.per_direction_weight["SHORT"] >= tally.per_direction_weight["LONG"]
