import time
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import select

from liquidhub.config import settings
from liquidhub.db.engine import session_scope
from liquidhub.db.models import OHLCV, MarketPair, ShadowTrade, SystemEvent, WorkerSignal
from liquidhub.db.models import WorkerRegistry as WorkerRow
from liquidhub.features.engine import latest_features
from liquidhub.feedback.loop import reward_punish
from liquidhub.feedback.scorer import score_mature_trades
from liquidhub.logging import get_logger
from liquidhub.voting.engine import WorkerVotingEngine
from liquidhub.workers.base import MarketContext, Signal
from liquidhub.workers.registry import WorkerRegistry

log = get_logger("paper.orchestrator")


def _load_candles(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    with session_scope() as s:
        pair = s.execute(
            select(MarketPair).where(
                MarketPair.exchange == settings.exchange,
                MarketPair.symbol == symbol,
            )
        ).scalar_one_or_none()
        if pair is None:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        stmt = (
            select(OHLCV)
            .where(OHLCV.pair_id == pair.id, OHLCV.timeframe == timeframe)
            .order_by(OHLCV.timestamp.desc())
            .limit(limit)
        )
        rows = list(s.execute(stmt).scalars())
        data = [
            {
                "ts": r.timestamp,
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close),
                "volume": float(r.volume),
            }
            for r in reversed(rows)
        ]
    if not data:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(data).set_index("ts")
    return df


def _load_symbols() -> list[str]:
    with session_scope() as s:
        rows = s.execute(
            select(MarketPair.symbol).where(
                MarketPair.exchange == settings.exchange,
                MarketPair.active.is_(True),
            ).order_by(MarketPair.symbol)
        ).scalars().all()
        return list(rows)


def _latest_close(candles: dict[str, pd.DataFrame]) -> float | None:
    for tf in settings.timeframes:
        df = candles.get(tf)
        if df is not None and not df.empty and "close" in df.columns:
            return float(df["close"].iloc[-1])
    return None


def _build_registry() -> WorkerRegistry:
    from liquidhub.ml.models_registry import available_timeframes
    from liquidhub.workers.accumulation_breakout import AccumulationWorker, BreakoutWorker
    from liquidhub.workers.bb import BollingerBandsWorker
    from liquidhub.workers.btc_cycle import BTCCycleWorker
    from liquidhub.workers.candlestick import CandlestickWorker
    from liquidhub.workers.ma import MAWorker
    from liquidhub.workers.macd import MACDWorker
    from liquidhub.workers.ml_xgboost import XGBoostWorker
    from liquidhub.workers.momentum import MomentumWorker
    from liquidhub.workers.regime import RegimeWorker
    from liquidhub.workers.rsi import RSIWorker
    from liquidhub.workers.volume import VolumeWorker

    reg = WorkerRegistry()

    for tf in available_timeframes("xgboost"):
        if tf in settings.timeframes:
            reg.register(XGBoostWorker(timeframe=tf))

    base_tf = "15m" if "15m" in settings.timeframes else settings.timeframes[0]
    for W in (RSIWorker, MACDWorker, MAWorker, BollingerBandsWorker, VolumeWorker, MomentumWorker, CandlestickWorker):
        reg.register(W(timeframe=base_tf))

    for tf in available_timeframes("accumulation"):
        if tf in settings.timeframes:
            reg.register(AccumulationWorker(timeframe=tf))

    for tf in available_timeframes("breakout"):
        if tf in settings.timeframes:
            reg.register(BreakoutWorker(timeframe=tf))

    if "15m" in settings.timeframes:
        reg.register(BTCCycleWorker())
        reg.register(RegimeWorker())

    reg.load_weights_from_db()
    return reg


def tick(registry: WorkerRegistry, voting: WorkerVotingEngine) -> None:
    now = datetime.now(UTC)
    symbols = _load_symbols()
    if not symbols:
        log.warning("no active symbols found in market_pairs for exchange=%s", settings.exchange)
        return

    results: list[dict] = []

    for symbol in symbols:
        candles = {tf: _load_candles(symbol, tf) for tf in settings.timeframes}
        if all(df.empty for df in candles.values()):
            continue

        features = {tf: latest_features(df) for tf, df in candles.items() if not df.empty}
        ctx = MarketContext(symbol=symbol, ts=now, candles=candles, features=features)

        signals = [s for w in registry.active() if (s := w.vote(ctx)) is not None]
        if not signals:
            continue

        result = voting.tally(signals)

        annotations: dict[str, str] = {}
        for sig in signals:
            wn = sig.worker
            if wn.startswith("accumulation_") or wn.startswith("breakout_") or wn.startswith("btc_cycle"):
                annotations[wn] = f"{sig.direction} @{sig.confidence:.0%}"

        results.append({
            "symbol": symbol,
            "direction": result.winner,
            "mark": result.mark,
            "consensus": result.consensus,
            "strength": result.strength,
            "agreement": result.agreement,
            "total": result.total_workers,
            "decision": result.decision,
            "annotations": annotations,
            "signals": signals,
        })

        with session_scope() as s:
            workers_by_name = {r.name: r for r in s.execute(select(WorkerRow)).scalars()}
            for sig in signals:
                row = workers_by_name.get(sig.worker)
                if row is None:
                    continue
                s.add(
                    WorkerSignal(
                        ts=now,
                        symbol=symbol,
                        worker_id=row.id,
                        direction=sig.direction,
                        confidence=sig.confidence,
                        reason=sig.reason,
                    )
                )
                if not sig.worker.startswith(("accumulation_", "breakout_", "btc_cycle")):
                    entry_price = _latest_close(candles)
                    if entry_price is not None:
                        s.add(
                            ShadowTrade(
                                ts=now,
                                symbol=symbol,
                                worker_id=row.id,
                                direction=sig.direction,
                                entry_price=entry_price,
                                confidence=sig.confidence,
                                timeframe=settings.shadow_score_timeframe,
                            )
                        )
            s.add(
                SystemEvent(
                    level="INFO",
                    source="orchestrator",
                    message=f"{symbol}: {result.decision}",
                    payload={
                        "mark": result.mark,
                        "winner": result.winner,
                        "consensus": result.consensus,
                        "strength": result.strength,
                        "agreement": result.agreement,
                        "total": result.total_workers,
                    },
                )
            )

    if results:
        results.sort(key=lambda r: r["mark"], reverse=True)
        _log_ranking(results, now)
    else:
        log.info("tick %s: no coins with data", now.strftime("%H:%M"))

    n_scored = score_mature_trades()
    if n_scored:
        n_adj = reward_punish(registry)
        log.info("feedback: scored=%d adjusted=%d", n_scored, n_adj)


def _log_ranking(results: list[dict], now: datetime) -> None:
    log.info("=" * 72)
    log.info("RANKING %s  (%d coins evaluated)", now.strftime("%H:%M UTC"), len(results))
    log.info("-" * 72)
    for r in results:
        log.info("Symbol: %-15s | Decision: %-6s | Mark: %5.3f", r["symbol"], r["decision"], r["mark"])
        for sig in r["signals"]:
            log.info("  -> Worker: %-20s | Dir: %-5s | Conf: %5.2f", sig.worker, sig.direction, sig.confidence)
    log.info("=" * 72)


def main_loop() -> None:
    log.info("paper orchestrator starting")
    registry = _build_registry()
    voting = WorkerVotingEngine(registry)

    if not registry.active():
        log.warning(
            "no workers registered — running empty ticks. "
            "Wire workers in orchestrator._build_registry()."
        )

    while True:
        try:
            tick(registry, voting)
        except Exception as e:
            log.exception("tick failed: %s", e)
        time.sleep(settings.tick_interval_seconds)
