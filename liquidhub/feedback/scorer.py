from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from liquidhub.config import settings
from liquidhub.db.engine import session_scope
from liquidhub.db.models import OHLCV, MarketPair, ShadowTrade, WorkerRegistry as WorkerRow
from liquidhub.logging import get_logger

log = get_logger("feedback.scorer")

_TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}


def score_mature_trades() -> int:
    now = datetime.now(UTC)
    tf = settings.shadow_score_timeframe
    tf_minutes = _TF_MINUTES.get(tf, 15)
    horizon_minutes = tf_minutes * settings.shadow_horizon_candles
    cutoff = now - timedelta(minutes=horizon_minutes)

    scored = 0
    with session_scope() as s:
        open_trades: list[ShadowTrade] = list(
            s.execute(
                select(ShadowTrade).where(
                    ShadowTrade.status == "OPEN",
                    ShadowTrade.scored.is_(False),
                    ShadowTrade.timeframe == tf,
                    ShadowTrade.ts <= cutoff,
                )
            ).scalars()
        )

        if not open_trades:
            return 0

        pair_cache: dict[str, int] = {}
        for trade in open_trades:
            if trade.symbol not in pair_cache:
                pair = s.execute(
                    select(MarketPair).where(
                        MarketPair.exchange == settings.exchange,
                        MarketPair.symbol == trade.symbol,
                    )
                ).scalar_one_or_none()
                pair_cache[trade.symbol] = pair.id if pair else -1

            pair_id = pair_cache.get(trade.symbol, -1)
            if pair_id == -1:
                continue

            horizon_ts = trade.ts + timedelta(minutes=horizon_minutes)
            candle = s.execute(
                select(OHLCV)
                .where(
                    OHLCV.pair_id == pair_id,
                    OHLCV.timeframe == tf,
                    OHLCV.timestamp >= horizon_ts,
                )
                .order_by(OHLCV.timestamp.asc())
                .limit(1)
            ).scalar_one_or_none()

            if candle is None:
                continue

            exit_price = float(candle.close)
            pnl_pct = _compute_pnl(trade.direction, trade.entry_price, exit_price)

            trade.exit_price = exit_price
            trade.exit_ts = candle.timestamp
            trade.pnl_pct = pnl_pct
            trade.status = "CLOSED"
            trade.scored = True
            scored += 1

    if scored:
        log.info("scored %d shadow trades", scored)
    return scored


def _compute_pnl(direction: str, entry: float, exit_: float) -> float:
    if entry == 0:
        return 0.0
    pct = (exit_ - entry) / entry
    return pct if direction == "LONG" else -pct
