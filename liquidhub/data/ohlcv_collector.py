import time
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from liquidhub.config import settings
from liquidhub.data.exchange import fetch_ohlcv, get_fallback_exchange
from liquidhub.db.engine import session_scope
from liquidhub.db.models import OHLCV, MarketPair
from liquidhub.logging import get_logger

log = get_logger("data.ohlcv_collector")

TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}


def _resolve_pair_id(session, exchange: str, symbol: str) -> int:
    row = session.execute(
        select(MarketPair).where(
            MarketPair.exchange == exchange, MarketPair.symbol == symbol
        )
    ).scalar_one_or_none()
    if row is not None:
        return row.id
    base = symbol.split("/")[0] if "/" in symbol else None
    quote_part = symbol.split("/")[1] if "/" in symbol else ""
    quote = quote_part.split(":")[0] if quote_part else None
    row = MarketPair(exchange=exchange, symbol=symbol, base=base, quote=quote, active=True)
    session.add(row)
    session.flush()
    return row.id


def _persist(exchange: str, symbol: str, timeframe: str, rows: list[list[float]]) -> int:
    if not rows:
        return 0
    with session_scope() as s:
        pair_id = _resolve_pair_id(s, exchange, symbol)
        payload = [
            {
                "pair_id": pair_id,
                "timeframe": timeframe,
                "timestamp": datetime.fromtimestamp(r[0] / 1000, tz=UTC).replace(tzinfo=None),
                "open": r[1],
                "high": r[2],
                "low": r[3],
                "close": r[4],
                "volume": r[5],
            }
            for r in rows
        ]
        stmt = insert(OHLCV).values(payload)
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["pair_id", "timeframe", "timestamp"]
        ).returning(OHLCV.timestamp)
        return len(s.execute(stmt).all())


def collect_once(symbols: list[str], timeframes: list[str]) -> None:
    exchange = settings.exchange
    for sym in symbols:
        for tf in timeframes:
            rows = None
            try:
                rows = fetch_ohlcv(sym, tf, limit=500)
            except Exception as e:
                log.warning("collect %s %s primary failed: %s", sym, tf, e)
            if not rows:
                try:
                    rows = get_fallback_exchange().fetch_ohlcv(sym, timeframe=tf, limit=500)
                    if rows:
                        log.info("%s %s: filled from fallback (binanceusdm)", sym, tf)
                except Exception as e:
                    log.warning("collect %s %s fallback failed: %s", sym, tf, e)
                    continue
            if rows:
                n = _persist(exchange, sym, tf, rows)
                log.info("%s %s: %d new candles", sym, tf, n)


def _active_symbols(exchange: str) -> list[str]:
    with session_scope() as s:
        rows = s.execute(
            select(MarketPair.symbol).where(
                MarketPair.exchange == exchange, MarketPair.active.is_(True)
            )
        ).all()
        return [r[0] for r in rows]


def run_collector(timeframes: list[str] | None = None, once: bool = False) -> None:
    tfs = timeframes or settings.timeframes
    smallest = min((TF_SECONDS[t] for t in tfs if t in TF_SECONDS), default=60)
    while True:
        collect_once(_active_symbols(settings.exchange), tfs)
        if once:
            return
        time.sleep(max(30, smallest // 2))
