"""
OHLCV backfill: walk every active (pair, timeframe) and fill gaps.

For each pair × timeframe:
  - Inspect existing timestamps in `ohlcv_data`.
  - Compute head/internal/tail gaps relative to a lookback window.
  - Page through `fetch_ohlcv(since=...)` in CHUNK-sized requests until each
    gap is filled. Best-effort: failures on one window are logged and skipped.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from liquidhub.config import settings
from liquidhub.data.exchange import get_exchange, get_fallback_exchange
from liquidhub.data.ohlcv_collector import TF_SECONDS, _persist
from liquidhub.db.engine import session_scope
from liquidhub.db.models import OHLCV, MarketPair
from liquidhub.logging import get_logger

log = get_logger("data.ohlcv_backfill")

DEFAULT_LOOKBACK_DAYS = 30
CHUNK = 500
GAP_TOLERANCE = 1.5  # treat consecutive ts gaps > 1.5x interval as missing
SLEEP_BETWEEN_REQUESTS = 0.25


def _existing_timestamps(pair_id: int, timeframe: str) -> list[datetime]:
    with session_scope() as s:
        rows = s.execute(
            select(OHLCV.timestamp)
            .where(OHLCV.pair_id == pair_id, OHLCV.timeframe == timeframe)
            .order_by(OHLCV.timestamp)
        ).all()
        return [r[0] for r in rows]


def _to_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=UTC).timestamp() * 1000)


def _find_gaps(
    timestamps: list[datetime],
    tf_seconds: int,
    lookback_start: datetime,
    now: datetime,
) -> list[tuple[int, int]]:
    interval = timedelta(seconds=tf_seconds)
    gaps: list[tuple[int, int]] = []
    if not timestamps:
        return [(_to_ms(lookback_start), _to_ms(now))]
    if timestamps[0] > lookback_start + interval:
        gaps.append((_to_ms(lookback_start), _to_ms(timestamps[0])))
    for prev, cur in zip(timestamps, timestamps[1:]):
        if (cur - prev) > interval * GAP_TOLERANCE:
            gaps.append((_to_ms(prev + interval), _to_ms(cur)))
    if (now - timestamps[-1]) > interval:
        gaps.append((_to_ms(timestamps[-1] + interval), _to_ms(now)))
    return gaps


def backfill_pair_tf(
    exchange_name: str,
    pair_id: int,
    symbol: str,
    timeframe: str,
    lookback_days: int,
) -> int:
    if timeframe not in TF_SECONDS:
        log.warning("%s %s: unknown timeframe, skipping", symbol, timeframe)
        return 0
    tf_s = TF_SECONDS[timeframe]
    now = datetime.now(UTC).replace(tzinfo=None)
    lookback_start = now - timedelta(days=lookback_days)

    existing = _existing_timestamps(pair_id, timeframe)
    gaps = _find_gaps(existing, tf_s, lookback_start, now)
    if not gaps:
        log.info("%s %s: no gaps", symbol, timeframe)
        return 0

    ex = get_exchange()
    fb = get_fallback_exchange()
    total_new = 0
    for start_ms, end_ms in gaps:
        cursor = start_ms
        windows = 0
        while cursor < end_ms:
            rows = None
            try:
                rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=CHUNK)
            except Exception as e:
                log.warning("%s %s: primary fetch since=%d failed: %s", symbol, timeframe, cursor, e)
            if not rows:
                try:
                    rows = fb.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=CHUNK)
                    if rows:
                        log.info("%s %s: filled gap from fallback (binanceusdm)", symbol, timeframe)
                except Exception as e:
                    log.warning("%s %s: fallback fetch since=%d failed: %s", symbol, timeframe, cursor, e)
                    break
            if not rows:
                break
            n = _persist(exchange_name, symbol, timeframe, rows)
            total_new += max(n, 0)
            last_ms = rows[-1][0]
            if last_ms <= cursor:
                break
            cursor = last_ms + tf_s * 1000
            windows += 1
            time.sleep(SLEEP_BETWEEN_REQUESTS)
        log.info(
            "%s %s: gap [%s..%s] -> %d windows fetched",
            symbol,
            timeframe,
            datetime.fromtimestamp(start_ms / 1000, tz=UTC).isoformat(timespec="minutes"),
            datetime.fromtimestamp(end_ms / 1000, tz=UTC).isoformat(timespec="minutes"),
            windows,
        )
    log.info("%s %s: backfill complete, ~%d candles inserted", symbol, timeframe, total_new)
    return total_new


def run_backfill(
    timeframes: list[str] | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> None:
    tfs = timeframes or settings.timeframes
    exchange_name = settings.exchange
    with session_scope() as s:
        pairs = s.execute(
            select(MarketPair.id, MarketPair.symbol).where(
                MarketPair.exchange == exchange_name, MarketPair.active.is_(True)
            )
        ).all()
    log.info("backfill start: %d pair(s), tfs=%s, lookback=%dd", len(pairs), tfs, lookback_days)
    grand_total = 0
    for pair_id, symbol in pairs:
        for tf in tfs:
            try:
                grand_total += backfill_pair_tf(
                    exchange_name, pair_id, symbol, tf, lookback_days
                )
            except Exception as e:
                log.exception("backfill %s %s failed: %s", symbol, tf, e)
    log.info("backfill done: ~%d candles inserted total", grand_total)
