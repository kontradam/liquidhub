"""
Derivatives collector: Open Interest, Funding Rate, Long/Short Ratio.

Writes to:
  - open_interest_snapshots (pair_id, timestamp, open_interest_base, open_interest_quote, mark_price)
  - funding_rate_snapshots  (pair_id, timestamp, funding_rate, next_funding_at, mark_price)
  - long_short_ratio_snapshots (pair_id, timestamp, ls_ratio, long_pct, short_pct, source)

Replaces the old trading-system derivatives_collector.py which was
orphaned when that tree was deleted. Symbols are read from market_pairs
(exchange=settings.exchange, active=true).
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from sqlalchemy import select, text

from liquidhub.config import settings
from liquidhub.data.exchange import get_exchange, get_fallback_exchange
from liquidhub.db.engine import session_scope
from liquidhub.db.models import MarketPair
from liquidhub.logging import get_logger

FALLBACK_SOURCE = "binanceusdm"

log = get_logger("data.derivatives_collector")

POLL_SECONDS = 60


def _active_symbols(exchange: str) -> list[tuple[int, str]]:
    with session_scope() as s:
        rows = s.execute(
            select(MarketPair.id, MarketPair.symbol).where(
                MarketPair.exchange == exchange, MarketPair.active.is_(True)
            )
        ).all()
        return [(r[0], r[1]) for r in rows]


def _ms_to_dt(ms: int | float | None) -> datetime | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(float(ms) / 1000.0, tz=UTC).replace(tzinfo=None)


def _collect_oi(pair_id: int, symbol: str, ex) -> bool:
    try:
        r = ex.fetch_open_interest(symbol)
    except Exception as e:
        log.warning("OI %s: fetch failed: %s", symbol, e)
        return False

    ts = _ms_to_dt(r.get("timestamp")) or datetime.now(UTC).replace(tzinfo=None)
    oi_base = r.get("openInterestAmount")
    oi_quote = r.get("openInterestValue")
    mark = r.get("markPrice")
    if oi_base is None and oi_quote is None:
        log.warning("OI %s: no values in response", symbol)
        return False

    with session_scope() as s:
        s.execute(
            text(
                "INSERT INTO open_interest_snapshots "
                "(pair_id, timestamp, open_interest_base, open_interest_quote, mark_price) "
                "VALUES (:pid, :ts, :b, :q, :mp) "
                "ON CONFLICT (pair_id, timestamp) DO NOTHING"
            ),
            {"pid": pair_id, "ts": ts, "b": oi_base, "q": oi_quote, "mp": mark},
        )
    return True


def _collect_funding(pair_id: int, symbol: str, ex) -> bool:
    try:
        r = ex.fetch_funding_rate(symbol)
    except Exception as e:
        log.warning("FR %s: fetch failed: %s", symbol, e)
        return False

    rate = r.get("fundingRate")
    if rate is None:
        log.warning("FR %s: no fundingRate", symbol)
        return False

    ts = _ms_to_dt(r.get("timestamp")) or datetime.now(UTC).replace(tzinfo=None)
    next_funding = _ms_to_dt(r.get("fundingTimestamp") or r.get("nextFundingTimestamp"))
    mark = r.get("markPrice")
    if mark is None:
        info = r.get("info") or {}
        try:
            mark = float(info.get("markPrice")) if info.get("markPrice") else None
        except (TypeError, ValueError):
            mark = None

    with session_scope() as s:
        s.execute(
            text(
                "INSERT INTO funding_rate_snapshots "
                "(pair_id, timestamp, funding_rate, next_funding_at, mark_price) "
                "VALUES (:pid, :ts, :r, :nf, :mp) "
                "ON CONFLICT (pair_id, timestamp) DO NOTHING"
            ),
            {"pid": pair_id, "ts": ts, "r": float(rate), "nf": next_funding, "mp": mark},
        )
    return True


def _collect_lsr(pair_id: int, symbol: str, ex, source: str | None = None) -> bool:
    src = source or settings.exchange
    for attempt in range(3):
        try:
            rows = ex.fetch_long_short_ratio_history(symbol, timeframe="5m", limit=2)
            break
        except Exception as e:
            err_str = str(e).lower()
            if "40054" in err_str:
                if "empty" in err_str:
                    log.debug("LSR %s: exchange reports empty data (40054)", symbol)
                    return False
                if attempt < 2:
                    wait = 2 * (attempt + 1)
                    log.debug("LSR %s: rate limited (40054), retrying in %ds", symbol, wait)
                    time.sleep(wait)
                    continue
            log.warning("LSR %s: fetch failed: %s", symbol, e)
            return False
    else:
        return False

    if not rows:
        return False

    latest = rows[-1]
    info = latest.get("info") or {}
    try:
        long_pct = float(info.get("longAccountRatio")) if info.get("longAccountRatio") else None
        short_pct = float(info.get("shortAccountRatio")) if info.get("shortAccountRatio") else None
    except (TypeError, ValueError):
        long_pct = short_pct = None

    # Existing DB convention: ls_ratio = long_pct / short_pct (not the raw Bitget diff field).
    if long_pct and short_pct and short_pct > 0:
        ls_ratio = long_pct / short_pct
    else:
        ls_ratio = latest.get("longShortRatio")
    if ls_ratio is None:
        return False

    ts = _ms_to_dt(latest.get("timestamp")) or datetime.now(UTC).replace(tzinfo=None)
    with session_scope() as s:
        s.execute(
            text(
                "INSERT INTO long_short_ratio_snapshots "
                "(pair_id, timestamp, ls_ratio, long_pct, short_pct, source) "
                "VALUES (:pid, :ts, :r, :lp, :sp, :src) "
                "ON CONFLICT (pair_id, timestamp, source) DO NOTHING"
            ),
            {"pid": pair_id, "ts": ts, "r": float(ls_ratio),
             "lp": long_pct, "sp": short_pct, "src": src},
        )
    return True


def collect_once() -> dict[str, int]:
    ex = get_exchange()
    fb = get_fallback_exchange()
    pairs = _active_symbols(settings.exchange)
    counts = {"oi": 0, "fr": 0, "lsr": 0, "oi_fb": 0, "fr_fb": 0, "lsr_fb": 0}
    for pair_id, symbol in pairs:
        if _collect_oi(pair_id, symbol, ex):
            counts["oi"] += 1
        elif _collect_oi(pair_id, symbol, fb):
            counts["oi_fb"] += 1
        if _collect_funding(pair_id, symbol, ex):
            counts["fr"] += 1
        elif _collect_funding(pair_id, symbol, fb):
            counts["fr_fb"] += 1
        if _collect_lsr(pair_id, symbol, ex):
            counts["lsr"] += 1
        elif _collect_lsr(pair_id, symbol, fb, source=FALLBACK_SOURCE):
            counts["lsr_fb"] += 1
        time.sleep(0.5)
    log.info(
        "tick: pairs=%d oi=%d(+%d fb) fr=%d(+%d fb) lsr=%d(+%d fb)",
        len(pairs),
        counts["oi"], counts["oi_fb"],
        counts["fr"], counts["fr_fb"],
        counts["lsr"], counts["lsr_fb"],
    )
    return counts


def run_collector(once: bool = False, poll_seconds: int = POLL_SECONDS) -> None:
    while True:
        try:
            collect_once()
        except Exception:
            log.exception("collect_once failed")
        if once:
            return
        time.sleep(poll_seconds)


if __name__ == "__main__":
    import sys
    run_collector(once=("--once" in sys.argv))
