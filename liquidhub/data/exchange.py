from functools import cache
from typing import Any

import ccxt

from liquidhub.config import settings
from liquidhub.logging import get_logger

log = get_logger("data.exchange")


@cache
def get_exchange() -> ccxt.Exchange:
    cls = getattr(ccxt, settings.exchange)
    creds: dict[str, Any] = {
        "apiKey": settings.exchange_api_key or None,
        "secret": settings.exchange_api_secret or None,
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    }
    if settings.exchange_api_passphrase:
        creds["password"] = settings.exchange_api_passphrase
    ex = cls({k: v for k, v in creds.items() if v is not None})
    log.info("initialized exchange: %s (swap)", settings.exchange)
    return ex


def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 500) -> list[list[float]]:
    ex = get_exchange()
    return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)


@cache
def get_fallback_exchange() -> ccxt.Exchange:
    """Public Binance USDT-M futures instance, used when the primary exchange
    has no data for a derivatives endpoint (OI/funding/LSR)."""
    ex = ccxt.binanceusdm({"enableRateLimit": True})
    log.info("initialized fallback exchange: binanceusdm")
    return ex
