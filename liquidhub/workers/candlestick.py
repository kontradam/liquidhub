"""Candlestick price-action worker — "the trader sitting at the candles".

Pure technical (no ML model): reads classic reversal/continuation candlestick
patterns on the latest candle, confirmed by recent trend context and relative
volume, and votes a direction + confidence by pattern strength.

Patterns: bullish/bearish engulfing, hammer, shooting star, marubozu
(continuation), doji (indecision/weak reversal). One instance per timeframe,
named candlestick_<tf> to match the accumulation_/breakout_ convention.
"""

from __future__ import annotations

from liquidhub.workers.base import MarketContext, Signal, Worker


class CandlestickWorker(Worker):
    kind = "technical"

    _DEFAULT_WEIGHTS = {"1m": 0.5, "5m": 1.0, "15m": 1.5, "4h": 1.0, "1d": 0.8}

    def __init__(self, timeframe: str = "15m"):
        self.timeframe = timeframe
        self.name = f"candlestick_{timeframe}"
        self.default_weight = self._DEFAULT_WEIGHTS.get(timeframe, 1.0)
        self.required_timeframes = (timeframe,)

    def vote(self, ctx: MarketContext) -> Signal:
        df = ctx.candles.get(self.timeframe)
        if df is None or len(df) < 20:
            return Signal(worker=self.name, direction="LONG", confidence=0.0,
                          reason={"skip": "insufficient data"})

        o = df["open"].astype(float); h = df["high"].astype(float)
        l = df["low"].astype(float); c = df["close"].astype(float)
        v = df["volume"].astype(float)

        o1, h1, l1, c1 = o.iloc[-1], h.iloc[-1], l.iloc[-1], c.iloc[-1]
        o0, c0 = o.iloc[-2], c.iloc[-2]

        rng = max(h1 - l1, 1e-9)
        body = abs(c1 - o1)
        upper = h1 - max(c1, o1)
        lower = min(c1, o1) - l1
        body_frac = body / rng
        upper_frac = upper / rng
        lower_frac = lower / rng

        # recent trend context (close vs mean of the prior ~6 candles)
        recent = c.iloc[-7:-1]
        ref = float(recent.mean()) if len(recent) else c0
        trend_up = c0 > ref
        trend_dn = c0 < ref
        vol_ratio = float(v.iloc[-1] / max(v.iloc[-20:].mean(), 1e-9))

        bull = c1 > o1
        prev_bull = c0 > o0
        prev_bear = c0 < o0

        # --- pattern recognition (most specific first) ---
        if bull and prev_bear and c1 >= o0 and o1 <= c0 and body_frac > 0.5:
            direction, pat, conf = "LONG", "bullish_engulfing", 0.60 + 0.30 * body_frac
        elif (not bull) and prev_bull and o1 >= c0 and c1 <= o0 and body_frac > 0.5:
            direction, pat, conf = "SHORT", "bearish_engulfing", 0.60 + 0.30 * body_frac
        elif lower_frac > 0.5 and body_frac < 0.35 and upper_frac < 0.2 and trend_dn:
            direction, pat, conf = "LONG", "hammer", 0.55 + 0.40 * lower_frac
        elif upper_frac > 0.5 and body_frac < 0.35 and lower_frac < 0.2 and trend_up:
            direction, pat, conf = "SHORT", "shooting_star", 0.55 + 0.40 * upper_frac
        elif body_frac > 0.8:
            direction, pat, conf = ("LONG" if bull else "SHORT"), "marubozu", 0.50 + 0.30 * body_frac
        elif body_frac < 0.1:
            # doji = indecision; lean as a weak reversal of the prevailing move
            direction, pat, conf = ("SHORT" if trend_up else "LONG"), "doji", 0.20
        else:
            direction, pat, conf = ("LONG" if bull else "SHORT"), "weak_body", 0.20 + 0.30 * body_frac

        # volume confirmation
        if vol_ratio > 1.3:
            conf += 0.10
        conf = max(0.0, min(1.0, conf))

        return Signal(worker=self.name, direction=direction, confidence=float(conf),
                      reason={"pattern": pat, "body_frac": round(body_frac, 3),
                              "upper_frac": round(upper_frac, 3), "lower_frac": round(lower_frac, 3),
                              "vol_ratio": round(vol_ratio, 2)})
