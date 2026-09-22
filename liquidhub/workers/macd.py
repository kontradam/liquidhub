from liquidhub.workers.base import MarketContext, Signal, Worker


class MACDWorker(Worker):
    name = "macd_worker"
    kind = "technical"
    default_weight = 0.5
    required_timeframes = ("15m",)

    def __init__(self, timeframe: str = "15m", fast: int = 12, slow: int = 26, signal: int = 9):
        self.timeframe = timeframe
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def vote(self, ctx: MarketContext) -> Signal:
        df = ctx.candles.get(self.timeframe)
        if df is None or len(df) < self.slow + self.signal:
            return Signal(worker=self.name, direction="LONG", confidence=0.0,
                          reason={"skip": "insufficient data"})

        c = df["close"]
        ema_fast = c.ewm(span=self.fast, adjust=False).mean()
        ema_slow = c.ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
        hist = macd_line - signal_line

        h_now = float(hist.iloc[-1])
        h_prev = float(hist.iloc[-2])
        slow_now = float(ema_slow.iloc[-1]) or 1e-9

        crossed_up = h_prev <= 0 < h_now
        crossed_down = h_prev >= 0 > h_now
        magnitude = min(1.0, abs(h_now) / abs(slow_now) * 100)

        if crossed_up:
            direction, conf = "LONG", max(0.55, magnitude)
        elif crossed_down:
            direction, conf = "SHORT", max(0.55, magnitude)
        elif h_now > 0:
            direction, conf = "LONG", max(0.3, magnitude * 0.8)
        else:
            direction, conf = "SHORT", max(0.3, magnitude * 0.8)

        return Signal(
            worker=self.name, direction=direction, confidence=float(min(1.0, conf)),
            reason={"hist": h_now, "signal": float(signal_line.iloc[-1])},
        )
