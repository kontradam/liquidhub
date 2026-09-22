from liquidhub.workers.base import MarketContext, Signal, Worker


class MAWorker(Worker):
    """Classic SMA 20/50 crossover."""

    name = "ma_worker"
    kind = "technical"
    default_weight = 3.0
    required_timeframes = ("15m",)

    def __init__(self, timeframe: str = "15m", fast: int = 20, slow: int = 50):
        self.timeframe = timeframe
        self.fast = fast
        self.slow = slow

    def vote(self, ctx: MarketContext) -> Signal:
        df = ctx.candles.get(self.timeframe)
        if df is None or len(df) < self.slow + 1:
            return Signal(worker=self.name, direction="LONG", confidence=0.0,
                          reason={"skip": "insufficient data"})

        c = df["close"]
        sma_fast = c.rolling(self.fast).mean()
        sma_slow = c.rolling(self.slow).mean()
        price = float(c.iloc[-1])
        f = float(sma_fast.iloc[-1])
        s = float(sma_slow.iloc[-1])

        if price > f > s:
            direction, conf = "LONG", min(1.0, 0.6 + (f - s) / s)
        elif price < f < s:
            direction, conf = "SHORT", min(1.0, 0.6 + (s - f) / s)
        elif price > s:
            direction, conf = "LONG", 0.4
        elif price < s:
            direction, conf = "SHORT", 0.4
        else:
            direction, conf = "LONG", 0.3

        return Signal(
            worker=self.name, direction=direction, confidence=float(abs(conf)),
            reason={"price": price, "sma_fast": f, "sma_slow": s},
        )
