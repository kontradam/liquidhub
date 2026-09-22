from liquidhub.workers.base import MarketContext, Signal, Worker


class MomentumWorker(Worker):
    """Rate-of-change momentum."""

    name = "momentum_worker"
    kind = "technical"
    default_weight = 3.0
    required_timeframes = ("15m",)

    def __init__(self, timeframe: str = "15m", period: int = 10, threshold: float = 0.005):
        self.timeframe = timeframe
        self.period = period
        self.threshold = threshold

    def vote(self, ctx: MarketContext) -> Signal:
        df = ctx.candles.get(self.timeframe)
        if df is None or len(df) < self.period + 1:
            return Signal(worker=self.name, direction="LONG", confidence=0.0,
                          reason={"skip": "insufficient data"})

        c = df["close"]
        roc = float((c.iloc[-1] - c.iloc[-1 - self.period]) / c.iloc[-1 - self.period])

        if roc > self.threshold:
            direction, conf = "LONG", min(1.0, 0.5 + abs(roc) / (self.threshold * 4))
        elif roc < -self.threshold:
            direction, conf = "SHORT", min(1.0, 0.5 + abs(roc) / (self.threshold * 4))
        else:
            direction = "LONG" if roc >= 0 else "SHORT"
            conf = 0.3 + abs(roc) / self.threshold * 0.2

        return Signal(
            worker=self.name, direction=direction, confidence=float(conf),
            reason={"roc": roc, "period": self.period},
        )
