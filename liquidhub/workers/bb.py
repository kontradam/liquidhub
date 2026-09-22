from liquidhub.workers.base import MarketContext, Signal, Worker


class BollingerBandsWorker(Worker):
    """Mean-reversion on Bollinger Band touches."""

    name = "bb_worker"
    kind = "technical"
    default_weight = 3.0
    required_timeframes = ("15m",)

    def __init__(self, timeframe: str = "15m", period: int = 20, num_std: float = 2.0):
        self.timeframe = timeframe
        self.period = period
        self.num_std = num_std

    def vote(self, ctx: MarketContext) -> Signal:
        df = ctx.candles.get(self.timeframe)
        if df is None or len(df) < self.period + 1:
            return Signal(worker=self.name, direction="LONG", confidence=0.0,
                          reason={"skip": "insufficient data"})

        c = df["close"]
        mid = c.rolling(self.period).mean()
        std = c.rolling(self.period).std()
        upper = mid + self.num_std * std
        lower = mid - self.num_std * std

        price = float(c.iloc[-1])
        u = float(upper.iloc[-1])
        l = float(lower.iloc[-1])
        m = float(mid.iloc[-1])
        band = u - l or 1e-9
        pos = (price - l) / band  # 0..1 normalized position inside the band

        if price <= l:
            direction, conf = "LONG", min(1.0, 0.7 + (l - price) / band)
        elif price >= u:
            direction, conf = "SHORT", min(1.0, 0.7 + (price - u) / band)
        else:
            direction = "LONG" if pos < 0.5 else "SHORT"
            conf = 0.3 + abs(pos - 0.5)

        return Signal(
            worker=self.name, direction=direction, confidence=float(max(0.0, conf)),
            reason={"price": price, "upper": u, "lower": l, "mid": m, "pct_b": pos},
        )
