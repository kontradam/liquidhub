import pandas as pd

from liquidhub.workers.base import MarketContext, Signal, Worker


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    return 100 - 100 / (1 + rs)


class RSIWorker(Worker):
    name = "rsi_worker"
    kind = "technical"
    default_weight = 3.0
    required_timeframes = ("15m",)

    def __init__(self, timeframe: str = "15m", oversold: float = 30, overbought: float = 70):
        self.timeframe = timeframe
        self.oversold = oversold
        self.overbought = overbought

    def vote(self, ctx: MarketContext) -> Signal:
        df = ctx.candles.get(self.timeframe)
        if df is None or df.empty or len(df) < 20:
            return Signal(worker=self.name, direction="LONG", confidence=0.0,
                          reason={"skip": "insufficient data"})
        r = float(rsi(df["close"]).iloc[-1])

        if r < self.oversold:
            direction = "LONG"
            conf = min(1.0, (self.oversold - r) / self.oversold + 0.5)
        elif r > self.overbought:
            direction = "SHORT"
            conf = min(1.0, (r - self.overbought) / (100 - self.overbought) + 0.5)
        else:
            direction = "LONG" if r < 50 else "SHORT"
            conf = 0.25 + abs(r - 50) / 100

        return Signal(worker=self.name, direction=direction, confidence=float(conf),
                      reason={"rsi": r})
