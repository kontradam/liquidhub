from liquidhub.workers.base import MarketContext, Signal, Worker


class VolumeWorker(Worker):
    """Confirms direction when volume spikes above its rolling average."""

    name = "volume_worker"
    kind = "technical"
    default_weight = 3.0
    required_timeframes = ("15m",)

    def __init__(self, timeframe: str = "15m", period: int = 20, spike_mult: float = 1.5):
        self.timeframe = timeframe
        self.period = period
        self.spike_mult = spike_mult

    def vote(self, ctx: MarketContext) -> Signal:
        df = ctx.candles.get(self.timeframe)
        if df is None or len(df) < self.period + 1:
            return Signal(worker=self.name, direction="LONG", confidence=0.0,
                          reason={"skip": "insufficient data"})

        v_now = float(df["volume"].iloc[-1])
        v_avg = float(df["volume"].rolling(self.period).mean().iloc[-1]) or 1e-9
        ratio = v_now / v_avg

        close_now = float(df["close"].iloc[-1])
        close_prev = float(df["close"].iloc[-2])
        price_up = close_now > close_prev

        if ratio >= self.spike_mult:
            direction = "LONG" if price_up else "SHORT"
            conf = min(1.0, 0.5 + (ratio - self.spike_mult) * 0.2)
        else:
            direction = "LONG" if price_up else "SHORT"
            conf = 0.2 + min(0.3, ratio / self.spike_mult * 0.3)

        return Signal(
            worker=self.name, direction=direction, confidence=float(conf),
            reason={"ratio": ratio, "vol": v_now, "avg": v_avg},
        )
