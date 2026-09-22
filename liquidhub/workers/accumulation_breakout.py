from liquidhub.features.ab_features import compute_ab_features, AB_FEATURES
from liquidhub.logging import get_logger
from liquidhub.ml.inference import predict_proba
from liquidhub.workers.base import MarketContext, Signal, Worker

log = get_logger("worker.accumulation_breakout")


class AccumulationWorker(Worker):
    kind = "forecast"

    _DEFAULT_WEIGHTS = {"1m": 0.5, "5m": 1.0, "15m": 1.5, "4h": 1.0, "1d": 0.8}

    def __init__(self, timeframe: str):
        self.timeframe = timeframe
        self.name = f"accumulation_{timeframe}"
        self.default_weight = self._DEFAULT_WEIGHTS.get(timeframe, 1.0)
        self.required_timeframes = (timeframe,)

    def vote(self, ctx: MarketContext) -> Signal:
        df = ctx.candles.get(self.timeframe)
        if df is None or len(df) < 50:
            return Signal(worker=self.name, direction="SHORT", confidence=0.0,
                          reason={"skip": "insufficient data"})

        feats = compute_ab_features(df)
        if feats.isna().any():
            return Signal(worker=self.name, direction="SHORT", confidence=0.0,
                          reason={"skip": "incomplete features"})

        try:
            probs = predict_proba(feats, self.timeframe, kind="accumulation")
        except FileNotFoundError:
            return Signal(worker=self.name, direction="SHORT", confidence=0.0,
                          reason={"skip": "model missing"})
        except Exception as e:
            log.exception("accumulation %s inference failed: %s", self.timeframe, e)
            return Signal(worker=self.name, direction="SHORT", confidence=0.0,
                          reason={"error": str(e)})

        prob = probs["positive"]
        direction = "LONG" if prob >= 0.5 else "SHORT"
        confidence = prob if direction == "LONG" else 1 - prob

        return Signal(worker=self.name, direction=direction, confidence=confidence,
                      reason={"accumulation_prob": prob})


class BreakoutWorker(Worker):
    kind = "forecast"

    _DEFAULT_WEIGHTS = {"1m": 0.5, "5m": 1.0, "15m": 1.5, "4h": 1.0, "1d": 0.8}

    def __init__(self, timeframe: str):
        self.timeframe = timeframe
        self.name = f"breakout_{timeframe}"
        self.default_weight = self._DEFAULT_WEIGHTS.get(timeframe, 1.0)
        self.required_timeframes = (timeframe,)

    def vote(self, ctx: MarketContext) -> Signal:
        df = ctx.candles.get(self.timeframe)
        if df is None or len(df) < 50:
            return Signal(worker=self.name, direction="SHORT", confidence=0.0,
                          reason={"skip": "insufficient data"})

        feats = compute_ab_features(df)
        if feats.isna().any():
            return Signal(worker=self.name, direction="SHORT", confidence=0.0,
                          reason={"skip": "incomplete features"})

        try:
            probs = predict_proba(feats, self.timeframe, kind="breakout")
        except FileNotFoundError:
            return Signal(worker=self.name, direction="SHORT", confidence=0.0,
                          reason={"skip": "model missing"})
        except Exception as e:
            log.exception("breakout %s inference failed: %s", self.timeframe, e)
            return Signal(worker=self.name, direction="SHORT", confidence=0.0,
                          reason={"error": str(e)})

        prob = probs["positive"]
        direction = "LONG" if prob >= 0.5 else "SHORT"
        confidence = prob if direction == "LONG" else 1 - prob

        return Signal(worker=self.name, direction=direction, confidence=confidence,
                      reason={"breakout_prob": prob})
