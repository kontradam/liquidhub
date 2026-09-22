from liquidhub.features.engine import XGB_FEATURES
from liquidhub.logging import get_logger
from liquidhub.ml.inference import predict_binary_direction
from liquidhub.workers.base import MarketContext, Signal, Worker

log = get_logger("worker.xgb")


class XGBoostWorker(Worker):
    """One XGBoost worker per timeframe. Binary LONG/SHORT output."""

    _DEFAULT_WEIGHTS = {"1m": 2.0, "5m": 0.9, "15m": 2.9, "1h": 3.0, "4h": 3.0, "1d": 2.0}
    kind = "ml"

    def __init__(self, timeframe: str, min_confidence: float = 0.35):
        self.timeframe = timeframe
        self.name = f"xgboost_{timeframe}_v1"
        self.default_weight = self._DEFAULT_WEIGHTS.get(timeframe, 1.0)
        self.required_timeframes = (timeframe,)
        self.min_confidence = min_confidence

    def vote(self, ctx: MarketContext) -> Signal:
        feats = ctx.features.get(self.timeframe)
        if feats is None or feats.isna().any():
            return Signal(worker=self.name, direction="LONG", confidence=0.0,
                          reason={"skip": "no features"})

        feats = feats.reindex(list(XGB_FEATURES))

        try:
            direction, confidence, probs = predict_binary_direction(
                feats, self.timeframe, min_confidence=self.min_confidence
            )
        except FileNotFoundError as e:
            log.warning("model missing for %s: %s", self.timeframe, e)
            return Signal(worker=self.name, direction="LONG", confidence=0.0,
                          reason={"skip": "model missing"})
        except Exception as e:
            log.exception("xgb %s inference failed: %s", self.timeframe, e)
            return Signal(worker=self.name, direction="LONG", confidence=0.0,
                          reason={"error": str(e)})

        return Signal(worker=self.name, direction=direction, confidence=confidence, reason=probs)
