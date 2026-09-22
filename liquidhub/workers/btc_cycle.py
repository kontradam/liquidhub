import joblib
import numpy as np

from liquidhub.features.cycle_features import compute_cycle_features
from liquidhub.logging import get_logger
from liquidhub.ml.models_registry import CYCLE_CONFIGS
from liquidhub.workers.base import MarketContext, Signal, Worker

log = get_logger("worker.btc_cycle")

MODELS_DIR = "/home/adas/MotherlodeApp/LiquidhubIndicatorSystem/models/btc_cycle"


class BTCCycleWorker(Worker):
    kind = "forecast"
    name = "btc_cycle4"
    default_weight = 0.5

    def __init__(self):
        self.required_timeframes = ("15m",)

    def vote(self, ctx: MarketContext) -> Signal:
        if "BTC" not in ctx.symbol.upper():
            return Signal(worker=self.name, direction="SHORT", confidence=0.0,
                          reason={"skip": "BTC only"})

        cfg = CYCLE_CONFIGS.get("cycle4_15m")
        if cfg is None:
            return Signal(worker=self.name, direction="SHORT", confidence=0.0,
                          reason={"skip": "no cycle config"})

        tf = cfg["timeframe"]
        df = ctx.candles.get(tf)
        if df is None or len(df) < 200:
            return Signal(worker=self.name, direction="SHORT", confidence=0.0,
                          reason={"skip": "insufficient data"})

        feats = compute_cycle_features(df, cfg["cycle_start"], cfg["avg_duration_days"])

        model_path = f"{MODELS_DIR}/xgboost_cycle4_15m_v1.pkl"
        imputer_path = f"{MODELS_DIR}/xgboost_cycle4_15m_v1_imputer.pkl"

        try:
            model = joblib.load(model_path)
            imputer = joblib.load(imputer_path)
        except FileNotFoundError as e:
            return Signal(worker=self.name, direction="SHORT", confidence=0.0,
                          reason={"skip": f"model missing: {e}"})

        X = imputer.transform(feats.values.reshape(1, -1))
        probs = model.predict_proba(X)[0]

        if len(probs) == 3:
            p_short, p_hold, p_long = float(probs[0]), float(probs[1]), float(probs[2])
        else:
            p_short, p_hold, p_long = float(probs[0]), 0.0, float(probs[1])

        direction = "LONG" if p_long >= p_short else "SHORT"
        confidence = max(p_long, p_short)

        return Signal(
            worker=self.name,
            direction=direction,
            confidence=confidence,
            reason={
                "cycle": "cycle4",
                "p_long": round(p_long, 3),
                "p_short": round(p_short, 3),
                "p_hold": round(p_hold, 3),
                "completion_pct": feats.get("cycle_completion_pct", 0),
            },
        )
