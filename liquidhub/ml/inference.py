from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from liquidhub.logging import get_logger
from liquidhub.ml.models_registry import model_paths

log = get_logger("ml.inference")

CLASS_IDX_3CLASS = {"SHORT": 0, "HOLD": 1, "LONG": 2}
CLASS_IDX_BINARY = {0: 0, 1: 1}


@lru_cache(maxsize=32)
def load(timeframe: str, kind: str = "xgboost") -> tuple[object, object]:
    paths = model_paths(timeframe, kind)
    if not paths.model.exists() or not paths.imputer.exists():
        raise FileNotFoundError(f"missing model for {kind} {timeframe}: {paths.model}")
    model = joblib.load(paths.model)
    imputer = joblib.load(paths.imputer)
    log.info("loaded %s %s model from %s", kind, timeframe, paths.model)
    return model, imputer


def _is_3class(model) -> bool:
    n = getattr(model, "n_classes_", None)
    return n is not None and n == 3


def predict_proba(features: pd.Series, timeframe: str, kind: str = "xgboost") -> dict[str, float]:
    model, imputer = load(timeframe, kind)
    X = imputer.transform(features.values.reshape(1, -1))
    probs = model.predict_proba(X)[0]

    if kind in ("accumulation", "breakout"):
        return {"positive": float(probs[1]), "negative": float(probs[0])}

    if _is_3class(model):
        p_short = float(probs[CLASS_IDX_3CLASS["SHORT"]])
        p_hold = float(probs[CLASS_IDX_3CLASS["HOLD"]])
        p_long = float(probs[CLASS_IDX_3CLASS["LONG"]])
        total = p_short + p_long
        if total < 1e-9:
            return {"SHORT": 0.5, "LONG": 0.5}
        return {"SHORT": p_short / (total + p_hold * 0.5), "LONG": p_long / (total + p_hold * 0.5)}

    return {"SHORT": float(probs[0]), "LONG": float(probs[1])}


def predict_binary_direction(features: pd.Series, timeframe: str, kind: str = "xgboost",
                             min_confidence: float = 0.35) -> tuple[str, float, dict[str, float]]:
    probs = predict_proba(features, timeframe, kind)
    direction = "LONG" if probs["LONG"] >= probs["SHORT"] else "SHORT"
    confidence = probs[direction]
    if confidence < min_confidence:
        direction = "LONG" if probs["LONG"] >= probs["SHORT"] else "SHORT"
    return direction, confidence, probs


def sanity_check_shape(features: pd.Series, timeframe: str) -> bool:
    _model, imputer = load(timeframe)
    expected = getattr(imputer, "n_features_in_", None)
    if expected is not None and len(features) != expected:
        log.error("feature mismatch %s: got %d, expected %d", timeframe, len(features), expected)
        return False
    return True
