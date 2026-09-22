import numpy as np
import pandas as pd

from liquidhub.features.engine import V9_FEATURES, compute_features, latest_features


def _synthetic_ohlcv(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 0.5, size=n))
    high = close + rng.uniform(0, 0.5, size=n)
    low = close - rng.uniform(0, 0.5, size=n)
    open_ = close + rng.normal(0, 0.1, size=n)
    volume = rng.uniform(100, 1000, size=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_compute_features_emits_v9_columns_in_order():
    feats = compute_features(_synthetic_ohlcv())
    assert tuple(feats.columns) == V9_FEATURES
    assert len(feats) == 200


def test_latest_features_returns_series():
    s = latest_features(_synthetic_ohlcv())
    assert list(s.index) == list(V9_FEATURES)
    # the last row should have no NaNs for features that don't need long windows
    assert not np.isnan(s["pct_change_5"])


def test_empty_input_returns_empty_frame():
    feats = compute_features(pd.DataFrame(columns=["open", "high", "low", "close", "volume"]))
    assert feats.empty
    assert list(feats.columns) == list(V9_FEATURES)
