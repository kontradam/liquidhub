import numpy as np
import pandas as pd

from liquidhub.logging import get_logger

log = get_logger("features.ab")

AB_FEATURES: tuple[str, ...] = (
    "bb_width", "bb_width_pct", "bb_width_slope", "bb_position",
    "atr_ratio", "ema_distance", "rsi", "rsi_range10",
    "macd_hist_std", "price_range20", "vol_ratio", "vol_trend",
    "vol_spike", "pct_change5", "pct_change10",
    "candles_since_high", "candles_since_low",
    "dist_to_resistance", "dist_to_support", "body_ratio_avg10",
)


def compute_ab_features(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) < 50:
        return pd.Series(dtype=float, index=list(AB_FEATURES))

    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    v = df["volume"]

    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    std20 = c.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_width = (bb_upper - bb_lower) / sma20.replace(0, np.nan)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()

    hl = h - l
    hc = (h - c.shift(1)).abs()
    lc = (l - c.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr14 = tr.ewm(com=13, min_periods=14).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi14 = 100 - (100 / (1 + rs))

    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    vol_sma20 = v.rolling(20).mean()

    rolling_high20 = h.rolling(20).max()
    rolling_low20 = l.rolling(20).min()

    body_ratio = ((c - o) / o.replace(0, np.nan)).abs()

    out = pd.Series(index=list(AB_FEATURES), dtype=float)

    out["bb_width"] = float(bb_width.iloc[-1]) if not bb_width.empty else np.nan
    out["bb_width_pct"] = float(bb_width.iloc[-1]) / 20 if not bb_width.empty else np.nan

    if len(bb_width) >= 5:
        recent = bb_width.iloc[-5:]
        out["bb_width_slope"] = float((recent.iloc[-1] - recent.iloc[0]) / len(recent))
    else:
        out["bb_width_slope"] = 0.0

    band_range = bb_upper.iloc[-1] - bb_lower.iloc[-1]
    out["bb_position"] = float((c.iloc[-1] - bb_lower.iloc[-1]) / (band_range + 1e-10))

    out["atr_ratio"] = float(atr14.iloc[-1] / (c.iloc[-1] + 1e-10))
    out["ema_distance"] = float((c.iloc[-1] - ema12.iloc[-1]) / (ema12.iloc[-1] + 1e-10))
    out["rsi"] = float(rsi14.iloc[-1])

    rsi_window = rsi14.iloc[-10:] if len(rsi14) >= 10 else rsi14
    out["rsi_range10"] = float(rsi_window.max() - rsi_window.min())

    out["macd_hist_std"] = float(macd_hist.rolling(20).std().iloc[-1]) if len(macd_hist) >= 20 else 0.0
    out["price_range20"] = float((rolling_high20.iloc[-1] - rolling_low20.iloc[-1]) / (rolling_low20.iloc[-1] + 1e-10))
    out["vol_ratio"] = float(v.iloc[-1] / (vol_sma20.iloc[-1] + 1e-10))
    out["vol_trend"] = float(vol_sma20.iloc[-1] / (vol_sma20.shift(20).iloc[-1] + 1e-10) - 1) if len(vol_sma20) >= 20 else 0.0

    vol_threshold = vol_sma20.iloc[-1] * 2.0 if not vol_sma20.empty else 0
    out["vol_spike"] = 1.0 if v.iloc[-1] > vol_threshold else 0.0

    out["pct_change5"] = float(c.pct_change(5).iloc[-1]) if len(c) >= 6 else 0.0
    out["pct_change10"] = float(c.pct_change(10).iloc[-1]) if len(c) >= 11 else 0.0

    recent_high_idx = h.iloc[-20:].idxmax() if len(h) >= 20 else h.idxmax()
    recent_low_idx = l.iloc[-20:].idxmin() if len(l) >= 20 else l.idxmin()
    out["candles_since_high"] = float(len(h.loc[recent_high_idx:].iloc[1:])) if recent_high_idx in h.index else 20.0
    out["candles_since_low"] = float(len(l.loc[recent_low_idx:].iloc[1:])) if recent_low_idx in l.index else 20.0

    out["dist_to_resistance"] = float((rolling_high20.iloc[-1] - c.iloc[-1]) / (c.iloc[-1] + 1e-10))
    out["dist_to_support"] = float((c.iloc[-1] - rolling_low20.iloc[-1]) / (c.iloc[-1] + 1e-10))

    out["body_ratio_avg10"] = float(body_ratio.rolling(10).mean().iloc[-1]) if len(body_ratio) >= 10 else 0.0

    return out.reindex(list(AB_FEATURES))
