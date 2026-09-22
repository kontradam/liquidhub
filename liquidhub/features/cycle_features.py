import numpy as np
import pandas as pd

from liquidhub.logging import get_logger

log = get_logger("features.cycle")

CYCLE_FEATURES: tuple[str, ...] = (
    "pct_change_1", "pct_change_5", "roc_10", "roc_20", "roc_50",
    "close_vs_sma20", "close_vs_sma50", "close_vs_sma200",
    "sma20_vs_sma50", "sma50_vs_sma200",
    "close_vs_ema12", "close_vs_ema26", "ema12_vs_ema26",
    "rsi_14", "rsi_21",
    "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_position",
    "atr_14_pct",
    "stoch_k", "stoch_d",
    "volume_ratio", "volatility_20",
    "candle_body", "candle_wick_top", "candle_wick_bot", "high_low_range",
    "higher_high", "lower_low",
    "days_since_cycle_start_norm", "pct_return_from_cycle_start",
    "distance_from_cycle_ath", "cycle_completion_pct",
)


def _rsi(s: pd.Series, period: int) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_cycle_features(
    df: pd.DataFrame,
    cycle_start: str,
    avg_duration_days: float,
) -> pd.Series:
    if df is None or len(df) < 200:
        return pd.Series(dtype=float, index=list(CYCLE_FEATURES))

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]

    out = pd.Series(index=list(CYCLE_FEATURES), dtype=float)

    out["pct_change_1"] = float(close.pct_change(1).iloc[-1])
    out["pct_change_5"] = float(close.pct_change(5).iloc[-1]) if len(close) >= 6 else 0.0
    out["roc_10"] = float(close.pct_change(10).iloc[-1]) if len(close) >= 11 else 0.0
    out["roc_20"] = float(close.pct_change(20).iloc[-1]) if len(close) >= 21 else 0.0
    out["roc_50"] = float(close.pct_change(50).iloc[-1]) if len(close) >= 51 else 0.0

    sma20 = close.rolling(20, min_periods=20).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    s20 = float(sma20.iloc[-1]) if not sma20.iloc[-1] != sma20.iloc[-1] else close.iloc[-1]
    s50 = float(sma50.iloc[-1]) if not sma50.iloc[-1] != sma50.iloc[-1] else close.iloc[-1]
    s200 = float(sma200.iloc[-1]) if not sma200.iloc[-1] != sma200.iloc[-1] else close.iloc[-1]
    e12 = float(ema12.iloc[-1])
    e26 = float(ema26.iloc[-1])
    c = float(close.iloc[-1])

    out["close_vs_sma20"] = (c - s20) / (s20 + 1e-10)
    out["close_vs_sma50"] = (c - s50) / (s50 + 1e-10)
    out["close_vs_sma200"] = (c - s200) / (s200 + 1e-10)
    out["sma20_vs_sma50"] = (s20 - s50) / (s50 + 1e-10)
    out["sma50_vs_sma200"] = (s50 - s200) / (s200 + 1e-10)
    out["close_vs_ema12"] = (c - e12) / (e12 + 1e-10)
    out["close_vs_ema26"] = (c - e26) / (e26 + 1e-10)
    out["ema12_vs_ema26"] = (e12 - e26) / (e26 + 1e-10)

    out["rsi_14"] = float(_rsi(close, 14).iloc[-1])
    out["rsi_21"] = float(_rsi(close, 21).iloc[-1]) if len(close) >= 22 else 50.0

    macd_line = ema12 - ema26
    macd_sig = macd_line.ewm(span=9, adjust=False).mean()
    macd_h = macd_line - macd_sig
    out["macd"] = float(macd_line.iloc[-1]) / (c + 1e-10)
    out["macd_signal"] = float(macd_sig.iloc[-1]) / (c + 1e-10)
    out["macd_hist"] = float(macd_h.iloc[-1]) / (c + 1e-10)

    bb_std = close.rolling(20, min_periods=20).std()
    bb_upper = sma20 + 2 * bb_std
    bb_lower = sma20 - 2 * bb_std
    bb_range = float(bb_upper.iloc[-1] - bb_lower.iloc[-1])
    out["bb_width"] = bb_range / (s20 + 1e-10)
    out["bb_position"] = (c - float(bb_lower.iloc[-1])) / (bb_range + 1e-10)

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean()
    out["atr_14_pct"] = float(atr14.iloc[-1]) / (c + 1e-10) if not atr14.iloc[-1] != atr14.iloc[-1] else 0.0

    ll14 = low.rolling(14, min_periods=14).min()
    hh14 = high.rolling(14, min_periods=14).max()
    stoch_range = hh14 - ll14
    stoch_k_fast = 100.0 * (close - ll14) / (stoch_range + 1e-10)
    stoch_k = stoch_k_fast.rolling(3, min_periods=3).mean()
    stoch_d = stoch_k.rolling(3, min_periods=3).mean()
    out["stoch_k"] = float(stoch_k.iloc[-1]) if not stoch_k.iloc[-1] != stoch_k.iloc[-1] else 50.0
    out["stoch_d"] = float(stoch_d.iloc[-1]) if not stoch_d.iloc[-1] != stoch_d.iloc[-1] else 50.0

    vol_sma20 = volume.rolling(20, min_periods=20).mean()
    out["volume_ratio"] = float(volume.iloc[-1]) / (float(vol_sma20.iloc[-1]) + 1e-10)
    out["volatility_20"] = float(close.pct_change().rolling(20, min_periods=20).std().iloc[-1]) if len(close) >= 21 else 0.0

    hl_range = float(high.iloc[-1] - low.iloc[-1]) or 1e-10
    body_high = max(float(open_.iloc[-1]), c)
    body_low = min(float(open_.iloc[-1]), c)
    o_val = float(open_.iloc[-1])
    h_val = float(high.iloc[-1])
    l_val = float(low.iloc[-1])

    out["candle_body"] = (c - o_val) / (c + 1e-10)
    out["candle_wick_top"] = (h_val - body_high) / hl_range
    out["candle_wick_bot"] = (body_low - l_val) / hl_range
    out["high_low_range"] = (h_val - l_val) / (c + 1e-10)
    out["higher_high"] = 1.0 if h_val > float(high.iloc[-2]) else 0.0
    out["lower_low"] = 1.0 if l_val < float(low.iloc[-2]) else 0.0

    # Cycle-aware features
    cycle_start_ts = pd.Timestamp(cycle_start, tz="UTC") if df.index.tz else pd.Timestamp(cycle_start)
    last_ts = df.index[-1]
    days_since = (last_ts - cycle_start_ts).total_seconds() / 86400.0

    out["days_since_cycle_start_norm"] = min(days_since / avg_duration_days, 1.0) if days_since > 0 else 0.0

    cycle_start_price = float(close.iloc[0]) if len(close) > 0 else c
    out["pct_return_from_cycle_start"] = (c - cycle_start_price) / (cycle_start_price + 1e-10)

    running_ath = float(high.max())
    out["distance_from_cycle_ath"] = (c - running_ath) / (running_ath + 1e-10)

    out["cycle_completion_pct"] = min(days_since / avg_duration_days * 100.0, 100.0) if days_since > 0 else 0.0

    return out.reindex(list(CYCLE_FEATURES))
