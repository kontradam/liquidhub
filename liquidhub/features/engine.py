import numpy as np
import pandas as pd

XGB_FEATURES: tuple[str, ...] = (
    "pct_change_1", "pct_change_5", "pct_change_10", "pct_change_20",
    "close_vs_sma20", "close_vs_sma50", "close_vs_sma200",
    "sma20_vs_sma50", "sma50_vs_sma200",
    "close_vs_ema12", "close_vs_ema26", "ema12_vs_ema26",
    "rsi14",
    "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_position",
    "atr14_pct",
    "volatility_20",
    "volume_ratio",
    "volume_sma20_trend",
    "obv_change",
    "candle_body",
    "candle_wick_top",
    "candle_wick_bot",
    "high_low_range",
    "higher_high",
    "lower_low",
)


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    hl = h - l
    hc = (h - c.shift(1)).abs()
    lc = (l - c.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def _obv(c: pd.Series, v: pd.Series) -> pd.Series:
    direction = np.sign(c.diff()).fillna(0)
    return (direction * v).cumsum()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=list(XGB_FEATURES))

    c = df["close"]
    o = df["open"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    out = pd.DataFrame(index=df.index)

    out["pct_change_1"] = c.pct_change(1)
    out["pct_change_5"] = c.pct_change(5)
    out["pct_change_10"] = c.pct_change(10)
    out["pct_change_20"] = c.pct_change(20)

    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    ema12 = _ema(c, 12)
    ema26 = _ema(c, 26)

    out["close_vs_sma20"] = c / sma20 - 1
    out["close_vs_sma50"] = c / sma50 - 1
    out["close_vs_sma200"] = c / sma200 - 1
    out["sma20_vs_sma50"] = sma20 / sma50 - 1
    out["sma50_vs_sma200"] = sma50 / sma200 - 1
    out["close_vs_ema12"] = c / ema12 - 1
    out["close_vs_ema26"] = c / ema26 - 1
    out["ema12_vs_ema26"] = ema12 / ema26 - 1

    out["rsi14"] = _rsi(c, 14)

    macd_line = ema12 - ema26
    macd_signal = _ema(macd_line, 9)
    out["macd"] = macd_line / c
    out["macd_signal"] = macd_signal / c
    out["macd_hist"] = (macd_line - macd_signal) / c

    bb_std = c.rolling(20).std()
    bb_upper = sma20 + 2 * bb_std
    bb_lower = sma20 - 2 * bb_std
    out["bb_width"] = (bb_upper - bb_lower) / sma20
    out["bb_position"] = (c - bb_lower) / (bb_upper - bb_lower + 1e-10)

    atr14 = _atr(h, l, c, 14)
    out["atr14_pct"] = atr14 / c

    out["volatility_20"] = c.pct_change().rolling(20).std()

    vol_sma20 = v.rolling(20).mean()
    out["volume_ratio"] = v / vol_sma20.replace(0, np.nan)
    out["volume_sma20_trend"] = vol_sma20 / vol_sma20.shift(20) - 1

    obv = _obv(c, v)
    out["obv_change"] = obv.pct_change(10)

    o_safe = o.replace(0, np.nan)
    out["candle_body"] = (c - o) / o_safe
    out["candle_wick_top"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / o_safe
    out["candle_wick_bot"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / o_safe
    out["high_low_range"] = (h - l) / o_safe

    out["higher_high"] = (h > h.shift(1)).astype(float)
    out["lower_low"] = (l < l.shift(1)).astype(float)

    return out[list(XGB_FEATURES)]


def latest_features(df: pd.DataFrame) -> pd.Series:
    feats = compute_features(df)
    return feats.iloc[-1] if not feats.empty else pd.Series(dtype=float, index=list(XGB_FEATURES))
