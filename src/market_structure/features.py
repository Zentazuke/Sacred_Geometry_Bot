"""Per-candle features: ATR, rolling volatility, returns, volume z-score, and
trend/volatility regimes. All causal (no lookahead): every value at row i uses
only rows <= i."""
from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).rolling(period, min_periods=period).mean()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["atr_14"] = atr(out, 14)
    out["atr_50"] = atr(out, 50)

    logret = np.log(out["close"]).diff()
    out["return_1"] = out["close"].pct_change(1)
    out["return_5"] = out["close"].pct_change(5)
    out["return_10"] = out["close"].pct_change(10)
    out["rolling_vol_20"] = logret.rolling(20, min_periods=20).std()
    out["rolling_vol_100"] = logret.rolling(100, min_periods=100).std()

    vmean = out["volume"].rolling(20, min_periods=20).mean()
    vstd = out["volume"].rolling(20, min_periods=20).std()
    out["volume_zscore"] = (out["volume"] - vmean) / vstd

    # trend regime: sign of (fast SMA - slow SMA)
    sma_fast = out["close"].rolling(20, min_periods=20).mean()
    sma_slow = out["close"].rolling(50, min_periods=50).mean()
    out["trend_regime"] = np.sign(sma_fast - sma_slow)  # -1 / 0 / +1

    # volatility regime: current vol vs its own median
    vol_med = out["rolling_vol_100"].expanding(min_periods=100).median()
    out["volatility_regime"] = np.where(out["rolling_vol_20"] > vol_med, "high", "low")

    out["distance_from_recent_high"] = out["close"] / out["high"].rolling(50, min_periods=1).max() - 1
    out["distance_from_recent_low"] = out["close"] / out["low"].rolling(50, min_periods=1).min() - 1
    return out
