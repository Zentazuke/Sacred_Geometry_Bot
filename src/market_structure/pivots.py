"""Pivot detection — the foundation of all geometry. Two methods:

  * fractal: high[i] greater than N bars either side. Confirms `right` bars late,
    so `confirmed_index` = i + right. Geometry must not use it before then.
  * zigzag: alternating swing points confirmed once price reverses by
    `zigzag_atr_mult` * ATR from the running extreme. `confirmed_index` = the bar
    that triggered the reversal.

Every pivot carries both its `bar_index` (where the extreme actually sits) and
its `confirmed_index` (the first bar at which the bot could have known about it).
Downstream code keys on `confirmed_index` to stay lookahead-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import atr as atr_series

PIVOT_COLUMNS = [
    "pivot_id", "symbol", "timeframe", "timestamp", "bar_index", "pivot_type",
    "price", "strength", "left_bars", "right_bars", "atr", "confirmed_at",
    "confirmed_index", "method",
]


def _safe(symbol: str) -> str:
    return symbol.replace("/", "_")


def _row(symbol, timeframe, df, bar_index, ptype, price, left, right,
         atr_val, confirmed_index, method) -> dict:
    return {
        "pivot_id": f"{_safe(symbol)}|{timeframe}|{method}|{ptype}|{bar_index}",
        "symbol": symbol,
        "timeframe": timeframe,
        "timestamp": df["timestamp"].iloc[bar_index],
        "bar_index": int(bar_index),
        "pivot_type": ptype,
        "price": float(price),
        "strength": int(right),
        "left_bars": int(left),
        "right_bars": int(right),
        "atr": float(atr_val) if not np.isnan(atr_val) else None,
        "confirmed_at": df["timestamp"].iloc[confirmed_index],
        "confirmed_index": int(confirmed_index),
        "method": method,
    }


def fractal_pivots(df: pd.DataFrame, symbol: str, timeframe: str,
                   left: int = 5, right: int = 5, atr_period: int = 14) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    n = len(df)
    a = atr_series(df, atr_period)
    highs, lows = df["high"].values, df["low"].values
    out: list[dict] = []
    for i in range(left, n - right):
        win_h = highs[i - left:i + right + 1]
        win_l = lows[i - left:i + right + 1]
        if highs[i] == win_h.max() and (win_h == highs[i]).sum() == 1:
            out.append(_row(symbol, timeframe, df, i, "high", highs[i], left, right,
                            a.iloc[i], i + right, "fractal"))
        if lows[i] == win_l.min() and (win_l == lows[i]).sum() == 1:
            out.append(_row(symbol, timeframe, df, i, "low", lows[i], left, right,
                            a.iloc[i], i + right, "fractal"))
    res = pd.DataFrame(out, columns=PIVOT_COLUMNS)
    return res.sort_values("confirmed_index").reset_index(drop=True)


def zigzag_pivots(df: pd.DataFrame, symbol: str, timeframe: str,
                  atr_mult: float = 2.0, atr_period: int = 14) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    n = len(df)
    a = atr_series(df, atr_period).bfill()
    highs, lows = df["high"].values, df["low"].values

    out: list[dict] = []
    trend: str | None = None          # 'up' => seeking a high; 'down' => seeking a low
    ext_idx = atr_period              # start once ATR is warm
    ext_high = highs[ext_idx]
    ext_low = lows[ext_idx]

    for i in range(atr_period + 1, n):
        thr = atr_mult * a.iloc[ext_idx]
        if trend is None:
            # establish initial direction from the first threshold-sized move
            if highs[i] - ext_low >= thr:
                trend = "up"; ext_high, ext_idx = highs[i], i
            elif ext_high - lows[i] >= thr:
                trend = "down"; ext_low, ext_idx = lows[i], i
            continue

        if trend == "up":
            if highs[i] > ext_high:
                ext_high, ext_idx = highs[i], i
            elif ext_high - lows[i] >= thr:
                out.append(_row(symbol, timeframe, df, ext_idx, "high", ext_high,
                                0, 0, a.iloc[ext_idx], i, "zigzag"))
                trend = "down"; ext_low, ext_idx = lows[i], i
        else:  # trend == 'down'
            if lows[i] < ext_low:
                ext_low, ext_idx = lows[i], i
            elif highs[i] - ext_low >= thr:
                out.append(_row(symbol, timeframe, df, ext_idx, "low", ext_low,
                                0, 0, a.iloc[ext_idx], i, "zigzag"))
                trend = "up"; ext_high, ext_idx = highs[i], i

    res = pd.DataFrame(out, columns=PIVOT_COLUMNS)
    return res.sort_values("confirmed_index").reset_index(drop=True)


def detect_pivots(df: pd.DataFrame, symbol: str, timeframe: str, method: str,
                  cfg: dict) -> pd.DataFrame:
    atr_period = int(cfg.get("atr_period", 14))
    if method == "zigzag":
        return zigzag_pivots(df, symbol, timeframe,
                             atr_mult=float(cfg.get("zigzag_atr_mult", 2.0)),
                             atr_period=atr_period)
    if method == "fractal":
        m = cfg.get("medium", {"left": 5, "right": 5})
        return fractal_pivots(df, symbol, timeframe, left=int(m["left"]),
                              right=int(m["right"]), atr_period=atr_period)
    raise ValueError(f"Unknown pivot method: {method}")
