"""Fibonacci retracement geometry + golden-pocket event detection.

A retracement zone is defined as a band [lo_ratio, hi_ratio] of a leg's range,
measured back from the terminal pivot B. For a long leg (low A -> high B) the
band sits *below* B; an event fires when price pulls back DOWN into it. For a
short leg the band sits above B and price must pull back UP into it.

Lookahead safety: a leg is only scanned from `known_index` (the bar B was
confirmed), and the event's reference price is the *close* of the entry bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import EVENT_COLUMNS, Leg, legs_from_pivots, to_json

# Standard retracement / extension ratios (used for level rendering & metadata).
RETRACEMENT_RATIOS = [0.236, 0.382, 0.5, 0.618, 0.707, 0.786, 0.886, 1.0]
EXTENSION_RATIOS = [1.272, 1.414, 1.618, 2.0, 2.618]
GOLDEN_POCKET = (0.618, 0.786)


def retracement_levels(a_price: float, b_price: float,
                       ratios=RETRACEMENT_RATIOS) -> dict[float, float]:
    """Map ratio -> price. Works for both directions (b - r*range)."""
    rng = b_price - a_price
    return {r: b_price - r * rng for r in ratios}


def _safe(symbol: str) -> str:
    return symbol.replace("/", "_")


def detect_zone_events(df: pd.DataFrame, symbol: str, timeframe: str,
                       legs: list[Leg], lo_ratio: float, hi_ratio: float,
                       subtype: str, control_kind: str | None = None,
                       max_scan: int = 150, atr_col: str = "atr_14") -> pd.DataFrame:
    """Fire one event per leg at the first bar whose range enters [lo, hi] band.

    `lo_ratio`/`hi_ratio` are retracement fractions; the lower-priced edge of the
    band for a long leg is the *higher* ratio (deeper retracement)."""
    rows: list[dict] = []
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atr = df[atr_col].values if atr_col in df.columns else np.full(len(df), np.nan)
    n = len(df)

    for leg in legs:
        rng = leg.b_price - leg.a_price  # signed: +ve for long leg, -ve for short
        # band edges in price space
        edge1 = leg.b_price - lo_ratio * rng
        edge2 = leg.b_price - hi_ratio * rng
        zone_low, zone_high = min(edge1, edge2), max(edge1, edge2)
        level_price = leg.b_price - ((lo_ratio + hi_ratio) / 2.0) * rng

        start = leg.known_index + 1
        stop = min(n, start + max_scan)
        for i in range(start, stop):
            if leg.direction == "long":
                entered = lows[i] <= zone_high          # pulled down into band
                invalidated = lows[i] < leg.a_price     # broke the swing origin
            else:
                entered = highs[i] >= zone_low          # pulled up into band
                invalidated = highs[i] > leg.a_price
            if entered:
                px = float(closes[i])
                a14 = atr[i] if not np.isnan(atr[i]) else np.nan
                rows.append({
                    "event_id": f"{_safe(symbol)}|{timeframe}|{subtype}|{leg.b_id}",
                    "symbol": symbol, "timeframe": timeframe,
                    "timestamp": df["timestamp"].iloc[i],
                    "geometry_type": "fibonacci", "geometry_subtype": subtype,
                    "direction": "long" if leg.direction == "long" else "short",
                    "level_price": float(level_price), "current_price": px,
                    "distance_pct": float((px - level_price) / level_price),
                    "distance_atr": float((px - level_price) / a14) if a14 and not np.isnan(a14) else None,
                    "anchor_data": to_json({"a": leg.a_id, "b": leg.b_id,
                                            "a_price": leg.a_price, "b_price": leg.b_price}),
                    "confluence_score": 1.0,
                    "metadata": to_json({"lo_ratio": lo_ratio, "hi_ratio": hi_ratio,
                                         "zone_low": zone_low, "zone_high": zone_high}),
                    "created_at": pd.Timestamp.now("UTC"),
                    "is_control": control_kind is not None,
                    "control_kind": control_kind,
                    "bar_index": i,
                    "entry_price": px,
                    "target_price": leg.b_price,          # take profit = swing terminal
                    "stop_price": leg.a_price,            # stop = swing origin
                })
                break
            if invalidated:
                break
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def golden_pocket_events(df: pd.DataFrame, symbol: str, timeframe: str,
                         pivots: pd.DataFrame, **kw) -> pd.DataFrame:
    legs = legs_from_pivots(pivots)
    return detect_zone_events(df, symbol, timeframe, legs,
                              GOLDEN_POCKET[0], GOLDEN_POCKET[1],
                              subtype="golden_pocket", control_kind=None, **kw)
