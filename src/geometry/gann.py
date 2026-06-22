"""Gann angles, ATR-normalised (the only honest way — raw Gann depends on chart
scaling, which is meaningless). An angle is anchored at a confirmed pivot and
rises/falls at `slope` ATR-units per bar:

    level(t) = P0 + s * slope * (t - t0) * ATR0

where s = +1 for a support fan from a swing low (→ long on a touch) and -1 for a
resistance fan from a swing high (→ short). An event fires at the first bar whose
close comes within `tol_atr` ATRs of the line.

The control the plan demands is *random slopes*: if randomly-angled lines get
touched and bounce just as often, Gann is dead for that market. `random_slopes=
True` swaps each slope for a uniform random one, keeping anchors and counts.

Stops/targets are ATR-based (angles give no natural level), so Gann trades are
directly comparable through the same labeller and backtest engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import EVENT_COLUMNS, to_json

# Classic Gann fan, as ATR-units-per-bar slopes.
DEFAULT_SLOPES = {
    "1x8": 0.125, "1x4": 0.25, "1x3": 1 / 3, "1x2": 0.5, "1x1": 1.0,
    "2x1": 2.0, "3x1": 3.0, "4x1": 4.0,
}
STOP_ATR = 1.0
TARGET_ATR = 2.0


def _safe(symbol: str) -> str:
    return symbol.replace("/", "_")


def gann_events(df: pd.DataFrame, symbol: str, timeframe: str, pivots: pd.DataFrame,
                slopes: dict | None = None, tol_atr: float = 0.25,
                max_project: int = 200, atr_col: str = "atr_14",
                random_slopes: bool = False, seed: int = 0,
                subtype: str = "gann", control_kind: str | None = None) -> pd.DataFrame:
    slopes = slopes or DEFAULT_SLOPES
    close = df["close"].values
    atr = df[atr_col].values
    ts = df["timestamp"].values
    n = len(df)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    piv = pivots.sort_values("confirmed_index").reset_index(drop=True)
    for _, p in piv.iterrows():
        t0 = int(p["bar_index"])
        known = int(p["confirmed_index"])
        p0 = float(p["price"])
        atr0 = float(p["atr"]) if p["atr"] and not np.isnan(p["atr"]) else np.nan
        if np.isnan(atr0) or atr0 <= 0:
            continue
        s = 1.0 if p["pivot_type"] == "low" else -1.0
        direction = "long" if p["pivot_type"] == "low" else "short"

        for label, base_slope in slopes.items():
            slope = float(rng.uniform(0.1, 4.0)) if random_slopes else base_slope
            start = known + 1
            end = min(n, t0 + max_project + 1)
            if start >= end:
                continue
            t = np.arange(start, end)
            bars = t - t0
            level = p0 + s * slope * bars * atr0
            atr_t = atr[t]
            near = np.abs(close[t] - level) <= tol_atr * atr_t
            valid = near & ~np.isnan(atr_t)
            if not valid.any():
                continue
            hit = t[np.argmax(valid)]          # first touch bar (global index)
            a = float(atr[hit])
            if np.isnan(a) or a <= 0:
                continue
            entry = float(close[hit])
            lvl = float(p0 + s * slope * (hit - t0) * atr0)
            if direction == "long":
                stop, target = entry - STOP_ATR * a, entry + TARGET_ATR * a
            else:
                stop, target = entry + STOP_ATR * a, entry - TARGET_ATR * a
            rows.append({
                "event_id": f"{_safe(symbol)}|{timeframe}|{subtype}|{p['pivot_id']}|{label}",
                "symbol": symbol, "timeframe": timeframe, "timestamp": ts[hit],
                "geometry_type": "gann", "geometry_subtype": f"{subtype}_{label}",
                "direction": direction, "level_price": lvl, "current_price": entry,
                "distance_pct": float((entry - lvl) / lvl),
                "distance_atr": float((entry - lvl) / a),
                "anchor_data": to_json({"anchor": p["pivot_id"], "p0": p0,
                                        "slope": slope, "atr0": atr0}),
                "confluence_score": 1.0,
                "metadata": to_json({"slope_label": label, "tol_atr": tol_atr}),
                "created_at": pd.Timestamp.now("UTC"),
                "is_control": control_kind is not None,
                "control_kind": control_kind, "bar_index": int(hit),
                "entry_price": entry, "target_price": float(target),
                "stop_price": float(stop),
            })
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)
