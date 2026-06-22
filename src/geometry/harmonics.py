"""Harmonic patterns (Gartley, Bat, Butterfly, Crab) — the most respectable
sacred geometry, because the ratios are strict and falsifiable.

A pattern is five consecutive alternating pivots X-A-B-C-D whose leg ratios fall
inside published Fibonacci bands. When D completes, it marks a Potential Reversal
Zone: go long if D is a swing low, short if D is a swing high. Stop sits just
beyond D; target is the 0.618 retracement of the CD leg (per the plan).

Lookahead-safe: a pattern is only emitted at D's `confirmed_index`, and entry is
that bar's close.

Control (`random_direction=True`): same D-points, but the trade direction is a
coin flip and the stop/target are generic ATR levels. If real harmonic direction
calls don't beat a coin flip at the same points, harmonics predict nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import EVENT_COLUMNS, to_json

# ratio bands: ab_xa = |AB|/|XA|, bc_ab = |BC|/|AB|, cd_bc = |CD|/|BC|,
# ad_xa = |XD|/|XA| (where D's retracement/extension of the XA leg).
PATTERNS = {
    "gartley":   {"ab_xa": (0.55, 0.68), "bc_ab": (0.38, 0.89), "cd_bc": (1.10, 1.70), "ad_xa": (0.74, 0.84)},
    "bat":       {"ab_xa": (0.38, 0.52), "bc_ab": (0.38, 0.89), "cd_bc": (1.55, 2.70), "ad_xa": (0.84, 0.92)},
    "butterfly": {"ab_xa": (0.74, 0.82), "bc_ab": (0.38, 0.89), "cd_bc": (1.55, 2.30), "ad_xa": (1.24, 1.62)},
    "crab":      {"ab_xa": (0.36, 0.65), "bc_ab": (0.38, 0.89), "cd_bc": (2.10, 3.70), "ad_xa": (1.50, 1.70)},
}
STOP_BUF_ATR = 0.5
TARGET_CD_RETRACE = 0.618


def _safe(symbol: str) -> str:
    return symbol.replace("/", "_")


def _match(ratios: dict) -> tuple[str | None, float]:
    """Return (best pattern name, score in [0,1]) or (None, 0) if no band match."""
    best, best_score = None, 0.0
    for name, bands in PATTERNS.items():
        ok, closeness = True, []
        for key, (lo, hi) in bands.items():
            v = ratios[key]
            if not (lo <= v <= hi):
                ok = False
                break
            mid, half = (lo + hi) / 2, (hi - lo) / 2
            closeness.append(1.0 - abs(v - mid) / half)
        if ok:
            score = float(np.mean(closeness))
            if score > best_score:
                best, best_score = name, score
    return best, best_score


def harmonic_events(df: pd.DataFrame, symbol: str, timeframe: str, pivots: pd.DataFrame,
                    atr_col: str = "atr_14", random_direction: bool = False,
                    seed: int = 0, subtype: str = "harmonic",
                    control_kind: str | None = None) -> pd.DataFrame:
    piv = pivots.sort_values("confirmed_index").reset_index(drop=True)
    close = df["close"].values
    atr = df[atr_col].values
    ts = df["timestamp"].values
    n = len(df)
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(4, len(piv)):
        X, A, B, C, D = (piv.iloc[i - 4], piv.iloc[i - 3], piv.iloc[i - 2],
                         piv.iloc[i - 1], piv.iloc[i])
        if not (X["pivot_type"] != A["pivot_type"] != B["pivot_type"]
                != C["pivot_type"] != D["pivot_type"]):
            continue
        xa = abs(A["price"] - X["price"])
        ab = abs(B["price"] - A["price"])
        bc = abs(C["price"] - B["price"])
        cd = abs(D["price"] - C["price"])
        if min(xa, ab, bc, cd) <= 0:
            continue
        # XAD retracement/extension is measured from A: |A-D| / |XA|
        # (Gartley ~0.786, Bat ~0.886, Crab ~1.618, ...).
        ratios = {"ab_xa": ab / xa, "bc_ab": bc / ab, "cd_bc": cd / bc,
                  "ad_xa": abs(A["price"] - D["price"]) / xa}
        name, score = _match(ratios)
        if name is None:
            continue

        known = int(D["confirmed_index"])
        if known >= n:
            continue
        a = atr[known]
        if np.isnan(a) or a <= 0:
            continue
        entry = float(close[known])
        long = D["pivot_type"] == "low"

        if random_direction:
            long = bool(rng.integers(0, 2))
            stop = entry - 1.0 * a if long else entry + 1.0 * a
            target = entry + 2.0 * a if long else entry - 2.0 * a
            direction = "long" if long else "short"
        else:
            direction = "long" if long else "short"
            if long:
                stop = float(D["price"]) - STOP_BUF_ATR * a
                target = float(D["price"]) + TARGET_CD_RETRACE * (float(C["price"]) - float(D["price"]))
            else:
                stop = float(D["price"]) + STOP_BUF_ATR * a
                target = float(D["price"]) - TARGET_CD_RETRACE * (float(D["price"]) - float(C["price"]))
        if abs(entry - stop) <= 0:
            continue

        rows.append({
            "event_id": f"{_safe(symbol)}|{timeframe}|{subtype}|{D['pivot_id']}",
            "symbol": symbol, "timeframe": timeframe, "timestamp": ts[known],
            "geometry_type": "harmonic", "geometry_subtype": f"{subtype}_{name}",
            "direction": direction, "level_price": float(D["price"]),
            "current_price": entry,
            "distance_pct": float((entry - D["price"]) / D["price"]),
            "distance_atr": float((entry - D["price"]) / a),
            "anchor_data": to_json({"X": X["pivot_id"], "A": A["pivot_id"],
                                    "B": B["pivot_id"], "C": C["pivot_id"], "D": D["pivot_id"]}),
            "confluence_score": float(score),
            "metadata": to_json({"pattern": name, "score": round(score, 3), **{k: round(v, 3) for k, v in ratios.items()}}),
            "created_at": pd.Timestamp.now("UTC"),
            "is_control": control_kind is not None, "control_kind": control_kind,
            "bar_index": known, "entry_price": entry,
            "target_price": float(target), "stop_price": float(stop),
        })
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)
