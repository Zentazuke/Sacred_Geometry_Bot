"""Stability checks. The golden-pocket rule has no fitted parameters, so there is
nothing to "train" — but performance can still be a fluke of one period. We slice
trades chronologically and report each slice, plus a simple in/out-of-sample
split, so a single lucky quarter can't masquerade as edge."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import compute_metrics


def segment_metrics(trades: pd.DataFrame, risk_pct: float,
                    n_segments: int = 4) -> pd.DataFrame:
    """Split trades into n equal-count chronological segments; metrics per slice."""
    if trades.empty:
        return pd.DataFrame()
    t = trades.sort_values("entry_time").reset_index(drop=True)
    bounds = np.array_split(np.arange(len(t)), n_segments)
    rows = []
    for k, idx in enumerate(bounds, start=1):
        if len(idx) == 0:
            continue
        seg = t.iloc[idx]
        m = compute_metrics(seg, risk_pct)
        rows.append({
            "segment": k,
            "from": seg["entry_time"].min(),
            "to": seg["entry_time"].max(),
            "n_trades": m["n_trades"],
            "win_rate": m["win_rate"],
            "expectancy_r": m["expectancy_r"],
            "total_return": m["total_return"],
        })
    return pd.DataFrame(rows)


def in_out_of_sample(trades: pd.DataFrame, risk_pct: float) -> dict:
    """First half vs second half by time."""
    if trades.empty:
        return {}
    t = trades.sort_values("entry_time").reset_index(drop=True)
    mid = len(t) // 2
    return {
        "in_sample": compute_metrics(t.iloc[:mid], risk_pct),
        "out_of_sample": compute_metrics(t.iloc[mid:], risk_pct),
    }
