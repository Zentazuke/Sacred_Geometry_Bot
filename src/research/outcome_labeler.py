"""Outcome labelling: for every geometry event, measure what happened *after* it.

All returns are *directional* — sign-adjusted so that a favourable move is
positive whether the event was long or short. This is what lets us ask "did the
level produce edge?" instead of "did price go up?".

  return_h : directional forward return at horizon h
  mfe_W    : max favourable excursion over W bars (best the trade could do)
  mae_W    : max adverse excursion over W bars (worst drawdown, <= 0)
  hit_target / hit_stop : which of the leg's TP/SL was touched first within 50 bars
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = [1, 3, 5, 10, 20, 50]
EXCURSION_WINDOWS = [10, 20]
RESOLVE_HORIZON = 50

OUTCOME_COLUMNS = (
    ["event_id"]
    + [f"return_{h}" for h in HORIZONS]
    + [f"mfe_{w}" for w in EXCURSION_WINDOWS]
    + [f"mae_{w}" for w in EXCURSION_WINDOWS]
    + ["hit_target", "hit_stop", "bars_evaluated", "labeled_at"]
)


def label_events(events: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Return one outcome row per event. Events must carry bar_index, direction,
    entry_price, target_price, stop_price."""
    if events.empty:
        return pd.DataFrame(columns=OUTCOME_COLUMNS)

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)
    now = pd.Timestamp.now("UTC")
    rows = []

    for _, ev in events.iterrows():
        i = int(ev["bar_index"])
        entry = float(ev["entry_price"])
        sign = 1.0 if ev["direction"] == "long" else -1.0
        bars_left = n - 1 - i
        row: dict = {"event_id": ev["event_id"], "bars_evaluated": int(bars_left),
                     "labeled_at": now}

        for h in HORIZONS:
            j = i + h
            row[f"return_{h}"] = float(sign * (close[j] - entry) / entry) if j < n else np.nan

        for w in EXCURSION_WINDOWS:
            end = min(n, i + 1 + w)
            if end <= i + 1:
                row[f"mfe_{w}"] = np.nan
                row[f"mae_{w}"] = np.nan
                continue
            seg_hi = high[i + 1:end]
            seg_lo = low[i + 1:end]
            if sign > 0:
                row[f"mfe_{w}"] = float((seg_hi.max() - entry) / entry)
                row[f"mae_{w}"] = float((seg_lo.min() - entry) / entry)
            else:
                row[f"mfe_{w}"] = float((entry - seg_lo.min()) / entry)
                row[f"mae_{w}"] = float((entry - seg_hi.max()) / entry)

        row["hit_target"], row["hit_stop"] = _resolve(
            i, n, high, low, sign, float(ev["target_price"]), float(ev["stop_price"]))
        rows.append(row)

    return pd.DataFrame(rows, columns=OUTCOME_COLUMNS)


def _resolve(i, n, high, low, sign, target, stop):
    """Walk forward up to RESOLVE_HORIZON bars; report first of TP/SL touched.
    If both touch in the same bar, count the stop first (conservative)."""
    end = min(n, i + 1 + RESOLVE_HORIZON)
    for j in range(i + 1, end):
        if sign > 0:
            hit_stop = low[j] <= stop
            hit_target = high[j] >= target
        else:
            hit_stop = high[j] >= stop
            hit_target = low[j] <= target
        if hit_stop:
            return False, True
        if hit_target:
            return True, False
    return False, False
