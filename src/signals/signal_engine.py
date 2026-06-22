"""Signal refinement — turn raw geometry *events* into better *trade plans*.

A geometry touch is not a trade. The plan's own checklist says: confirm the
trend, wait for a rejection candle, place a sane stop, demand reward:risk. This
module applies those filters/adjustments to an events frame before it reaches the
backtest engine. Each improvement is independently switchable so we can attribute
any change in edge to a specific idea, not a soup of them.

All operations stay lookahead-free: they only read the signal bar (`bar_index`)
and the levels already known at that bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..geometry.base import to_json


def refine_events(events: pd.DataFrame, feat: pd.DataFrame, params) -> pd.DataFrame:
    """Filter and re-plan events according to the improvement flags on `params`."""
    if events.empty:
        return events

    trend = feat["trend_regime"].values if "trend_regime" in feat else None
    atr = feat["atr_14"].values
    close = feat["close"].values
    out = []

    for _, ev in events.iterrows():
        i = int(ev["bar_index"])
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue
        long = ev["direction"] == "long"
        entry = float(ev["entry_price"])

        # 1) trend filter — only trade with the regime
        if params.trend_filter and trend is not None:
            if long and trend[i] < 0:
                continue
            if (not long) and trend[i] > 0:
                continue

        # 2) confirmation — require a rejection: the signal bar closes back on the
        #    trade's side of the level (not still falling through a long pocket)
        if params.confirm:
            lvl = float(ev["level_price"])
            if long and close[i] < lvl:
                continue
            if (not long) and close[i] > lvl:
                continue

        # 3) stop placement — widen beyond the swing origin and/or floor at k*ATR
        stop = float(ev["stop_price"])
        if long:
            stop = stop - params.stop_buffer_atr * a
            if params.atr_stop_floor > 0:
                stop = min(stop, entry - params.atr_stop_floor * a)
        else:
            stop = stop + params.stop_buffer_atr * a
            if params.atr_stop_floor > 0:
                stop = max(stop, entry + params.atr_stop_floor * a)

        # 4) target — optionally an R-multiple instead of the swing terminal
        target = float(ev["target_price"])
        risk = abs(entry - stop)
        if params.target_r > 0 and risk > 0:
            target = entry + params.target_r * risk if long else entry - params.target_r * risk

        row = ev.to_dict()
        row["stop_price"] = stop
        row["target_price"] = target
        row["metadata"] = to_json({"refined": True})
        out.append(row)

    return pd.DataFrame(out, columns=events.columns)
