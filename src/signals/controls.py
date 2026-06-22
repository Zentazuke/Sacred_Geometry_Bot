"""Control / baseline event generators. The whole point of the project: a sacred
level only earns belief if it beats these. All controls share the golden
pocket's mechanics so the comparison is apples-to-apples."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..geometry.base import EVENT_COLUMNS, Leg, legs_from_pivots, to_json
from ..geometry.fibonacci import GOLDEN_POCKET, detect_zone_events

POCKET_WIDTH = GOLDEN_POCKET[1] - GOLDEN_POCKET[0]   # 0.168


def _safe(symbol: str) -> str:
    return symbol.replace("/", "_")


def random_zone_events(df, symbol, timeframe, legs: list[Leg], rng: np.random.Generator,
                       **kw) -> pd.DataFrame:
    """Each leg gets its own random retracement band (same width) in 0.2-0.9."""
    frames = []
    for leg in legs:
        lo = float(rng.uniform(0.2, 0.9 - POCKET_WIDTH))
        hi = lo + POCKET_WIDTH
        frames.append(detect_zone_events(df, symbol, timeframe, [leg], lo, hi,
                                         subtype="random_zone", control_kind="random_zone", **kw))
    return _concat(frames)


def fixed_zone_events(df, symbol, timeframe, legs, center: float, label: str, **kw):
    lo = max(0.0, center - POCKET_WIDTH / 2)
    hi = lo + POCKET_WIDTH
    return detect_zone_events(df, symbol, timeframe, legs, lo, hi,
                              subtype=label, control_kind=label, **kw)


def scrambled_ratio_events(df, symbol, timeframe, legs, rng: np.random.Generator, **kw):
    """One randomised band, fixed for the whole run (scrambled, not per-leg random)."""
    lo = float(rng.uniform(0.2, 0.9 - POCKET_WIDTH))
    hi = lo + POCKET_WIDTH
    return detect_zone_events(df, symbol, timeframe, legs, lo, hi,
                              subtype="scrambled_ratios", control_kind="scrambled_ratios", **kw)


def random_entry_events(df, symbol, timeframe, n_events: int, rng: np.random.Generator,
                        atr_col: str = "atr_14", horizon_pad: int = 60) -> pd.DataFrame:
    """N entries at random bars; direction follows the trend regime; synthetic
    ATR-based target/stop so hit-rate metrics stay defined."""
    n = len(df)
    if n_events == 0 or n <= horizon_pad + 50:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    candidates = np.arange(50, n - horizon_pad)
    pick = rng.choice(candidates, size=min(n_events, len(candidates)), replace=False)
    rows = []
    trend = df["trend_regime"].values if "trend_regime" in df.columns else np.zeros(n)
    atr = df[atr_col].values
    closes = df["close"].values
    for i in sorted(pick):
        direction = "long" if trend[i] >= 0 else "short"
        px = float(closes[i])
        a = atr[i] if not np.isnan(atr[i]) else px * 0.01
        if direction == "long":
            target, stop = px + 1.5 * a, px - 1.0 * a
        else:
            target, stop = px - 1.5 * a, px + 1.0 * a
        rows.append({
            "event_id": f"{_safe(symbol)}|{timeframe}|random_entry|{i}",
            "symbol": symbol, "timeframe": timeframe, "timestamp": df["timestamp"].iloc[i],
            "geometry_type": "control", "geometry_subtype": "random_entry",
            "direction": direction, "level_price": px, "current_price": px,
            "distance_pct": 0.0, "distance_atr": 0.0,
            "anchor_data": to_json({}), "confluence_score": 0.0,
            "metadata": to_json({"atr": float(a)}), "created_at": pd.Timestamp.now("UTC"),
            "is_control": True, "control_kind": "random_entry", "bar_index": int(i),
            "entry_price": px, "target_price": float(target), "stop_price": float(stop),
        })
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def build_controls(df, symbol, timeframe, pivots, control_list: list[str],
                   n_golden: int, seed: int = 42, **kw) -> dict[str, pd.DataFrame]:
    """Return {control_kind: events_df} for the requested controls."""
    rng = np.random.default_rng(seed)
    legs = legs_from_pivots(pivots)
    out: dict[str, pd.DataFrame] = {}
    for ctrl in control_list:
        if ctrl == "random_zone":
            out[ctrl] = random_zone_events(df, symbol, timeframe, legs, rng, **kw)
        elif ctrl == "scrambled_ratios":
            out[ctrl] = scrambled_ratio_events(df, symbol, timeframe, legs, rng, **kw)
        elif ctrl == "random_entry":
            out[ctrl] = random_entry_events(df, symbol, timeframe, n_golden, rng)
        elif ctrl.startswith("fixed_zone_"):
            center = float(ctrl.rsplit("_", 1)[1])
            out[ctrl] = fixed_zone_events(df, symbol, timeframe, legs, center, ctrl, **kw)
    return out


def _concat(frames):
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.concat(frames, ignore_index=True)
