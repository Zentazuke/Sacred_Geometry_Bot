"""Shared geometry vocabulary. Every geometry module emits the same kind of
event so the research engine can treat sacred levels and control levels
identically. New modules (Gann, harmonics, ...) implement GeometryModule."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd

EVENT_COLUMNS = [
    "event_id", "symbol", "timeframe", "timestamp", "geometry_type",
    "geometry_subtype", "direction", "level_price", "current_price",
    "distance_pct", "distance_atr", "anchor_data", "confluence_score",
    "metadata", "created_at", "is_control", "control_kind", "bar_index",
    # extra columns the outcome labeler needs (not all persisted to duckdb):
    "entry_price", "target_price", "stop_price",
]


@dataclass
class Leg:
    """A directional swing between two consecutive confirmed pivots."""
    a_index: int          # origin pivot bar index
    b_index: int          # terminal pivot bar index
    a_price: float
    b_price: float
    direction: str        # 'long' (low->high leg) or 'short' (high->low leg)
    known_index: int      # first bar at which the leg could be used (B confirmed)
    a_id: str
    b_id: str

    @property
    def range(self) -> float:
        return abs(self.b_price - self.a_price)


def legs_from_pivots(pivots: pd.DataFrame) -> list[Leg]:
    """Pair consecutive alternating pivots (in confirmation order) into legs."""
    piv = pivots.sort_values("confirmed_index").reset_index(drop=True)
    legs: list[Leg] = []
    for i in range(1, len(piv)):
        a, b = piv.iloc[i - 1], piv.iloc[i]
        if a["pivot_type"] == b["pivot_type"]:
            continue  # need alternating high/low to form a clean leg
        direction = "long" if (a["pivot_type"] == "low" and b["pivot_type"] == "high") else "short"
        legs.append(Leg(
            a_index=int(a["bar_index"]), b_index=int(b["bar_index"]),
            a_price=float(a["price"]), b_price=float(b["price"]),
            direction=direction, known_index=int(b["confirmed_index"]),
            a_id=a["pivot_id"], b_id=b["pivot_id"],
        ))
    return legs


def to_json(obj: Any) -> str:
    return json.dumps(obj, default=str)


class GeometryModule(Protocol):
    name: str

    def generate_events(self, df: pd.DataFrame, pivots: pd.DataFrame,
                        **kwargs) -> pd.DataFrame:
        ...
