"""Data-quality checks: detect missing candles (gaps) in a sorted series."""
from __future__ import annotations

import pandas as pd

from .exchange_client import timeframe_ms


def find_gaps(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Return rows describing gaps where consecutive candles are > 1 interval apart."""
    if len(df) < 2:
        return pd.DataFrame(columns=["gap_start", "gap_end", "missing"])
    tf = pd.Timedelta(milliseconds=timeframe_ms(timeframe))
    ts = df["timestamp"].reset_index(drop=True)
    delta = ts.diff()
    bad = delta[delta > tf]
    rows = []
    for idx, gap in bad.items():
        missing = int(gap / tf) - 1
        rows.append({"gap_start": ts[idx - 1], "gap_end": ts[idx], "missing": missing})
    return pd.DataFrame(rows)


def is_stale(df: pd.DataFrame, timeframe: str, max_age_intervals: int = 2) -> bool:
    """True if the most recent candle is older than max_age_intervals * tf."""
    if df.empty:
        return True
    tf = pd.Timedelta(milliseconds=timeframe_ms(timeframe))
    age = pd.Timestamp.now("UTC") - df["timestamp"].iloc[-1]
    return age > max_age_intervals * tf
