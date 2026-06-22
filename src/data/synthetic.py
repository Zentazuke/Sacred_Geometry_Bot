"""Offline candle generator. Lets the whole research loop run with no network /
no keys, so the pipeline can be developed and tested deterministically.

The series is a geometric random walk with occasional trends and mean-reverting
pullbacks, which produces realistic-looking swings for pivots & Fibonacci."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .candle_collector import CANDLE_COLUMNS
from .exchange_client import timeframe_ms


def generate_candles(symbol: str, timeframe: str, n: int = 3000, seed: int = 0,
                     start_price: float = 30000.0) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash((symbol, timeframe, seed))) % (2**32))
    tf_ms = timeframe_ms(timeframe)

    # Drift regime that flips occasionally -> trends + reversals (good for swings).
    drift = np.zeros(n)
    d = 0.0
    for i in range(n):
        if rng.random() < 0.01:                # ~1% chance to flip regime each bar
            d = rng.normal(0, 0.0008)
        drift[i] = d
    vol = 0.012
    rets = drift + rng.normal(0, vol, n)
    close = start_price * np.exp(np.cumsum(rets))

    # Build OHLC around the close path.
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]
    bar_vol = np.abs(rng.normal(0, vol, n)) * close
    high = np.maximum(open_, close) + bar_vol
    low = np.minimum(open_, close) - bar_vol
    volume = np.abs(rng.normal(1000, 300, n))

    end = pd.Timestamp.now("UTC").floor("h")
    ts = pd.date_range(end=end, periods=n, freq=pd.Timedelta(milliseconds=tf_ms), tz="UTC")

    df = pd.DataFrame({
        "timestamp": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    return df[CANDLE_COLUMNS]
