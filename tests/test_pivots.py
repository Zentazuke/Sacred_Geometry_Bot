import numpy as np
import pandas as pd

from src.market_structure.pivots import fractal_pivots, zigzag_pivots


def _df(prices):
    n = len(prices)
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    p = np.asarray(prices, float)
    return pd.DataFrame({
        "timestamp": ts, "open": p, "high": p + 0.5, "low": p - 0.5,
        "close": p, "volume": np.ones(n),
    })


def test_fractal_detects_swing_high_and_confirms_late():
    # peak at index 5; left=right=2 -> confirmed at index 7
    prices = [10, 11, 12, 13, 14, 20, 14, 13, 12, 11, 10]
    df = _df(prices)
    piv = fractal_pivots(df, "BTC/USDT", "1h", left=2, right=2, atr_period=3)
    highs = piv[piv["pivot_type"] == "high"]
    assert (highs["bar_index"] == 5).any()
    row = highs[highs["bar_index"] == 5].iloc[0]
    assert row["confirmed_index"] == 7          # i + right
    assert row["confirmed_index"] > row["bar_index"]   # no lookahead


def test_fractal_detects_swing_low():
    prices = [20, 18, 16, 14, 12, 5, 12, 14, 16, 18, 20]
    df = _df(prices)
    piv = fractal_pivots(df, "BTC/USDT", "1h", left=2, right=2, atr_period=3)
    lows = piv[piv["pivot_type"] == "low"]
    assert (lows["bar_index"] == 5).any()


def test_zigzag_alternates_and_is_causal():
    # clear up then down then up move, large enough to trip a 2*ATR reversal
    up = list(np.linspace(100, 200, 40))
    down = list(np.linspace(200, 120, 40))
    up2 = list(np.linspace(120, 260, 40))
    df = _df(up + down + up2)
    piv = zigzag_pivots(df, "BTC/USDT", "1h", atr_mult=2.0, atr_period=14)
    assert len(piv) >= 2
    types = list(piv["pivot_type"])
    # consecutive pivots must alternate high/low
    assert all(types[i] != types[i + 1] for i in range(len(types) - 1))
    # confirmation never precedes the extreme bar
    assert (piv["confirmed_index"] >= piv["bar_index"]).all()
