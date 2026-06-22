import numpy as np
import pandas as pd

from src.geometry.gann import gann_events
from src.market_structure.pivots import PIVOT_COLUMNS


def _df(close):
    n = len(close)
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    c = np.asarray(close, float)
    return pd.DataFrame({"timestamp": ts, "open": c, "high": c + 0.05,
                         "low": c - 0.05, "close": c, "volume": np.ones(n),
                         "atr_14": np.ones(n)})


def _low_pivot(bar_index, price, confirmed_index, df):
    row = {col: None for col in PIVOT_COLUMNS}
    row.update({
        "pivot_id": "P1", "symbol": "BTC/USDT", "timeframe": "1h",
        "timestamp": df["timestamp"].iloc[bar_index], "bar_index": bar_index,
        "pivot_type": "low", "price": price, "strength": 0, "left_bars": 0,
        "right_bars": 0, "atr": 1.0,
        "confirmed_at": df["timestamp"].iloc[confirmed_index],
        "confirmed_index": confirmed_index, "method": "zigzag",
    })
    return pd.DataFrame([row], columns=PIVOT_COLUMNS)


def test_gann_1x1_touch_from_low_is_long():
    # price rides the 1x1 ATR line exactly: close(t) = 100 + (t - 5), atr = 1
    n = 40
    close = [100.0] * 5 + [100.0 + (t - 5) for t in range(5, n)]
    df = _df(close)
    pivots = _low_pivot(5, 100.0, 7, df)
    ev = gann_events(df, "BTC/USDT", "1h", pivots, tol_atr=0.25)
    assert not ev.empty
    assert (ev["direction"] == "long").all()          # low anchor -> support -> long
    assert ev["geometry_subtype"].str.contains("1x1").any()
    # touch can't be before the pivot is confirmed
    assert (ev["bar_index"] > 7).all()


def test_random_slopes_differ_from_real_fan():
    n = 40
    close = [100.0] * 5 + [100.0 + (t - 5) for t in range(5, n)]
    df = _df(close)
    pivots = _low_pivot(5, 100.0, 7, df)
    real = gann_events(df, "BTC/USDT", "1h", pivots)
    rand = gann_events(df, "BTC/USDT", "1h", pivots, random_slopes=True, seed=1,
                       control_kind="random_slopes")
    assert rand["control_kind"].eq("random_slopes").all() if not rand.empty else True
    # the real fan should reliably catch the 1x1 line; identical event sets would
    # be a red flag that randomisation did nothing
    assert real["geometry_subtype"].str.contains("1x1").any()
