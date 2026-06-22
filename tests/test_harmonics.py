import numpy as np
import pandas as pd

from src.geometry.harmonics import harmonic_events
from src.market_structure.pivots import PIVOT_COLUMNS


def _df(n=80):
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    c = np.full(n, 100.0)
    return pd.DataFrame({"timestamp": ts, "open": c, "high": c + 1, "low": c - 1,
                         "close": c, "volume": np.ones(n), "atr_14": np.ones(n)})


def _piv(rows, df):
    out = []
    for bar, ptype, price, conf in rows:
        r = {col: None for col in PIVOT_COLUMNS}
        r.update({"pivot_id": f"P{bar}", "symbol": "BTC/USDT", "timeframe": "1h",
                  "timestamp": df["timestamp"].iloc[bar], "bar_index": bar,
                  "pivot_type": ptype, "price": price, "strength": 0, "left_bars": 0,
                  "right_bars": 0, "atr": 1.0, "confirmed_at": df["timestamp"].iloc[conf],
                  "confirmed_index": conf, "method": "zigzag"})
        out.append(r)
    return pd.DataFrame(out, columns=PIVOT_COLUMNS)


def test_detects_a_textbook_gartley():
    # Bullish Gartley alternates L-H-L-H-L; D is the final low. Ratios:
    # ab_xa=.618, bc_ab=.6, cd_bc=1.45 -> ad_xa~.786 (all in Gartley bands).
    df = _df()
    X, A = 100.0, 200.0                  # XA = 100 (up)
    B = A - 0.618 * 100                  # ab_xa = .618 ; B = 138.2 (low)
    C = B + 0.6 * (A - B)                # bc_ab = .6 ; C = 175.28 (high)
    D = A - 0.785 * 100                  # ad_xa = .785 ; D = 121.5 (low)
    df.loc[47, ["open", "high", "low", "close"]] = [124, 127, 123, 125]  # entry bar
    rows = [(5, "low", X, 7), (15, "high", A, 17), (25, "low", B, 27),
            (35, "high", C, 37), (45, "low", D, 47)]
    piv = _piv(rows, df)
    ev = harmonic_events(df, "BTC/USDT", "1h", piv)
    assert not ev.empty
    r = ev.iloc[0]
    assert "gartley" in r["geometry_subtype"]
    assert r["direction"] == "long"           # D is a low -> bullish
    assert r["bar_index"] == 47               # emitted at D confirmation
    assert r["stop_price"] < r["entry_price"] # stop below for a long


def test_no_pattern_when_ratios_are_off():
    df = _df()
    rows = [(5, "high", 200, 7), (15, "low", 100, 17), (25, "high", 105, 27),
            (35, "low", 102, 37), (45, "low", 50, 47)]   # nonsense ratios
    piv = _piv(rows, df)
    assert harmonic_events(df, "BTC/USDT", "1h", piv).empty


def test_random_direction_control_flips_and_stays_defined():
    df = _df()
    X, A = 100.0, 200.0
    B = A - 0.618 * 100
    C = B + 0.6 * (A - B)
    D = A - 0.785 * 100
    rows = [(5, "low", X, 7), (15, "high", A, 17), (25, "low", B, 27),
            (35, "high", C, 37), (45, "low", D, 47)]
    piv = _piv(rows, df)
    ctrl = harmonic_events(df, "BTC/USDT", "1h", piv, random_direction=True,
                           seed=3, control_kind="coin_flip")
    assert not ctrl.empty
    r = ctrl.iloc[0]
    assert r["control_kind"] == "coin_flip"
    assert r["direction"] in ("long", "short")
    assert r["target_price"] != r["entry_price"]   # a usable trade plan
