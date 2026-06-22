import numpy as np
import pandas as pd

from src.geometry.base import Leg
from src.geometry.fibonacci import GOLDEN_POCKET, detect_zone_events, retracement_levels


def test_retracement_levels_long_leg():
    # A=100 -> B=200, range 100. 0.618 retr => 200 - 61.8 = 138.2
    levels = retracement_levels(100.0, 200.0)
    assert abs(levels[0.618] - 138.2) < 1e-6
    assert abs(levels[0.5] - 150.0) < 1e-6


def _df_from_path(path):
    n = len(path)
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    p = np.asarray(path, float)
    return pd.DataFrame({
        "timestamp": ts, "open": p, "high": p + 0.1, "low": p - 0.1,
        "close": p, "volume": np.ones(n), "atr_14": np.ones(n),
    })


def test_golden_pocket_event_fires_on_pullback():
    # leg low=100 (idx0) -> high=200 (idx10), known at idx10. Golden pocket for
    # this leg is 121.4 .. 138.2; pull price back DOWN into it (dip to 130).
    path = ([100] * 1 + list(np.linspace(100, 200, 10))
            + [190, 170, 130, 130, 160, 180])  # dips to 130 -> inside pocket
    df = _df_from_path(path)
    leg = Leg(a_index=0, b_index=10, a_price=100.0, b_price=200.0,
              direction="long", known_index=10, a_id="A", b_id="B")
    ev = detect_zone_events(df, "BTC/USDT", "1h", [leg],
                            GOLDEN_POCKET[0], GOLDEN_POCKET[1], subtype="golden_pocket")
    assert len(ev) == 1
    row = ev.iloc[0]
    assert row["direction"] == "long"
    assert row["bar_index"] > leg.known_index          # event after B confirmed
    assert row["stop_price"] == 100.0 and row["target_price"] == 200.0


def test_no_event_when_price_never_retraces():
    path = list(np.linspace(100, 200, 12)) + list(np.linspace(200, 260, 12))
    df = _df_from_path(path)
    leg = Leg(a_index=0, b_index=11, a_price=100.0, b_price=200.0,
              direction="long", known_index=11, a_id="A", b_id="B")
    ev = detect_zone_events(df, "BTC/USDT", "1h", [leg],
                            GOLDEN_POCKET[0], GOLDEN_POCKET[1], subtype="golden_pocket")
    assert len(ev) == 0
