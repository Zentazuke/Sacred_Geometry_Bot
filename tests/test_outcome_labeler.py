import numpy as np
import pandas as pd

from src.geometry.base import EVENT_COLUMNS
from src.research.outcome_labeler import label_events


def _df(path):
    n = len(path)
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    p = np.asarray(path, float)
    return pd.DataFrame({
        "timestamp": ts, "open": p, "high": p + 0.001, "low": p - 0.001,
        "close": p, "volume": np.ones(n),
    })


def _event(bar_index, direction, entry, target, stop):
    row = {c: None for c in EVENT_COLUMNS}
    row.update({
        "event_id": "E1", "symbol": "BTC/USDT", "timeframe": "1h",
        "direction": direction, "bar_index": bar_index, "entry_price": entry,
        "target_price": target, "stop_price": stop,
    })
    return pd.DataFrame([row], columns=EVENT_COLUMNS)


def test_long_favourable_return_is_positive_when_price_rises():
    df = _df([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111])
    ev = _event(0, "long", entry=100.0, target=110.0, stop=95.0)
    out = label_events(ev, df).iloc[0]
    assert out["return_10"] > 0
    assert out["hit_target"] and not out["hit_stop"]


def test_short_return_sign_is_directional():
    # price falls -> a SHORT should show a positive directional return
    df = _df([100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89])
    ev = _event(0, "short", entry=100.0, target=90.0, stop=105.0)
    out = label_events(ev, df).iloc[0]
    assert out["return_10"] > 0
    assert out["hit_target"] and not out["hit_stop"]


def test_mae_is_non_positive():
    df = _df([100, 98, 102, 99, 104, 101, 106, 108, 110, 112, 114, 116])
    ev = _event(0, "long", entry=100.0, target=120.0, stop=90.0)
    out = label_events(ev, df).iloc[0]
    assert out["mae_10"] <= 0
    assert out["mfe_10"] >= 0
