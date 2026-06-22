import numpy as np
import pandas as pd

from src.backtest.engine import BacktestParams
from src.geometry.base import EVENT_COLUMNS
from src.signals.signal_engine import refine_events


def _feat(n=30, trend=1.0):
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    c = np.linspace(100, 130, n)
    return pd.DataFrame({"timestamp": ts, "open": c, "high": c + 1, "low": c - 1,
                         "close": c, "volume": np.ones(n), "atr_14": np.ones(n),
                         "trend_regime": np.full(n, trend)})


def _event(bar_index, direction, entry, level, stop, target):
    row = {col: None for col in EVENT_COLUMNS}
    row.update({"event_id": "E1", "symbol": "BTC/USDT", "timeframe": "1h",
                "direction": direction, "bar_index": bar_index, "entry_price": entry,
                "level_price": level, "stop_price": stop, "target_price": target,
                "control_kind": None})
    return pd.DataFrame([row], columns=EVENT_COLUMNS)


def test_trend_filter_drops_countertrend():
    feat = _feat(trend=1.0)                      # bullish regime
    ev = _event(10, "short", entry=120, level=120, stop=125, target=110)
    out = refine_events(ev, feat, BacktestParams(trend_filter=True))
    assert out.empty                              # a short in an uptrend is dropped


def test_trend_filter_keeps_with_trend():
    feat = _feat(trend=1.0)
    ev = _event(10, "long", entry=120, level=120, stop=115, target=130)
    out = refine_events(ev, feat, BacktestParams(trend_filter=True))
    assert len(out) == 1


def test_atr_stop_floor_widens_tight_stop():
    feat = _feat()
    entry = feat["close"].iloc[10]
    # original stop only 0.2 away; ATR=1, floor=2 -> stop must move to entry-2
    ev = _event(10, "long", entry=entry, level=entry, stop=entry - 0.2, target=entry + 5)
    out = refine_events(ev, feat, BacktestParams(atr_stop_floor=2.0)).iloc[0]
    assert abs(out["stop_price"] - (entry - 2.0)) < 1e-9


def test_target_r_sets_reward_multiple():
    feat = _feat()
    entry = feat["close"].iloc[10]
    ev = _event(10, "long", entry=entry, level=entry, stop=entry - 2.0, target=entry + 1)
    out = refine_events(ev, feat, BacktestParams(target_r=3.0)).iloc[0]
    # risk = 2.0 -> target = entry + 3*2 = entry+6
    assert abs(out["target_price"] - (entry + 6.0)) < 1e-9
