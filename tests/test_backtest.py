import numpy as np
import pandas as pd

from src.backtest.engine import BacktestParams, simulate_trades
from src.backtest.metrics import compute_metrics
from src.geometry.base import EVENT_COLUMNS

NO_COST = BacktestParams(fee_bps=0, slippage_bps=0, risk_pct=0.01, max_hold=50)


def _df(opens, highs, lows, closes):
    n = len(opens)
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "open": opens, "high": highs,
                         "low": lows, "close": closes, "volume": np.ones(n)})


def _event(bar_index, direction, entry, target, stop):
    row = {c: None for c in EVENT_COLUMNS}
    row.update({"event_id": "E1", "symbol": "BTC/USDT", "timeframe": "1h",
                "direction": direction, "bar_index": bar_index, "entry_price": entry,
                "target_price": target, "stop_price": stop, "control_kind": None})
    return pd.DataFrame([row], columns=EVENT_COLUMNS)


def test_enters_at_next_bar_open():
    # signal on bar 0; entry must be bar 1's open (101), not bar 0
    df = _df([100, 101, 102, 110, 111], [100, 101, 111, 111, 112],
             [100, 100, 101, 109, 110], [100, 101, 110, 110, 111])
    ev = _event(0, "long", entry=101.0, target=110.0, stop=96.0)
    t = simulate_trades(ev, df, NO_COST).iloc[0]
    assert t["entry_price"] == 101.0


def test_target_hit_gives_correct_R():
    # entry 100, stop 95 (risk 5%), target 110 (reward 10%) -> +2R, no costs
    df = _df([100, 100, 100, 100], [100, 101, 111, 111],
             [100, 99, 98, 98], [100, 100, 110, 110])
    ev = _event(0, "long", entry=100.0, target=110.0, stop=95.0)
    t = simulate_trades(ev, df, NO_COST).iloc[0]
    assert t["exit_reason"] == "target"
    assert abs(t["r_net"] - 2.0) < 1e-6


def test_stop_hit_gives_minus_one_R():
    df = _df([100, 100, 100], [100, 100, 100], [100, 94, 94], [100, 95, 95])
    ev = _event(0, "long", entry=100.0, target=120.0, stop=95.0)
    t = simulate_trades(ev, df, NO_COST).iloc[0]
    assert t["exit_reason"] == "stop"
    assert abs(t["r_net"] + 1.0) < 1e-6


def test_costs_reduce_R():
    df = _df([100, 100, 100, 100], [100, 101, 111, 111],
             [100, 99, 98, 98], [100, 100, 110, 110])
    ev = _event(0, "long", entry=100.0, target=110.0, stop=95.0)
    with_cost = BacktestParams(fee_bps=10, slippage_bps=5, risk_pct=0.01)
    t = simulate_trades(ev, df, with_cost).iloc[0]
    # 0.30% round-trip cost over a 5% stop = 0.06R drag -> 2.0 - 0.06
    assert abs(t["r_net"] - (2.0 - 0.003 / 0.05)) < 1e-6


def test_metrics_expectancy_matches_mean_R():
    trades = pd.DataFrame({
        "entry_time": pd.date_range("2024-01-01", periods=4, freq="1D", tz="UTC"),
        "exit_time": pd.date_range("2024-01-02", periods=4, freq="1D", tz="UTC"),
        "r_net": [2.0, -1.0, -1.0, 2.0], "exit_reason": ["target", "stop", "stop", "target"],
        "bars_held": [3, 2, 2, 3],
    })
    m = compute_metrics(trades, risk_pct=0.01)
    assert abs(m["expectancy_r"] - 0.5) < 1e-9
    assert m["win_rate"] == 0.5
    assert m["n_trades"] == 4
