import numpy as np
import pandas as pd

from src.research.leaderboard import _oos, _stable


def _trades(r_values):
    n = len(r_values)
    return pd.DataFrame({
        "entry_time": pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC"),
        "exit_time": pd.date_range("2024-01-02", periods=n, freq="1D", tz="UTC"),
        "r_net": r_values, "exit_reason": ["target"] * n, "bars_held": [1] * n,
    })


def test_oos_splits_in_and_out_of_sample():
    # first half all losers, second half all winners
    t = _trades([-1.0] * 10 + [2.0] * 10)
    oin, oout = _oos(t, risk_pct=0.01)
    assert oin < 0 < oout


def test_stable_requires_edge_and_both_halves_positive():
    good = {"trades": 100, "edge_vs_rand": 0.05, "oos_in": 0.1, "oos_out": 0.08}
    assert _stable(good, floor=0.0)

    # negative geometry edge over same-market random -> not real even if positive
    no_edge = {"trades": 100, "edge_vs_rand": -0.02, "oos_in": 0.1, "oos_out": 0.08}
    assert not _stable(no_edge, floor=0.0)

    # collapses out of sample
    overfit = {"trades": 100, "edge_vs_rand": 0.05, "oos_in": 0.2, "oos_out": -0.05}
    assert not _stable(overfit, floor=0.0)

    # too few trades
    thin = {"trades": 20, "edge_vs_rand": 0.05, "oos_in": 0.1, "oos_out": 0.08}
    assert not _stable(thin, floor=0.0)
