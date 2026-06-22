"""Trade-level performance metrics + an equity curve. Risk-normalised: each trade
risks `risk_pct` of equity, so the equity multiplier per trade is (1 + risk_pct *
R_net). This makes results comparable across symbols with different price scales."""
from __future__ import annotations

import numpy as np
import pandas as pd

METRIC_KEYS = [
    "n_trades", "win_rate", "expectancy_r", "profit_factor", "avg_win_r",
    "avg_loss_r", "total_return", "annual_return", "max_drawdown", "calmar",
    "sharpe", "sortino", "longest_losing_streak", "avg_bars_held",
    "target_rate", "stop_rate", "timeout_rate",
]


def equity_curve(trades: pd.DataFrame, risk_pct: float) -> np.ndarray:
    """Compounded equity (starting at 1.0) from sequential per-trade R outcomes."""
    if trades.empty:
        return np.array([1.0])
    per_trade = risk_pct * trades.sort_values("entry_time")["r_net"].values
    return np.concatenate([[1.0], np.cumprod(1.0 + per_trade)])


def _max_drawdown(eq: np.ndarray) -> float:
    peak = np.maximum.accumulate(eq)
    return float((1.0 - eq / peak).max())


def _streak(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def compute_metrics(trades: pd.DataFrame, risk_pct: float,
                    years: float | None = None) -> dict:
    if trades.empty:
        return {k: float("nan") for k in METRIC_KEYS} | {"n_trades": 0}

    t = trades.sort_values("entry_time").reset_index(drop=True)
    r = t["r_net"].values
    wins = r[r > 0]
    losses = r[r <= 0]
    eq = equity_curve(t, risk_pct)
    per_trade = risk_pct * r

    if years is None:
        span = (t["exit_time"].max() - t["entry_time"].min())
        years = max(span / np.timedelta64(365, "D"), 1e-9)
    total_return = float(eq[-1] - 1.0)
    annual_return = float((eq[-1]) ** (1.0 / years) - 1.0) if eq[-1] > 0 else -1.0
    mdd = _max_drawdown(eq)

    downside = per_trade[per_trade < 0]
    sharpe = float(per_trade.mean() / per_trade.std() * np.sqrt(len(r) / years)) \
        if per_trade.std() > 0 else float("nan")
    sortino = float(per_trade.mean() / downside.std() * np.sqrt(len(r) / years)) \
        if len(downside) > 1 and downside.std() > 0 else float("nan")

    reason = t["exit_reason"].value_counts(normalize=True)
    return {
        "n_trades": int(len(t)),
        "win_rate": float((r > 0).mean()),
        "expectancy_r": float(r.mean()),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf"),
        "avg_win_r": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_r": float(losses.mean()) if len(losses) else 0.0,
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": mdd,
        "calmar": float(annual_return / mdd) if mdd > 0 else float("inf"),
        "sharpe": sharpe,
        "sortino": sortino,
        "longest_losing_streak": _streak(r <= 0),
        "avg_bars_held": float(t["bars_held"].mean()),
        "target_rate": float(reason.get("target", 0.0)),
        "stop_rate": float(reason.get("stop", 0.0)),
        "timeout_rate": float(reason.get("timeout", 0.0)),
    }
