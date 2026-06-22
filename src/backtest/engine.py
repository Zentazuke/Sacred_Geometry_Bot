"""Event-driven trade simulator. Turns geometry events into trades and resolves
each one bar by bar, the way a live bot could actually have done it.

Lookahead safety:
  * The signal fires on bar `bar_index` (its close). We ENTER at the NEXT bar's
    OPEN — never using information from the signal bar's future.
  * Stop / target are resolved by walking forward bar by bar. If a single bar's
    range spans BOTH stop and target, the stop is assumed first (pessimistic).
  * Costs (fees + slippage) are charged on the round trip.

Each event already carries its trade plan: `direction`, `entry_price` (reference
only), `stop_price` (swing origin) and `target_price` (swing terminal).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADE_COLUMNS = [
    "event_id", "symbol", "timeframe", "control_kind", "direction",
    "entry_time", "entry_price", "exit_time", "exit_price", "stop_price",
    "target_price", "bars_held", "exit_reason", "rr_planned",
    "gross_return", "net_return", "r_net",
]


@dataclass
class BacktestParams:
    fee_bps: float = 10.0          # per side; Binance spot taker ~0.10%
    slippage_bps: float = 5.0      # per side
    risk_pct: float = 0.005        # fraction of equity risked per trade
    max_hold: int = 50             # bars before a timeout exit at close
    min_rr: float = 0.0            # skip trades whose planned reward:risk < this

    @property
    def round_trip_cost(self) -> float:
        return 2.0 * (self.fee_bps + self.slippage_bps) / 1e4


def simulate_trades(events: pd.DataFrame, df: pd.DataFrame,
                    params: BacktestParams) -> pd.DataFrame:
    """Resolve every event into at most one trade. Returns a tidy trades frame."""
    if events.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)

    op = df["open"].values
    hi = df["high"].values
    lo = df["low"].values
    cl = df["close"].values
    ts = df["timestamp"].values
    n = len(df)
    cost = params.round_trip_cost
    rows = []

    for _, ev in events.iterrows():
        sig = int(ev["bar_index"])
        entry_idx = sig + 1                       # enter at next bar's open
        if entry_idx >= n:
            continue
        entry = float(op[entry_idx])
        stop = float(ev["stop_price"])
        target = float(ev["target_price"])
        sign = 1.0 if ev["direction"] == "long" else -1.0

        risk_unit = abs(entry - stop) / entry
        reward_unit = abs(target - entry) / entry
        if risk_unit <= 0:
            continue
        rr = reward_unit / risk_unit
        if rr < params.min_rr:
            continue

        exit_idx, exit_px, reason = _resolve(entry_idx, n, hi, lo, cl,
                                             sign, stop, target, params.max_hold)
        gross = sign * (exit_px - entry) / entry
        net = gross - cost
        r_net = net / risk_unit
        rows.append({
            "event_id": ev["event_id"], "symbol": ev["symbol"],
            "timeframe": ev["timeframe"], "control_kind": ev.get("control_kind"),
            "direction": ev["direction"], "entry_time": ts[entry_idx],
            "entry_price": entry, "exit_time": ts[exit_idx], "exit_price": exit_px,
            "stop_price": stop, "target_price": target,
            "bars_held": exit_idx - entry_idx, "exit_reason": reason,
            "rr_planned": rr, "gross_return": gross, "net_return": net,
            "r_net": r_net,
        })

    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


def _resolve(entry_idx, n, hi, lo, cl, sign, stop, target, max_hold):
    """Walk forward; return (exit_index, exit_price, reason)."""
    end = min(n, entry_idx + max_hold + 1)
    for j in range(entry_idx, end):
        if sign > 0:
            if lo[j] <= stop:
                return j, stop, "stop"
            if hi[j] >= target:
                return j, target, "target"
        else:
            if hi[j] >= stop:
                return j, stop, "stop"
            if lo[j] <= target:
                return j, target, "target"
    last = end - 1
    return last, float(cl[last]), "timeout"
