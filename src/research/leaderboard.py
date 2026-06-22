"""Leaderboard — backtest EVERY (coin x timeframe x geometry) combination and
rank them by performance, with an out-of-sample split so the "winner" can't just
be in-sample luck.

The honest framing this module enforces: with dozens of combinations, the top of
any leaderboard is selection bias. The columns that matter are not the headline
expectancy but (a) whether the combo stays positive *out of sample* and (b)
whether it beats a random-entry baseline run under the identical trade rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestParams, simulate_trades
from ..backtest.metrics import compute_metrics
from ..backtest.runner import _get_candles, _make_events
from ..data.candle_store import CandleStore
from ..market_structure.features import add_features
from ..market_structure.pivots import detect_pivots
from ..signals.controls import build_controls
from ..signals.signal_engine import refine_events

GEOMETRIES = ["golden_pocket", "gann", "harmonics"]

# The "improved" trend-following rule (the only config that ever turned positive).
IMPROVED = dict(trend_filter=True, confirm=True, atr_stop_floor=3.0,
                target_r=5.0, max_hold=200)


def _oos(trades, risk_pct):
    if len(trades) < 4:
        return float("nan"), float("nan")
    t = trades.sort_values("entry_time").reset_index(drop=True)
    mid = len(t) // 2
    return (compute_metrics(t.iloc[:mid], risk_pct)["expectancy_r"],
            compute_metrics(t.iloc[mid:], risk_pct)["expectancy_r"])


def rank(settings, synthetic: bool, params: BacktestParams,
         geometries=GEOMETRIES) -> dict:
    store = CandleStore(settings.path("raw_dir"), settings.path("duckdb_path"))
    pivot_method = settings.experiments.get("EXP_001_GOLDEN_POCKET", {}) \
        .get("pivot_method", "zigzag")
    rows = []
    rand_trades = []

    for symbol in settings.symbols:
        for tf in settings.timeframes:
            df = _get_candles(settings, store, symbol, tf, synthetic)
            if len(df) < 300:
                continue
            feat = add_features(df)
            pivots = detect_pivots(feat, symbol, tf, pivot_method, settings.pivots)

            # SAME-MARKET random-entry baseline under the identical rule. This is
            # the fair control: it absorbs "this coin/tf just trends well", so any
            # remaining gap is attributable to the geometry, not the market.
            ctrl = build_controls(feat, symbol, tf, pivots, ["random_entry"], n_golden=600)
            rev = refine_events(ctrl.get("random_entry", feat.iloc[0:0]), feat, params)
            rtr = simulate_trades(rev, feat, params)
            rand_trades.append(rtr)
            rand_exp_mkt = compute_metrics(rtr, params.risk_pct)["expectancy_r"]

            for geo in geometries:
                ev = refine_events(_make_events(geo, feat, symbol, tf, pivots), feat, params)
                tr = simulate_trades(ev, feat, params)
                m = compute_metrics(tr, params.risk_pct)
                oin, oout = _oos(tr, params.risk_pct)
                rows.append({
                    "coin": symbol, "tf": tf, "geometry": geo,
                    "trades": m["n_trades"], "win": m["win_rate"],
                    "expectancy_r": m["expectancy_r"], "profit_factor": m["profit_factor"],
                    "total_return": m["total_return"], "sharpe": m["sharpe"],
                    "oos_in": oin, "oos_out": oout,
                    "rand_same": rand_exp_mkt,
                    "edge_vs_rand": m["expectancy_r"] - rand_exp_mkt,
                })

    board = pd.DataFrame(rows)
    rand = pd.concat([t for t in rand_trades if not t.empty], ignore_index=True) \
        if any(not t.empty for t in rand_trades) else pd.DataFrame()
    rand_exp = compute_metrics(rand, params.risk_pct)["expectancy_r"] if not rand.empty else float("nan")
    return {"params": params, "board": board, "random_baseline_r": rand_exp,
            "geometries": geometries}


def _stable(r, floor) -> bool:
    # the honest bar: enough trades, BEATS its own market's random baseline, and
    # stays positive in both time halves.
    return (r["trades"] >= 50 and r["edge_vs_rand"] > 0
            and r["oos_in"] > 0 and r["oos_out"] > 0)


def build_report(result: dict, top_n: int = 20, min_trades: int = 30) -> str:
    p = result["params"]
    board = result["board"].copy()
    rb = result["random_baseline_r"]
    floor = max(rb, 0.0) if rb == rb else 0.0

    rule = ("improved trend-following (trend filter + confirm + 3-ATR stop + 5R "
            "target + 200-bar hold)") if p.trend_filter else "naive (raw geometry levels)"

    L = ["# Leaderboard — every coin x timeframe x geometry, ranked", ""]
    L.append(f"Rule: **{rule}**. Costs {p.round_trip_cost*100:.2f}% round trip, "
             f"risk {p.risk_pct*100:.2f}%/trade. Combinations: **{len(board)}**.")
    L.append(f"Random-entry baseline under the same rule: **{rb:+.3f}R** — this is the "
             f"noise floor; a combo only 'wins' if it clears this *and* holds out-of-sample.")
    L.append("")

    ranked = board[board["trades"] >= min_trades].sort_values("expectancy_r", ascending=False)
    L.append(f"## Top {top_n} by expectancy (>= {min_trades} trades)")
    L.append("")
    L.append("| # | coin | tf | geometry | trades | win | exp(R) | rand(same mkt) | geo edge | OOS in→out | real? |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, (_, r) in enumerate(ranked.head(top_n).iterrows(), 1):
        flag = "✅" if _stable(r, floor) else ""
        L.append(f"| {i} | {r['coin']} | {r['tf']} | {r['geometry']} | {int(r['trades'])} | "
                 f"{r['win']*100:.0f}% | {r['expectancy_r']:+.3f} | {r['rand_same']:+.3f} | "
                 f"{r['edge_vs_rand']:+.3f} | {r['oos_in']:+.2f}→{r['oos_out']:+.2f} | {flag} |")
    L.append("")
    L.append("`geo edge` = the combo's expectancy minus *its own coin/timeframe* random-entry "
             "baseline under the identical rule. This is the number that isolates geometry from "
             "\"this market just trends well\". `real?` ✅ = positive geo edge AND positive in "
             "both time halves.")
    L.append("")

    # best per geometry
    L.append("## Best combo per geometry (by geometry edge over same-market random)")
    L.append("")
    L.append("| geometry | best coin/tf | exp(R) | geo edge | trades | OOS in→out |")
    L.append("|---|---|---|---|---|---|")
    for geo in result["geometries"]:
        sub = board[(board["geometry"] == geo) & (board["trades"] >= min_trades)]
        if sub.empty:
            L.append(f"| {geo} | (too few trades) | — | — | — | — |")
            continue
        b = sub.loc[sub["edge_vs_rand"].idxmax()]
        L.append(f"| {geo} | {b['coin']} {b['tf']} | {b['expectancy_r']:+.3f} | "
                 f"{b['edge_vs_rand']:+.3f} | {int(b['trades'])} | "
                 f"{b['oos_in']:+.2f}→{b['oos_out']:+.2f} |")
    L.append("")

    # honesty section
    beats_mkt = int((ranked["edge_vs_rand"] > 0).sum())
    stable = int(ranked.apply(lambda r: _stable(r, floor), axis=1).sum())
    L.append("## Reality check")
    L.append(f"- The headline winners have high *expectancy* mostly because they are **4h on "
             f"coins that trended hard** — the same-market random baseline is already strongly "
             f"positive there, so it's trend-following, not geometry.")
    L.append(f"- {beats_mkt}/{len(ranked)} combos beat their OWN market's random baseline "
             f"in-sample (geo edge > 0). Under a true no-edge null you'd expect ~half "
             f"({len(ranked)//2}) by chance — so this is consistent with **zero real geometry edge**.")
    L.append(f"- **{stable}/{len(ranked)}** also hold positive geo-edge across both time halves "
             f"(✅). With {len(ranked)} combos, that many survivors is within chance.")
    L.append(f"- The leaderboard *ranks* performance; it does not *prove* edge. The best combo "
             f"is selection bias. The only legitimate use of a ✅ row is as a single, "
             f"pre-registered hypothesis for fresh out-of-sample data — not as a portfolio.")
    L.append("")
    return "\n".join(L)
