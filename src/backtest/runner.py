"""Run the golden-pocket backtest across all symbols/timeframes, compare it
against a random-entry control and buy-and-hold, and render a report.

This is the Phase 8 deliverable: it answers "do golden-pocket *trades* make money
after fees and slippage?" — not just "do levels react?"."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data.candle_collector import CandleCollector
from ..data.candle_store import CandleStore
from ..data.exchange_client import ExchangeClient
from ..data.synthetic import generate_candles
from ..geometry.fibonacci import golden_pocket_events
from ..market_structure.features import add_features
from ..market_structure.pivots import detect_pivots
from ..signals.controls import build_controls
from .engine import BacktestParams, simulate_trades
from .metrics import compute_metrics
from .walk_forward import in_out_of_sample, segment_metrics


def _get_candles(settings, store, symbol, timeframe, synthetic, synth_n=10000):
    if synthetic:
        return generate_candles(symbol, timeframe, n=synth_n)
    df = store.load_candles(symbol, timeframe)
    if df.empty:
        # market data from production (public); testnet has no candle history
        client = ExchangeClient(settings.exchange_name, sandbox=False)
        df = CandleCollector(client).backfill(symbol, timeframe, limit=settings.backfill_limit)
        store.upsert_candles(symbol, timeframe, df)
    return df


def run(settings, synthetic: bool, params: BacktestParams) -> dict:
    store = CandleStore(settings.path("raw_dir"), settings.path("duckdb_path"))
    exp = settings.experiments.get("EXP_001_GOLDEN_POCKET", {})
    pivot_method = exp.get("pivot_method", "zigzag")

    strat_trades, ctrl_trades = [], []
    buyhold, coverage = [], []

    for symbol in settings.symbols:
        for tf in settings.timeframes:
            df = _get_candles(settings, store, symbol, tf, synthetic)
            if len(df) < 300:
                continue
            feat = add_features(df)
            pivots = detect_pivots(feat, symbol, tf, pivot_method, settings.pivots)

            golden = golden_pocket_events(feat, symbol, tf, pivots)
            strat_trades.append(simulate_trades(golden, feat, params))

            controls = build_controls(feat, symbol, tf, pivots, ["random_entry"],
                                      n_golden=len(golden))
            ctrl_trades.append(simulate_trades(controls.get("random_entry",
                               golden.iloc[0:0]), feat, params))

            span_years = (feat["timestamp"].iloc[-1] - feat["timestamp"].iloc[0]) \
                / np.timedelta64(365, "D")
            bh = feat["close"].iloc[-1] / feat["close"].iloc[0] - 1.0
            buyhold.append({"market": f"{symbol} {tf}", "return": bh, "years": span_years})
            coverage.append({"market": f"{symbol} {tf}", "candles": len(feat),
                             "years": round(span_years, 2),
                             "from": feat["timestamp"].iloc[0],
                             "to": feat["timestamp"].iloc[-1]})

    strat = _concat(strat_trades)
    ctrl = _concat(ctrl_trades)
    years = np.mean([c["years"] for c in coverage]) if coverage else 1.0

    return {
        "params": params,
        "coverage": pd.DataFrame(coverage),
        "buyhold": pd.DataFrame(buyhold),
        "strategy_trades": strat,
        "control_trades": ctrl,
        "strategy_metrics": compute_metrics(strat, params.risk_pct, years),
        "control_metrics": compute_metrics(ctrl, params.risk_pct, years),
        "segments": segment_metrics(strat, params.risk_pct, n_segments=4),
        "oos": in_out_of_sample(strat, params.risk_pct),
        "avg_years": years,
    }


def _concat(frames):
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------
def _row(name, m):
    def pct(x):
        return f"{x * 100:+.1f}%" if x == x else "n/a"
    return (f"| {name} | {m['n_trades']} | {m['win_rate'] * 100:.1f}% | "
            f"{m['expectancy_r']:+.3f}R | {m['profit_factor']:.2f} | "
            f"{pct(m['total_return'])} | {pct(m['annual_return'])} | "
            f"{pct(m['max_drawdown'])} | {m['sharpe']:.2f} |")


def build_report(result: dict) -> str:
    p = result["params"]
    sm, cm = result["strategy_metrics"], result["control_metrics"]
    L = []
    L.append("# EXP_001 — Golden-Pocket Backtest")
    L.append("")
    L.append(f"Costs: {p.fee_bps:.0f} bps fee + {p.slippage_bps:.0f} bps slippage per side "
             f"(round trip {p.round_trip_cost * 100:.2f}%). "
             f"Risk/trade: {p.risk_pct * 100:.2f}%. Max hold: {p.max_hold} bars. "
             f"Min reward:risk: {p.min_rr:.2f}.")
    L.append(f"Entry = next-bar open after the signal (no lookahead). "
             f"Avg coverage: {result['avg_years']:.2f} years/market.")
    L.append("")

    L.append("## Data coverage")
    L.append("")
    L.append("| market | candles | years | from | to |")
    L.append("|---|---|---|---|---|")
    for _, r in result["coverage"].iterrows():
        L.append(f"| {r['market']} | {r['candles']} | {r['years']} | "
                 f"{str(r['from'])[:10]} | {str(r['to'])[:10]} |")
    L.append("")

    L.append("## Headline (pooled across all markets)")
    L.append("")
    L.append("| strategy | trades | win | expectancy | PF | total | annual | maxDD | Sharpe |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    L.append(_row("golden_pocket", sm))
    L.append(_row("random_entry", cm))
    L.append("")
    bh = result["buyhold"]
    if not bh.empty:
        L.append(f"Buy & hold (equal-weight avg over the same windows): "
                 f"{bh['return'].mean() * 100:+.1f}% total.")
        L.append("")

    L.append("## Exit breakdown (golden pocket)")
    L.append(f"- target hit: {sm.get('target_rate', 0) * 100:.1f}%  |  "
             f"stop hit: {sm.get('stop_rate', 0) * 100:.1f}%  |  "
             f"timeout: {sm.get('timeout_rate', 0) * 100:.1f}%")
    L.append(f"- avg bars held: {sm.get('avg_bars_held', float('nan')):.1f}  |  "
             f"longest losing streak: {sm.get('longest_losing_streak', 0)}")
    L.append("")

    seg = result["segments"]
    if not seg.empty:
        L.append("## Stability — chronological quartiles")
        L.append("")
        L.append("| segment | from | to | trades | win | expectancy | period return |")
        L.append("|---|---|---|---|---|---|---|")
        for _, r in seg.iterrows():
            L.append(f"| Q{r['segment']} | {str(r['from'])[:10]} | {str(r['to'])[:10]} | "
                     f"{r['n_trades']} | {r['win_rate'] * 100:.1f}% | "
                     f"{r['expectancy_r']:+.3f}R | {r['total_return'] * 100:+.1f}% |")
        L.append("")

    oos = result["oos"]
    if oos:
        i, o = oos["in_sample"], oos["out_of_sample"]
        L.append("## In-sample vs out-of-sample (time split)")
        L.append(f"- in-sample  expectancy: {i['expectancy_r']:+.3f}R  "
                 f"(win {i['win_rate'] * 100:.1f}%, {i['n_trades']} trades)")
        L.append(f"- out-sample expectancy: {o['expectancy_r']:+.3f}R  "
                 f"(win {o['win_rate'] * 100:.1f}%, {o['n_trades']} trades)")
        L.append("")

    L.append("## Verdict")
    L.append(_verdict(sm, cm))
    L.append("")
    return "\n".join(L)


def _verdict(sm, cm) -> str:
    if sm["n_trades"] < 100:
        return f"INSUFFICIENT TRADES — only {sm['n_trades']}. Harvest more candles."
    beats_control = sm["expectancy_r"] > cm["expectancy_r"]
    profitable = sm["expectancy_r"] > 0 and sm["total_return"] > 0
    if profitable and beats_control:
        return ("PROMISING — positive expectancy after costs AND beats random entries. "
                "Pressure-test with walk-forward across more assets before believing it.")
    if not profitable and beats_control:
        return ("MIXED — loses money after costs but still less badly than random entries. "
                "There may be a weak reaction effect; the trade rule (stops/targets) is not "
                "harvesting it. Worth iterating on exits, not yet tradable.")
    return ("NOT TRADABLE — negative expectancy after costs and no edge over random entries. "
            "The expected fate of most sacred ideas; exactly why we backtest before trading.")
