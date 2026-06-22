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
from ..geometry.gann import gann_events
from ..geometry.harmonics import harmonic_events
from ..market_structure.features import add_features
from ..market_structure.pivots import detect_pivots
from ..signals.controls import build_controls
from ..signals.signal_engine import refine_events
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


def _make_events(geometry, feat, symbol, tf, pivots):
    if geometry == "gann":
        return gann_events(feat, symbol, tf, pivots)
    if geometry == "harmonics":
        return harmonic_events(feat, symbol, tf, pivots)
    return golden_pocket_events(feat, symbol, tf, pivots)


def run(settings, synthetic: bool, params: BacktestParams,
        geometry: str = "golden_pocket") -> dict:
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

            events = _make_events(geometry, feat, symbol, tf, pivots)
            events = refine_events(events, feat, params)
            strat_trades.append(simulate_trades(events, feat, params))

            controls = build_controls(feat, symbol, tf, pivots, ["random_entry"],
                                      n_golden=len(events))
            ctrl_trades.append(simulate_trades(controls.get("random_entry",
                               events.iloc[0:0]), feat, params))

            span_years = (feat["timestamp"].iloc[-1] - feat["timestamp"].iloc[0]) \
                / np.timedelta64(365, "D")
            bh = feat["close"].iloc[-1] / feat["close"].iloc[0] - 1.0
            buyhold.append({"coin": symbol, "tf": tf, "market": f"{symbol} {tf}",
                            "return": bh, "years": span_years})
            coverage.append({"market": f"{symbol} {tf}", "candles": len(feat),
                             "years": round(span_years, 2),
                             "from": feat["timestamp"].iloc[0],
                             "to": feat["timestamp"].iloc[-1]})

    strat = _concat(strat_trades)
    ctrl = _concat(ctrl_trades)
    years = np.mean([c["years"] for c in coverage]) if coverage else 1.0

    return {
        "geometry": geometry,
        "params": params,
        "coverage": pd.DataFrame(coverage),
        "buyhold": pd.DataFrame(buyhold),
        "strategy_trades": strat,
        "control_trades": ctrl,
        "strategy_metrics": compute_metrics(strat, params.risk_pct, years),
        "control_metrics": compute_metrics(ctrl, params.risk_pct, years),
        "per_market_metrics": _grouped_metrics(strat, ["symbol", "timeframe"], params.risk_pct),
        "per_coin_metrics": _grouped_metrics(strat, ["symbol"], params.risk_pct),
        "segments": segment_metrics(strat, params.risk_pct, n_segments=4),
        "oos": in_out_of_sample(strat, params.risk_pct),
        "avg_years": years,
    }


def _grouped_metrics(trades, by, risk_pct) -> pd.DataFrame:
    """Full metric set computed independently for each group (coin or coin+tf)."""
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for key, g in trades.groupby(by):
        label = key if isinstance(key, str) else " ".join(key)
        m = compute_metrics(g, risk_pct)
        gross_pnl_R = float(g["r_net"].sum())          # cumulative R, before sizing
        rows.append({"group": label, **m, "sum_r": gross_pnl_R})
    return pd.DataFrame(rows).sort_values("expectancy_r", ascending=False)


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


def _metrics_table(df: pd.DataFrame, group_header: str) -> list[str]:
    """Render a full-metric table, one row per group, sorted by expectancy."""
    L = []
    cols = (f"| {group_header} | trades | win | expectancy | PF | sumR | total | "
            f"annual | maxDD | Calmar | Sharpe | Sortino | tgt/stop/to |")
    L.append(cols)
    L.append("|" + "---|" * 13)
    for _, r in df.iterrows():
        L.append(
            f"| {r['group']} | {int(r['n_trades'])} | {r['win_rate']*100:.1f}% | "
            f"{r['expectancy_r']:+.3f}R | {r['profit_factor']:.2f} | {r['sum_r']:+.0f}R | "
            f"{r['total_return']*100:+.1f}% | {r['annual_return']*100:+.1f}% | "
            f"{r['max_drawdown']*100:.1f}% | {r['calmar']:.2f} | {r['sharpe']:.2f} | "
            f"{r['sortino']:.2f} | "
            f"{r['target_rate']*100:.0f}/{r['stop_rate']*100:.0f}/{r['timeout_rate']*100:.0f}% |")
    return L


def build_report(result: dict) -> str:
    p = result["params"]
    sm, cm = result["strategy_metrics"], result["control_metrics"]
    geo = result.get("geometry", "golden_pocket")
    L = []
    L.append(f"# Backtest — `{geo}` strategy, per-coin P/L")
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
    L.append(_row(geo, sm))
    L.append(_row("random_entry", cm))
    L.append("")
    bh = result["buyhold"]
    if not bh.empty:
        L.append(f"Buy & hold (equal-weight avg over the same windows): "
                 f"{bh['return'].mean() * 100:+.1f}% total.")
        L.append("")

    # ---- per-coin P/L (the headline question) ---------------------------
    pcm = result.get("per_coin_metrics")
    if pcm is not None and not pcm.empty:
        bh_by_coin = (result["buyhold"].groupby("coin")["return"].mean().to_dict()
                      if not bh.empty else {})
        L.append("## Profit / loss per coin (1h + 4h combined)")
        L.append("")
        L += _metrics_table(pcm, "coin")
        L.append("")
        L.append("Buy & hold per coin over the same window (the benchmark every strategy "
                 "is competing with):")
        L.append("")
        L.append("| coin | buy & hold |")
        L.append("|---|---|")
        for coin, ret in sorted(bh_by_coin.items(), key=lambda kv: -kv[1]):
            L.append(f"| {coin} | {ret*100:+.0f}% |")
        L.append("")

    pmm = result.get("per_market_metrics")
    if pmm is not None and not pmm.empty:
        L.append("## Profit / loss per coin × timeframe")
        L.append("")
        L += _metrics_table(pmm, "market")
        L.append("")

    L.append(f"## Exit breakdown (pooled, {geo})")
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

    L.append("## Metrics glossary")
    L += _GLOSSARY
    L.append("")

    L.append("## Verdict")
    L.append(_verdict(sm, cm))
    L.append("")
    return "\n".join(L)


_GLOSSARY = [
    "",
    "- **trades** — number of completed trades (one per geometry signal that found an entry).",
    "- **win** — fraction of trades with a positive net return after costs.",
    "- **expectancy (R)** — average profit per trade in units of *risk* (R). 1R = the amount "
    "risked, i.e. the distance from entry to stop. +0.2R means each trade nets a fifth of what "
    "it risked, on average. This is the single most important number; it is scale-free and "
    "directly comparable across coins.",
    "- **PF (profit factor)** — gross R won / gross R lost. >1 is profitable; 1.0 is break-even.",
    "- **sumR** — cumulative R across all the group's trades (total edge harvested, before "
    "position sizing).",
    "- **total** — compounded account return, risking `risk/trade` of equity per trade. NOTE: "
    "this assumes one account trading that group's signals sequentially, so it punishes "
    "over-trading; judge edge on *expectancy*, not this.",
    "- **annual** — total return annualised over the group's time span.",
    "- **maxDD (max drawdown)** — largest peak-to-trough equity drop, as a %. Lower is better.",
    "- **Calmar** — annual return / max drawdown. Reward per unit of worst-case pain.",
    "- **Sharpe** — mean per-trade return / its standard deviation, annualised. Risk-adjusted "
    "return; >1 is good, negative means losing.",
    "- **Sortino** — like Sharpe but only penalises *downside* volatility.",
    "- **tgt/stop/to** — share of trades exited at target / stop / timeout (max-hold reached).",
    "- **buy & hold** — return from simply buying at the start and holding to the end; the "
    "benchmark any active strategy must beat to justify the effort and risk.",
]


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
