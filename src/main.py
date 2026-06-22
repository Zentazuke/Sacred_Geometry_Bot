"""Sacred Geometry Bot — command-line entry point.

Subcommands:
  backfill        Harvest historical candles into parquet + duckdb (the Historian).
  observe         One pass: features -> pivots -> geometry events -> outcomes (no trades).
  experiment001   Run EXP_001_GOLDEN_POCKET end-to-end and write a research report
                  (the Scientist). Use --synthetic to run fully offline with no keys.

This MVP never trades. The Trader personality (Phase 9, Binance testnet) is gated
behind ALLOW_LIVE_TRADING and not wired here on purpose.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running as `python src/main.py` or `python -m src.main`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import load_settings
    from src.data.candle_collector import CandleCollector
    from src.data.candle_store import CandleStore
    from src.data.data_quality import find_gaps
    from src.data.exchange_client import ExchangeClient
    from src.data.synthetic import generate_candles
    from src.geometry.fibonacci import golden_pocket_events
    from src.market_structure.features import add_features
    from src.market_structure.pivots import detect_pivots
    from src.research import reports
    from src.research.outcome_labeler import label_events
    from src.signals.controls import build_controls
else:  # pragma: no cover
    from .config import load_settings
    from .data.candle_collector import CandleCollector
    from .data.candle_store import CandleStore
    from .data.data_quality import find_gaps
    from .data.exchange_client import ExchangeClient
    from .data.synthetic import generate_candles
    from .geometry.fibonacci import golden_pocket_events
    from .market_structure.features import add_features
    from .market_structure.pivots import detect_pivots
    from .research import reports
    from .research.outcome_labeler import label_events
    from .signals.controls import build_controls


def get_store(settings) -> CandleStore:
    return CandleStore(settings.path("raw_dir"), settings.path("duckdb_path"))


def get_candles(settings, store, symbol, timeframe, synthetic: bool,
                synth_n: int = 3000) -> pd.DataFrame:
    if synthetic:
        return generate_candles(symbol, timeframe, n=synth_n)
    df = store.load_candles(symbol, timeframe)
    if df.empty:
        # Market data always comes from PRODUCTION (public, keyless). The sandbox
        # flag governs trading only — testnet has almost no candle history.
        client = ExchangeClient(settings.exchange_name, sandbox=False)
        collector = CandleCollector(client)
        df = collector.backfill(symbol, timeframe, limit=settings.backfill_limit)
        store.upsert_candles(symbol, timeframe, df)
    return df


def cmd_backfill(settings, args):
    store = get_store(settings)
    # production endpoint for candle history (sandbox/testnet has none)
    client = ExchangeClient(settings.exchange_name, sandbox=False)
    collector = CandleCollector(client)
    limit = args.limit or settings.backfill_limit
    for symbol in settings.symbols:
        for tf in settings.timeframes:
            try:
                df = collector.backfill(symbol, tf, limit=limit)
            except Exception as exc:  # one bad/illiquid symbol shouldn't kill the run
                print(f"{symbol} {tf}: SKIPPED ({type(exc).__name__}: {exc})")
                continue
            store.upsert_candles(symbol, tf, df)
            gaps = find_gaps(df, tf)
            span = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) if len(df) else None
            yrs = f"{span.days / 365:.2f}y" if span is not None else "—"
            print(f"{symbol} {tf}: {len(df)} candles ({yrs}), {len(gaps)} gaps")


def cmd_rank(settings, args):
    from src.backtest.engine import BacktestParams
    from src.research import leaderboard

    kw = dict(leaderboard.IMPROVED) if args.rule == "improved" else {}
    params = BacktestParams(**kw)
    geoms = leaderboard.GEOMETRIES if args.geometry == "all" else [args.geometry]
    result = leaderboard.rank(settings, args.synthetic, params, geometries=geoms)
    text = leaderboard.build_report(result)
    out_path = Path(settings.path("duckdb_path")).parent / "reports" / "LEADERBOARD.md"
    reports.write_report(text, out_path)
    print(text)
    print(f"\nReport written to: {out_path}")


def cmd_sweep(settings, args):
    from src.research import sweep

    result = sweep.run(settings, args.synthetic, horizon=args.horizon)
    text = sweep.build_report(result)
    out_path = Path(settings.path("duckdb_path")).parent / "reports" / "MYTH_DETECTOR.md"
    reports.write_report(text, out_path)
    print(text)
    print(f"\nReport written to: {out_path}")


def cmd_backtest(settings, args):
    from src.backtest.engine import BacktestParams
    from src.backtest import runner

    params = BacktestParams(fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
                            risk_pct=args.risk_pct, max_hold=args.max_hold,
                            min_rr=args.min_rr, trend_filter=args.trend_filter,
                            confirm=args.confirm, atr_stop_floor=args.atr_stop_floor,
                            stop_buffer_atr=args.stop_buffer_atr, target_r=args.target_r)
    result = runner.run(settings, args.synthetic, params, geometry=args.geometry)
    text = runner.build_report(result)
    out_path = (Path(settings.path("duckdb_path")).parent / "reports"
                / f"BACKTEST_{args.geometry}.md")
    reports.write_report(text, out_path)
    print(text)
    print(f"\nReport written to: {out_path}")


def _run_pipeline(settings, store, symbol, tf, synthetic, controls_list, persist=True):
    """Returns (golden_outcomes, {control: outcomes}, n_events) for one market."""
    df = get_candles(settings, store, symbol, tf, synthetic)
    if len(df) < 200:
        return None
    feat = add_features(df)
    pivots = detect_pivots(feat, symbol, tf, settings.experiments
                           .get("EXP_001_GOLDEN_POCKET", {}).get("pivot_method", "zigzag"),
                           settings.pivots)

    golden = golden_pocket_events(feat, symbol, tf, pivots)
    golden_out = label_events(golden, feat)

    controls = build_controls(feat, symbol, tf, pivots, controls_list,
                              n_golden=len(golden))
    control_outs = {k: label_events(v, feat) for k, v in controls.items()}

    if persist and not golden.empty:
        store.write_df("pivots", pivots, replace_keys=["pivot_id"])
        persist_cols = [c for c in golden.columns]
        store.write_df("geometry_events", golden, replace_keys=["event_id"])
        store.write_df("event_outcomes", golden_out, replace_keys=["event_id"])

    return golden, golden_out, control_outs


def cmd_observe(settings, args):
    store = get_store(settings)
    exp = settings.experiments.get("EXP_001_GOLDEN_POCKET", {})
    controls_list = exp.get("controls", [])
    total = 0
    for symbol in settings.symbols:
        for tf in settings.timeframes:
            res = _run_pipeline(settings, store, symbol, tf, args.synthetic, controls_list)
            if res is None:
                print(f"{symbol} {tf}: insufficient data")
                continue
            golden, golden_out, _ = res
            total += len(golden)
            print(f"{symbol} {tf}: {len(golden)} golden-pocket events logged")
    print(f"Total golden-pocket events: {total}")


def cmd_experiment001(settings, args):
    store = get_store(settings)
    exp = settings.experiments.get("EXP_001_GOLDEN_POCKET", {})
    controls_list = exp.get("controls", [])
    horizon = 10

    golden_frames, control_frames = [], {c: [] for c in controls_list}
    for symbol in settings.symbols:
        for tf in settings.timeframes:
            res = _run_pipeline(settings, store, symbol, tf, args.synthetic,
                                controls_list, persist=not args.synthetic)
            if res is None:
                continue
            _, golden_out, control_outs = res
            golden_frames.append(golden_out)
            for c, out in control_outs.items():
                control_frames[c].append(out)

    if not golden_frames:
        print("No events produced — check data.")
        return

    golden_all = pd.concat(golden_frames, ignore_index=True)
    control_all = {c: pd.concat(f, ignore_index=True) for c, f in control_frames.items() if f}

    text = reports.build_report("EXP_001_GOLDEN_POCKET", horizon, golden_all, control_all)
    out_path = Path(settings.path("duckdb_path")).parent / "reports" / "EXP_001_GOLDEN_POCKET.md"
    reports.write_report(text, out_path)
    print(text)
    print(f"\nReport written to: {out_path}")


def main(argv=None):
    # Windows consoles default to cp1252; reports contain unicode (Δ, ...).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(prog="sacred-geometry-bot")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_b = sub.add_parser("backfill", help="harvest historical candles")
    p_b.add_argument("--limit", type=int, default=None,
                     help="candles per market (overrides config; ~8760 = 1y of 1h)")
    p_b.set_defaults(func=cmd_backfill)

    p_o = sub.add_parser("observe", help="one observer pass, log events + outcomes")
    p_o.add_argument("--synthetic", action="store_true", help="use offline generated candles")
    p_o.set_defaults(func=cmd_observe)

    p_e = sub.add_parser("experiment001", help="run golden-pocket experiment + report")
    p_e.add_argument("--synthetic", action="store_true", help="use offline generated candles")
    p_e.set_defaults(func=cmd_experiment001)

    p_t = sub.add_parser("backtest", help="simulate golden-pocket trades + report")
    p_t.add_argument("--synthetic", action="store_true", help="use offline generated candles")
    p_t.add_argument("--fee-bps", type=float, default=10.0, dest="fee_bps")
    p_t.add_argument("--slippage-bps", type=float, default=5.0, dest="slippage_bps")
    p_t.add_argument("--risk-pct", type=float, default=0.005, dest="risk_pct")
    p_t.add_argument("--max-hold", type=int, default=50, dest="max_hold")
    p_t.add_argument("--min-rr", type=float, default=0.0, dest="min_rr")
    p_t.add_argument("--geometry", choices=["golden_pocket", "gann", "harmonics"],
                     default="golden_pocket", help="which geometry to trade")
    p_t.add_argument("--trend-filter", action="store_true", dest="trend_filter",
                     help="only trade with the trend regime")
    p_t.add_argument("--confirm", action="store_true",
                     help="require a rejection close back out of the zone")
    p_t.add_argument("--atr-stop-floor", type=float, default=0.0, dest="atr_stop_floor",
                     help="widen stop to at least this many ATRs from entry")
    p_t.add_argument("--stop-buffer-atr", type=float, default=0.0, dest="stop_buffer_atr",
                     help="push stop this many ATRs beyond the swing origin")
    p_t.add_argument("--target-r", type=float, default=0.0, dest="target_r",
                     help="target = entry +/- target_r * risk (else geometry terminal)")
    p_t.set_defaults(func=cmd_backtest)

    p_s = sub.add_parser("sweep", help="grid coins x timeframes x geometry; myth detector")
    p_s.add_argument("--synthetic", action="store_true", help="use offline generated candles")
    p_s.add_argument("--horizon", type=int, default=10, help="forward bars for bounce metric")
    p_s.set_defaults(func=cmd_sweep)

    p_r = sub.add_parser("rank", help="backtest every coin x timeframe x geometry, ranked")
    p_r.add_argument("--synthetic", action="store_true", help="use offline generated candles")
    p_r.add_argument("--rule", choices=["improved", "naive"], default="improved",
                     help="trade rule: improved trend-following (default) or raw geometry")
    p_r.add_argument("--geometry", choices=["all", "golden_pocket", "gann", "harmonics"],
                     default="all")
    p_r.set_defaults(func=cmd_rank)

    args = parser.parse_args(argv)
    settings = load_settings()
    args.func(settings, args)


if __name__ == "__main__":
    main()
