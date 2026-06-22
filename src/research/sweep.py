"""Myth Detector — sweep many (geometry variant x coin x timeframe) combinations
and rank them by edge over a matched control, with bootstrap CIs and an explicit
multiple-testing correction.

The danger of "trying as many combinations as you can to find a pattern" is that
with enough combinations SOMETHING always looks good by chance. So every combo is
scored against its own control, and we report both the naive 5% bar AND the
Bonferroni-corrected bar. A pattern only counts if it clears the corrected bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.runner import _get_candles
from ..data.candle_store import CandleStore
from ..geometry.base import legs_from_pivots
from ..geometry.fibonacci import detect_zone_events
from ..geometry.gann import gann_events
from ..market_structure.features import add_features
from ..market_structure.pivots import detect_pivots
from . import stats
from .outcome_labeler import label_events

MIN_EVENTS = 200

# Geometry variants to sweep. Each fib variant is a retracement band; its control
# is a random band of the SAME width. Gann's control is random slopes.
VARIANTS = [
    {"name": "fib_golden_0.618-0.786", "kind": "fib", "lo": 0.618, "hi": 0.786},
    {"name": "fib_deep_0.786-1.000", "kind": "fib", "lo": 0.786, "hi": 1.0},
    {"name": "fib_mid_0.500-0.618", "kind": "fib", "lo": 0.5, "hi": 0.618},
    {"name": "fib_shallow_0.382-0.500", "kind": "fib", "lo": 0.382, "hi": 0.5},
    {"name": "gann_fan", "kind": "gann"},
]


# A control must be the AVERAGE of many random draws, not one. A single random
# band / random-slope fan can be anomalously weak by luck and manufacture a fake
# edge (this actually happened on SOL 1h in an earlier single-draw version). We
# pool K draws into one larger control sample.
CONTROL_DRAWS = 8


def _random_band(df, symbol, tf, legs, width, rng, **kw):
    """Per-leg random retracement band of the given width, in [0.2, 0.95]."""
    frames = []
    for leg in legs:
        lo = float(rng.uniform(0.2, 0.95 - width))
        frames.append(detect_zone_events(df, symbol, tf, [leg], lo, lo + width,
                                         subtype="rand_band", control_kind="rand_band", **kw))
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else df.iloc[0:0]


def _gen(variant, feat, symbol, tf, pivots, legs, rng):
    """Return (event_df, pooled_control_df) for one variant on one market.
    The control pools CONTROL_DRAWS independent random draws."""
    draws = []
    if variant["kind"] == "fib":
        lo, hi = variant["lo"], variant["hi"]
        ev = detect_zone_events(feat, symbol, tf, legs, lo, hi, subtype=variant["name"])
        for _ in range(CONTROL_DRAWS):
            draws.append(_random_band(feat, symbol, tf, legs, hi - lo, rng))
    else:  # gann: real fan vs pooled random-slope fans
        ev = gann_events(feat, symbol, tf, pivots, subtype="gann")
        for k in range(CONTROL_DRAWS):
            draws.append(gann_events(feat, symbol, tf, pivots, random_slopes=True,
                                     seed=1000 + k, control_kind="random_slopes"))
    return ev, _concat(draws)


def run(settings, synthetic: bool, horizon: int = 10, n_boot: int = 2000) -> dict:
    store = CandleStore(settings.path("raw_dir"), settings.path("duckdb_path"))
    pivot_method = settings.experiments.get("EXP_001_GOLDEN_POCKET", {}) \
        .get("pivot_method", "zigzag")
    rng = np.random.default_rng(2024)

    # outcomes[(variant, tf)] = {"ev": [frames], "ctrl": [frames]}
    pooled: dict = {}
    per_market: list[dict] = []

    for symbol in settings.symbols:
        for tf in settings.timeframes:
            df = _get_candles(settings, store, symbol, tf, synthetic)
            if len(df) < 300:
                continue
            feat = add_features(df)
            pivots = detect_pivots(feat, symbol, tf, pivot_method, settings.pivots)
            legs = legs_from_pivots(pivots)

            for variant in VARIANTS:
                ev, ctrl = _gen(variant, feat, symbol, tf, pivots, legs, rng)
                ev_out = label_events(ev, feat)
                ctrl_out = label_events(ctrl, feat)
                key = (variant["name"], tf)
                pooled.setdefault(key, {"ev": [], "ctrl": []})
                pooled[key]["ev"].append(ev_out)
                pooled[key]["ctrl"].append(ctrl_out)

                rec = _score(variant["name"], symbol, tf, ev_out, ctrl_out,
                             horizon, n_boot, do_perm=False)
                per_market.append(rec)

    pooled_recs = []
    for (name, tf), d in pooled.items():
        ev_out = _concat(d["ev"])
        ctrl_out = _concat(d["ctrl"])
        pooled_recs.append(_score(name, "ALL", tf, ev_out, ctrl_out,
                                  horizon, n_boot, do_perm=True))

    n_tests = len(per_market) + len(pooled_recs)
    return {
        "horizon": horizon,
        "coins": settings.symbols,
        "timeframes": settings.timeframes,
        "variants": [v["name"] for v in VARIANTS],
        "n_tests": n_tests,
        "bonferroni_alpha": 0.05 / max(n_tests, 1),
        "pooled": sorted(pooled_recs, key=lambda r: r["edge"], reverse=True),
        "per_market": sorted(per_market, key=lambda r: r["edge"], reverse=True),
    }


def _score(name, coin, tf, ev_out, ctrl_out, horizon, n_boot, do_perm):
    es = stats.group_summary(ev_out, horizon)
    cs = stats.group_summary(ctrl_out, horizon)
    col = f"return_{horizon}"
    a = ev_out[col].values if not ev_out.empty else np.array([])
    b = ctrl_out[col].values if not ctrl_out.empty else np.array([])
    boot = stats.bootstrap_diff(a, b, n_boot=n_boot, statistic="bounce_rate")
    p = stats.permutation_pvalue(a, b, n_perm=n_boot, statistic="bounce_rate") if do_perm else np.nan
    return {
        "variant": name, "coin": coin, "tf": tf,
        "n_ev": es["evaluated"], "bounce_ev": es["bounce_rate"],
        "n_ctrl": cs["evaluated"], "bounce_ctrl": cs["bounce_rate"],
        "edge": boot["diff"], "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
        "p": p,
    }


def _survivor(rec, alpha) -> bool:
    return (rec["n_ev"] >= MIN_EVENTS and rec["ci_low"] > 0
            and (np.isnan(rec["p"]) or rec["p"] < alpha))


def _concat(frames):
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------
def build_report(result: dict) -> str:
    h = result["horizon"]
    L = ["# Myth Detector — geometry sweep", ""]
    L.append(f"Combinations tested: **{result['n_tests']}** "
             f"({len(result['variants'])} variants x {len(result['coins'])} coins x "
             f"{len(result['timeframes'])} timeframes, pooled + per-market).")
    L.append(f"Metric: favourable-bounce rate at {h} bars vs a matched control "
             f"(random band / random slopes). Edge = bounce_event − bounce_control.")
    L.append(f"Multiple-testing: naive bar p<0.05; Bonferroni bar "
             f"p<{result['bonferroni_alpha']:.4f}. A pattern must clear the corrected bar.")
    L.append("")

    L.append("## Pooled across all coins (the powerful test)")
    L.append("")
    L.append("| variant | tf | events | bounce | control | edge | 95% CI | p | verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in result["pooled"]:
        L.append(_line(r, result["bonferroni_alpha"]))
    L.append("")

    survivors = [r for r in result["per_market"]
                 if _survivor(r, 0.05)]   # per-market gated on CI (no perm p)
    L.append("## Per-market hits (CI excludes 0, n>=200)")
    L.append("")
    if not survivors:
        L.append("_None._ No single coin/timeframe/variant shows a bounce edge whose "
                 "bootstrap 95% CI clears zero. With this many combinations, finding "
                 "nothing is itself a strong (negative) result.")
    else:
        L.append("| variant | coin | tf | events | bounce | control | edge | 95% CI |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in survivors:
            L.append(f"| {r['variant']} | {r['coin']} | {r['tf']} | {r['n_ev']} | "
                     f"{r['bounce_ev']*100:.1f}% | {r['bounce_ctrl']*100:.1f}% | "
                     f"{r['edge']*100:+.1f}pp | [{r['ci_low']*100:+.1f}, {r['ci_high']*100:+.1f}] |")
        L.append("")
        L.append(f"> {len(survivors)} per-market hit(s) out of {len(result['per_market'])} "
                 f"tests. At p<0.05 you'd expect ~{0.05*len(result['per_market']):.0f} false "
                 f"positives by chance alone — treat these as leads to retest out-of-sample, "
                 f"not discoveries.")
    L.append("")

    L.append("## Verdict")
    pooled_survivors = [r for r in result["pooled"]
                        if _survivor(r, result["bonferroni_alpha"])]
    if pooled_survivors:
        names = ", ".join(f"{r['variant']} {r['tf']}" for r in pooled_survivors)
        L.append(f"SURVIVED correction: {names}. Pooled across all coins these beat their "
                 f"control with a CI excluding zero AND p below the Bonferroni bar. Worth a "
                 f"dedicated out-of-sample backtest before believing it.")
    else:
        L.append("NO pattern survived multiple-testing correction. Across 6 coins, 2 "
                 "timeframes and 5 geometry variants, none produced a bounce edge robust "
                 "enough to clear the Bonferroni bar. The wizard robe stays in the closet — "
                 "exactly the outcome the controls exist to enforce.")
        # surface the strongest sub-threshold lead so it isn't buried
        leads = [r for r in result["pooled"]
                 if r["n_ev"] >= MIN_EVENTS and r["ci_low"] > 0]
        if leads:
            best = max(leads, key=lambda r: r["edge"])
            L.append("")
            L.append(f"Closest lead: **{best['variant']} {best['tf']}** — "
                     f"+{best['edge']*100:.1f}pp bounce vs control, CI "
                     f"[{best['ci_low']*100:+.1f}, {best['ci_high']*100:+.1f}], p={best['p']:.3f} "
                     f"over {best['n_ev']} events. It clears the naive bar but not the "
                     f"corrected one, and a {best['edge']*100:.1f}pp edge on a bounce *rate* "
                     f"is unlikely to survive trading costs (bounce direction is not profit). "
                     f"A real follow-up would be a dedicated out-of-sample backtest of this "
                     f"single hypothesis — not another sweep.")
    L.append("")
    return "\n".join(L)


def _line(r, alpha):
    if r["n_ev"] < MIN_EVENTS:
        verdict = "thin"
    elif _survivor(r, alpha):
        verdict = "**SURVIVES**"
    elif r["ci_low"] > 0:
        verdict = "naive-only"
    else:
        verdict = "no edge"
    p = f"{r['p']:.3f}" if not np.isnan(r["p"]) else "—"
    return (f"| {r['variant']} | {r['tf']} | {r['n_ev']} | {r['bounce_ev']*100:.1f}% | "
            f"{r['bounce_ctrl']*100:.1f}% | {r['edge']*100:+.1f}pp | "
            f"[{r['ci_low']*100:+.1f}, {r['ci_high']*100:+.1f}] | {p} | {verdict} |")
