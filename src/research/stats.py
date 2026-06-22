"""Statistics for comparing a sacred signal against controls. Honest, blunt:
bounce rates, mean forward returns, bootstrap CIs and a permutation p-value on
the difference. Nothing here cares whether the geometry is "beautiful"."""
from __future__ import annotations

import numpy as np
import pandas as pd


def group_summary(outcomes: pd.DataFrame, horizon: int = 10) -> dict:
    """Headline stats for one set of events' outcomes."""
    col = f"return_{horizon}"
    r = outcomes[col].dropna().values
    mfe = outcomes.get("mfe_10", pd.Series(dtype=float)).dropna().values
    mae = outcomes.get("mae_10", pd.Series(dtype=float)).dropna().values
    n = len(r)
    return {
        "events": int(len(outcomes)),
        "evaluated": int(n),
        "bounce_rate": float((r > 0).mean()) if n else float("nan"),
        "mean_return": float(r.mean()) if n else float("nan"),
        "median_return": float(np.median(r)) if n else float("nan"),
        "mfe_10": float(mfe.mean()) if len(mfe) else float("nan"),
        "mae_10": float(mae.mean()) if len(mae) else float("nan"),
        "hit_target_rate": float(outcomes["hit_target"].mean()) if "hit_target" in outcomes else float("nan"),
        "hit_stop_rate": float(outcomes["hit_stop"].mean()) if "hit_stop" in outcomes else float("nan"),
    }


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 5000,
                   seed: int = 7, statistic="mean") -> dict:
    """Bootstrap 95% CI for stat(a) - stat(b). statistic in {mean, bounce_rate}."""
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)

    def stat(x):
        return (x > 0).mean() if statistic == "bounce_rate" else x.mean()

    point = stat(a) - stat(b)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        sa = a[rng.integers(0, len(a), len(a))]
        sb = b[rng.integers(0, len(b), len(b))]
        diffs[k] = stat(sa) - stat(sb)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(point), "ci_low": float(lo), "ci_high": float(hi)}


def permutation_pvalue(a: np.ndarray, b: np.ndarray, n_perm: int = 5000,
                       seed: int = 11, statistic="mean") -> float:
    """Two-sided permutation test on the difference in the statistic."""
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)

    def stat(x):
        return (x > 0).mean() if statistic == "bounce_rate" else x.mean()

    observed = abs(stat(a) - stat(b))
    pool = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(stat(pool[:na]) - stat(pool[na:])) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def compare(golden_out: pd.DataFrame, control_out: pd.DataFrame,
            horizon: int = 10) -> dict:
    """Compare golden vs one control on the horizon return."""
    col = f"return_{horizon}"
    a = golden_out[col].values
    b = control_out[col].values
    boot_mean = bootstrap_diff(a, b, statistic="mean")
    boot_bounce = bootstrap_diff(a, b, statistic="bounce_rate")
    return {
        "mean_return_diff": boot_mean,
        "bounce_rate_diff": boot_bounce,
        "p_mean": permutation_pvalue(a, b, statistic="mean"),
        "p_bounce": permutation_pvalue(a, b, statistic="bounce_rate"),
        "beats_control": boot_bounce["ci_low"] > 0,   # CI on bounce-rate diff excludes 0
    }
