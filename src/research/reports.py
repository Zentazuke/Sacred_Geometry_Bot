"""Render a brutally honest research report from a signal vs its controls."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import stats


def _fmt_pct(x) -> str:
    try:
        return f"{x * 100:+.2f}%"
    except (TypeError, ValueError):
        return "  n/a"


def build_report(experiment_id: str, horizon: int,
                 golden_out: pd.DataFrame,
                 control_outs: dict[str, pd.DataFrame],
                 min_events: int = 200) -> str:
    lines: list[str] = []
    g = stats.group_summary(golden_out, horizon)

    lines.append(f"# {experiment_id} — research report")
    lines.append("")
    lines.append(f"Horizon for comparison: {horizon} bars\n")
    lines.append("## Golden pocket (sacred signal)")
    lines.append(f"- events: {g['events']}  (evaluated: {g['evaluated']})")
    lines.append(f"- bounce rate: {g['bounce_rate'] * 100:.1f}%")
    lines.append(f"- mean / median forward return: {_fmt_pct(g['mean_return'])} / {_fmt_pct(g['median_return'])}")
    lines.append(f"- avg MFE10 / MAE10: {_fmt_pct(g['mfe_10'])} / {_fmt_pct(g['mae_10'])}")
    lines.append(f"- hit target / hit stop: {g['hit_target_rate'] * 100:.1f}% / {g['hit_stop_rate'] * 100:.1f}%")
    lines.append("")

    lines.append("## Controls")
    lines.append("")
    header = "| control | events | bounce | mean ret | Δbounce vs golden (95% CI) | p | golden wins? |"
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|")

    all_beaten = True
    enough = g["evaluated"] >= min_events
    for name, out in control_outs.items():
        cs = stats.group_summary(out, horizon)
        cmp = stats.compare(golden_out, out, horizon)
        bd = cmp["bounce_rate_diff"]
        wins = cmp["beats_control"]
        all_beaten = all_beaten and wins
        ci = f"{bd['diff'] * 100:+.1f}pp [{bd['ci_low'] * 100:+.1f}, {bd['ci_high'] * 100:+.1f}]"
        lines.append(
            f"| {name} | {cs['evaluated']} | {cs['bounce_rate'] * 100:.1f}% | "
            f"{_fmt_pct(cs['mean_return'])} | {ci} | {cmp['p_bounce']:.3f} | "
            f"{'YES' if wins else 'no'} |"
        )

    lines.append("")
    lines.append("## Verdict")
    if not enough:
        verdict = (f"INSUFFICIENT DATA — only {g['evaluated']} evaluated events "
                   f"(need >= {min_events}). Harvest more candles before believing anything.")
    elif all_beaten:
        verdict = ("PROMISING — golden pocket beats every control's bounce rate with a "
                   "bootstrap 95% CI excluding zero. Worth backtesting (Phase 8).")
    else:
        verdict = ("NOT PROVEN — golden pocket does NOT clear every control out of sample. "
                   "Beautiful, but not (yet) tradable. This is the expected result for most "
                   "sacred ideas, and exactly why we measure.")
    lines.append(verdict)
    lines.append("")
    return "\n".join(lines)


def write_report(text: str, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
