# Sacred Geometry Bot

A **research machine** that tests whether sacred-geometry-derived price levels
produce measurable edge, or are just beautiful nonsense. It can later trade on
testnet — but only after the research engine proves an idea deserves it.

> The bot has three personalities: **Historian** (harvest candles/pivots/events),
> **Scientist** (test geometry against random/control baselines), and **Trader**
> (testnet/paper only, gated behind `ALLOW_LIVE_TRADING`). This MVP ships the
> Historian and the Scientist. The Trader is intentionally not wired up yet.

## What's built (Phases 1–8)

- **Data** — CCXT Binance candle harvest → Parquet + DuckDB (`src/data`)
- **Market structure** — causal features (ATR, vol, regimes) + fractal & ZigZag
  pivots with explicit confirmation delay, so geometry stays lookahead-free
  (`src/market_structure`)
- **Geometry** — Fibonacci retracement + golden-pocket detection, and
  ATR-normalised **Gann angles** (fan from pivots, tested vs random slopes)
  (`src/geometry`)
- **Controls** — random zones, fixed 0.5/0.7 zones, scrambled ratios, random
  entries (`src/signals/controls.py`)
- **Research** — outcome labeller (directional forward returns, MFE/MAE, TP/SL
  resolution), bootstrap + permutation stats, honest markdown report
  (`src/research`)
- **Backtest (Phase 8)** — event-driven trade simulator with next-bar-open
  entries, fees + slippage, risk-normalised metrics (expectancy, profit factor,
  Sharpe/Sortino, Calmar, max drawdown), chronological walk-forward and
  in/out-of-sample split, vs a random-entry control and buy-and-hold
  (`src/backtest`)
- **Myth Detector sweep** — grids {coin × timeframe × geometry variant},
  scores each against a *pooled* random control (random bands / random slopes,
  averaged over many draws), with bootstrap CIs and a Bonferroni multiple-testing
  bar (`src/research/sweep.py`)
- **Tests** — pivots, Fibonacci, Gann, outcome labelling, backtest engine +
  metrics (`tests/`, 16 tests)

## Quick start

```bash
# one-time setup: create the venv (Python 3.11) and install deps
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

# run the headline experiment fully OFFLINE (no keys, no network):
.venv/Scripts/python.exe -m src.main experiment001 --synthetic

# run against real Binance data (public, keyless):
.venv/Scripts/python.exe -m src.main backfill            # ~1.4y of 1h, 5.5y of 4h
.venv/Scripts/python.exe -m src.main experiment001

# Phase 8 — backtest TRADES after fees + slippage (golden_pocket or gann):
.venv/Scripts/python.exe -m src.main backtest --geometry gann
# improved trend-following rule (trend filter + wide stops + let winners run):
.venv/Scripts/python.exe -m src.main backtest --geometry gann --trend-filter \
    --confirm --atr-stop-floor 4 --target-r 4 --max-hold 200 --min-rr 2

# Myth Detector — sweep every coin x timeframe x geometry variant:
.venv/Scripts/python.exe -m src.main sweep
```

### Can the rule be improved? Yes — but the geometry still adds nothing

The naive rule loses (−0.59R). Adding the plan's own improvements — **trend filter,
rejection-confirmation, ATR-floored wide stops, and big R-multiple targets (let
winners run)** — walks expectancy all the way to **positive** (+0.05R golden
pocket, +0.07R gann, PF ~1.1, 4/6 coins green). Tempting!

But the decisive control kills it: apply the **same rules to random entries** and
they do just as well (+0.05R). Golden pocket is actually *worse* than random
(−0.007R edge); gann's +0.017R edge **collapses out-of-sample** (+0.13R →
+0.015R). So the positive expectancy is generic **trend-following risk
management**, not the geometry. The levels never earned their keep — exactly what
the controls exist to prove. (`src/signals/signal_engine.py`)

### Myth Detector findings (6 coins: BTC/ETH/SOL/DOGE/LINK/BNB, 1h + 4h, 1.4–5.5y)

> **No geometry survived multiple-testing correction.** 70 combinations were
> scored against pooled random controls.
> - **Fibonacci is dead** — golden pocket and every other retracement band show
>   no bounce edge vs random bands, on any coin or timeframe.
> - **Gann's 1h fan is the one near-miss**: +1.3pp favourable-bounce edge vs
>   *random slopes*, pooled across all coins (CI [+0.5,+2.1], p=0.001, n≈21k),
>   consistent across ETH/LINK individually. It clears the naive 5% bar but
>   **not** the Bonferroni bar (0.0007), only appears on 1h (4h: nothing), and a
>   1.3pp edge on a bounce *rate* won't survive trading costs — it's a lead to
>   retest out-of-sample, not a tradable edge.
>
> A cautionary tale lives in the git history: an early single-draw control
> flagged a fake +3.5pp "hit" on SOL 1h that **evaporated** once the control was
> averaged over many random draws. Control quality is everything.

### What the backtest found (BTC/ETH/SOL, 1h + 4h, ~1.4–5.5y, 0.30% round-trip cost)

> **NOT TRADABLE.** The naive golden-pocket rule (enter the 0.618–0.786 pocket,
> stop at the swing origin, target the swing terminal) returns **≈ −0.77R per
> trade after costs** — *worse* than random entries (−0.26R) and crushed by
> buy-and-hold (+790%). Stops hit 67% of the time; filtering for higher
> reward:risk makes it *worse* (deep-retracement setups are reversal traps).
> This is the expected fate of most sacred ideas — and exactly why we backtest
> before risking anything. Reports land in `data/reports/`.

Note: the pooled total-return / equity-curve figures assume one account trading
all six markets sequentially, so per-trade **expectancy** is the robust statistic
to compare on, not the headline total return.

The report lands in `data/reports/EXP_001_GOLDEN_POCKET.md` and prints to stdout.

### Run the tests

```bash
.venv/Scripts/python.exe -m pytest -q
```

## Experiment 001 — Does the golden pocket matter?

Hypothesis: price entering the **0.618–0.786** retracement zone of a confirmed
swing bounces more often than at random / scrambled zones. The report compares
the golden pocket's bounce rate against every control with a bootstrap 95% CI and
a permutation p-value, and prints a blunt verdict:

- **INSUFFICIENT DATA** — fewer than 200 evaluated events
- **PROMISING** — beats every control, CI excludes zero → worth backtesting
- **NOT PROVEN** — does not clear the controls (the expected fate of most sacred ideas)

## Lookahead safety (non-negotiable)

- Only closed candles are stored (`_drop_open_candle`).
- Pivots carry `confirmed_index`; legs are only scanned **after** the terminal
  pivot is confirmed.
- Event reference price is the **close** of the entry bar.
- Features are causal (row `i` uses only rows `<= i`).

## Roadmap (next phases)

Phase 9 Binance testnet execution + risk manager (gated by `ENVIRONMENT=testnet`
and `ALLOW_LIVE_TRADING=false`), Phase 10 dashboard + "Myth Detector". New
geometry modules (Gann, Square of Nine, harmonics, time cycles, circles, vortex)
each implement the same `GeometryModule` interface and reuse the controls,
labeller, and backtest engine unchanged.

## Project layout

```
config/        settings, symbols, experiment definitions (YAML)
src/data/      exchange client, collector, parquet+duckdb store, synthetic gen
src/market_structure/  features, pivots
src/geometry/  base interface, fibonacci, gann
src/signals/   controls / baselines
src/research/  outcome labeller, stats, reports, sweep (myth detector)
src/backtest/  trade engine, metrics, walk-forward, runner
src/main.py    CLI: backfill | observe | experiment001 | backtest | sweep
tests/         pytest suite
```
