# Sacred Geometry Bot

A **research machine** that tests whether sacred-geometry-derived price levels
produce measurable edge, or are just beautiful nonsense. It can later trade on
testnet — but only after the research engine proves an idea deserves it.

> The bot has three personalities: **Historian** (harvest candles/pivots/events),
> **Scientist** (test geometry against random/control baselines), and **Trader**
> (testnet/paper only, gated behind `ALLOW_LIVE_TRADING`). This MVP ships the
> Historian and the Scientist. The Trader is intentionally not wired up yet.

## What's built (MVP — Phases 1–7)

- **Data** — CCXT Binance candle harvest → Parquet + DuckDB (`src/data`)
- **Market structure** — causal features (ATR, vol, regimes) + fractal & ZigZag
  pivots with explicit confirmation delay, so geometry stays lookahead-free
  (`src/market_structure`)
- **Geometry** — Fibonacci retracement + golden-pocket event detection
  (`src/geometry`)
- **Controls** — random zones, fixed 0.5/0.7 zones, scrambled ratios, random
  entries (`src/signals/controls.py`)
- **Research** — outcome labeller (directional forward returns, MFE/MAE, TP/SL
  resolution), bootstrap + permutation stats, honest markdown report
  (`src/research`)
- **Tests** — pivots, Fibonacci, outcome labelling (`tests/`)

## Quick start

```bash
# one-time setup: create the venv (Python 3.11) and install deps
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

# run the headline experiment fully OFFLINE (no keys, no network):
.venv/Scripts/python.exe -m src.main experiment001 --synthetic

# run against real Binance data (public, keyless):
.venv/Scripts/python.exe -m src.main backfill
.venv/Scripts/python.exe -m src.main experiment001
```

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

Phase 8 backtest (vectorbt), Phase 9 Binance testnet execution + risk manager
(gated by `ENVIRONMENT=testnet` and `ALLOW_LIVE_TRADING=false`), Phase 10
dashboard + "Myth Detector". New geometry modules (Gann, Square of Nine,
harmonics, time cycles, circles, vortex) each implement the same
`GeometryModule` interface and reuse the controls + labeller unchanged.

## Project layout

```
config/        settings, symbols, experiment definitions (YAML)
src/data/      exchange client, collector, parquet+duckdb store, synthetic gen
src/market_structure/  features, pivots
src/geometry/  base interface, fibonacci
src/signals/   controls / baselines
src/research/  outcome labeller, stats, reports
src/main.py    CLI: backfill | observe | experiment001
tests/         pytest suite
```
