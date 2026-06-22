"""Storage: candles as partitioned Parquet, plus a DuckDB research database that
holds candles, pivots, geometry_events and event_outcomes for SQL queries."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def _safe(symbol: str) -> str:
    return symbol.replace("/", "_")


class CandleStore:
    def __init__(self, raw_dir: Path, duckdb_path: Path):
        self.raw_dir = Path(raw_dir)
        self.duckdb_path = Path(duckdb_path)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # --- parquet ----------------------------------------------------------
    def _parquet_path(self, symbol: str, timeframe: str) -> Path:
        return self.raw_dir / f"{_safe(symbol)}__{timeframe}.parquet"

    def upsert_candles(self, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        """Merge new candles into the parquet file, de-duped on timestamp."""
        if df.empty:
            return 0
        df = df.copy()
        df["symbol"] = symbol
        df["timeframe"] = timeframe
        path = self._parquet_path(symbol, timeframe)
        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
        df = (df.drop_duplicates(subset=["timestamp"], keep="last")
                .sort_values("timestamp")
                .reset_index(drop=True))
        df.to_parquet(path, index=False)
        return len(df)

    def load_candles(self, symbol: str, timeframe: str) -> pd.DataFrame:
        path = self._parquet_path(symbol, timeframe)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)

    # --- duckdb -----------------------------------------------------------
    def connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.duckdb_path))

    def _init_schema(self) -> None:
        con = self.connect()
        con.execute("""
            CREATE TABLE IF NOT EXISTS pivots (
                pivot_id TEXT PRIMARY KEY, symbol TEXT, timeframe TEXT,
                timestamp TIMESTAMP, pivot_type TEXT, price DOUBLE,
                strength INTEGER, left_bars INTEGER, right_bars INTEGER,
                atr DOUBLE, confirmed_at TIMESTAMP, method TEXT
            );
            CREATE TABLE IF NOT EXISTS geometry_events (
                event_id TEXT PRIMARY KEY, symbol TEXT, timeframe TEXT,
                timestamp TIMESTAMP, geometry_type TEXT, geometry_subtype TEXT,
                direction TEXT, level_price DOUBLE, current_price DOUBLE,
                distance_pct DOUBLE, distance_atr DOUBLE, anchor_data JSON,
                confluence_score DOUBLE, metadata JSON, created_at TIMESTAMP,
                is_control BOOLEAN, control_kind TEXT, bar_index INTEGER
            );
            CREATE TABLE IF NOT EXISTS event_outcomes (
                event_id TEXT PRIMARY KEY, return_1 DOUBLE, return_3 DOUBLE,
                return_5 DOUBLE, return_10 DOUBLE, return_20 DOUBLE, return_50 DOUBLE,
                mfe_10 DOUBLE, mae_10 DOUBLE, mfe_20 DOUBLE, mae_20 DOUBLE,
                hit_target BOOLEAN, hit_stop BOOLEAN, bars_evaluated INTEGER,
                labeled_at TIMESTAMP
            );
        """)
        con.close()

    def write_df(self, table: str, df: pd.DataFrame, replace_keys: list[str] | None = None) -> None:
        """Insert a DataFrame into a duckdb table, replacing rows on key collision."""
        if df.empty:
            return
        con = self.connect()
        con.register("incoming", df)
        cols = [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]
        usable = [c for c in cols if c in df.columns]
        if replace_keys:
            keys = ", ".join(replace_keys)
            con.execute(
                f"DELETE FROM {table} WHERE ({keys}) IN (SELECT {keys} FROM incoming)"
            )
        collist = ", ".join(usable)
        con.execute(f"INSERT INTO {table} ({collist}) SELECT {collist} FROM incoming")
        con.unregister("incoming")
        con.close()
