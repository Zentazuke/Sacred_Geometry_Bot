"""Candle harvesting: paginated backfill + latest-closed fetch. Returns tidy
DataFrames with a tz-aware UTC `timestamp` column. Only closed candles are kept."""
from __future__ import annotations

import time
from typing import Optional

import pandas as pd

from .exchange_client import ExchangeClient, timeframe_ms

CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _rows_to_df(rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop(columns=["ts"])
    return df[CANDLE_COLUMNS]


class CandleCollector:
    def __init__(self, client: ExchangeClient):
        self.client = client

    def backfill(self, symbol: str, timeframe: str, limit: int = 5000,
                 page: int = 1000) -> pd.DataFrame:
        """Walk forward from (now - limit*tf) until caught up, page by page."""
        tf_ms = timeframe_ms(timeframe)
        now_ms = self.client.exchange.milliseconds()
        since = now_ms - limit * tf_ms
        out: list[list] = []
        while since < now_ms:
            rows = self.client.fetch_ohlcv(symbol, timeframe, since=since, limit=page)
            if not rows:
                break
            out.extend(rows)
            last = rows[-1][0]
            if last <= since:  # no progress -> stop to avoid an infinite loop
                break
            since = last + tf_ms
            time.sleep(self.client.exchange.rateLimit / 1000)
        df = _rows_to_df(out).drop_duplicates(subset="timestamp").sort_values("timestamp")
        return self._drop_open_candle(df, timeframe)

    def fetch_latest_closed(self, symbol: str, timeframe: str, lookback: int = 200) -> pd.DataFrame:
        rows = self.client.fetch_ohlcv(symbol, timeframe, since=None, limit=lookback)
        df = _rows_to_df(rows).sort_values("timestamp")
        return self._drop_open_candle(df, timeframe)

    @staticmethod
    def _drop_open_candle(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Drop the still-forming candle: keep only bars whose close time has passed."""
        if df.empty:
            return df.reset_index(drop=True)
        tf = pd.Timedelta(milliseconds=timeframe_ms(timeframe))
        now = pd.Timestamp.now("UTC")
        closed = df[df["timestamp"] + tf <= now]
        return closed.reset_index(drop=True)
