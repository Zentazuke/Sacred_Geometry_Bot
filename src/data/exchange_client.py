"""Thin CCXT wrapper. Market data is keyless; sandbox mode is set immediately
after construction (CCXT requires set_sandbox_mode before any other call)."""
from __future__ import annotations

import os
from typing import Optional

import ccxt


# CCXT timeframe -> milliseconds, for gap detection and pagination.
TIMEFRAME_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


def timeframe_ms(tf: str) -> int:
    try:
        return TIMEFRAME_MS[tf]
    except KeyError as exc:
        raise ValueError(f"Unsupported timeframe: {tf}") from exc


class ExchangeClient:
    """Wraps a single ccxt exchange instance."""

    def __init__(self, name: str = "binance", sandbox: bool = True, trading: bool = False):
        cls = getattr(ccxt, name)
        params: dict = {"enableRateLimit": True}
        if trading:
            # keys only needed for placing orders; never for market data.
            params["apiKey"] = os.getenv("BINANCE_TESTNET_API_KEY", "")
            params["secret"] = os.getenv("BINANCE_TESTNET_API_SECRET", "")
        self.exchange = cls(params)
        if sandbox:
            # MUST be called immediately after construction, before any other call.
            self.exchange.set_sandbox_mode(True)
        self.name = name
        self.sandbox = sandbox

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: Optional[int] = None,
                    limit: int = 1000) -> list[list]:
        """Return raw ccxt OHLCV rows: [ts, open, high, low, close, volume]."""
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
