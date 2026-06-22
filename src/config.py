"""Configuration loading: YAML files + .env, with a single typed accessor."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional; env vars still work without it
    def load_dotenv(*_a, **_k):  # type: ignore
        return False

# Project root = parent of the directory holding this file's package (src/).
ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"


def _read_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass
class Settings:
    """Merged view of settings.yaml + symbols.yaml + .env, with paths resolved."""

    raw: dict[str, Any] = field(default_factory=dict)
    symbols_cfg: dict[str, Any] = field(default_factory=dict)
    experiments: dict[str, Any] = field(default_factory=dict)

    # --- convenience accessors -------------------------------------------
    @property
    def mode(self) -> str:
        return self.raw.get("mode", "observer")

    @property
    def exchange_name(self) -> str:
        return self.raw.get("exchange", {}).get("name", "binance")

    @property
    def sandbox(self) -> bool:
        return bool(self.raw.get("exchange", {}).get("sandbox", True))

    @property
    def symbols(self) -> list[str]:
        return list(self.symbols_cfg.get("symbols", []))

    @property
    def timeframes(self) -> list[str]:
        return list(self.symbols_cfg.get("timeframes", []))

    @property
    def backfill_limit(self) -> int:
        return int(self.symbols_cfg.get("backfill_limit", 5000))

    @property
    def pivots(self) -> dict[str, Any]:
        return self.raw.get("pivots", {})

    def path(self, key: str) -> Path:
        """Resolve a storage path from settings against the project root."""
        rel = self.raw.get("storage", {})[key]
        p = ROOT / rel
        return p

    @property
    def allow_live_trading(self) -> bool:
        # Two gates must BOTH be true; default false. The env var wins as a kill switch.
        cfg = bool(self.raw.get("safety", {}).get("allow_live_trading", False))
        env = os.getenv("ALLOW_LIVE_TRADING", "false").strip().lower() in {"1", "true", "yes"}
        return cfg and env


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    return Settings(
        raw=_read_yaml("settings.yaml"),
        symbols_cfg=_read_yaml("symbols.yaml"),
        experiments=_read_yaml("experiments.yaml"),
    )
