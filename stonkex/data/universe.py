"""Helpers to build multi-symbol candlestick datasets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from .datasets import CANDLESTICK_COLUMNS
from .preprocessing import DataDownloadConfig, download_candles, resample_timeframe, save_dataframe


# 70 heavily traded US equities across multiple sectors.
DEFAULT_SYMBOLS: List[str] = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "NVDA",
    "JPM",
    "V",
    "MA",
    "HD",
    "DIS",
    "NFLX",
    "KO",
    "PEP",
    "PFE",
    "MRK",
    "UNH",
    "VZ",
    "T",
    "INTC",
    "AMD",
    "CSCO",
    "ADBE",
    "CRM",
    "ORCL",
    "IBM",
    "QCOM",
    "TXN",
    "AVGO",
    "BA",
    "LMT",
    "CAT",
    "GE",
    "HON",
    "MMM",
    "UPS",
    "FDX",
    "NKE",
    "MCD",
    "SBUX",
    "COST",
    "WMT",
    "LOW",
    "TGT",
    "CVS",
    "WBA",
    "XOM",
    "CVX",
    "SLB",
    "COP",
    "OXY",
    "SPG",
    "PLD",
    "AMT",
    "NEE",
    "DUK",
    "SO",
    "D",
    "AEP",
    "BK",
    "GS",
    "MS",
    "AXP",
    "C",
    "TFC",
    "USB",
    "SCHW",
    "BLK",
]


@dataclass
class UniverseBuildResult:
    """Summary of the downloaded universe."""

    symbols: list[str]
    paths: list[Path]
    total_rows: int
    total_bytes: int


def load_symbol_universe(
    symbols: Optional[Sequence[str]] = None, *, symbols_file: Optional[Path] = None, min_symbols: int = 1
) -> list[str]:
    """Return a unique, upper-case list of ticker symbols to download."""

    collected: list[str] = []
    if symbols_file:
        if not symbols_file.exists():
            raise FileNotFoundError(symbols_file)
        with symbols_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                token = line.strip()
                if token:
                    collected.append(token)
    if symbols:
        collected.extend(symbols)
    if not collected:
        collected = DEFAULT_SYMBOLS.copy()
    cleaned = []
    seen = set()
    for symbol in collected:
        upper = symbol.strip().upper()
        if not upper:
            continue
        if upper in seen:
            continue
        seen.add(upper)
        cleaned.append(upper)
    if len(cleaned) < min_symbols:
        raise ValueError(
            f"Requested universe has {len(cleaned)} symbols which is below the required minimum of {min_symbols}"
        )
    return cleaned


def download_symbol_universe(
    symbols: Sequence[str],
    *,
    start: str,
    end: str,
    interval: str,
    output_dir: Path,
    resample: Optional[str] = None,
    float_precision: Optional[int] = 6,
    max_bytes: int = 1024 ** 3,
    resume: bool = True,
) -> UniverseBuildResult:
    """Download OHLCV candles for many symbols ensuring the dataset stays within the byte budget."""

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    total_bytes = 0
    total_rows = 0

    for symbol in symbols:
        target_path = output_dir / f"{symbol}_{interval}.csv"
        if resume and target_path.exists():
            size = target_path.stat().st_size
            total_bytes += size
            saved_paths.append(target_path)
            df = pd.read_csv(target_path)
            total_rows += len(df)
            continue

        config = DataDownloadConfig(symbol=symbol, start=start, end=end, interval=interval)
        frame = download_candles(config)
        if resample:
            frame = resample_timeframe(frame, resample)
        frame["symbol"] = symbol
        save_dataframe(frame, target_path, float_precision=float_precision)
        size = target_path.stat().st_size
        total_bytes += size
        total_rows += len(frame)
        if total_bytes > max_bytes:
            raise RuntimeError(
                f"Aborting download – combined CSV size {total_bytes} bytes exceeds the configured limit of {max_bytes} bytes"
            )
        saved_paths.append(target_path)

    return UniverseBuildResult(symbols=list(symbols), paths=saved_paths, total_rows=total_rows, total_bytes=total_bytes)


def load_universe_dataframe(paths: Iterable[Path]) -> pd.DataFrame:
    """Load a collection of CSV files into a single sorted dataframe."""

    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        if "symbol" not in frame.columns:
            frame["symbol"] = path.stem.split("_")[0].upper()
        missing = [col for col in ("datetime", *CANDLESTICK_COLUMNS) if col not in frame.columns]
        if missing:
            raise ValueError(f"File {path} is missing columns: {missing}")
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["datetime"])
        frames.append(frame)
    if not frames:
        raise ValueError("No CSV files provided to load_universe_dataframe")
    combined = pd.concat(frames, ignore_index=True)
    combined["symbol"] = combined["symbol"].astype(str).str.upper()
    combined = combined.dropna(subset=CANDLESTICK_COLUMNS)
    combined = combined.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    return combined
