"""Data download and preprocessing helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .datasets import CANDLESTICK_COLUMNS, ensure_datetime_index


@dataclass
class DataDownloadConfig:
    """Configuration for downloading OHLCV candles via yfinance."""

    symbol: str
    start: str
    end: str
    interval: str = "1h"
    auto_adjust: bool = True


def download_candles(config: DataDownloadConfig) -> pd.DataFrame:
    """Download OHLCV candles using yfinance."""

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - exercised in runtime environments
        raise RuntimeError("yfinance is required to download market data") from exc

    data = yf.download(
        tickers=config.symbol,
        start=config.start,
        end=config.end,
        interval=config.interval,
        auto_adjust=config.auto_adjust,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise ValueError("No data returned for the given symbol/timeframe")

    frame = data.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "close",
            "Volume": "volume",
        }
    ).reset_index()

    datetime_col = next((col for col in ("Datetime", "Date", "datetime") if col in frame.columns), None)
    if datetime_col is None:
        raise ValueError("Downloaded data does not contain a datetime column")
    frame = frame.rename(columns={datetime_col: "datetime"})
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)

    for column in CANDLESTICK_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame.loc[:, ["datetime", *CANDLESTICK_COLUMNS]].dropna()
    return frame


def resample_timeframe(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample an OHLCV dataframe to a coarser timeframe."""

    indexed = ensure_datetime_index(frame)
    aggregated = indexed.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna()
    aggregated = aggregated.reset_index()
    aggregated.rename(columns={"datetime": "datetime"}, inplace=True)
    return aggregated


def save_dataframe(frame: pd.DataFrame, output: Path, float_precision: Optional[int] = 6) -> None:
    """Persist a dataframe to disk as CSV, creating parent directories as needed."""

    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, float_format=f"%.{float_precision}f" if float_precision else None)
