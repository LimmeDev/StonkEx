"""CLI to download and preprocess candlestick data."""
from __future__ import annotations

import argparse
from pathlib import Path

from stonkex.data.preprocessing import DataDownloadConfig, download_candles, resample_timeframe, save_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and preprocess stock data")
    parser.add_argument("symbol", help="Ticker symbol, e.g. AAPL")
    parser.add_argument("start", help="Start date, e.g. 2020-01-01")
    parser.add_argument("end", help="End date, e.g. 2024-01-01")
    parser.add_argument("--interval", default="1h", help="Download interval (default: 1h)")
    parser.add_argument("--resample", default=None, help="Optional pandas resample rule (e.g. '4H')")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw.csv"),
        help="Path to save the processed dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DataDownloadConfig(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        interval=args.interval,
    )
    frame = download_candles(config)
    if args.resample:
        frame = resample_timeframe(frame, args.resample)
    save_dataframe(frame, args.output)
    print(f"Saved dataset to {args.output}")


if __name__ == "__main__":
    main()
