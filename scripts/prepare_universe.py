"""Download a multi-symbol OHLCV universe ready for training."""
from __future__ import annotations

import argparse
from pathlib import Path

from stonkex.data import (
    DEFAULT_SYMBOLS,
    download_symbol_universe,
    load_symbol_universe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OHLCV data for many tickers at once")
    parser.add_argument("start", help="Start date, e.g. 2015-01-01")
    parser.add_argument("end", help="End date, e.g. 2024-01-01")
    parser.add_argument("--interval", default="1h", help="Download interval (default: 1h)")
    parser.add_argument(
        "--resample",
        default=None,
        help="Optional pandas resample rule for coarser timeframes (e.g. '4H' or '1D')",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/universe"),
        help="Directory to store per-symbol CSV files",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Explicit space-separated ticker symbols. Overrides defaults when provided.",
    )
    parser.add_argument(
        "--symbols-file",
        type=Path,
        help="Path to a newline-delimited list of ticker symbols. Combined with --symbols when both are provided.",
    )
    parser.add_argument(
        "--min-symbols",
        type=int,
        default=70,
        help="Require at least this many tickers to satisfy the training universe.",
    )
    parser.add_argument(
        "--max-bytes",
        type=float,
        default=1.0,
        help="Maximum aggregate CSV size in gigabytes. Increase (<=3) for longer histories.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume logic and re-download data even if CSVs already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = load_symbol_universe(args.symbols, symbols_file=args.symbols_file, min_symbols=args.min_symbols)
    limit_bytes = int(min(max(args.max_bytes, 0.1), 3.0) * (1024**3))
    print(f"Preparing universe with {len(symbols)} tickers (limit: {limit_bytes / (1024**3):.2f} GB)")
    result = download_symbol_universe(
        symbols,
        start=args.start,
        end=args.end,
        interval=args.interval,
        output_dir=args.output_dir,
        resample=args.resample,
        max_bytes=limit_bytes,
        resume=not args.no_resume,
    )
    print("Downloaded files:")
    for path in result.paths:
        size_mb = path.stat().st_size / (1024**2)
        print(f"  {path} ({size_mb:.2f} MB)")
    print(
        "Summary: "
        f"{result.total_rows} rows across {len(result.symbols)} symbols, "
        f"total size {result.total_bytes / (1024**3):.2f} GB"
    )
    print("Default symbol universe:" if not args.symbols and not args.symbols_file else "Symbols used:")
    print(", ".join(result.symbols) if result.symbols else ", ".join(DEFAULT_SYMBOLS))


if __name__ == "__main__":
    main()
