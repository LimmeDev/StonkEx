# StonkEx

StonkEx is a GPU-optimized research sandbox for training transformer-based models on candlestick (OHLCV) data and visualizing forecasts in real time. It provides:

- **Data pipeline** for downloading, resampling, and windowing stock candles.
- **PyTorch training stack** with automatic mixed precision tuned for NVIDIA RTX 4080 GPUs and Intel i9-class CPUs.
- **Streamlit dashboards** to monitor training metrics and interactively preview model forecasts over translucent candlestick overlays.
- **Rich instrumentation** including JSONL logs, checkpointing, early stopping, and reproducible hyperparameter capture.

## Project structure

```
stonkex/
├── config.py               # Experiment configuration dataclasses
├── data/                   # Dataset + preprocessing utilities
├── inference/              # Model loading helpers
├── models/                 # Transformer architecture
├── training/               # Trainer, callbacks, utilities
├── visualization/          # Plotly candlestick helpers
app/
├── dashboard.py            # Training monitor UI (Streamlit)
├── predictor.py            # Forecast visualization UI (Streamlit)
scripts/
├── prepare_data.py         # Download & preprocess candles (yfinance)
├── train.py                # Main training CLI
├── evaluate.py             # Offline evaluation script
tests/                      # Pytest-based smoke tests
```

## Installation

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]
   ```

> **Tip:** For maximum throughput on an RTX 4080 Super, install the latest CUDA-enabled PyTorch wheel from https://pytorch.org/get-started/locally/.

## Data preparation

Use `scripts/prepare_data.py` to download OHLCV candles via *yfinance*:

```bash
python scripts/prepare_data.py AAPL 2020-01-01 2024-01-01 --interval 1h --output data/aapl_1h.csv
```

The script can optionally resample to a coarser interval using the `--resample` argument.

## Training

Launch training with GPU-optimized defaults:

```bash
python scripts/train.py \
  --data-path data/aapl_1h.csv \
  --input-window 192 \
  --horizon 24 \
  --batch-size 256 \
  --epochs 150 \
  --d-model 512 \
  --nhead 8 \
  --ff-dim 1024
```

### CPU / Hugging Face Spaces (zero GPU) tips

When running inside a CPU-only environment such as a zero-GPU Gradient Space on Hugging Face, use the CPU execution mode so the script automatically disables CUDA-only features and right-sizes batch sizes and dataloader workers:

```bash
python scripts/train.py \
  --data-path data/aapl_1h.csv \
  --device cpu \
  --batch-size 64 \
  --num-workers 0 \
  --mixed-precision False \
  --d-model 256 \
  --ff-dim 512 \
  --epochs 50
```

The trainer will automatically cap the batch size at 64 for CPU runs and skip CUDA AMP. On Spaces, set `STREAMLIT_SERVER_ADDRESS=0.0.0.0` and `STREAMLIT_SERVER_PORT=7860` before launching the Streamlit apps to expose them to the public interface.

Key features of the training pipeline:

- Automatic mixed precision (AMP) enabled by default (`--no-mixed-precision` to disable).
- Cosine annealing learning-rate scheduler with AdamW optimizer tuned for transformer workloads.
- Gradient clipping, gradient accumulation, and checkpointing under `artifacts/checkpoints/`.
- JSONL training logs at `artifacts/logs/training_log.jsonl` and hyperparameter snapshots at `artifacts/hparams.json`.
- Final artifact (`artifacts/final_model.pt`) bundles the model state, optimizer, scheduler, and normalization statistics for deployment.

Resume from a checkpoint using `--resume artifacts/checkpoints/epoch_XXXX.pt`.

## Monitoring UI

Start the Streamlit dashboard to watch training progress:

```bash
streamlit run app/dashboard.py
```

Select the artifact directory (`artifacts/` by default) and click **Refresh data** to update loss curves, live metrics, and hyperparameters.

## Forecast visualization UI

After training, preview translucent forecasts over candlesticks:

```bash
streamlit run app/predictor.py
```

Upload the generated `artifacts/final_model.pt` along with a CSV containing recent OHLCV candles. Adjust the input window and horizon to align with training settings. The app renders:

- A translucent dashed overlay representing the model’s predicted price trajectory.
- Candlestick chart of the recent history and (if available) realized future prices.
- Downloadable JSON with the forecast series for downstream integrations.

## Evaluation

Compute validation MSE for a saved checkpoint:

```bash
python scripts/evaluate.py --data-path data/aapl_1h.csv --model-path artifacts/final_model.pt
```

## Testing

Run the automated smoke tests:

```bash
pytest
```

These tests ensure dataset windowing, transformer outputs, and the trainer loop operate as expected.

## Troubleshooting & debugging

- Training emits verbose logs; adjust logging verbosity via `STONKEX_LOG_LEVEL` environment variable.
- `artifacts/logs/training_log.jsonl` can be tailed for live debugging or loaded into pandas for analysis.
- Checkpoint files store optimizer + scheduler states for full recovery.
- For GPU memory overflows, reduce `--batch-size` or `--d-model`; for CPU-bound dataloading, tweak `--num-workers`.

## License

This repository is provided as-is for experimentation.
