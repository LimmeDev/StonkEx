# Deploying StonkEx on Hugging Face Spaces

This guide covers how to launch training from a Hugging Face Space, both on CPU-only ("Zero GPU") instances and GPU-backed Gradient tiers.

## 1. Prepare the repository

1. Push this repository to a Hugging Face Space or fork an existing Space and upload the files.
2. Ensure `requirements.txt` and `pyproject.toml` are present so the build system installs dependencies automatically.
3. Optionally add a `start.sh` (see below) to automate multi-step workflows.

## 2. CPU / Zero GPU mode

If you are limited to CPU-only hardware, use conservative defaults:

```bash
python scripts/prepare_universe.py 2018-01-01 2024-01-01 \
  --interval 1h \
  --output-dir data/universe \
  --max-bytes 1.0

python scripts/train.py \
  --data-path data/universe \
  --device cpu \
  --batch-size 64 \
  --num-workers 0 \
  --mixed-precision False \
  --epochs 50
```

The training script automatically right-sizes batch sizes and disables CUDA-only features. You can expose the monitoring dashboard by running `streamlit run app/dashboard.py` in a second process (Spaces will serve whatever script is defined in `app.py` or `start.sh`).

## 3. GPU mode (recommended)

1. In the Space UI, navigate to **Settings → Hardware** and select a GPU tier (e.g., A10G or T4). Save the change so future builds attach the GPU.
2. Update or create `start.sh` so Spaces runs the workflow automatically. Example:

```bash
#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_universe.py 2015-01-01 2024-01-01 \
  --interval 1h \
  --output-dir data/universe \
  --max-bytes 2.0

python scripts/train.py \
  --data-path data/universe \
  --device cuda \
  --require-gpu \
  --epochs 120 \
  --checkpoint-every 5
```

3. Commit the script and make sure it has executable permissions (`chmod +x start.sh`).
4. Redeploy the Space; the logs will show the "Using device: CUDA:…" banner emitted by `scripts/train.py`. The run aborts if CUDA is missing and `--require-gpu` is set, preventing accidental CPU deployments.

## 4. Remote assistance limitations

The assistant cannot directly connect to or modify your Hugging Face Space. To let someone else update the code, grant them **Write** access to the Space through the Hugging Face UI or share a repository URL they can push to. If you provide a VS Code remote URL, ensure it is active during the session and exposes the repository root; no additional credentials can be handled here.

## 5. Monitoring and troubleshooting

- Use the Streamlit dashboard (`app/dashboard.py`) to track training losses in real time. Run it in a separate Space or process.
- Artifacts, checkpoints, and JSONL logs are written under `artifacts/`. Persist this directory by enabling the Space's persistent storage or periodically syncing to the Hub.
- If runs stop unexpectedly, review the Space build logs and confirm the GPU quota has not been exceeded.

Following these steps keeps GPU-backed training predictable while still allowing CPU fallbacks when necessary.
