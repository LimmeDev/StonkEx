"""Streamlit dashboard to monitor training progress in real time."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

DEFAULT_ARTIFACT_DIR = Path("artifacts")


def load_history(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        return pd.DataFrame(columns=["epoch", "step", "train_loss", "val_loss"])
    records = []
    with log_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        return pd.DataFrame(columns=["epoch", "step", "train_loss", "val_loss"])
    frame = pd.DataFrame(records)
    return frame.sort_values("epoch")


def load_hyperparameters(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        try:
            return json.load(fp)
        except json.JSONDecodeError:
            return {}


def main() -> None:
    st.set_page_config(page_title="StonkEx Training Dashboard", layout="wide")
    st.title("📈 StonkEx Training Monitor")

    artifact_dir = Path(st.sidebar.text_input("Artifact directory", value=str(DEFAULT_ARTIFACT_DIR)))
    log_path = artifact_dir / "logs" / "training_log.jsonl"
    hparams_path = artifact_dir / "hparams.json"

    if st.sidebar.button("Refresh data"):
        st.experimental_rerun()

    history = load_history(log_path)
    if history.empty:
        st.info("No training history found yet. Start `python scripts/train.py ...` to populate logs.")
        st.stop()

    latest = history.iloc[-1]
    col_epoch, col_train, col_val = st.columns(3)
    col_epoch.metric("Epoch", int(latest["epoch"]))
    col_train.metric("Train Loss", f"{latest['train_loss']:.6f}")
    if pd.notna(latest.get("val_loss")):
        col_val.metric("Validation Loss", f"{latest['val_loss']:.6f}")
    else:
        col_val.metric("Validation Loss", "N/A")

    chart_data = history.set_index("epoch")[["train_loss"]]
    if "val_loss" in history.columns:
        chart_data["val_loss"] = history.set_index("epoch")["val_loss"]
    st.line_chart(chart_data, height=320)

    st.subheader("Training Log")
    st.dataframe(history.tail(50))

    hparams = load_hyperparameters(hparams_path)
    if hparams:
        st.subheader("Hyperparameters")
        st.json(hparams)

    st.caption("Use the refresh button to update metrics while training is running.")


if __name__ == "__main__":
    main()
