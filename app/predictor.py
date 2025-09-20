"""Streamlit application for interactive forecasting visualization."""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import streamlit as st
import torch

from stonkex.data.datasets import CANDLESTICK_COLUMNS, SequenceWindow, ensure_datetime_index
from stonkex.models.transformer import TemporalTransformer, TransformerConfig
from stonkex.visualization.chart import candlestick_chart


st.set_page_config(page_title="StonkEx Predictor", layout="wide")
st.title("🔮 StonkEx Inference Console")

st.sidebar.header("Inputs")
model_file = st.sidebar.file_uploader("Model artifact (.pt)", type=["pt"])
data_file = st.sidebar.file_uploader("Candlestick CSV", type=["csv"])
input_window = st.sidebar.number_input("Input window", min_value=16, max_value=1024, value=192, step=8)
horizon = st.sidebar.number_input("Forecast horizon", min_value=1, max_value=256, value=24, step=1)
run_button = st.sidebar.button("Run inference")

def load_model(buffer: bytes, forecast_horizon: int) -> tuple[TemporalTransformer, dict | None]:
    artifact = torch.load(io.BytesIO(buffer), map_location="cpu")
    if isinstance(artifact, dict) and "model_state_dict" in artifact:
        model_config = artifact.get("model_config") or {}
        config = TransformerConfig(
            input_dim=model_config.get("input_dim", len(CANDLESTICK_COLUMNS)),
            d_model=model_config.get("d_model", 512),
            nhead=model_config.get("nhead", 8),
            num_layers=model_config.get("num_layers", 6),
            dim_feedforward=model_config.get("dim_feedforward", 1024),
            dropout=model_config.get("dropout", 0.1),
            horizon=model_config.get("horizon", forecast_horizon),
        )
        model = TemporalTransformer(config)
        model.load_state_dict(artifact["model_state_dict"])
        model.eval()
        return model, artifact.get("normalization")
    raise RuntimeError("Invalid model artifact format")


def prepare_frame(file_buffer: bytes) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(file_buffer))
    frame = ensure_datetime_index(frame).reset_index()
    return frame


if run_button:
    if not model_file or not data_file:
        st.warning("Please upload both a model artifact and a CSV dataset")
        st.stop()

    model, normalization = load_model(model_file.getvalue(), horizon)
    frame = prepare_frame(data_file.getvalue())

    if len(frame) < input_window + horizon:
        st.error("Dataset is too small for the selected window and horizon")
        st.stop()

    window = SequenceWindow(input_window=input_window, horizon=horizon)
    recent = frame.tail(window.input_window)
    if normalization and normalization.get("feature_columns"):
        feature_cols = [col for col in normalization["feature_columns"] if col in recent.columns]
    else:
        feature_cols = [col for col in CANDLESTICK_COLUMNS if col in recent.columns]
    features = recent[feature_cols]
    values = torch.tensor(features.to_numpy(dtype=np.float32)).unsqueeze(0)

    if normalization:
        mean = torch.tensor([normalization["mean"].get(col, float(features[col].mean())) for col in features.columns], dtype=torch.float32)
        std = torch.tensor([normalization["std"].get(col, float(features[col].std() or 1.0)) for col in features.columns], dtype=torch.float32)
        std[std == 0] = 1
        values = (values - mean) / std

    with torch.no_grad():
        prediction = model(values)
    prediction = prediction.squeeze(0).cpu().numpy()

    st.subheader("Forecast Preview")
    fig = candlestick_chart(frame.tail(window.input_window + horizon), pd.Series(prediction), prediction_horizon=horizon)
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download predictions",
        data=json.dumps({"forecast": prediction.tolist()}),
        file_name="stonkex_forecast.json",
        mime="application/json",
    )
else:
    st.info("Upload a trained model and dataset to visualize predictions.")
