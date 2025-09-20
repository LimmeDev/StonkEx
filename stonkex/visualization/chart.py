"""Visualization helpers using Plotly."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def candlestick_chart(
    frame: pd.DataFrame,
    predictions: Optional[pd.Series] = None,
    prediction_horizon: int = 0,
    title: str = "Candlestick Forecast",
) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=frame["datetime"],
                open=frame["open"],
                high=frame["high"],
                low=frame["low"],
                close=frame["close"],
                name="Observed",
            )
        ]
    )
    if predictions is not None and prediction_horizon > 0:
        interval = frame["datetime"].diff().median()
        if pd.isna(interval) or interval == pd.Timedelta(0):
            interval = pd.Timedelta(hours=1)
        future_index = frame["datetime"].iloc[-1] + interval * np.arange(1, prediction_horizon + 1)
        fig.add_trace(
            go.Scatter(
                x=future_index,
                y=predictions,
                mode="lines",
                name="Forecast",
                line=dict(color="rgba(0, 200, 150, 0.7)", width=2, dash="dash"),
            )
        )
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Price")
    return fig
