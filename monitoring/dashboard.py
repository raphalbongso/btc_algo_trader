"""Plotly real-time dashboard for strategy monitoring (Ch.7)."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_backtest_dashboard(
    results: pd.DataFrame,
    title: str = "BTC/USDT Backtest Results",
    output_html: Optional[str] = None,
):
    """Create an interactive Plotly dashboard from backtest results.

    Parameters
    ----------
    results : pd.DataFrame
        Must contain 'creturns' and 'cstrategy' columns.
    title : str
        Dashboard title.
    output_html : str, optional
        If provided, save to HTML file.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Cumulative Returns", "Position", "Drawdown"),
        row_heights=[0.5, 0.2, 0.3],
    )

    # Cumulative returns
    if "creturns" in results.columns:
        fig.add_trace(
            go.Scatter(x=results.index, y=results["creturns"], name="Buy & Hold",
                       line=dict(color="blue")),
            row=1, col=1,
        )
    if "cstrategy" in results.columns:
        fig.add_trace(
            go.Scatter(x=results.index, y=results["cstrategy"], name="Strategy",
                       line=dict(color="green")),
            row=1, col=1,
        )

    # Position
    if "position" in results.columns:
        fig.add_trace(
            go.Scatter(x=results.index, y=results["position"], name="Position",
                       line=dict(color="orange"), fill="tozeroy"),
            row=2, col=1,
        )

    # Drawdown
    if "cstrategy" in results.columns:
        cum = results["cstrategy"]
        peak = cum.cummax()
        dd = (cum - peak) / peak
        fig.add_trace(
            go.Scatter(x=results.index, y=dd, name="Drawdown",
                       line=dict(color="red"), fill="tozeroy"),
            row=3, col=1,
        )

    fig.update_layout(title=title, height=800, showlegend=True)

    if output_html:
        fig.write_html(output_html)
        logger.info(f"Dashboard saved to {output_html}")
    else:
        fig.show()


def create_equity_chart(
    equity_curve: list[float],
    title: str = "Equity Curve",
    output_html: Optional[str] = None,
):
    """Plot equity curve from live/paper trading."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=equity_curve, mode="lines", name="Equity",
        line=dict(color="green"),
    ))

    peak = np.maximum.accumulate(equity_curve)
    fig.add_trace(go.Scatter(
        y=peak, mode="lines", name="Peak",
        line=dict(color="gray", dash="dash"),
    ))

    fig.update_layout(title=title, xaxis_title="Tick", yaxis_title="USDT")

    if output_html:
        fig.write_html(output_html)
    else:
        fig.show()
