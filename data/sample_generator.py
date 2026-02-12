"""Synthetic data generation using Geometric Brownian Motion.

Provides BTC-like price data for testing when real data is unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_btc_data(
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    initial_price: float = 30000.0,
    mu: float = 0.05,
    sigma: float = 0.80,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic BTC price data using Geometric Brownian Motion.

    Parameters
    ----------
    start, end : str
        Date range.
    initial_price : float
        Starting BTC price.
    mu : float
        Annualized drift.
    sigma : float
        Annualized volatility (0.8 = 80%, realistic for BTC).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and columns: Open, High, Low, Close, Volume.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    dt = 1 / 365  # crypto trades 24/7
    daily_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)

    log_prices = np.log(initial_price) + np.cumsum(daily_returns)
    close = np.exp(log_prices)

    # Synthetic OHLV from close
    intraday_vol = rng.uniform(0.005, 0.025, n)
    high = close * (1 + intraday_vol)
    low = close * (1 - intraday_vol)
    open_ = close * (1 + rng.uniform(-0.01, 0.01, n))
    volume = rng.integers(1000, 50000, size=n).astype(float)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    df.index.name = "Date"
    return df


def generate_sample_data(
    symbol: str = "BTC-USD",
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate sample data compatible with Chapter 3 format.

    Returns a DataFrame with a single 'price' column containing close prices.
    """
    df = generate_btc_data(start=start, end=end, seed=seed)
    return pd.DataFrame({"price": df["Close"]})
