"""Shared test fixtures. NO real API calls are ever made in tests."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.sample_generator import generate_btc_data, generate_sample_data


@pytest.fixture
def btc_data() -> pd.DataFrame:
    """Synthetic BTC OHLCV data (2 years) with price/returns columns."""
    df = generate_btc_data(start="2022-01-01", end="2023-12-31", seed=42)
    df["price"] = df["Close"]
    df["returns"] = np.log(df["price"] / df["price"].shift(1))
    df.dropna(inplace=True)
    return df


@pytest.fixture
def btc_price_data() -> pd.DataFrame:
    """Synthetic BTC data with 'price' column (Ch.3 compat)."""
    return generate_sample_data(start="2022-01-01", end="2023-12-31", seed=42)


@pytest.fixture
def short_btc_data() -> pd.DataFrame:
    """Short synthetic data for quick tests."""
    df = generate_btc_data(start="2023-01-01", end="2023-06-30", seed=42)
    df["price"] = df["Close"]
    df["returns"] = np.log(df["price"] / df["price"].shift(1))
    df.dropna(inplace=True)
    return df


@pytest.fixture
def btc_minute_data() -> pd.DataFrame:
    """Synthetic BTC/USDT minute data (5000 bars)."""
    np.random.seed(123)
    n = 5000
    index = pd.date_range("2024-01-01", periods=n, freq="1min")
    prices = 50000 * np.exp(np.cumsum(np.random.normal(0, 0.0005, n)))
    return pd.DataFrame({
        "Open": prices * 0.9999,
        "High": prices * 1.001,
        "Low": prices * 0.999,
        "Close": prices,
        "price": prices,
        "Volume": np.random.randint(10, 1000, n),
    }, index=index)


@pytest.fixture
def log_returns(btc_data) -> pd.Series:
    """Log returns from BTC close prices."""
    return np.log(btc_data["Close"] / btc_data["Close"].shift(1)).dropna()


@pytest.fixture
def mock_mexc_exchange():
    """Mock CCXT MEXC exchange object."""
    exchange = MagicMock()
    exchange.fetch_ticker.return_value = {
        "bid": 50000.0, "ask": 50001.0, "last": 50000.5, "baseVolume": 1234.5,
    }
    exchange.fetch_balance.return_value = {
        "USDT": {"free": 10000.0, "total": 10000.0},
        "BTC": {"free": 0.1, "total": 0.1},
    }
    exchange.create_order.return_value = {
        "id": "test-order-123", "status": "closed",
        "filled": 0.001, "price": 50000.0,
    }
    exchange.markets = {
        "BTC/USDT": {"limits": {"amount": {"min": 0.00001}}},
        "BTC/USDT:USDT": {"limits": {"amount": {"min": 0.001}}},
    }
    exchange.cancel_all_orders.return_value = None
    return exchange
