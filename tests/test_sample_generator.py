"""Tests for data.sample_generator."""

import numpy as np
import pandas as pd
import pytest

from data.sample_generator import generate_btc_data, generate_sample_data


class TestGenerateBtcData:
    def test_returns_dataframe(self):
        df = generate_btc_data()
        assert isinstance(df, pd.DataFrame)

    def test_columns(self):
        df = generate_btc_data()
        assert set(df.columns) == {"Open", "High", "Low", "Close", "Volume"}

    def test_index_is_datetime(self):
        df = generate_btc_data()
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "Date"

    def test_positive_prices(self):
        df = generate_btc_data()
        assert (df["Close"] > 0).all()
        assert (df["High"] >= df["Low"]).all()

    def test_reproducibility(self):
        df1 = generate_btc_data(seed=123)
        df2 = generate_btc_data(seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = generate_btc_data(seed=1)
        df2 = generate_btc_data(seed=2)
        assert not df1["Close"].equals(df2["Close"])

    def test_date_range(self):
        df = generate_btc_data(start="2023-06-01", end="2023-06-30")
        assert df.index[0] >= pd.Timestamp("2023-06-01")
        assert df.index[-1] <= pd.Timestamp("2023-06-30")

    def test_initial_price_influence(self):
        df_low = generate_btc_data(initial_price=100, seed=42)
        df_high = generate_btc_data(initial_price=100000, seed=42)
        # Same seed, different initial: ratio should be ~1000
        ratio = df_high["Close"].iloc[0] / df_low["Close"].iloc[0]
        assert 900 < ratio < 1100


class TestGenerateSampleData:
    def test_returns_dataframe(self):
        df = generate_sample_data()
        assert isinstance(df, pd.DataFrame)

    def test_has_price_column(self):
        df = generate_sample_data()
        assert "price" in df.columns
        assert len(df.columns) == 1

    def test_positive_prices(self):
        df = generate_sample_data()
        assert (df["price"] > 0).all()
