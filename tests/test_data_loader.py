"""Tests for data.data_loader."""

import pandas as pd
import pytest

from data.data_loader import load_btc_data


class TestLoadBtcData:
    def test_returns_dataframe(self):
        df = load_btc_data(use_cache=False)
        assert isinstance(df, pd.DataFrame)

    def test_has_close_column(self):
        df = load_btc_data(use_cache=False)
        assert "Close" in df.columns

    def test_has_datetime_index(self):
        df = load_btc_data(use_cache=False)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_non_empty(self):
        df = load_btc_data(start="2023-01-01", end="2023-06-30", use_cache=False)
        assert len(df) > 50

    def test_positive_prices(self):
        df = load_btc_data(use_cache=False)
        assert (df["Close"] > 0).all()
