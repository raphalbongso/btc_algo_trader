"""Tests for backtesting.vectorized.lr_backtester."""

import numpy as np
import pandas as pd
import pytest

from backtesting.vectorized.lr_backtester import LRVectorBacktester


class TestLRVectorBacktester:
    def test_run_returns_dataframe(self, btc_data):
        bt = LRVectorBacktester(btc_data, lags=3)
        result = bt.run()
        assert isinstance(result, pd.DataFrame)

    def test_has_train_test_split(self, btc_data):
        bt = LRVectorBacktester(btc_data, lags=3, train_ratio=0.7)
        result = bt.run()
        assert "split" in result.columns
        assert set(result["split"].unique()) == {"train", "test"}

    def test_train_ratio(self, btc_data):
        bt = LRVectorBacktester(btc_data, lags=3, train_ratio=0.6)
        result = bt.run()
        train_pct = (result["split"] == "train").mean()
        assert 0.55 < train_pct < 0.65

    def test_positions_are_valid(self, btc_data):
        bt = LRVectorBacktester(btc_data, lags=3)
        result = bt.run()
        assert set(result["position"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_summary(self, btc_data):
        bt = LRVectorBacktester(btc_data, lags=3)
        bt.run()
        summary = bt.summary()
        assert "strategy_return_log" in summary
        assert "test_samples" in summary
        assert summary["lags"] == 3

    def test_works_with_price_column(self, btc_price_data):
        bt = LRVectorBacktester(btc_price_data, lags=3)
        result = bt.run()
        assert len(result) > 0
