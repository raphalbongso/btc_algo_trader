"""Tests for backtesting.vectorized.scikit_backtester."""

import numpy as np
import pandas as pd
import pytest

from backtesting.vectorized.scikit_backtester import ScikitVectorBacktester


class TestScikitVectorBacktester:
    def test_logistic_runs(self, btc_data):
        bt = ScikitVectorBacktester(btc_data, model_type="logistic", lags=3)
        result = bt.run()
        assert isinstance(result, pd.DataFrame)

    def test_adaboost_runs(self, btc_data):
        bt = ScikitVectorBacktester(btc_data, model_type="adaboost", lags=3)
        result = bt.run()
        assert isinstance(result, pd.DataFrame)

    def test_has_train_test_split(self, btc_data):
        bt = ScikitVectorBacktester(btc_data, lags=3)
        result = bt.run()
        assert "split" in result.columns

    def test_positions_are_valid(self, btc_data):
        bt = ScikitVectorBacktester(btc_data, lags=3)
        result = bt.run()
        # Positions should be -1, 0, or 1
        assert result["position"].isin([-1, 0, 1]).all()

    def test_summary(self, btc_data):
        bt = ScikitVectorBacktester(btc_data, lags=3)
        bt.run()
        summary = bt.summary()
        assert "model_type" in summary
        assert "lags" in summary
        assert summary["model_type"] == "logistic"

    def test_model_is_fitted(self, btc_data):
        bt = ScikitVectorBacktester(btc_data, lags=3)
        bt.run()
        assert bt.model is not None

    def test_works_with_price_column(self, btc_price_data):
        bt = ScikitVectorBacktester(btc_price_data, lags=3)
        result = bt.run()
        assert len(result) > 0
