"""Tests for backtesting.vectorized.sma_backtester."""

import numpy as np
import pandas as pd
import pytest

from backtesting.vectorized.sma_backtester import SMAVectorBacktester


class TestSMAVectorBacktester:
    def test_run_returns_dataframe(self, btc_data):
        bt = SMAVectorBacktester(btc_data, sma_short=10, sma_long=30)
        result = bt.run()
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self, btc_data):
        bt = SMAVectorBacktester(btc_data, sma_short=10, sma_long=30)
        result = bt.run()
        for col in ["returns", "position", "strategy_net", "creturns", "cstrategy"]:
            assert col in result.columns

    def test_positions_are_valid(self, btc_data):
        bt = SMAVectorBacktester(btc_data, sma_short=10, sma_long=30)
        result = bt.run()
        assert set(result["position"].unique()).issubset({-1, 0, 1})

    def test_cumulative_returns_start_near_one(self, btc_data):
        bt = SMAVectorBacktester(btc_data, sma_short=10, sma_long=30)
        result = bt.run()
        assert abs(result["creturns"].iloc[0] - 1.0) < 0.15
        assert abs(result["cstrategy"].iloc[0] - 1.0) < 0.15

    def test_summary_keys(self, btc_data):
        bt = SMAVectorBacktester(btc_data, sma_short=10, sma_long=30)
        bt.run()
        summary = bt.summary()
        assert "strategy_return" in summary
        assert "sharpe_ratio" in summary
        assert "n_trades" in summary

    def test_zero_tc_matches_gross(self, btc_data):
        bt = SMAVectorBacktester(btc_data, sma_short=10, sma_long=30, ptc=0.0)
        result = bt.run()
        # With no TC, strategy_net should equal strategy
        np.testing.assert_array_almost_equal(
            result["strategy"].values, result["strategy_net"].values
        )

    def test_works_with_price_column(self, btc_price_data):
        bt = SMAVectorBacktester(btc_price_data, sma_short=10, sma_long=30)
        result = bt.run()
        assert len(result) > 0

    def test_optimize(self, short_btc_data):
        bt = SMAVectorBacktester(short_btc_data, sma_short=10, sma_long=30)
        best_s, best_l, perf = bt.optimize(
            short_range=range(5, 16, 5), long_range=range(20, 41, 10)
        )
        assert isinstance(best_s, (int, np.integer))
        assert isinstance(best_l, (int, np.integer))
        assert best_s < best_l
