"""Tests for backtesting.vectorized.mom_backtester."""

import numpy as np
import pandas as pd
import pytest

from backtesting.vectorized.mom_backtester import MomVectorBacktester


class TestMomVectorBacktester:
    def test_run_returns_dataframe(self, btc_data):
        bt = MomVectorBacktester(btc_data, momentum=15)
        result = bt.run()
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self, btc_data):
        bt = MomVectorBacktester(btc_data, momentum=15)
        result = bt.run()
        for col in ["returns", "position", "strategy_net", "creturns", "cstrategy"]:
            assert col in result.columns

    def test_positions_are_valid(self, btc_data):
        bt = MomVectorBacktester(btc_data, momentum=15)
        result = bt.run()
        assert set(result["position"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_summary(self, btc_data):
        bt = MomVectorBacktester(btc_data, momentum=15)
        bt.run()
        summary = bt.summary()
        assert "strategy_return" in summary
        assert "momentum" in summary
        assert summary["momentum"] == 15

    def test_optimize(self, short_btc_data):
        bt = MomVectorBacktester(short_btc_data, momentum=10)
        best_mom, perf = bt.optimize(momentum_range=range(5, 26, 5))
        assert isinstance(best_mom, (int, np.integer))
        assert 5 <= best_mom <= 25

    def test_works_with_price_column(self, btc_price_data):
        bt = MomVectorBacktester(btc_price_data, momentum=15)
        result = bt.run()
        assert len(result) > 0
