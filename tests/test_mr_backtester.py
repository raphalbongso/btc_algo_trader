"""Tests for backtesting.vectorized.mr_backtester."""

import numpy as np
import pandas as pd
import pytest

from backtesting.vectorized.mr_backtester import MRVectorBacktester


class TestMRVectorBacktester:
    def test_run_returns_dataframe(self, btc_data):
        bt = MRVectorBacktester(btc_data, window=20, threshold=1.0)
        result = bt.run()
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self, btc_data):
        bt = MRVectorBacktester(btc_data, window=20, threshold=1.0)
        result = bt.run()
        for col in ["returns", "position", "strategy_net", "creturns", "cstrategy"]:
            assert col in result.columns

    def test_positions_are_valid(self, btc_data):
        bt = MRVectorBacktester(btc_data, window=20, threshold=1.0)
        result = bt.run()
        assert result["position"].isin([-1, 0, 1]).all()

    def test_summary(self, btc_data):
        bt = MRVectorBacktester(btc_data, window=20, threshold=1.0)
        bt.run()
        summary = bt.summary()
        assert "strategy_return" in summary

    def test_inherits_from_mom(self):
        from backtesting.vectorized.mom_backtester import MomVectorBacktester
        assert issubclass(MRVectorBacktester, MomVectorBacktester)

    def test_different_thresholds(self, btc_data):
        bt_low = MRVectorBacktester(btc_data, window=20, threshold=0.5)
        bt_high = MRVectorBacktester(btc_data, window=20, threshold=2.0)
        r_low = bt_low.run()
        r_high = bt_high.run()
        # Lower threshold should trigger more trades
        assert r_low["trades"].sum() >= r_high["trades"].sum()
