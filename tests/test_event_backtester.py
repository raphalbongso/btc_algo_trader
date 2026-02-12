"""Tests for event-based backtesters."""

import numpy as np
import pandas as pd
import pytest

from backtesting.event_based.backtest_base import BacktestBase
from backtesting.event_based.backtest_long_only import BacktestLongOnly
from backtesting.event_based.backtest_long_short import BacktestLongShort
from strategies.sma_strategy import SMAStrategy
from strategies.momentum_strategy import MomentumStrategy


class TestBacktestLongOnly:
    @pytest.fixture
    def strategy(self):
        return SMAStrategy(sma_short=10, sma_long=30)

    def test_runs_and_returns_summary(self, btc_data, strategy):
        bt = BacktestLongOnly(btc_data, strategy=strategy, initial_capital=100_000)
        summary = bt.run()
        assert isinstance(summary, dict)
        assert "total_return" in summary

    def test_portfolio_values_tracked(self, btc_data, strategy):
        bt = BacktestLongOnly(btc_data, strategy=strategy)
        bt.run()
        assert len(bt.portfolio_values) == len(btc_data)

    def test_initial_capital(self, btc_data, strategy):
        bt = BacktestLongOnly(btc_data, strategy=strategy, initial_capital=50_000)
        summary = bt.run()
        assert summary["initial_capital"] == 50_000

    def test_with_transaction_costs(self, btc_data, strategy):
        bt_free = BacktestLongOnly(btc_data, strategy=strategy, ftc=0, ptc=0)
        bt_costly = BacktestLongOnly(btc_data, strategy=strategy, ftc=1.0, ptc=0.002)
        s_free = bt_free.run()
        s_costly = bt_costly.run()
        # With TC, final value should be lower
        assert s_costly["final_value"] <= s_free["final_value"]


class TestBacktestLongShort:
    @pytest.fixture
    def strategy(self):
        return MomentumStrategy(window=15)

    def test_runs_and_returns_summary(self, btc_data, strategy):
        bt = BacktestLongShort(btc_data, strategy=strategy, initial_capital=100_000)
        summary = bt.run()
        assert isinstance(summary, dict)
        assert "total_return" in summary

    def test_portfolio_values_tracked(self, btc_data, strategy):
        bt = BacktestLongShort(btc_data, strategy=strategy)
        bt.run()
        assert len(bt.portfolio_values) == len(btc_data)

    def test_trades_logged(self, btc_data, strategy):
        bt = BacktestLongShort(btc_data, strategy=strategy)
        bt.run()
        assert bt.trades > 0
        assert len(bt.trade_log) > 0

    def test_summary_has_metrics(self, btc_data, strategy):
        bt = BacktestLongShort(btc_data, strategy=strategy)
        summary = bt.run()
        for key in ["annualized_return", "sharpe_ratio", "max_drawdown", "n_trades"]:
            assert key in summary
