"""Tests for live.btc_trader using paper mode."""

import pandas as pd
import pytest

from config.settings import TradingConfig
from strategies.sma_strategy import SMAStrategy
from strategies.momentum_strategy import MomentumStrategy
from live.btc_trader import BTCTrader


class TestBTCTrader:
    @pytest.fixture
    def strategy(self):
        return SMAStrategy(sma_short=10, sma_long=30)

    @pytest.fixture
    def trading_config(self):
        return TradingConfig(
            units=0.001,
            initial_capital=1000.0,
            ptc=0.0005,
        )

    def test_paper_mode_init(self, strategy, trading_config):
        trader = BTCTrader(
            strategy=strategy, mode="paper", trading_config=trading_config,
        )
        assert trader.mode == "paper"

    def test_invalid_mode_raises(self, strategy):
        with pytest.raises(ValueError, match="mode must be"):
            BTCTrader(strategy=strategy, mode="invalid")

    def test_run_on_data_returns_summary(self, strategy, trading_config, btc_data):
        trader = BTCTrader(
            strategy=strategy, mode="paper", trading_config=trading_config,
        )
        summary = trader.run_on_data(btc_data)
        assert isinstance(summary, dict)
        assert "n_trades" in summary

    def test_paper_trades_are_recorded(self, strategy, trading_config, btc_data):
        trader = BTCTrader(
            strategy=strategy, mode="paper", trading_config=trading_config,
        )
        summary = trader.run_on_data(btc_data)
        assert summary["n_trades"] > 0

    def test_paper_final_equity_positive(self, strategy, trading_config, btc_data):
        trader = BTCTrader(
            strategy=strategy, mode="paper", trading_config=trading_config,
        )
        summary = trader.run_on_data(btc_data)
        assert summary["final_equity"] > 0

    def test_close_out_resets_position(self, strategy, trading_config, btc_data):
        trader = BTCTrader(
            strategy=strategy, mode="paper", trading_config=trading_config,
        )
        trader.run_on_data(btc_data)
        assert trader.order_manager.position.side == 0

    def test_momentum_strategy_runs(self, trading_config, btc_data):
        strategy = MomentumStrategy(window=10)
        trader = BTCTrader(
            strategy=strategy, mode="paper", trading_config=trading_config,
        )
        summary = trader.run_on_data(btc_data)
        assert summary["n_trades"] >= 0

    def test_get_summary_empty_before_run(self, strategy, trading_config):
        trader = BTCTrader(
            strategy=strategy, mode="paper", trading_config=trading_config,
        )
        summary = trader.get_summary()
        assert summary["n_trades"] == 0
