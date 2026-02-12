"""Tests for execution.paper_executor."""

import pytest

from execution.paper_executor import PaperExecutor


class TestPaperExecutor:
    @pytest.fixture
    def executor(self):
        return PaperExecutor(initial_usdt=10000.0, ptc=0.0005, ftc=0.0)

    def test_initial_balance(self, executor):
        bal = executor.get_balance()
        assert bal["USDT_free"] == 10000.0
        assert bal["BTC_free"] == 0.0

    def test_market_buy(self, executor):
        executor.set_price(50000.0)
        order = executor.market_buy(0.1)
        assert order is not None
        assert order["status"] == "closed"
        assert executor._btc == 0.1
        assert executor._usdt < 10000.0  # Deducted cost + TC

    def test_market_sell(self, executor):
        executor.set_price(50000.0)
        executor.market_buy(0.1)
        order = executor.market_sell(0.1)
        assert order is not None
        assert executor._btc == 0.0

    def test_insufficient_funds_buy(self, executor):
        executor.set_price(50000.0)
        order = executor.market_buy(1.0)  # 50000 > 10000
        assert order is None

    def test_insufficient_btc_sell(self, executor):
        order = executor.market_sell(0.1)  # No BTC held
        assert order is None

    def test_transaction_cost_applied(self, executor):
        executor.set_price(50000.0)
        executor.market_buy(0.1)
        # Cost: 0.1 * 50000 = 5000, TC: 5000 * 0.0005 = 2.5
        expected = 10000 - 5000 - 2.5
        assert abs(executor._usdt - expected) < 0.01

    def test_get_ticker(self, executor):
        executor.set_price(60000.0)
        ticker = executor.get_ticker()
        assert ticker["last"] == 60000.0
        assert ticker["bid"] < ticker["ask"]

    def test_trade_log(self, executor):
        executor.set_price(50000.0)
        executor.market_buy(0.01)
        executor.market_sell(0.01)
        assert len(executor.trade_log) == 2
        assert executor.trade_log[0]["side"] == "buy"
        assert executor.trade_log[1]["side"] == "sell"

    def test_cancel_all(self, executor):
        assert executor.cancel_all_orders() is True

    def test_limit_buy(self, executor):
        order = executor.limit_buy(0.01, 49000.0)
        assert order is not None
        assert executor._btc == 0.01

    def test_limit_sell(self, executor):
        executor.set_price(50000.0)
        executor.market_buy(0.01)
        order = executor.limit_sell(0.01, 51000.0)
        assert order is not None
        assert executor._btc == 0.0
