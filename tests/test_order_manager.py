"""Tests for execution.order_manager."""

import pytest

from execution.paper_executor import PaperExecutor
from execution.order_manager import OrderManager


class TestOrderManager:
    @pytest.fixture
    def manager(self):
        executor = PaperExecutor(initial_usdt=10000.0, ptc=0.0005)
        return OrderManager(
            executor=executor,
            max_position_size=0.1,
            max_drawdown_pct=0.15,
            initial_capital=10000.0,
        )

    def test_initial_state(self, manager):
        assert manager.position.side == 0
        assert not manager.is_halted

    def test_execute_long(self, manager):
        manager.executor.set_price(50000.0)
        manager.execute_signal(1, 0.01, 50000.0)
        assert manager.position.side == 1
        assert manager.position.amount == 0.01

    def test_execute_short(self, manager):
        manager.executor.set_price(50000.0)
        manager.execute_signal(-1, 0.01, 50000.0)
        assert manager.position.side == -1

    def test_execute_flat(self, manager):
        manager.executor.set_price(50000.0)
        manager.execute_signal(1, 0.01, 50000.0)
        manager.execute_signal(0, 0.01, 51000.0)
        assert manager.position.side == 0

    def test_no_trade_when_same_signal(self, manager):
        manager.executor.set_price(50000.0)
        result = manager.execute_signal(0, 0.01, 50000.0)
        assert result is None  # Already flat

    def test_clamp_position_size(self, manager):
        manager.executor.set_price(50000.0)
        # Try to trade 1.0 BTC but max is 0.1
        manager.execute_signal(1, 1.0, 50000.0)
        assert manager.position.amount == 0.1

    def test_close_all(self, manager):
        manager.executor.set_price(50000.0)
        manager.execute_signal(1, 0.01, 50000.0)
        manager.close_all(51000.0)
        assert manager.position.side == 0

    def test_trades_recorded(self, manager):
        manager.executor.set_price(50000.0)
        manager.execute_signal(1, 0.01, 50000.0)
        manager.executor.set_price(51000.0)
        manager.execute_signal(0, 0.01, 51000.0)
        assert len(manager.trades) == 1
        assert manager.trades[0]["pnl"] > 0  # Bought at 50k, sold at 51k

    def test_summary(self, manager):
        manager.executor.set_price(50000.0)
        manager.execute_signal(1, 0.01, 50000.0)
        manager.executor.set_price(51000.0)
        manager.execute_signal(0, 0.01, 51000.0)
        summary = manager.summary()
        assert summary["n_trades"] == 1
        assert summary["total_pnl"] > 0

    def test_risk_check_normal(self, manager):
        assert manager.check_risk() is True

    def test_empty_summary(self, manager):
        summary = manager.summary()
        assert summary["n_trades"] == 0
