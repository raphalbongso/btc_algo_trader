"""Tests for execution.okx_executor (using mocked CCXT)."""

from unittest.mock import patch, MagicMock

import ccxt
import pytest

from execution.okx_executor import OKXExecutor


@pytest.fixture
def mock_exchange(mock_okx_exchange):
    """Patch ccxt.okx to return our mock."""
    return mock_okx_exchange


class TestOKXExecutor:
    def _create_executor(self, mock_exchange, trading_type="spot"):
        """Create executor with mocked exchange."""
        with patch("execution.okx_executor.ccxt.okx") as mock_cls:
            mock_cls.return_value = mock_exchange
            mock_exchange.urls = {}  # No sandbox URL
            mock_exchange.load_markets.return_value = None
            executor = OKXExecutor(
                api_key="test_key",
                secret_key="test_secret",
                passphrase="test_passphrase",
                trading_type=trading_type,
                sandbox=False,
            )
        return executor

    def test_spot_symbol(self, mock_exchange):
        executor = self._create_executor(mock_exchange, "spot")
        assert executor.symbol == "BTC/USDT"

    def test_futures_symbol(self, mock_exchange):
        executor = self._create_executor(mock_exchange, "swap")
        assert executor.symbol == "BTC/USDT:USDT"

    def test_get_balance(self, mock_exchange):
        executor = self._create_executor(mock_exchange)
        bal = executor.get_balance()
        assert bal["USDT_free"] == 10000.0
        assert bal["BTC_free"] == 0.1

    def test_get_ticker(self, mock_exchange):
        executor = self._create_executor(mock_exchange)
        ticker = executor.get_ticker()
        assert ticker["bid"] == 50000.0
        assert ticker["ask"] == 50001.0

    def test_market_buy(self, mock_exchange):
        executor = self._create_executor(mock_exchange)
        order = executor.market_buy(0.001)
        assert order is not None
        assert order["id"] == "test-order-123"
        mock_exchange.create_order.assert_called_once()

    def test_market_sell(self, mock_exchange):
        executor = self._create_executor(mock_exchange)
        order = executor.market_sell(0.001)
        assert order is not None

    def test_limit_buy(self, mock_exchange):
        executor = self._create_executor(mock_exchange)
        order = executor.limit_buy(0.001, 49000.0)
        assert order is not None

    def test_limit_sell(self, mock_exchange):
        executor = self._create_executor(mock_exchange)
        order = executor.limit_sell(0.001, 51000.0)
        assert order is not None

    def test_below_minimum_order(self, mock_exchange):
        mock_exchange.markets = {
            "BTC/USDT": {"limits": {"amount": {"min": 0.01}}},
        }
        executor = self._create_executor(mock_exchange)
        order = executor.market_buy(0.001)  # Below 0.01 minimum
        assert order is None

    def test_insufficient_funds(self, mock_exchange):
        mock_exchange.create_order.side_effect = ccxt.InsufficientFunds("No funds")
        executor = self._create_executor(mock_exchange)
        order = executor.market_buy(0.001)
        assert order is None

    def test_invalid_order(self, mock_exchange):
        mock_exchange.create_order.side_effect = ccxt.InvalidOrder("Bad order")
        executor = self._create_executor(mock_exchange)
        order = executor.market_buy(0.001)
        assert order is None

    def test_cancel_all(self, mock_exchange):
        executor = self._create_executor(mock_exchange)
        assert executor.cancel_all_orders() is True

    def test_no_api_key_raises(self):
        with pytest.raises(ValueError, match="OKX_API_KEY"):
            with patch("execution.okx_executor.os.environ.get", return_value=""):
                OKXExecutor(api_key="", secret_key="", passphrase="")
