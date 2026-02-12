"""MEXC order execution via CCXT.

Supports spot (BTC/USDT) and futures (BTC/USDT:USDT perpetual).
Market and limit orders, position management, leverage configuration.

CRITICAL:
- Always use enableRateLimit=True
- Validate order sizes against exchange minimums
- Handle partial fills and order failures
- Log every order attempt and result
"""

from __future__ import annotations

import os
import time
import logging
from typing import Dict, Literal, Optional

import ccxt

from execution.broker_base import BrokerBase

logger = logging.getLogger(__name__)


class MEXCExecutor(BrokerBase):
    """MEXC spot + futures execution engine."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        trading_type: Literal["spot", "swap"] = "spot",
        leverage: int = 1,
        margin_mode: str = "isolated",
        sandbox: bool = True,
    ):
        self._api_key = api_key or os.environ.get("MEXC_API_KEY", "")
        self._secret_key = secret_key or os.environ.get("MEXC_SECRET_KEY", "")
        self._trading_type = trading_type
        self._leverage = leverage

        if not self._api_key:
            raise ValueError("MEXC_API_KEY is required")

        self._exchange = ccxt.mexc({
            "apiKey": self._api_key,
            "secret": self._secret_key,
            "enableRateLimit": True,
            "options": {"defaultType": trading_type},
        })

        if sandbox:
            if self._exchange.urls.get("test"):
                self._exchange.set_sandbox_mode(True)
                logger.info("SANDBOX MODE ENABLED")
            else:
                logger.warning("MEXC sandbox not available -- using LIVE with caution")

        self._exchange.load_markets()

        # Configure futures leverage/margin
        if trading_type == "swap" and leverage > 1:
            try:
                self._exchange.set_leverage(leverage, "BTC/USDT:USDT")
                self._exchange.set_margin_mode(margin_mode, "BTC/USDT:USDT")
                logger.info(f"Futures: leverage={leverage}x, margin={margin_mode}")
            except Exception as e:
                logger.error(f"Failed to set leverage/margin: {e}")

    @property
    def symbol(self) -> str:
        if self._trading_type == "swap":
            return "BTC/USDT:USDT"
        return "BTC/USDT"

    def get_balance(self) -> Dict[str, float]:
        try:
            bal = self._exchange.fetch_balance()
            return {
                "USDT_free": float(bal.get("USDT", {}).get("free", 0)),
                "USDT_total": float(bal.get("USDT", {}).get("total", 0)),
                "BTC_free": float(bal.get("BTC", {}).get("free", 0)),
                "BTC_total": float(bal.get("BTC", {}).get("total", 0)),
            }
        except ccxt.BaseError as e:
            logger.error(f"Balance fetch failed: {e}")
            return {}

    def get_ticker(self) -> Dict[str, float]:
        try:
            t = self._exchange.fetch_ticker(self.symbol)
            return {
                "bid": float(t["bid"] or 0),
                "ask": float(t["ask"] or 0),
                "last": float(t["last"] or 0),
                "volume": float(t.get("baseVolume", 0)),
            }
        except ccxt.BaseError as e:
            logger.error(f"Ticker fetch failed: {e}")
            return {}

    def get_positions(self) -> list:
        """Fetch open futures positions."""
        if self._trading_type != "swap":
            return []
        try:
            return self._exchange.fetch_positions([self.symbol])
        except ccxt.BaseError as e:
            logger.error(f"Position fetch failed: {e}")
            return []

    def market_buy(self, amount: float, reduce_only: bool = False) -> Optional[Dict]:
        return self._place_order("buy", "market", amount, reduce_only=reduce_only)

    def market_sell(self, amount: float, reduce_only: bool = False) -> Optional[Dict]:
        return self._place_order("sell", "market", amount, reduce_only=reduce_only)

    def limit_buy(self, amount: float, price: float) -> Optional[Dict]:
        return self._place_order("buy", "limit", amount, price=price)

    def limit_sell(self, amount: float, price: float) -> Optional[Dict]:
        return self._place_order("sell", "limit", amount, price=price)

    def close_position(self, amount: float, side: str = "sell") -> Optional[Dict]:
        """Close futures position with reduceOnly."""
        return self._place_order(side, "market", amount, reduce_only=True)

    def cancel_all_orders(self) -> bool:
        try:
            self._exchange.cancel_all_orders(self.symbol)
            logger.info(f"All orders cancelled for {self.symbol}")
            return True
        except ccxt.BaseError as e:
            logger.error(f"Cancel all failed: {e}")
            return False

    def _place_order(
        self,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
        reduce_only: bool = False,
    ) -> Optional[Dict]:
        """Core order placement with validation, retry, and logging."""
        market = self._exchange.markets.get(self.symbol)
        if market:
            min_amount = market["limits"]["amount"]["min"]
            if min_amount and amount < min_amount:
                logger.error(f"Order amount {amount} below minimum {min_amount}")
                return None

        params = {}
        if reduce_only and self._trading_type == "swap":
            params["reduceOnly"] = True

        max_retries = 3
        for attempt in range(max_retries):
            try:
                order = self._exchange.create_order(
                    self.symbol, order_type, side, amount, price, params
                )
                logger.info(
                    f"ORDER PLACED | {side.upper()} {order_type} | "
                    f"{amount} {self.symbol} | "
                    f"id={order['id']} | status={order['status']}"
                )
                return order

            except ccxt.InsufficientFunds as e:
                logger.error(f"INSUFFICIENT FUNDS: {e}")
                return None

            except ccxt.InvalidOrder as e:
                logger.error(f"INVALID ORDER: {e}")
                return None

            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
                wait = 2 ** attempt
                logger.warning(
                    f"Network error (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait}s"
                )
                time.sleep(wait)

            except ccxt.BaseError as e:
                logger.error(f"Exchange error: {e}")
                return None

        logger.error(f"Order failed after {max_retries} retries")
        return None
