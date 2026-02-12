"""Paper trading executor -- simulated execution with no exchange connection."""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

from execution.broker_base import BrokerBase

logger = logging.getLogger(__name__)


class PaperExecutor(BrokerBase):
    """Simulated execution for paper trading.

    Tracks a virtual portfolio with configurable transaction costs.
    No exchange connection is made.

    Parameters
    ----------
    initial_usdt : float
        Starting USDT balance.
    ptc : float
        Proportional transaction cost.
    ftc : float
        Fixed transaction cost per trade.
    """

    def __init__(
        self,
        initial_usdt: float = 1000.0,
        ptc: float = 0.0005,
        ftc: float = 0.0,
    ):
        self._usdt = initial_usdt
        self._btc = 0.0
        self._ptc = ptc
        self._ftc = ftc
        self._last_price = 50000.0  # Default until updated
        self._order_count = 0
        self._trade_log: list[Dict] = []

    def set_price(self, price: float) -> None:
        """Update the simulated current price."""
        self._last_price = price

    def get_balance(self) -> Dict[str, float]:
        return {
            "USDT_free": self._usdt,
            "USDT_total": self._usdt + self._btc * self._last_price,
            "BTC_free": self._btc,
            "BTC_total": self._btc,
        }

    def get_ticker(self) -> Dict[str, float]:
        spread = self._last_price * 0.0001
        return {
            "bid": self._last_price - spread,
            "ask": self._last_price + spread,
            "last": self._last_price,
            "volume": 0,
        }

    def market_buy(self, amount: float, **kwargs) -> Optional[Dict]:
        cost = amount * self._last_price
        tc = cost * self._ptc + self._ftc
        if cost + tc > self._usdt:
            logger.warning(f"Paper: insufficient funds for buy {amount} BTC")
            return None

        self._usdt -= cost + tc
        self._btc += amount
        return self._log_order("buy", "market", amount, self._last_price, tc)

    def market_sell(self, amount: float, **kwargs) -> Optional[Dict]:
        if amount > self._btc and not kwargs.get("reduce_only"):
            logger.warning(f"Paper: insufficient BTC for sell {amount}")
            return None

        proceeds = amount * self._last_price
        tc = proceeds * self._ptc + self._ftc
        self._usdt += proceeds - tc
        self._btc -= amount
        return self._log_order("sell", "market", amount, self._last_price, tc)

    def limit_buy(self, amount: float, price: float) -> Optional[Dict]:
        # Paper: execute immediately at limit price
        cost = amount * price
        tc = cost * self._ptc + self._ftc
        if cost + tc > self._usdt:
            return None
        self._usdt -= cost + tc
        self._btc += amount
        return self._log_order("buy", "limit", amount, price, tc)

    def limit_sell(self, amount: float, price: float) -> Optional[Dict]:
        if amount > self._btc:
            return None
        proceeds = amount * price
        tc = proceeds * self._ptc + self._ftc
        self._usdt += proceeds - tc
        self._btc -= amount
        return self._log_order("sell", "limit", amount, price, tc)

    def cancel_all_orders(self) -> bool:
        return True

    def _log_order(
        self, side: str, order_type: str, amount: float, price: float, tc: float
    ) -> Dict:
        self._order_count += 1
        order = {
            "id": f"paper-{self._order_count}",
            "status": "closed",
            "side": side,
            "type": order_type,
            "amount": amount,
            "filled": amount,
            "price": price,
            "cost": amount * price,
            "tc": tc,
        }
        self._trade_log.append(order)
        logger.info(
            f"PAPER | {side.upper()} {order_type} | {amount} BTC @ {price:.2f} | TC={tc:.4f}"
        )
        return order

    @property
    def trade_log(self) -> list[Dict]:
        return self._trade_log
