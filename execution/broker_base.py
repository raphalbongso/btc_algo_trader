"""Abstract broker interface for execution engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional


class BrokerBase(ABC):
    """Abstract interface for all execution backends."""

    @abstractmethod
    def get_balance(self) -> Dict[str, float]:
        """Fetch account balances."""
        ...

    @abstractmethod
    def get_ticker(self) -> Dict[str, float]:
        """Fetch current bid/ask/last for the symbol."""
        ...

    @abstractmethod
    def market_buy(self, amount: float, **kwargs) -> Optional[Dict]:
        """Place a market buy order."""
        ...

    @abstractmethod
    def market_sell(self, amount: float, **kwargs) -> Optional[Dict]:
        """Place a market sell order."""
        ...

    @abstractmethod
    def limit_buy(self, amount: float, price: float) -> Optional[Dict]:
        """Place a limit buy order."""
        ...

    @abstractmethod
    def limit_sell(self, amount: float, price: float) -> Optional[Dict]:
        """Place a limit sell order."""
        ...

    @abstractmethod
    def cancel_all_orders(self) -> bool:
        """Cancel all open orders."""
        ...
