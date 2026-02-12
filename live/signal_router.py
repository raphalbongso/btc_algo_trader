"""Routes strategy signals to execution, with optional ZeroMQ logging."""

from __future__ import annotations

import logging
from typing import Dict, Optional

from strategies.base import StrategyBase
from execution.order_manager import OrderManager

logger = logging.getLogger(__name__)


class SignalRouter:
    """Routes signals from one or more strategies to the order manager.

    Supports:
    - Single strategy routing
    - Multi-strategy ensemble (majority vote)
    - Signal filtering (ignore weak signals)
    """

    def __init__(
        self,
        order_manager: OrderManager,
        strategies: list[StrategyBase] | None = None,
        min_agreement: float = 0.5,
    ):
        self.order_manager = order_manager
        self.strategies = strategies or []
        self.min_agreement = min_agreement

    def add_strategy(self, strategy: StrategyBase) -> None:
        self.strategies.append(strategy)

    def route(self, data, current_price: float, trade_amount: float) -> Optional[Dict]:
        """Generate ensemble signal and route to execution.

        Parameters
        ----------
        data : pd.DataFrame
            Current market data window.
        current_price : float
            Latest price.
        trade_amount : float
            BTC amount per trade.

        Returns
        -------
        dict or None
            Order result.
        """
        if not self.strategies:
            logger.warning("No strategies configured")
            return None

        signals = []
        for s in self.strategies:
            sig = s.generate_signal(data)
            if hasattr(sig, "iloc"):
                sig = int(sig.iloc[-1])
            signals.append(sig)

        # Majority vote
        avg_signal = sum(signals) / len(signals)
        if avg_signal > self.min_agreement:
            final_signal = 1
        elif avg_signal < -self.min_agreement:
            final_signal = -1
        else:
            final_signal = 0

        logger.debug(
            f"Signals: {signals} -> avg={avg_signal:.2f} -> final={final_signal}"
        )

        return self.order_manager.execute_signal(
            final_signal, trade_amount, current_price
        )
