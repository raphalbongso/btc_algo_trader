"""Event-based long-only backtester. Position in {0, 1}."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.event_based.backtest_base import BacktestBase
from strategies.base import StrategyBase


class BacktestLongOnly(BacktestBase):
    """Long-only event-based backtester.

    Position is either 0 (flat) or 1 (long).

    Parameters
    ----------
    data : pd.DataFrame
        Price data.
    strategy : StrategyBase
        Strategy that generates signals.
    initial_capital : float
        Starting capital.
    ftc : float
        Fixed transaction cost.
    ptc : float
        Proportional transaction cost.
    trading_days : int
        Annualization factor.
    verbose : bool
        Print trade details.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy: StrategyBase,
        initial_capital: float = 100_000.0,
        ftc: float = 0.0,
        ptc: float = 0.001,
        trading_days: int = 365,
        verbose: bool = False,
    ):
        super().__init__(data, initial_capital, ftc, ptc, trading_days, verbose)
        self.strategy = strategy

    def run(self) -> dict:
        """Run the long-only backtest bar by bar."""
        signals = self.strategy.generate_signal(self.data)

        for bar in range(len(self.prices)):
            sig = signals.iloc[bar] if bar < len(signals) else 0

            # Map signal to position: long-only means {0, 1}
            target_position = 1 if sig > 0 else 0
            current_units = 1 if self.position > 0 else 0

            if target_position == 1 and current_units == 0:
                # Buy 1 unit
                units_to_buy = max(1, int(self.cash * 0.95 / self._get_price(bar)))
                self._execute_trade(bar, units_to_buy)
            elif target_position == 0 and current_units == 1:
                # Sell all
                self._close_position(bar)

            self.portfolio_values.append(self._portfolio_value(bar))

        # Close any open position at the end
        if self.position > 0:
            self._close_position(len(self.prices) - 1)
            self.portfolio_values[-1] = self._portfolio_value(len(self.prices) - 1)

        return self.summary()
