"""Event-based long/short backtester. Position in {-1, 0, 1}."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.event_based.backtest_base import BacktestBase
from strategies.base import StrategyBase


class BacktestLongShort(BacktestBase):
    """Long/short event-based backtester.

    Position can be -1 (short), 0 (flat), or 1 (long).

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
        """Run the long/short backtest bar by bar."""
        signals = self.strategy.generate_signal(self.data)

        # Calculate unit size based on initial capital
        unit_size = max(1, int(self.initial_capital * 0.95 / self.prices[0]))

        for bar in range(len(self.prices)):
            sig = signals.iloc[bar] if bar < len(signals) else 0

            # Target position: -1, 0, or 1 (in unit_size multiples)
            target_units = int(sig) * unit_size
            diff = target_units - self.position

            if diff != 0:
                self._execute_trade(bar, diff)

            self.portfolio_values.append(self._portfolio_value(bar))

        # Close at the end
        if self.position != 0:
            self._close_position(len(self.prices) - 1)
            self.portfolio_values[-1] = self._portfolio_value(len(self.prices) - 1)

        return self.summary()
