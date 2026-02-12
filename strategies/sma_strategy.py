"""Simple Moving Average crossover strategy."""

from __future__ import annotations

import pandas as pd

from strategies.base import StrategyBase


class SMAStrategy(StrategyBase):
    """SMA crossover: go long when short SMA > long SMA, short otherwise.

    Parameters
    ----------
    sma_short : int
        Short-window SMA period (default 20).
    sma_long : int
        Long-window SMA period (default 50).
    """

    def __init__(self, sma_short: int = 20, sma_long: int = 50):
        self.sma_short = sma_short
        self.sma_long = sma_long

    @property
    def required_history(self) -> int:
        return self.sma_long

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        price = self._get_price(data)
        sma_s = price.rolling(self.sma_short).mean()
        sma_l = price.rolling(self.sma_long).mean()

        signal = pd.Series(0, index=data.index, dtype=int)
        signal[sma_s > sma_l] = 1
        signal[sma_s <= sma_l] = -1

        # Shift to avoid look-ahead bias
        return signal.shift(1).fillna(0).astype(int)
