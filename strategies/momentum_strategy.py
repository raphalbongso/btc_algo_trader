"""Momentum strategy based on rolling mean of log returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base import StrategyBase


class MomentumStrategy(StrategyBase):
    """Go long when rolling mean of log returns is positive, short otherwise.

    Parameters
    ----------
    window : int
        Lookback window for rolling mean (default 20).
    """

    def __init__(self, window: int = 20):
        self.window = window

    @property
    def required_history(self) -> int:
        return self.window

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        price = self._get_price(data)
        log_returns = np.log(price / price.shift(1))
        rolling_mean = log_returns.rolling(self.window).mean()

        signal = pd.Series(0, index=data.index, dtype=int)
        signal[rolling_mean > 0] = 1
        signal[rolling_mean <= 0] = -1

        return signal.shift(1).fillna(0).astype(int)
