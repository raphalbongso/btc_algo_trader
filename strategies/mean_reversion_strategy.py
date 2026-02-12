"""Mean reversion strategy with threshold."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base import StrategyBase


class MeanReversionStrategy(StrategyBase):
    """Mean reversion: short when price is above SMA+threshold, long when below.

    Parameters
    ----------
    window : int
        SMA lookback window (default 25).
    threshold : float
        Standard deviations from mean to trigger signal (default 1.0).
    """

    def __init__(self, window: int = 25, threshold: float = 1.0):
        self.window = window
        self.threshold = threshold

    @property
    def required_history(self) -> int:
        return self.window

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        price = self._get_price(data)
        log_returns = np.log(price / price.shift(1))

        rolling_mean = log_returns.rolling(self.window).mean()
        rolling_std = log_returns.rolling(self.window).std()

        # Z-score of current return relative to rolling window
        z_score = (log_returns - rolling_mean) / rolling_std

        signal = pd.Series(0, index=data.index, dtype=int)
        # Mean reversion: go short when price is high, long when low
        signal[z_score > self.threshold] = -1
        signal[z_score < -self.threshold] = 1

        return signal.shift(1).fillna(0).astype(int)
