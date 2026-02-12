"""Vectorized mean reversion backtester (inherits from MomVectorBacktester)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.vectorized.mom_backtester import MomVectorBacktester


class MRVectorBacktester(MomVectorBacktester):
    """Mean reversion backtester.

    Inherits from MomVectorBacktester, overrides signal generation.
    Shorts when price is above rolling mean + threshold, longs when below.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'Close' or 'price' column.
    window : int
        Lookback window for rolling stats.
    threshold : float
        Z-score threshold for entry.
    ptc : float
        Proportional transaction cost.
    trading_days : int
        Annualization factor.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        window: int = 25,
        threshold: float = 1.0,
        ptc: float = 0.001,
        trading_days: int = 365,
    ):
        self.window = window
        self.threshold = threshold
        # Initialize parent with momentum=window (reuse data prep)
        super().__init__(data=data, momentum=window, ptc=ptc, trading_days=trading_days)

    def run(self) -> pd.DataFrame:
        data = self.data.copy()

        rolling_mean = data["returns"].rolling(self.window).mean()
        rolling_std = data["returns"].rolling(self.window).std()
        z_score = (data["returns"] - rolling_mean) / rolling_std
        data.dropna(inplace=True)
        z_score = z_score.reindex(data.index)

        # Mean reversion: short high, long low
        data["position"] = pd.Series(0, index=data.index, dtype=float)
        data.loc[z_score > self.threshold, "position"] = -1
        data.loc[z_score < -self.threshold, "position"] = 1
        # Forward fill positions (hold until opposite signal)
        data["position"] = data["position"].replace(0, np.nan).ffill().fillna(0)
        data["position"] = data["position"].shift(1).fillna(0)

        data["strategy"] = data["position"] * data["returns"]
        data["trades"] = data["position"].diff().fillna(0).abs()
        data["strategy_net"] = data["strategy"] - data["trades"] * self.ptc

        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy_net"].cumsum().apply(np.exp)

        self.results = data
        return data

    def optimize(
        self, window_range: range, threshold_range: list[float] | None = None
    ) -> tuple[int, float, float]:
        """Optimize window and threshold.

        Returns (best_window, best_threshold, best_performance).
        """
        if threshold_range is None:
            threshold_range = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

        best = (-np.inf, 0, 0.0)
        for w in window_range:
            for t in threshold_range:
                bt = MRVectorBacktester(
                    self.data[[self._price_col]].copy(),
                    window=w, threshold=t,
                    ptc=self.ptc, trading_days=self.trading_days,
                )
                results = bt.run()
                perf = results["cstrategy"].iloc[-1]
                if perf > best[0]:
                    best = (perf, w, t)
        return best[1], best[2], best[0]
