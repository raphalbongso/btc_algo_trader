"""Vectorized SMA crossover backtester."""

from __future__ import annotations

import numpy as np
import pandas as pd


class SMAVectorBacktester:
    """Vectorized backtester for SMA crossover strategy.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'Close' or 'price' column.
    sma_short : int
        Short SMA window.
    sma_long : int
        Long SMA window.
    ptc : float
        Proportional transaction cost per trade.
    trading_days : int
        Annualization factor (365 for crypto).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        sma_short: int = 20,
        sma_long: int = 50,
        ptc: float = 0.001,
        trading_days: int = 365,
    ):
        self.data = data.copy()
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.ptc = ptc
        self.trading_days = trading_days
        self._price_col = "Close" if "Close" in data.columns else "price"
        self._prepare_data()

    def _prepare_data(self) -> None:
        self.data["returns"] = np.log(
            self.data[self._price_col] / self.data[self._price_col].shift(1)
        )
        self.data["SMA_short"] = self.data[self._price_col].rolling(self.sma_short).mean()
        self.data["SMA_long"] = self.data[self._price_col].rolling(self.sma_long).mean()
        self.data.dropna(inplace=True)

    def run(self) -> pd.DataFrame:
        """Run the backtest and return results DataFrame."""
        data = self.data.copy()

        # Signal: 1 when short SMA > long SMA, -1 otherwise
        data["position"] = np.where(data["SMA_short"] > data["SMA_long"], 1, -1)
        # Shift to avoid look-ahead bias
        data["position"] = data["position"].shift(1).fillna(0)

        data["strategy"] = data["position"] * data["returns"]

        # Transaction costs on position changes
        data["trades"] = data["position"].diff().fillna(0).abs()
        data["strategy_net"] = data["strategy"] - data["trades"] * self.ptc

        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy_net"].cumsum().apply(np.exp)

        self.results = data
        return data

    def optimize(
        self, short_range: range, long_range: range
    ) -> tuple[int, int, float]:
        """Brute-force optimize SMA parameters.

        Returns
        -------
        tuple
            (best_short, best_long, best_performance)
        """
        best = (-np.inf, 0, 0)
        for s in short_range:
            for l in long_range:
                if s >= l:
                    continue
                bt = SMAVectorBacktester(
                    self.data[[self._price_col]].copy(),
                    sma_short=s, sma_long=l,
                    ptc=self.ptc, trading_days=self.trading_days,
                )
                results = bt.run()
                perf = results["cstrategy"].iloc[-1]
                if perf > best[0]:
                    best = (perf, s, l)
        return best[1], best[2], best[0]

    def summary(self) -> dict:
        """Return performance summary."""
        if not hasattr(self, "results"):
            self.run()
        r = self.results
        n_years = len(r) / self.trading_days
        strategy_return = r["cstrategy"].iloc[-1] - 1
        buy_hold_return = r["creturns"].iloc[-1] - 1
        ann_return = (1 + strategy_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        ann_vol = r["strategy_net"].std() * np.sqrt(self.trading_days)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        n_trades = r["trades"].sum()

        return {
            "strategy_return": strategy_return,
            "buy_hold_return": buy_hold_return,
            "outperformance": strategy_return - buy_hold_return,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "n_trades": n_trades,
            "sma_short": self.sma_short,
            "sma_long": self.sma_long,
        }
