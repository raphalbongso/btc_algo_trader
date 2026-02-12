"""Vectorized momentum backtester."""

from __future__ import annotations

import numpy as np
import pandas as pd


class MomVectorBacktester:
    """Vectorized backtester for momentum strategy.

    Goes long when rolling mean of log returns > 0, short otherwise.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'Close' or 'price' column.
    momentum : int
        Lookback window for rolling mean.
    ptc : float
        Proportional transaction cost.
    trading_days : int
        Annualization factor.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        momentum: int = 20,
        ptc: float = 0.001,
        trading_days: int = 365,
    ):
        self.data = data.copy()
        self.momentum = momentum
        self.ptc = ptc
        self.trading_days = trading_days
        self._price_col = "Close" if "Close" in data.columns else "price"
        self._prepare_data()

    def _prepare_data(self) -> None:
        self.data["returns"] = np.log(
            self.data[self._price_col] / self.data[self._price_col].shift(1)
        )
        self.data.dropna(inplace=True)

    def run(self) -> pd.DataFrame:
        data = self.data.copy()
        data["rolling_mean"] = data["returns"].rolling(self.momentum).mean()
        data.dropna(inplace=True)

        data["position"] = np.sign(data["rolling_mean"])
        data["position"] = data["position"].shift(1).fillna(0)

        data["strategy"] = data["position"] * data["returns"]
        data["trades"] = data["position"].diff().fillna(0).abs()
        data["strategy_net"] = data["strategy"] - data["trades"] * self.ptc

        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy_net"].cumsum().apply(np.exp)

        self.results = data
        return data

    def optimize(self, momentum_range: range) -> tuple[int, float]:
        """Brute-force optimize momentum window.

        Returns (best_momentum, best_performance).
        """
        best = (-np.inf, 0)
        for m in momentum_range:
            bt = MomVectorBacktester(
                self.data[[self._price_col]].copy(),
                momentum=m, ptc=self.ptc, trading_days=self.trading_days,
            )
            results = bt.run()
            perf = results["cstrategy"].iloc[-1]
            if perf > best[0]:
                best = (perf, m)
        return best[1], best[0]

    def summary(self) -> dict:
        if not hasattr(self, "results"):
            self.run()
        r = self.results
        n_years = len(r) / self.trading_days
        strategy_return = r["cstrategy"].iloc[-1] - 1
        buy_hold_return = r["creturns"].iloc[-1] - 1
        ann_return = (1 + strategy_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        ann_vol = r["strategy_net"].std() * np.sqrt(self.trading_days)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        return {
            "strategy_return": strategy_return,
            "buy_hold_return": buy_hold_return,
            "outperformance": strategy_return - buy_hold_return,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "n_trades": r["trades"].sum(),
            "momentum": self.momentum,
        }
