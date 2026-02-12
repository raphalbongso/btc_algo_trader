"""Vectorized linear regression backtester."""

from __future__ import annotations

import numpy as np
import pandas as pd


class LRVectorBacktester:
    """Vectorized backtester using linear regression on lagged returns.

    Predicts direction of next return using OLS on lagged features.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'Close' or 'price' column.
    lags : int
        Number of lagged features.
    train_ratio : float
        Temporal train/test split.
    ptc : float
        Proportional transaction cost.
    trading_days : int
        Annualization factor.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        lags: int = 5,
        train_ratio: float = 0.7,
        ptc: float = 0.001,
        trading_days: int = 365,
    ):
        self.data = data.copy()
        self.lags = lags
        self.train_ratio = train_ratio
        self.ptc = ptc
        self.trading_days = trading_days
        self._price_col = "Close" if "Close" in data.columns else "price"
        self._prepare_data()

    def _prepare_data(self) -> None:
        self.data["returns"] = np.log(
            self.data[self._price_col] / self.data[self._price_col].shift(1)
        )
        for lag in range(1, self.lags + 1):
            self.data[f"lag_{lag}"] = self.data["returns"].shift(lag)
        self.data.dropna(inplace=True)

    def _get_feature_cols(self) -> list[str]:
        return [f"lag_{i}" for i in range(1, self.lags + 1)]

    def run(self) -> pd.DataFrame:
        data = self.data.copy()
        feature_cols = self._get_feature_cols()

        split = int(len(data) * self.train_ratio)
        train = data.iloc[:split]
        test = data.iloc[split:]

        # Train-only normalization
        mu = train[feature_cols].mean()
        std = train[feature_cols].std()

        X_train = (train[feature_cols] - mu) / std
        y_train = np.sign(train["returns"])

        # Simple OLS via numpy
        X_matrix = np.column_stack([np.ones(len(X_train)), X_train.values])
        beta = np.linalg.lstsq(X_matrix, y_train.values, rcond=None)[0]

        # Predict on full dataset
        X_all = (data[feature_cols] - mu) / std
        X_all_matrix = np.column_stack([np.ones(len(X_all)), X_all.values])
        predictions = X_all_matrix @ beta

        data["position"] = np.sign(predictions)
        data["position"] = data["position"].shift(1).fillna(0)

        data["strategy"] = data["position"] * data["returns"]
        data["trades"] = data["position"].diff().fillna(0).abs()
        data["strategy_net"] = data["strategy"] - data["trades"] * self.ptc

        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy_net"].cumsum().apply(np.exp)

        # Mark train/test split
        data["split"] = "train"
        data.iloc[split:, data.columns.get_loc("split")] = "test"

        self.results = data
        return data

    def optimize(self, lag_range: range) -> tuple[int, float]:
        """Optimize number of lags. Returns (best_lags, best_performance)."""
        best = (-np.inf, 0)
        for l in lag_range:
            bt = LRVectorBacktester(
                self.data[[self._price_col]].copy(),
                lags=l, train_ratio=self.train_ratio,
                ptc=self.ptc, trading_days=self.trading_days,
            )
            results = bt.run()
            # Evaluate on test set only
            test = results[results["split"] == "test"]
            perf = test["strategy_net"].sum()
            if perf > best[0]:
                best = (perf, l)
        return best[1], best[0]

    def summary(self) -> dict:
        if not hasattr(self, "results"):
            self.run()
        r = self.results
        test = r[r["split"] == "test"]
        n_years = len(test) / self.trading_days

        strategy_return = test["strategy_net"].sum()
        buy_hold_return = test["returns"].sum()

        return {
            "strategy_return_log": strategy_return,
            "buy_hold_return_log": buy_hold_return,
            "outperformance_log": strategy_return - buy_hold_return,
            "test_samples": len(test),
            "n_years": n_years,
            "lags": self.lags,
        }
