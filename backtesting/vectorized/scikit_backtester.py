"""Vectorized backtester using sklearn classifiers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression


class ScikitVectorBacktester:
    """Vectorized backtester using sklearn classification models.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'Close' or 'price' column.
    model_type : str
        "logistic" or "adaboost".
    lags : int
        Number of lagged return features.
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
        model_type: str = "logistic",
        lags: int = 5,
        train_ratio: float = 0.7,
        ptc: float = 0.001,
        trading_days: int = 365,
    ):
        self.data = data.copy()
        self.model_type = model_type
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
        self.data["direction"] = np.sign(self.data["returns"]).astype(int)

    def _create_model(self):
        if self.model_type == "adaboost":
            return AdaBoostClassifier(n_estimators=50, random_state=42)
        return LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    def _get_feature_cols(self) -> list[str]:
        return [f"lag_{i}" for i in range(1, self.lags + 1)]

    def run(self) -> pd.DataFrame:
        data = self.data.copy()
        feature_cols = self._get_feature_cols()

        split = int(len(data) * self.train_ratio)
        train = data.iloc[:split]

        # Training-only normalization
        mu = train[feature_cols].mean()
        std = train[feature_cols].std()

        X_train = (train[feature_cols] - mu) / std
        y_train = train["direction"]

        model = self._create_model()
        model.fit(X_train, y_train)

        X_all = (data[feature_cols] - mu) / std
        data["position"] = model.predict(X_all)
        data["position"] = data["position"].shift(1).fillna(0)

        data["strategy"] = data["position"] * data["returns"]
        data["trades"] = data["position"].diff().fillna(0).abs()
        data["strategy_net"] = data["strategy"] - data["trades"] * self.ptc

        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy_net"].cumsum().apply(np.exp)

        data["split"] = "train"
        data.iloc[split:, data.columns.get_loc("split")] = "test"

        self.results = data
        self.model = model
        return data

    def optimize(self, lag_range: range) -> tuple[int, float]:
        """Optimize number of lags. Returns (best_lags, best_performance)."""
        best = (-np.inf, 0)
        for l in lag_range:
            bt = ScikitVectorBacktester(
                self.data[[self._price_col]].copy(),
                model_type=self.model_type, lags=l,
                train_ratio=self.train_ratio,
                ptc=self.ptc, trading_days=self.trading_days,
            )
            results = bt.run()
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

        return {
            "strategy_return_log": test["strategy_net"].sum(),
            "buy_hold_return_log": test["returns"].sum(),
            "outperformance_log": test["strategy_net"].sum() - test["returns"].sum(),
            "test_samples": len(test),
            "model_type": self.model_type,
            "lags": self.lags,
            "train_accuracy": (
                r[r["split"] == "train"]["position"].shift(-1) == r[r["split"] == "train"]["direction"]
            ).mean(),
        }
