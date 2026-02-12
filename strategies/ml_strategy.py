"""Machine learning strategy using sklearn classifiers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

from strategies.base import StrategyBase


class MLStrategy(StrategyBase):
    """Classification-based strategy using sklearn.

    Trains on lagged log returns to predict direction.

    Parameters
    ----------
    model_type : str
        "logistic" or "adaboost".
    lags : int
        Number of lagged return features.
    train_ratio : float
        Fraction of data used for training (temporal split).
    """

    def __init__(
        self,
        model_type: str = "logistic",
        lags: int = 5,
        train_ratio: float = 0.7,
    ):
        self.model_type = model_type
        self.lags = lags
        self.train_ratio = train_ratio
        self.model = None
        self._fit_cols: list[str] = []

    @property
    def required_history(self) -> int:
        return self.lags + 1

    def _create_model(self):
        if self.model_type == "adaboost":
            return AdaBoostClassifier(n_estimators=50, random_state=42)
        return LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create lagged log-return features."""
        df = data.copy()
        price = self._get_price(df)
        df["log_returns"] = np.log(price / price.shift(1))
        df["direction"] = np.sign(df["log_returns"]).astype(int)

        for lag in range(1, self.lags + 1):
            df[f"lag_{lag}"] = df["log_returns"].shift(lag)

        return df.dropna()

    def _get_feature_cols(self) -> list[str]:
        return [f"lag_{i}" for i in range(1, self.lags + 1)]

    def fit(self, data: pd.DataFrame) -> None:
        """Train the model on the given data.

        Uses temporal split — only trains on the first train_ratio of data.
        Normalization stats are computed on training data only.
        """
        df = self.prepare_features(data)
        feature_cols = self._get_feature_cols()
        self._fit_cols = feature_cols

        split = int(len(df) * self.train_ratio)
        train = df.iloc[:split]

        # Training-only normalization
        self._mu = train[feature_cols].mean()
        self._std = train[feature_cols].std()

        X_train = (train[feature_cols] - self._mu) / self._std
        y_train = train["direction"]

        self.model = self._create_model()
        self.model.fit(X_train, y_train)

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals. Calls fit() if model not yet trained."""
        if self.model is None:
            self.fit(data)

        df = self.prepare_features(data)
        feature_cols = self._get_feature_cols()

        X = (df[feature_cols] - self._mu) / self._std
        predictions = self.model.predict(X)

        signal = pd.Series(predictions, index=df.index, dtype=int)
        # Reindex to original data index, fill with 0
        signal = signal.reindex(data.index, fill_value=0)

        return signal.shift(1).fillna(0).astype(int)
