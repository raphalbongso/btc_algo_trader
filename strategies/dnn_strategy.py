"""Deep Neural Network strategy using TensorFlow/Keras."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base import StrategyBase


class DNNStrategy(StrategyBase):
    """DNN classification strategy.

    Uses a feed-forward neural network to predict price direction
    from lagged log returns.

    Parameters
    ----------
    lags : int
        Number of lagged return features.
    hidden_units : tuple[int, ...]
        Neurons per hidden layer.
    dropout : float
        Dropout rate.
    epochs : int
        Training epochs.
    train_ratio : float
        Temporal train/test split ratio.
    """

    def __init__(
        self,
        lags: int = 5,
        hidden_units: tuple[int, ...] = (64, 32),
        dropout: float = 0.3,
        epochs: int = 50,
        train_ratio: float = 0.7,
    ):
        self.lags = lags
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.epochs = epochs
        self.train_ratio = train_ratio
        self.model = None
        self._mu = None
        self._std = None

    @property
    def required_history(self) -> int:
        return self.lags + 1

    def _build_model(self, input_dim: int):
        import tensorflow as tf

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(self.hidden_units[0], activation="relu",
                                        input_shape=(input_dim,)))
        model.add(tf.keras.layers.Dropout(self.dropout))

        for units in self.hidden_units[1:]:
            model.add(tf.keras.layers.Dense(units, activation="relu"))
            model.add(tf.keras.layers.Dropout(self.dropout))

        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        price = self._get_price(df)
        df["log_returns"] = np.log(price / price.shift(1))
        df["direction"] = np.where(df["log_returns"] > 0, 1, 0)

        for lag in range(1, self.lags + 1):
            df[f"lag_{lag}"] = df["log_returns"].shift(lag)

        return df.dropna()

    def _get_feature_cols(self) -> list[str]:
        return [f"lag_{i}" for i in range(1, self.lags + 1)]

    def fit(self, data: pd.DataFrame) -> None:
        """Train the DNN. Normalization uses training data only."""
        df = self.prepare_features(data)
        feature_cols = self._get_feature_cols()

        split = int(len(df) * self.train_ratio)
        train = df.iloc[:split]

        self._mu = train[feature_cols].mean()
        self._std = train[feature_cols].std()

        X_train = (train[feature_cols] - self._mu) / self._std
        y_train = train["direction"]

        self.model = self._build_model(len(feature_cols))
        self.model.fit(X_train, y_train, epochs=self.epochs,
                       batch_size=32, verbose=0, validation_split=0.15)

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        if self.model is None:
            self.fit(data)

        df = self.prepare_features(data)
        feature_cols = self._get_feature_cols()

        X = (df[feature_cols] - self._mu) / self._std
        proba = self.model.predict(X, verbose=0).flatten()
        predictions = np.where(proba > 0.5, 1, -1)

        signal = pd.Series(predictions, index=df.index, dtype=int)
        signal = signal.reindex(data.index, fill_value=0)

        return signal.shift(1).fillna(0).astype(int)
