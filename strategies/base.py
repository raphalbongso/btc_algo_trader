"""Abstract base class for all trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class StrategyBase(ABC):
    """Base class for trading strategies.

    All strategies must implement generate_signal() and declare
    how much historical data they need via required_history.
    """

    @property
    @abstractmethod
    def required_history(self) -> int:
        """Minimum number of bars needed before generating a signal."""
        ...

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from price data.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain at least a 'Close' or 'price' column.

        Returns
        -------
        pd.Series
            Signal series with values in {-1, 0, 1}.
            Signals are shifted by 1 to avoid look-ahead bias.
        """
        ...

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optional feature engineering step.

        Override in subclasses that need derived features (e.g., ML strategies).
        Default implementation returns log returns.
        """
        df = data.copy()
        price_col = "Close" if "Close" in df.columns else "price"
        df["returns"] = df[price_col].pct_change()
        df["log_returns"] = pd.Series(
            data=__import__("numpy").log(df[price_col] / df[price_col].shift(1)),
            index=df.index,
        )
        return df

    def _get_price(self, data: pd.DataFrame) -> pd.Series:
        """Extract price series from DataFrame (supports 'Close' or 'price')."""
        if "Close" in data.columns:
            return data["Close"]
        if "price" in data.columns:
            return data["price"]
        raise ValueError("DataFrame must contain 'Close' or 'price' column")
