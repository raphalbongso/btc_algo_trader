"""Event-based backtesting base class with dual transaction costs."""

from __future__ import annotations

import numpy as np
import pandas as pd


class BacktestBase:
    """Base class for event-based backtesting.

    Simulates bar-by-bar trading with both fixed and proportional
    transaction costs.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'Close' or 'price' column.
    initial_capital : float
        Starting capital.
    ftc : float
        Fixed transaction cost per trade (e.g., $1.00).
    ptc : float
        Proportional transaction cost (e.g., 0.001 = 0.1%).
    trading_days : int
        Annualization factor.
    verbose : bool
        Print trade details.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100_000.0,
        ftc: float = 0.0,
        ptc: float = 0.001,
        trading_days: int = 365,
        verbose: bool = False,
    ):
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.ftc = ftc
        self.ptc = ptc
        self.trading_days = trading_days
        self.verbose = verbose

        self._price_col = "Close" if "Close" in data.columns else "price"
        self.prices = self.data[self._price_col].values
        self.dates = self.data.index

        # State
        self.position = 0  # Current units held
        self.trades = 0
        self.cash = initial_capital
        self.portfolio_values: list[float] = []
        self.trade_log: list[dict] = []

    def _get_price(self, bar: int) -> float:
        return self.prices[bar]

    def _get_date(self, bar: int):
        return self.dates[bar]

    def _portfolio_value(self, bar: int) -> float:
        return self.cash + self.position * self._get_price(bar)

    def _execute_trade(self, bar: int, units: int) -> None:
        """Execute a trade at the given bar.

        Parameters
        ----------
        bar : int
            Current bar index.
        units : int
            Number of units to trade (positive=buy, negative=sell).
        """
        price = self._get_price(bar)
        cost = units * price
        tc = self.ftc + abs(cost) * self.ptc

        self.cash -= cost + tc
        self.position += units
        self.trades += 1

        if self.verbose:
            action = "BUY" if units > 0 else "SELL"
            print(
                f"{self._get_date(bar)} | {action} {abs(units)} @ {price:.2f} "
                f"| TC={tc:.2f} | Cash={self.cash:.2f}"
            )

        self.trade_log.append({
            "date": self._get_date(bar),
            "action": "BUY" if units > 0 else "SELL",
            "units": abs(units),
            "price": price,
            "tc": tc,
            "cash": self.cash,
            "position": self.position,
        })

    def _close_position(self, bar: int) -> None:
        """Close any open position."""
        if self.position != 0:
            self._execute_trade(bar, -self.position)

    def run(self) -> dict:
        """Run the backtest. Must be implemented by subclasses."""
        raise NotImplementedError

    def summary(self) -> dict:
        """Compute performance summary from portfolio_values."""
        if not self.portfolio_values:
            return {}

        pv = np.array(self.portfolio_values)
        returns = np.diff(np.log(pv))

        final_value = pv[-1]
        total_return = (final_value / self.initial_capital) - 1
        n_years = len(pv) / self.trading_days

        ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        ann_vol = returns.std() * np.sqrt(self.trading_days) if len(returns) > 1 else 0
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # Max drawdown
        peak = np.maximum.accumulate(pv)
        drawdown = (pv - peak) / peak
        max_dd = drawdown.min()

        return {
            "initial_capital": self.initial_capital,
            "final_value": final_value,
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "n_trades": self.trades,
            "n_bars": len(pv),
        }
