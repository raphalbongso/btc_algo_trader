"""Order lifecycle management, position tracking, and risk limits."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from execution.broker_base import BrokerBase

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Tracks a single position."""
    side: int = 0          # +1 long, -1 short, 0 flat
    entry_price: float = 0.0
    amount: float = 0.0
    unrealized_pnl: float = 0.0


class OrderManager:
    """Manages order lifecycle, position tracking, and risk limits.

    Parameters
    ----------
    executor : BrokerBase
        The execution backend (OKX or paper).
    max_position_size : float
        Maximum BTC position size.
    max_drawdown_pct : float
        Circuit breaker: halt trading if drawdown exceeds this.
    initial_capital : float
        Starting capital for drawdown calculation.
    """

    def __init__(
        self,
        executor: BrokerBase,
        max_position_size: float = 0.01,
        max_drawdown_pct: float = 0.15,
        initial_capital: float = 1000.0,
    ):
        self.executor = executor
        self.max_position_size = max_position_size
        self.max_drawdown_pct = max_drawdown_pct
        self.initial_capital = initial_capital

        self.position = Position()
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [initial_capital]
        self._halted = False

    @property
    def is_halted(self) -> bool:
        return self._halted

    def update_equity(self, current_price: float) -> float:
        """Update equity curve with current mark-to-market value."""
        if self.position.side != 0:
            self.position.unrealized_pnl = (
                (current_price - self.position.entry_price)
                * self.position.amount
                * self.position.side
            )

        balance = self.executor.get_balance()
        equity = balance.get("USDT_total", self.equity_curve[-1])
        self.equity_curve.append(equity)
        return equity

    def check_risk(self) -> bool:
        """Check if drawdown exceeds max allowed. Returns True if OK."""
        if len(self.equity_curve) < 2:
            return True

        eq = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = (eq[-1] - peak[-1]) / peak[-1] if peak[-1] > 0 else 0

        if abs(dd) > self.max_drawdown_pct:
            logger.critical(
                f"MAX DRAWDOWN BREACHED: {dd:.2%} > {self.max_drawdown_pct:.2%}"
            )
            self._halted = True
            return False
        return True

    def execute_signal(self, signal: int, amount: float, current_price: float) -> Optional[Dict]:
        """Execute a trade based on signal vs current position.

        Parameters
        ----------
        signal : int
            Target position: +1 (long), -1 (short), 0 (flat).
        amount : float
            Trade size in BTC.
        current_price : float
            Current market price (for paper executor).

        Returns
        -------
        dict or None
            Order result, or None if no trade needed.
        """
        if self._halted:
            logger.warning("Trading halted due to risk limit -- forcing flat")
            signal = 0

        if signal == self.position.side:
            return None  # No change

        # Validate position size
        if amount > self.max_position_size:
            logger.warning(
                f"Amount {amount} exceeds max {self.max_position_size}, clamping"
            )
            amount = self.max_position_size

        order = None

        # Close existing position first
        if self.position.side == 1:
            order = self.executor.market_sell(self.position.amount)
        elif self.position.side == -1:
            order = self.executor.market_buy(self.position.amount)

        if self.position.side != 0:
            self._record_close(current_price)

        # Open new position
        if signal == 1:
            order = self.executor.market_buy(amount)
            self.position = Position(side=1, entry_price=current_price, amount=amount)
        elif signal == -1:
            order = self.executor.market_sell(amount)
            self.position = Position(side=-1, entry_price=current_price, amount=amount)
        else:
            self.position = Position()

        return order

    def close_all(self, current_price: float) -> None:
        """Close all positions."""
        if self.position.side == 1:
            self.executor.market_sell(self.position.amount)
        elif self.position.side == -1:
            self.executor.market_buy(self.position.amount)
        self._record_close(current_price)
        self.position = Position()
        self.executor.cancel_all_orders()
        logger.info("All positions closed")

    def _record_close(self, exit_price: float) -> None:
        if self.position.side == 0:
            return
        pnl = (
            (exit_price - self.position.entry_price)
            * self.position.amount
            * self.position.side
        )
        self.trades.append({
            "side": self.position.side,
            "entry": self.position.entry_price,
            "exit": exit_price,
            "amount": self.position.amount,
            "pnl": pnl,
        })

    def summary(self) -> Dict:
        """Return trading summary."""
        if not self.trades:
            return {"n_trades": 0}

        pnls = [t["pnl"] for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        return {
            "n_trades": len(self.trades),
            "total_pnl": sum(pnls),
            "win_rate": len(wins) / len(pnls) if pnls else 0,
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "largest_win": max(pnls),
            "largest_loss": min(pnls),
            "final_equity": self.equity_curve[-1] if self.equity_curve else 0,
        }
