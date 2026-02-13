"""Unified BTC/USDT Trading Bot for OKX.

Combines Ch.7 streaming, Ch.8 momentum trading, Ch.10 ML + Kelly sizing.

Trading modes:
- 'paper': Simulated execution (no exchange connection)
- 'spot':  OKX spot trading (BTC/USDT)
- 'futures': OKX USDT-M perpetual futures (BTC/USDT:USDT)

CRITICAL: In 'spot'/'futures' modes this trades REAL money on OKX.
Always test with 'paper' mode first, then small amounts.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
import ccxt

from config.settings import AppConfig, OKXConfig, TradingConfig, load_config
from strategies.base import StrategyBase
from execution.okx_executor import OKXExecutor
from execution.paper_executor import PaperExecutor
from execution.order_manager import OrderManager

logger = logging.getLogger(__name__)


class BTCTrader:
    """Unified BTC trading bot targeting OKX exchange."""

    def __init__(
        self,
        strategy: StrategyBase,
        mode: str = "paper",
        trading_config: Optional[TradingConfig] = None,
        okx_config: Optional[OKXConfig] = None,
        zmq_publish: bool = False,
    ):
        if mode not in ("paper", "spot", "futures"):
            raise ValueError("mode must be 'paper', 'spot', or 'futures'")

        self.strategy = strategy
        self.mode = mode
        self.config = trading_config or TradingConfig()
        self.okx_config = okx_config or OKXConfig()

        # State
        self.bar_data = pd.DataFrame()
        self._tick_count = 0

        # Build executor
        if mode == "paper":
            executor = PaperExecutor(
                initial_usdt=self.config.initial_capital,
                ptc=self.config.ptc,
            )
        else:
            executor = OKXExecutor(
                trading_type="spot" if mode == "spot" else "swap",
                leverage=int(self.okx_config.leverage),
                margin_mode=self.okx_config.margin_mode,
                sandbox=self.okx_config.sandbox,
            )

        # Auto-detect real balance for live modes
        initial_capital = self.config.initial_capital
        if mode != "paper":
            try:
                bal = executor.get_balance()
                real_usdt = bal.get("USDT_free", 0)
                if real_usdt > 0:
                    initial_capital = real_usdt
                    logger.info(f"Detected account balance: {real_usdt:.2f} USDT")
            except Exception as e:
                logger.warning(f"Could not fetch balance, using config: {e}")

        self.order_manager = OrderManager(
            executor=executor,
            max_position_size=self.config.units * self.config.max_leverage,
            max_drawdown_pct=self.config.max_drawdown_pct,
            initial_capital=initial_capital,
        )

        # CCXT for public data (no key needed)
        self._exchange = ccxt.okx({"enableRateLimit": True})

        # ZeroMQ publisher
        self._zmq_socket = None
        if zmq_publish:
            import zmq
            ctx = zmq.Context()
            self._zmq_socket = ctx.socket(zmq.PUB)
            self._zmq_socket.bind("tcp://127.0.0.1:5555")

        logger.info(f"BTCTrader initialized: mode={mode}")

    def run_polling(self, intervals: int = 1000, sleep_sec: float = 5.0) -> None:
        """Main trading loop using REST polling.

        For each interval:
        1. Fetch latest OHLCV from OKX
        2. Generate signal from strategy
        3. Risk check
        4. Execute if signal changed
        """
        logger.info(f"Starting polling: {intervals} intervals, {sleep_sec}s sleep")

        for i in range(intervals):
            try:
                self._poll_once()
            except ccxt.BaseError as e:
                logger.error(f"Exchange error in loop: {e}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")

            time.sleep(sleep_sec)

        self.close_out()

    def _poll_once(self) -> None:
        """Single polling iteration."""
        history_needed = self.strategy.required_history + 10
        ohlcv = self._exchange.fetch_ohlcv(
            "BTC/USDT", self.config.bar_length, limit=min(history_needed, 1000)
        )
        if not ohlcv:
            return

        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df["price"] = df["Close"]
        df["returns"] = np.log(df["price"] / df["price"].shift(1))
        df.dropna(inplace=True)
        self.bar_data = df

        if len(self.bar_data) < self.strategy.required_history:
            logger.debug(
                f"Warming up: {len(self.bar_data)}/{self.strategy.required_history}"
            )
            return

        # Generate signal
        signal = self.strategy.generate_signal(self.bar_data)
        if isinstance(signal, pd.Series):
            signal = int(signal.iloc[-1])

        # Risk check
        current_price = float(self.bar_data["price"].iloc[-1])
        self.order_manager.update_equity(current_price)

        if not self.order_manager.check_risk():
            logger.critical("RISK LIMIT HIT -- forcing neutral")
            signal = 0

        # Spot/paper cannot short
        if self.mode in ("spot", "paper") and signal == -1:
            logger.warning("Cannot short on spot/paper -- going neutral")
            signal = 0

        # Update paper executor's price
        if self.mode == "paper":
            self.order_manager.executor.set_price(current_price)

        # Execute
        self.order_manager.execute_signal(signal, self.config.units, current_price)
        self._tick_count += 1

        if self._tick_count % 10 == 0:
            equity = self.order_manager.equity_curve[-1]
            pos = self.order_manager.position.side
            logger.info(
                f"Tick {self._tick_count} | Price: {current_price:.2f} | "
                f"Pos: {pos} | Equity: {equity:.2f}"
            )

        # ZMQ publish
        if self._zmq_socket:
            msg = (
                f"TICK | {dt.datetime.now()} | price={current_price:.2f} | "
                f"signal={signal} | pos={self.order_manager.position.side}"
            )
            self._zmq_socket.send_string(msg)

    def run_on_data(self, data: pd.DataFrame) -> Dict:
        """Run strategy on historical data (for paper backtesting via live engine).

        Parameters
        ----------
        data : pd.DataFrame
            Must have 'Close' or 'price' column.

        Returns
        -------
        dict
            Trading summary from order manager.
        """
        price_col = "Close" if "Close" in data.columns else "price"

        for i in range(self.strategy.required_history, len(data)):
            window = data.iloc[:i + 1]
            signal = self.strategy.generate_signal(window)
            if isinstance(signal, pd.Series):
                signal = int(signal.iloc[-1])

            current_price = float(window[price_col].iloc[-1])

            if self.mode == "paper":
                self.order_manager.executor.set_price(current_price)

            if self.mode in ("spot", "paper") and signal == -1:
                signal = 0

            self.order_manager.update_equity(current_price)
            if not self.order_manager.check_risk():
                signal = 0

            self.order_manager.execute_signal(signal, self.config.units, current_price)

        # Close out
        final_price = float(data[price_col].iloc[-1])
        self.order_manager.close_all(final_price)

        return self.order_manager.summary()

    def close_out(self) -> None:
        """Close all positions and shut down."""
        if len(self.bar_data) > 0:
            price = float(self.bar_data["price"].iloc[-1])
        else:
            price = 0

        if self.mode == "paper":
            self.order_manager.executor.set_price(price)

        self.order_manager.close_all(price)
        logger.info("BTCTrader shut down")

    def get_summary(self) -> Dict:
        return self.order_manager.summary()
