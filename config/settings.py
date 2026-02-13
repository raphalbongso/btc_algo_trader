"""Centralized configuration for OKX BTC trading bot.

Priority: (1) Environment variables -> (2) .env file -> (3) Defaults.
API keys NEVER in code. Always load from environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class OKXConfig:
    """OKX exchange credentials and settings."""
    api_key: str = field(default_factory=lambda: os.environ.get("OKX_API_KEY", ""))
    secret_key: str = field(default_factory=lambda: os.environ.get("OKX_SECRET_KEY", ""))
    passphrase: str = field(default_factory=lambda: os.environ.get("OKX_PASSPHRASE", ""))
    trading_type: str = "swap"          # 'spot' or 'swap' (futures)
    leverage: int = 1                   # 1 = no leverage (spot), >1 = futures
    margin_mode: str = "isolated"       # 'isolated' or 'cross'
    sandbox: bool = True                # USE SANDBOX/TESTNET FIRST


@dataclass(frozen=True)
class TradingConfig:
    """Trading parameters."""
    symbol: str = "BTC/USDT"                # Spot symbol
    futures_symbol: str = "BTC/USDT:USDT"   # Futures symbol
    bar_length: str = "1m"                   # OHLCV timeframe
    momentum_window: int = 6
    sma_short: int = 42
    sma_long: int = 252
    ml_lags: int = 6
    ml_window: int = 20
    units: float = 0.001                     # BTC amount per trade
    initial_capital: float = 1000.0          # USDT
    ftc: float = 0.0                         # Fixed transaction cost
    ptc: float = 0.001                       # Proportional TC (OKX spot taker: 0.1%)
    ptc_futures: float = 0.0005              # Futures taker: 0.05%
    trading_days: int = 365                  # Crypto trades 24/7
    risk_free_rate: float = 0.0
    max_leverage: float = 5.0               # Max leverage for futures
    kelly_fraction: float = 0.5             # Half-Kelly
    max_drawdown_pct: float = 0.15          # 15% max drawdown circuit breaker
    features: List[str] = field(
        default_factory=lambda: ["return", "sma", "min", "max", "vol", "mom"]
    )


@dataclass(frozen=True)
class ZMQConfig:
    """ZeroMQ monitoring settings."""
    pub_port: int = 5555
    bind_address: str = "127.0.0.1"  # localhost only -- NOT 0.0.0.0

    @property
    def address(self) -> str:
        return f"tcp://{self.bind_address}:{self.pub_port}"


@dataclass(frozen=True)
class AppConfig:
    okx: OKXConfig = field(default_factory=OKXConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    zmq: ZMQConfig = field(default_factory=ZMQConfig)


def load_config() -> AppConfig:
    """Load all configuration objects with validation."""
    okx = OKXConfig()
    trading = TradingConfig()
    zmq = ZMQConfig()

    if not okx.api_key and not okx.sandbox:
        raise ValueError("OKX_API_KEY required for live trading")
    if trading.max_leverage > 20:
        raise ValueError("Max leverage capped at 20x for safety")

    return AppConfig(okx=okx, trading=trading, zmq=zmq)
