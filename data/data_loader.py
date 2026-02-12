"""Multi-source BTC/USDT data loader.

Fallback chain:
  Level 1: Local pickle cache
  Level 2: MEXC OHLCV via CCXT (live exchange data, no key needed)
  Level 3: yfinance ('BTC-USD')
  Level 4: Synthetic GBM (WARNING logged)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from data.sample_generator import generate_btc_data

logger = get_logger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


def _cache_path(symbol: str, start: str, end: str, timeframe: str = "1d") -> Path:
    """Deterministic cache file path."""
    key = f"{symbol}_{timeframe}_{start}_{end}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    safe_sym = symbol.replace("/", "_").replace(":", "_")
    return CACHE_DIR / f"{safe_sym}_{h}.pkl"


def load_btc_data(
    symbol: str = "BTC/USDT",
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    timeframe: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load BTC/USDT OHLCV data with 4-level fallback chain.

    Parameters
    ----------
    symbol : str
        Trading pair (CCXT format).
    start, end : str
        Date range (YYYY-MM-DD).
    timeframe : str
        OHLCV timeframe ('1m', '5m', '1h', '1d').
    use_cache : bool
        Whether to use/update the pickle cache.

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with DatetimeIndex and columns:
        Open, High, Low, Close, Volume, price, returns.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = _cache_path(symbol, start, end, timeframe)

    # Level 1: Cache
    if use_cache and cache_file.exists():
        logger.info("loading_from_cache", path=str(cache_file))
        return pd.read_pickle(cache_file)

    # Level 2: MEXC via CCXT
    try:
        df = _fetch_mexc_ohlcv(symbol, start, end, timeframe)
        if len(df) > 50:
            if use_cache:
                df.to_pickle(cache_file)
                logger.info("cached_data", path=str(cache_file))
            return df
        logger.warning("mexc_insufficient_data", rows=len(df))
    except Exception as e:
        logger.warning("mexc_fetch_failed", error=str(e))

    # Level 3: yfinance
    try:
        import yfinance as yf

        yf_symbol = "BTC-USD"
        logger.info("downloading_from_yfinance", symbol=yf_symbol, start=start, end=end)
        df = yf.download(yf_symbol, start=start, end=end, auto_adjust=True, progress=False)
        if df is not None and len(df) > 50:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index.name = "Date"
            df["price"] = df["Close"]
            df["returns"] = np.log(df["price"] / df["price"].shift(1))
            df.dropna(inplace=True)
            if use_cache:
                df.to_pickle(cache_file)
                logger.info("cached_data", path=str(cache_file))
            return df
        logger.warning("yfinance_insufficient_data", rows=len(df) if df is not None else 0)
    except Exception as e:
        logger.warning("yfinance_failed", error=str(e))

    # Level 4: Synthetic GBM
    logger.warning("all_sources_failed_using_synthetic_data")
    df = generate_btc_data(start=start, end=end)
    df["price"] = df["Close"]
    df["returns"] = np.log(df["price"] / df["price"].shift(1))
    df.dropna(inplace=True)
    if use_cache:
        df.to_pickle(cache_file)
    return df


def _fetch_mexc_ohlcv(
    symbol: str,
    start: str,
    end: str,
    timeframe: str,
) -> pd.DataFrame:
    """Fetch OHLCV from MEXC via CCXT. No API key needed for public data."""
    import ccxt

    exchange = ccxt.mexc({"enableRateLimit": True})
    exchange.load_markets()

    since = exchange.parse8601(f"{start}T00:00:00Z")
    end_ts = exchange.parse8601(f"{end}T00:00:00Z")

    all_ohlcv = []
    logger.info("fetching_mexc_ohlcv", symbol=symbol, timeframe=timeframe)

    while since < end_ts:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1  # Next ms after last candle

    if not all_ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="first")]

    df["price"] = df["Close"]
    df["returns"] = np.log(df["price"] / df["price"].shift(1))
    df.dropna(inplace=True)

    logger.info("mexc_ohlcv_fetched", bars=len(df))
    return df
