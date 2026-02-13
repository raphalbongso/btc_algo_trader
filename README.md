# BTC Algorithmic Trading Bot for OKX

A production-grade BTC/USDT algorithmic trading system targeting the **OKX exchange** via **CCXT**. Based on all 10 chapters of *Python for Algorithmic Trading* (Yves Hilpisch, O'Reilly), adapted for cryptocurrency spot and futures trading.

**Target exchange:** OKX (spot + futures)
**Primary pair:** BTC/USDT
**Broker library:** CCXT (`ccxt.okx()`)
**Trading modes:** Paper → OKX Spot → OKX Futures

> **Disclaimer:** This system is for educational purposes. Cryptocurrency trading involves significant risk of financial loss. Futures trading with leverage can result in losses exceeding your initial deposit. Always start with paper trading and small amounts. Never trade money you cannot afford to lose.

---

## Architecture

```
btc_algo_trader/
├── config/                 # OKXConfig, TradingConfig, ZMQConfig, structlog
├── data/                   # OKX OHLCV via CCXT, yfinance fallback, GBM synthetic,
│                           #   WebSocket client, ZeroMQ tick server, HDF5/SQLite storage
├── strategies/             # SMA, momentum, mean reversion, ML (sklearn),
│                           #   DNN (Keras), ensemble voting
├── backtesting/
│   ├── vectorized/         # SMA, momentum, MR, LR, scikit backtesters
│   ├── event_based/        # Long-only and long/short with dual TC (ftc + ptc)
│   └── performance.py      # Sharpe, Sortino, drawdown, VaR, CVaR, Kelly
├── execution/              # OKXExecutor (CCXT), PaperExecutor, OrderManager
├── live/                   # BTCTrader (unified bot), SignalRouter
├── monitoring/             # ZMQ logger, strategy monitor, Plotly dashboard, alerts
├── deployment/             # Automated strategy runner, remote monitoring
├── scripts/                # CLI tools
├── tests/                  # 131 tests (mocked CCXT, no real API calls)
├── Dockerfile
└── docker-compose.yml
```

## Quick Start

### 1. Install

```bash
git clone https://github.com/raphalbongso/btc_algo_trader.git
cd btc_algo_trader
pip install -r requirements.txt
```

### 2. Configure (for live trading only)

```bash
cp .env.example .env
# Edit .env: add your OKX API key, secret, and passphrase
# Get keys at: https://www.okx.com/account/my-api
```

### 3. Backtest (no API key needed)

```bash
python scripts/run_backtest.py --strategy sma --sma-short 20 --sma-long 50
python scripts/run_backtest.py --strategy momentum --momentum 10
python scripts/run_backtest.py --strategy scikit --model adaboost --lags 6
python scripts/run_backtest.py --strategy mr --mr-window 25 --mr-threshold 1.0
python scripts/run_backtest.py --strategy event-short --momentum 20
```

### 4. Kelly Criterion Sizing

```bash
python scripts/kelly_sizing.py --start 2023-01-01 --end 2024-12-31
python scripts/kelly_sizing.py --start 2023-01-01 --end 2024-12-31 --simulate
```

### 5. Optimize Parameters

```bash
python scripts/optimize_params.py --strategy sma
python scripts/optimize_params.py --strategy momentum
```

### 6. Train ML Model

```bash
python scripts/train_model.py --model adaboost --lags 6 --output models/btc_algo.pkl
```

### 7. Paper Trade (no API key needed)

```bash
python scripts/run_live.py --mode paper --strategy momentum --intervals 500
```

### 8. Live Trade (REAL MONEY)

```bash
# Spot (start small!)
python scripts/run_live.py --mode spot --strategy momentum --intervals 100

# Futures (understand the risk!)
python scripts/run_live.py --mode futures --strategy ensemble --leverage 3 --intervals 100
```

### 9. Monitor (separate terminal)

```bash
python deployment/strategy_monitoring.py --port 5555
```

### 10. Run Tests

```bash
pytest tests/ -v --tb=short --timeout=60
```

### 11. Docker

```bash
docker build -t btc-okx-trader .
docker run --env-file .env btc-okx-trader
```

## Strategies

| Strategy | Description | Parameters |
|----------|-------------|------------|
| **SMA** | SMA crossover: long when short SMA > long SMA | `sma_short`, `sma_long` |
| **Momentum** | Long when rolling mean of log returns > 0 | `window` |
| **Mean Reversion** | Short when z-score > threshold, long when below | `window`, `threshold` |
| **ML (sklearn)** | LogisticRegression or AdaBoost on lagged returns | `model_type`, `lags` |
| **DNN** | Keras feed-forward network on lagged returns | `lags`, `hidden_units`, `epochs` |
| **Ensemble** | Majority, unanimous, or weighted voting across strategies | `mode`, `weights` |

All strategies use `.shift(1)` on signals to prevent look-ahead bias. ML strategies use temporal train/test splits with training-only normalization.

## Transaction Cost Model

| Mode | Fee | Default `ptc` |
|------|-----|---------------|
| OKX Spot (taker) | 0.1% | `0.001` |
| OKX Futures (taker) | 0.05% | `0.0005` |
| Event-based | Fixed + proportional | `ftc=0`, `ptc=0.001` |

## Data Fallback Chain

The data loader tries four sources in order:

1. **Local pickle cache** (fastest)
2. **OKX OHLCV via CCXT** (real exchange data, no API key needed)
3. **yfinance** (`BTC-USD`)
4. **Synthetic GBM** (sigma=0.8, always works)

## Risk Management

- **Max drawdown circuit breaker** (15% default) — halts all trading
- **Position size clamping** — never exceeds `units * max_leverage`
- **Half-Kelly sizing** — conservative position sizing
- **VaR/CVaR monitoring** — tail risk awareness
- **Graceful shutdown** — `close_out()` on SIGINT/SIGTERM

## Key Design Decisions

- `TRADING_DAYS = 365` for crypto (24/7 markets)
- Modern APIs: `np.random.default_rng()`, `pd.concat()`, `estimator=` (not `base_estimator=`)
- Spot mode cannot short — signals of -1 are mapped to 0 (flat)
- Futures symbol uses CCXT format: `BTC/USDT:USDT`
- ZeroMQ binds to `127.0.0.1` only (not `0.0.0.0`)

## Chapter-to-Module Mapping

| Chapter | Topic | Module |
|---------|-------|--------|
| 1 | GBM, NumPy vectorization | `data/sample_generator.py` |
| 2 | Docker, cloud | `Dockerfile`, `config/` |
| 3 | Data sources, storage | `data/data_loader.py`, `data/storage.py` |
| 4 | Vectorized backtesting | `backtesting/vectorized/`, `strategies/` |
| 5 | ML prediction | `strategies/ml_strategy.py`, `strategies/dnn_strategy.py` |
| 6 | Event-based backtesting | `backtesting/event_based/` |
| 7 | Streaming, real-time | `data/okx_ws_client.py`, `monitoring/dashboard.py` |
| 8 | Live trading (Oanda → OKX) | `execution/okx_executor.py`, `live/btc_trader.py` |
| 9 | Crypto trading (FXCM → OKX) | `execution/okx_executor.py` (unified) |
| 10 | Kelly, deployment | `backtesting/performance.py`, `deployment/` |

## References

- Hilpisch, Y. J. (2020). *Python for Algorithmic Trading*. O'Reilly.
- [CCXT Library](https://github.com/ccxt/ccxt)
- [OKX API Docs](https://www.okx.com/docs-v5/en/)
