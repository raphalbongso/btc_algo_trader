"""MEXC WebSocket client for real-time tick data.

Uses raw websockets for streaming BTC/USDT trades.
Preferred over REST polling for live trading (lower latency).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

MEXC_WS_URL = "wss://wbs.mexc.com/ws"


class MEXCWebSocketClient:
    """WebSocket client for MEXC real-time data."""

    def __init__(self, on_tick: Optional[Callable] = None):
        self._on_tick = on_tick
        self._running = False

    async def subscribe_trades(self, symbol: str = "BTCUSDT") -> None:
        """Subscribe to real-time trade stream."""
        import websockets

        self._running = True

        async with websockets.connect(MEXC_WS_URL) as ws:
            sub_msg = {
                "method": "SUBSCRIPTION",
                "params": [f"spot@public.deals.v3.api@{symbol}"],
            }
            await ws.send(json.dumps(sub_msg))
            logger.info(f"Subscribed to {symbol} trades")

            while self._running:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30)
                    data = json.loads(msg)

                    if "d" in data and "deals" in data["d"]:
                        for deal in data["d"]["deals"]:
                            tick = {
                                "price": float(deal["p"]),
                                "quantity": float(deal["v"]),
                                "timestamp": int(deal["t"]),
                                "side": "buy" if deal["S"] == 1 else "sell",
                            }
                            if self._on_tick:
                                self._on_tick(tick)

                except asyncio.TimeoutError:
                    await ws.send(json.dumps({"method": "PING"}))
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    break

    def stop(self) -> None:
        """Stop the WebSocket client."""
        self._running = False


def run_ws_client(on_tick: Callable, symbol: str = "BTCUSDT") -> None:
    """Convenience function to run WebSocket client."""
    client = MEXCWebSocketClient(on_tick=on_tick)
    asyncio.run(client.subscribe_trades(symbol))
