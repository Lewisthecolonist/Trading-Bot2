import asyncio
import ccxt.async_support as ccxt
import numpy as np
from collections import deque
from sortedcontainers import SortedDict
from decimal import Decimal
import time

class OrderBook:
    def __init__(self, exchange, symbol, depth=10):
        self.exchange = exchange
        self.symbol = symbol
        self.bids = SortedDict()
        self.asks = SortedDict()
        self.depth = depth
        self.last_update_time = 0

    async def update(self):
        try:
            order_book = await self.exchange.fetch_order_book(self.symbol, self.depth)
            self.bids = SortedDict({Decimal(str(price)): Decimal(str(amount)) for price, amount in order_book['bids']})
            self.asks = SortedDict({Decimal(str(price)): Decimal(str(amount)) for price, amount in order_book['asks']})
            self.last_update_time = time.time()
        except Exception as e:
            print(f"Error updating order book: {e}")

    def get_mid_price(self):
        if self.bids and self.asks:
            return (self.bids.peekitem(-1)[0] + self.asks.peekitem(0)[0]) / 2
        return None

    def get_spread(self):
        if self.bids and self.asks:
            return (self.asks.peekitem(0)[0] - self.bids.peekitem(-1)[0]) / self.get_mid_price()
        return None

    def get_liquidity(self, depth=None):
        if depth is None:
            depth = self.depth
        bid_liquidity = sum(volume for _, volume in self.bids.items()[-depth:])
        ask_liquidity = sum(volume for _, volume in self.asks.items()[:depth])
        return bid_liquidity, ask_liquidity
