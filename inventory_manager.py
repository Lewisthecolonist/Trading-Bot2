import asyncio
import ccxt.async_support as ccxt
import numpy as np
from collections import deque
from sortedcontainers import SortedDict
from decimal import Decimal
import time

class InventoryManager:
    def __init__(self, config, exchange):
        self.config = config
        self.exchange = exchange
        self.btc_balance = Decimal('0')
        self.usdt_balance = Decimal('0')

    async def update_balances(self):
        try:
            balance = await self.exchange.fetch_balance()
            self.btc_balance = Decimal(str(balance['BTC']['free']))
            self.usdt_balance = Decimal(str(balance['USDT']['free']))
        except Exception as e:
            print(f"Error updating balances: {e}")

    def get_inventory_skew(self):
        total_value_in_btc = self.btc_balance + self.usdt_balance / self.config.get_btc_price()
        btc_ratio = self.btc_balance / total_value_in_btc if total_value_in_btc else Decimal('0')
        target_ratio = Decimal('0.5')  # Aim for 50/50 split between BTC and USDT
        return float((btc_ratio - target_ratio) * 2)  # Scale the skew
