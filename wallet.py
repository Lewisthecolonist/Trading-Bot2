import os
from decimal import Decimal
from typing import Dict, Optional
import ccxt.async_support as ccxt
from rate_limiter import RateLimiter

class Wallet:
    def __init__(self, exchange):
        self.exchange = exchange
        self.balances: Dict[str, Decimal] = {}
        self.rate_limiter = RateLimiter(rate=5, per=1.0)  # 5 calls per second

    async def connect(self):
        await self.rate_limiter.wait()
        try:
            await self.update_balances()
            print("Wallet connected successfully")
        except Exception as e:
            print(f"Failed to connect wallet: {e}")

    async def update_balances(self):
        try:
            await self.rate_limiter.wait()
            kraken_balances = await self.exchange.fetch_balance()
            for asset, balance in kraken_balances['free'].items():
                self.balances[asset] = Decimal(str(balance))
            print(f"Updated balances: {self.balances}")
        except Exception as e:
            print(f"Error updating balances: {e}")

    def get_balance(self, asset: str) -> Decimal:
        return self.balances.get(asset, Decimal('0'))

    async def withdraw(self, asset: str, amount: Decimal, address: str):
        try:
            await self.rate_limiter.wait()
            await self.exchange.withdraw(asset, amount, address)
            await self.update_balances()
            print(f"Withdrawn {amount} {asset} to {address}")
        except Exception as e:
            print(f"Error withdrawing from Kraken: {e}")

    async def place_order(self, symbol: str, order_type: str, side: str, amount: Decimal, price: Optional[Decimal] = None):
        try:
            await self.rate_limiter.wait()
            if order_type == 'market':
                order = await self.exchange.create_market_order(symbol, side, amount)
            elif order_type == 'limit':
                if price is None:
                    raise ValueError("Price must be specified for limit orders")
                order = await self.exchange.create_limit_order(symbol, side, amount, price)
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            print(f"Placed {order_type} {side} order for {amount} {symbol} at {price if price else 'market price'}")
            await self.update_balances()
            return order
        except Exception as e:
            print(f"Error placing order: {e}")
            return None

    async def cancel_order(self, order_id: str, symbol: str):
        try:
            await self.rate_limiter.wait()
            await self.exchange.cancel_order(order_id, symbol)
            print(f"Cancelled order {order_id} for {symbol}")
            await self.update_balances()
        except Exception as e:
            print(f"Error cancelling order: {e}")

    async def get_open_orders(self, symbol: Optional[str] = None):
        try:
            await self.rate_limiter.wait()
            open_orders = await self.exchange.fetch_open_orders(symbol)
            return open_orders
        except Exception as e:
            print(f"Error fetching open orders: {e}")
            return []

    async def get_order_history(self, symbol: Optional[str] = None, since: Optional[int] = None, limit: Optional[int] = None):
        try:
            await self.rate_limiter.wait()
            order_history = await self.exchange.fetch_closed_orders(symbol, since, limit)
            return order_history
        except Exception as e:
            print(f"Error fetching order history: {e}")
            return []

    async def get_deposit_address(self, asset: str):
        try:
            await self.rate_limiter.wait()
            deposit_address = await self.exchange.fetch_deposit_address(asset)
            return deposit_address
        except Exception as e:
            print(f"Error fetching deposit address for {asset}: {e}")
            return None

    async def get_ticker(self, symbol: str):
        try:
            await self.rate_limiter.wait()
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            print(f"Error fetching ticker for {symbol}: {e}")
            return None

    async def close(self):
        try:
            await self.exchange.close()
            print("Wallet closed successfully")
        except Exception as e:
            print(f"Error closing wallet: {e}")

