import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
from backtester import Backtester
from market_maker import MarketMaker
import time
from config import Config
import zipfile
import io
from wallet import Wallet
import os
import ccxt.async_support as ccxt
import datetime
from decimal import Decimal
from datetime import timedelta
from queue import Queue

class TradingSystem:
    def __init__(self, config, historical_data):
        self.config = Config
        self.historical_data = historical_data
        self.results_queue = Queue()
        self.backtester = Backtester(config, historical_data, self.results_queue)
        self.market_maker = MarketMaker(config, strategy_config_path='strategies.json', results_queue=self.results_queue)
        self.exchange = ccxt.kraken({
            'apiKey': os.getenv('KRAKEN_API_KEY'),
            'secret': os.getenv('KRAKEN_SECRET'),
            'enableRateLimit': True,
        })
        self.wallet = Wallet(self.exchange)
        self.is_running = False
        self.backtest_results = None
        self.mode = None
        self.loop = asyncio.get_event_loop()
        self.executor = ThreadPoolExecutor(max_workers=3)

    async def start(self):
        await self.wallet.connect()
        self.mode = await self.loop.run_in_executor(self.executor, self.get_user_choice)
        self.is_running = True

        tasks = []
        if self.mode in [1, 3]:
            tasks.append(self.loop.run_in_executor(self.executor, self.run_backtester))
        if self.mode in [2, 3]:
            tasks.append(self.run_market_maker())
        tasks.append(self.main_loop())
        tasks.append(self.process_results_queue())

        await asyncio.gather(*tasks)

    def get_user_choice(self):
        while True:
            print("Choose an option:")
            print("1. Run only the backtester")
            print("2. Run only the market maker")
            print("3. Run both in parallel")
            try:
                choice = int(input("Enter your choice (1-3): "))
                if choice in [1, 2, 3]:
                    return choice
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    async def stop(self):
        self.is_running = False
        await self.wallet.close()
        self.executor.shutdown(wait=True)

    def run_backtester(self):
        self.backtest_results = self.backtester.run()
        return self.backtest_results

    async def run_market_maker(self):
        while self.is_running:
            try:
                market_data = await self.get_latest_market_data()
                self.market_maker.update(market_data)
                await self.market_maker.execute_trades(self.wallet)
                await asyncio.sleep(self.config.MARKET_MAKER_UPDATE_INTERVAL)
            except Exception as e:
                print(f"Error in market maker: {e}")

    async def main_loop(self):
        while self.is_running:
            try:
                if self.mode in [1, 3]:
                    if self.backtest_results:
                        await self.process_backtest_results(self.backtest_results)
                        self.backtest_results = None
                elif self.mode == 2:
                    asyncio.sleep(1) # Prevent busy waiting
            except Exception as e:
                print(f"Error in main loop: {e}")

    async def process_backtest_results(self, results):
        if self.mode == 3:
            self.market_maker.update_strategy(results)

    async def process_results_queue(self):
        while self.is_running:
            try:
                if not self.results_queue.empty():
                    result = self.results_queue.get_nowait()
                    await self.process_backtest_results(result)
                else:
                    await asyncio.sleep(0.1)  # Short sleep to prevent busy waiting
            except Exception as e:
                print(f"Error processing results queue: {e}")

    async def get_latest_market_data(self):
        try:
            symbol = self.config.BASE_PARAMS['SYMBOL']
            
            # Fetch ticker data
            ticker = await self.exchange.fetch_ticker(symbol)
            
            # Fetch order book
            order_book = await self.exchange.fetch_order_book(symbol)
            
            # Fetch recent trades
            trades = await self.exchange.fetch_trades(symbol, limit=100)
            
            # Fetch OHLCV data for the last 24 hours
            since = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe='1h', since=since)
            
            # Calculate additional metrics
            vwap = sum(trade['price'] * trade['amount'] for trade in trades) / sum(trade['amount'] for trade in trades)
            volatility = self.calculate_volatility([candle[4] for candle in ohlcv])  # Using close prices
            
            market_data = {
                'symbol': symbol,
                'last': Decimal(str(ticker['last'])),
                'bid': Decimal(str(ticker['bid'])),
                'ask': Decimal(str(ticker['ask'])),
                'volume': Decimal(str(ticker['baseVolume'])),
                'timestamp': ticker['timestamp'],
                'vwap': Decimal(str(vwap)),
                'volatility': Decimal(str(volatility)),
                'order_book': {
                    'bids': [[Decimal(str(price)), Decimal(str(amount))] for price, amount in order_book['bids'][:5]],
                    'asks': [[Decimal(str(price)), Decimal(str(amount))] for price, amount in order_book['asks'][:5]]
                },
                'recent_trades': [
                    {
                        'price': Decimal(str(trade['price'])),
                        'amount': Decimal(str(trade['amount'])),
                        'side': trade['side'],
                        'timestamp': trade['timestamp']
                    } for trade in trades[:10]
                ],
                'ohlcv': [
                    {
                        'timestamp': candle[0],
                        'open': Decimal(str(candle[1])),
                        'high': Decimal(str(candle[2])),
                        'low': Decimal(str(candle[3])),
                        'close': Decimal(str(candle[4])),
                        'volume': Decimal(str(candle[5]))
                    } for candle in ohlcv
                ]
            }
            
            return market_data
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return None

    def calculate_volatility(self, prices):
        returns = pd.Series(prices).pct_change().dropna()
        return float(returns.std() * (252 ** 0.5))  # Annualized volatility