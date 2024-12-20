import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
from backtester import Backtester
from market_maker import MarketMaker
from wallet import Wallet
import time
from config import Config
import zipfile
import io
import os
import ccxt.async_support as ccxt
import datetime
from decimal import Decimal
from datetime import timedelta
from queue import Queue
from api_call_manager import APICallManager

class TradingSystem:
    def __init__(self, config, historical_data):
        self.config = config
        self.historical_data = historical_data
        self.results_queue = Queue()
        self.backtester = Backtester(self.config, historical_data, self.results_queue)
        self.exchange = ccxt.kraken({
            'apiKey': os.getenv('KRAKEN_API_KEY'),
            'secret': os.getenv('KRAKEN_PRIVATE_KEY'),
            'enableRateLimit': True,
        })
        self.wallet = Wallet(self.exchange)
        self.market_maker = MarketMaker(config, strategy_config_path='strategies.json')
        self.market_maker.set_wallet(self.wallet)
        self.is_running = False
        self.backtest_results = None
        self.mode = None
        self.loop = asyncio.get_event_loop()
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.start_time = None
        self.backtest_duration = float(config.BASE_PARAMS['BACKTEST_DURATION'])
        self.market_maker_duration = float(config.BASE_PARAMS['MARKET_MAKER_DURATION'])
        self.api_call_manager = APICallManager()
        asyncio.create_task(self.initialize_api_call_manager())

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

    async def start(self):
        await self.api_call_manager.load_state()
        self.api_call_manager.start_api_state_saver()
        self.mode = await self.loop.run_in_executor(self.executor, self.get_user_choice)
        if self.mode in [1, 3]:
            self.backtest_duration = await self.loop.run_in_executor(
                self.executor, 
                lambda: float(input("Enter backtester duration in hours (0 for indefinite): "))
            )
            self.backtest_duration *= 3600  # Convert hours to seconds

        if self.mode in [2, 3]:
            self.market_maker_duration = await self.loop.run_in_executor(
                self.executor, 
                lambda: float(input("Enter market maker duration in hours (0 for indefinite): "))
            )
            self.market_maker_duration *= 3600  # Convert hours to seconds
        self.is_running = True
        self.start_time = time.time()
        market_data = await self.market_maker.update_market_data()
        await self.market_maker.initialize(market_data)

    async def main_loop(self):
        tasks = []
        if self.mode in [1, 3]:
            tasks.append(self.run_backtester())
        if self.mode in [2, 3]:
            tasks.append(self.run_market_maker())
        tasks.append(self.process_results_queue())

        await asyncio.gather(*tasks)

    async def run_backtester(self):
        print("Starting backtest...")
        # Single run
        self.backtest_results = await self.loop.run_in_executor(self.executor, self.backtester.run)
        print("Backtest completed, processing results...")
        self.results_queue.put(self.backtest_results)
        await self.process_backtest_results(self.backtest_results)
        print("Results processed")
        await self.stop()
    async def run_market_maker(self):
        end_time = self.start_time + self.market_maker_duration if self.market_maker_duration > 0 else None
        while self.is_running and (end_time is None or time.time() < end_time):
            try:
                market_data = await self.get_latest_market_data()
                self.market_maker.update(market_data)
                await self.market_maker.execute_trades(self.wallet)
                await asyncio.sleep(self.config.MARKET_MAKER_UPDATE_INTERVAL)
            except Exception as e:
                print(f"Error in market maker: {e}")
        await self.stop()
        print("Market maker completed")

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

    async def stop(self):
        self.is_running = False
        await self.wallet.close()
        self.executor.shutdown(wait=True)
        self.api_call_manager.save_state()

    async def process_backtest_results(self, results):
        # Await the results coroutine first
        results_data = await results
    
        if results_data:
            # Calculate and print performance metrics
            total_return = results_data['total_return']
            sharpe_ratio = results_data['sharpe_ratio']
            max_drawdown = results_data['max_drawdown']
            win_rate = results_data['win_rate']
        
            print(f"Total Return: {total_return:.2%}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Max Drawdown: {max_drawdown:.2%}")
            print(f"Win Rate: {win_rate:.2%}")

            # Process trade distribution
            trade_counts = results_data['trade_counts']
            print("\nTrade Distribution:")
            for strategy, count in trade_counts.items():
                print(f"{strategy}: {count} trades")

            return results_data    
    def plot_equity_curve(self, equity_curve):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve)
        plt.title('Equity Curve')
        plt.xlabel('Trade Number')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.savefig('equity_curve.png')
        plt.close()
        print("Equity curve plot saved as 'equity_curve.png'")

    def save_results_to_file(self, results):
        import json
        with open('backtest_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        print("Backtest results saved to 'backtest_results.json'")

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

    async def initialize_api_call_manager(self):
        self.api_call_manager = APICallManager()
        await self.api_call_manager.load_state()

    async def initialize_strategies(self, market_data):
        try:
            # Ensure we await the strategy generation
            strategies = await self.strategy_generator.generate_strategies(market_data)
            for timeframe, strat_list in strategies.items():
                for strategy in strat_list:
                    self.strategy_manager.add_strategy(timeframe, strategy)
            return strategies
        except Exception as e:
            print.error(f"Error initializing strategies: {str(e)}")
            raise
