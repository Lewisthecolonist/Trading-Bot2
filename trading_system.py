import pandas as pd
import multiprocessing
import queue
from backtester import Backtester
from market_maker import MarketMaker
import threading
import time
from config import Config
import zipfile
import io
from wallet import Wallet
import decimal as Decimal
import web3 as Web3

class TradingSystem:
    def __init__(self, config, historical_data):
        self.config = Config
        self.historical_data = historical_data
        self.result_queue = multiprocessing.Queue()
        self.backtester = Backtester(config, historical_data, self.result_queue)
        self.market_maker = MarketMaker(config)
        self.is_running = False
        self.backtest_results = None
        self.mode = None

    def start(self):
        self.mode = self.get_user_choice()
        self.is_running = True

        if self.mode in [1, 3]:
            self.backtester_process = multiprocessing.Process(target=self.run_backtester)
            self.backtester_process.start()

        if self.mode in [2, 3]:
            self.market_maker_thread = threading.Thread(target=self.run_market_maker)
            self.market_maker_thread.start()

        self.main_thread = threading.Thread(target=self.main_loop)
        self.main_thread.start()

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

    def stop(self):
        self.is_running = False
        if hasattr(self, 'backtester_process'):
            self.backtester_process.terminate()
            self.backtester_process.join()
        if hasattr(self, 'market_maker_thread'):
            self.market_maker_thread.join()
        self.main_thread.join()

    def run_backtester(self):
        self.backtester.run()

    def run_market_maker(self):
        while self.is_running:
            try:
                self.market_maker.update(self.get_latest_market_data())
                self.market_maker.execute_trades()
                time.sleep(self.config.MARKET_MAKER_UPDATE_INTERVAL)
            except Exception as e:
                print(f"Error in market maker: {e}")

    def main_loop(self):
        while self.is_running:
            try:
                if self.mode in [1, 3]:
                    backtest_results = self.result_queue.get(timeout=1)
                    self.process_backtest_results(backtest_results)
                elif self.mode == 2:
                    time.sleep(1)  # Prevent busy waiting
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error in main loop: {e}")

    def process_backtest_results(self, results):
        self.backtest_results = results
        if self.mode == 3:
            self.market_maker.update_strategy(results)

    def get_latest_market_data(self):
        # Implement this method to fetch the latest market data
        pass
