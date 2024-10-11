import json
import asyncio
from strategy_generator import StrategyGenerator
from strategy import Strategy
from typing import Callable
import pandas as pd

class StrategyFactory:
    def __init__(self, market_maker, config, config_file_path):
        self.market_maker = market_maker
        self.config = config
        self.strategy_generator = StrategyGenerator(config)
        self.strategies = {}
        self.signal_methods = {}
        self.config_file_path = config_file_path
        self.load_strategy_config()
        asyncio.create_task(self.monitor_strategies())

    def load_strategy_config(self):
        try:
            with open(self.config_file_path, 'r') as f:
                content = f.read().strip()
                if content:
                    self.strategies = json.loads(content)
                else:
                    self.strategies = {}
        except FileNotFoundError:
            self.strategies = {}
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {self.config_file_path}. Creating a new empty configuration.")
            self.strategies = {}
            self.save_strategy_config()


    def save_strategy_config(self):
        with open(self.config_file_path, 'w') as f:
            json.dump(self.strategies, f, indent=4)

    async def generate_and_update_strategies(self):
        market_data = await self.market_maker.get_recent_data()
        new_strategies = self.strategy_generator.generate_strategies(market_data)
        
        for time_frame, strategies in new_strategies.items():
            for strategy in strategies:
                self.update_strategy(strategy.name, strategy)

    def update_strategy(self, strategy_name: str, strategy: Strategy):
        self.strategies[strategy_name] = strategy.__dict__
        self.signal_methods[strategy_name] = self.create_signal_method(strategy)
        self.save_strategy_config()

    def create_signal_method(self, strategy: Strategy) -> Callable:
        def signal_method(market_data: pd.DataFrame) -> float:
            return strategy.generate_signal(market_data)
        return signal_method

    def get_signal_method(self, strategy_name: str) -> Callable:
        return self.signal_methods.get(strategy_name, self.default_signal)

    @staticmethod
    def default_signal(market_data: pd.DataFrame) -> float:
        return 0

    async def monitor_strategies(self):
        while True:
            await asyncio.sleep(60)  # Check every minute
            current_strategies = set(self.strategies.keys())
            config_strategies = set(self.load_strategy_config().keys())

            if current_strategies != config_strategies:
                self.save_strategy_config()
                print("Strategy configuration updated.")

    def delete_strategy(self, strategy_name: str):
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            del self.signal_methods[strategy_name]
            self.save_strategy_config()
            print(f"Strategy {strategy_name} deleted.")
