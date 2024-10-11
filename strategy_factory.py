from strategy_generator import StrategyGenerator
from strategy import Strategy
from typing import Callable
import pandas as pd

class StrategyFactory:
    def __init__(self, market_maker, config):
        self.market_maker = market_maker
        self.config = config
        self.strategy_generator = StrategyGenerator(config)
        self.strategies = {}
        self.signal_methods = {}

    async def generate_and_update_strategies(self):
        market_data = await self.market_maker.get_recent_data()
        new_strategies = self.strategy_generator.generate_strategies(market_data)
        
        for time_frame, strategies in new_strategies.items():
            for strategy in strategies:
                self.update_strategy(strategy.name, strategy)

    def update_strategy(self, strategy_name: str, strategy: Strategy):
        self.strategies[strategy_name] = strategy
        self.signal_methods[strategy_name] = self.create_signal_method(strategy)

    def create_signal_method(self, strategy: Strategy) -> Callable:
        def signal_method(market_data: pd.DataFrame) -> float:
            return strategy.generate_signal(market_data)
        return signal_method

    def get_signal_method(self, strategy_name: str) -> Callable:
        return self.signal_methods.get(strategy_name, self.default_signal)

    @staticmethod
    def default_signal(market_data: pd.DataFrame) -> float:
        return 0
