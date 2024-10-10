from typing import Dict, List
from strategy import Strategy, TimeFrame
from strategy_generator import StrategyGenerator
from strategy_selector import StrategySelector
import pandas as pd

class StrategyManager:
    def __init__(self, config):
        self.config = config
        self.strategies: Dict[TimeFrame, List[Strategy]] = {tf: [] for tf in TimeFrame}
        self.active_strategies: Dict[TimeFrame, Strategy] = {}
        self.strategy_generator = StrategyGenerator(config)
        self.strategy_selector = StrategySelector(config)

    def generate_strategies(self, market_data: pd.DataFrame):
        self.strategies = self.strategy_generator.generate_strategies(market_data)

    def select_strategies(self, market_data: pd.DataFrame, strategy_performance: Dict[str, Dict]):
        self.active_strategies = self.strategy_selector.select_strategies(self.strategies, strategy_performance, market_data)

    def add_strategy(self, strategy: Strategy):
        self.strategies[strategy.time_frame].append(strategy)

    def remove_strategy(self, strategy: Strategy):
        self.strategies[strategy.time_frame].remove(strategy)

    def get_strategies(self, time_frame: TimeFrame) -> List[Strategy]:
        return self.strategies[time_frame]

    def get_active_strategy(self, time_frame: TimeFrame) -> Strategy:
        return self.active_strategies.get(time_frame)

    def update_strategy_performance(self, strategy: Strategy, performance: Dict[str, float]):
        strategy.update_performance(performance)

    def adjust_strategy_weights(self):
        # This method can be implemented to adjust weights of strategies based on their performance
        pass
