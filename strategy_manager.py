from typing import Dict, List
from strategy import Strategy, TimeFrame
import pandas as pd
import numpy as np
from strategy_selector import StrategySelector
from datetime import datetime, timedelta
from strategy_generator import StrategyGenerator

class StrategyManager:
    def __init__(self, config):
        self.strategies = {tf: [] for tf in TimeFrame}
        self.active_strategies = {}
        self.strategy_selector = StrategySelector(config)
        self.config = config
        self.protection_period = timedelta(hours=1)  # Adjust as needed

    async def initialize_strategies(self, strategy_generator: StrategyGenerator, market_data: pd.DataFrame):
        strategies = strategy_generator.generate_strategies(market_data)
        for time_frame, time_frame_strategies in strategies.items():
            for strategy in time_frame_strategies:
                self.add_strategy(strategy)
            if time_frame_strategies:
                best_strategy = max(time_frame_strategies, key=lambda s: s.performance.get('total_return', 0))
                self.set_active_strategy(time_frame, best_strategy)
        
        await self.log(f"Initialized strategies for all time frames")

    def add_strategy(self, strategy: Strategy):
        strategy.protected_until = datetime.now() + self.protection_period
        self.strategies[strategy.time_frame][strategy.name] = strategy

    def remove_strategy(self, strategy: Strategy):
        if (strategy not in self.active_strategies.values() and 
            datetime.now() > strategy.protected_until):
            if strategy.name in self.strategies[strategy.time_frame]:
                del self.strategies[strategy.time_frame][strategy.name]

    def update_strategy_protection(self):
        current_time = datetime.now()
        for time_frame in TimeFrame:
            for strategy in self.strategies[time_frame].values():
                if current_time > strategy.protected_until:
                    strategy.protected_until = None

    def set_active_strategy(self, time_frame: TimeFrame, strategy: Strategy):
        if strategy in self.strategies[time_frame]:
            self.active_strategies[time_frame] = strategy
            strategy.protected_until = datetime.now() + self.protection_period

    def get_active_strategies(self) -> Dict[TimeFrame, Strategy]:
        return self.active_strategies

    def get_all_strategies(self) -> List[Strategy]:
        return [strategy for strategies in self.strategies.values() for strategy in strategies]

    def get_strategies_by_timeframe(self, time_frame: TimeFrame) -> List[Strategy]:
        return self.strategies[time_frame]

    def update_strategy_performance(self, strategy: Strategy, performance_metrics: Dict):
        if strategy in self.get_all_strategies():
            strategy.update_performance(performance_metrics)

    def calculate_strategy_weight(self, strategy: Strategy) -> float:
        performance = strategy.performance
    
        # Define weights for each factor
        factor_weights = {
        'total_return': 0.2,
        'sharpe_ratio': 0.15,
        'max_drawdown': 0.1,
        'win_rate': 0.2,
        'profit_factor': 0.15,
        'calmar_ratio': 0.1,
        'recent_performance': 0.1
        }
    
        # Calculate the weighted score
        weighted_score = sum(
            factor_weights[factor] * performance.get(factor, 0)
            for factor in factor_weights
        )
    
        # Calculate strategy age in days
        strategy_age = (self.current_timestamp - strategy.creation_timestamp) / (24 * 60 * 60)
    
        # Novelty bonus for new strategies (first 30 days)
        novelty_bonus = max(0, 0.2 - (strategy_age / 30) * 0.2) if strategy_age <= 30 else 0
    
        # Age bonus for older strategies (40 to 70 days)
        if strategy_age > 40:
            age_bonus = min(0.3, (strategy_age - 40) / 30 * 0.3)  # Max 30% bonus over 30 days
        else:
            age_bonus = 0
    
        # Adjust for market volatility
        volatility_factor = 1 + (performance.get('volatility', 0) - self.average_market_volatility) / self.average_market_volatility
    
        # Combine all factors
        final_weight = (weighted_score + novelty_bonus + age_bonus) * volatility_factor
    
        return max(0, final_weight)  # Ensure non-negative weight


    def select_best_strategies(self, market_data: pd.DataFrame, asset_position_value: float):
        all_strategies = {tf: self.get_strategies_by_timeframe(tf) for tf in TimeFrame}
        strategy_performance = {s.name: s.performance for s in self.get_all_strategies()}
        
        suitable_strategies = self.strategy_selector.select_strategies(
            all_strategies,
            strategy_performance,
            market_data,
            asset_position_value
        )

        for time_frame in TimeFrame:
            weighted_strategies = [(strategy, self.calculate_strategy_weight(strategy)) for strategy in suitable_strategies[time_frame]]
            sorted_strategies = sorted(weighted_strategies, key=lambda x: x[1], reverse=True)
            if sorted_strategies:
                selected_strategy = sorted_strategies[0][0]
                self.set_active_strategy(time_frame, selected_strategy)

    def get_weighted_signal(self, time_frame: TimeFrame, market_data: pd.DataFrame) -> float:
        active_strategies = self.select_best_strategies(time_frame, market_data)
        if not active_strategies:
            return 0
        
        total_weight = sum(strategy.weight for strategy in active_strategies)
        weighted_signal = sum(strategy.generate_signal(market_data) * strategy.weight for strategy in active_strategies)
        
        return weighted_signal / total_weight if total_weight > 0 else 0
    
    def update_strategy_weights(self):
        for time_frame in TimeFrame:
            strategies = self.get_strategies_by_timeframe(time_frame)
            for strategy in strategies:
                strategy.weight = self.calculate_strategy_weight(strategy)