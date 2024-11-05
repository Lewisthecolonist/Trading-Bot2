from typing import Dict, List
from strategy import Strategy, TimeFrame
import pandas as pd
import numpy as np
from strategy_selector import StrategySelector
from datetime import datetime, timedelta
from strategy_generator import StrategyGenerator
from api_call_manager import APICallManager
import asyncio
import logging
from strategy import Strategy
from risk_manager import RiskManager

class StrategyManager:
    def __init__(self, config):
        self.strategies = {tf: {} for tf in TimeFrame}
        self.active_strategies = {}
        self.strategy_selector = StrategySelector(config)
        self.config = config
        self.protection_period = timedelta(hours=1)
        self.api_call_manager = APICallManager()
        self.logger = logging.getLogger(__name__)
        self.current_timestamp = datetime.now().timestamp()
        self.average_market_volatility = 0.0
        self.risk_manager = RiskManager(config)  # Add this line
    async def initialize_strategies(self, strategy_generator: StrategyGenerator, market_data: pd.DataFrame):
        if await self.api_call_manager.can_make_call():
            try:
                strategies = await strategy_generator.generate_strategies(market_data)
                self.strategies = {tf: {} for tf in TimeFrame}
                for time_frame, time_frame_strategies in strategies.items():
                    if time_frame_strategies:
                        for strategy in time_frame_strategies:
                            self.add_strategy(strategy)
                        best_strategy = max(time_frame_strategies, key=lambda s: s.performance.get('total_return', 0))
                        self.set_active_strategy(time_frame, best_strategy)
    
                self.logger.info(f"Initialized strategies for all time frames")
            except Exception as e:
                self.logger.error(f"Error initializing strategies: {str(e)}")
                self.strategies = {tf: {} for tf in TimeFrame}
        else:
            wait_time = await self.api_call_manager.time_until_reset()
            print(f"API call limit reached. Waiting for {wait_time:.2f} seconds.")
            await asyncio.sleep(wait_time)
            return await self.initialize_strategies(strategy_generator, market_data)
    def add_strategy(self, strategy: Strategy):
        try:
            if isinstance(strategy, dict):
                strategy = Strategy(
                    strategy_name=strategy.get('name', 'unnamed_strategy'),
                    description=strategy.get('description', 'No description'),
                    parameters=strategy.get('parameters', {'INITIAL_CAPITAL': 10000}),
                    favored_patterns=strategy.get('favored_patterns', ['trend_following']),
                    time_frame=strategy.get('time_frame', TimeFrame.SHORT_TERM)
                )
            elif not isinstance(strategy, Strategy):
                raise TypeError("Invalid strategy type")
                
            strategy.protected_until = datetime.now() + self.protection_period
            self.strategies[strategy.time_frame][strategy.name] = strategy
            
        except Exception as e:
            self.logger.error(f"Error adding strategy: {str(e)}")
    def remove_strategy(self, strategy: Strategy):
        try:
            if (strategy not in self.active_strategies.values() and 
                strategy.protected_until and datetime.now() > strategy.protected_until):
                if strategy.name in self.strategies[strategy.time_frame]:
                    del self.strategies[strategy.time_frame][strategy.name]
        except Exception as e:
            self.logger.error(f"Error removing strategy: {str(e)}")

    def update_strategy_protection(self):
        current_time = datetime.now()
        for time_frame in TimeFrame:
            for strategy in self.strategies[time_frame].values():
                if hasattr(strategy, 'protected_until') and strategy.protected_until and current_time > strategy.protected_until:
                    strategy.protected_until = None

    async def update_strategies(self, market_data: pd.DataFrame, time_frame: TimeFrame, strategy_generator: StrategyGenerator):
        if await self.api_call_manager.can_make_call():
            try:
                for strategy in self.get_strategies_by_timeframe(time_frame):
                    self.update_strategy_performance(strategy, strategy.calculate_performance(market_data))

                if len(self.get_strategies_by_timeframe(time_frame)) < self.config.BASE_PARAMS['MAX_STRATEGIES_PER_TIMEFRAME']:
                    new_strategies = await strategy_generator.generate_strategies(market_data)
                    for new_strategy in new_strategies:
                        self.add_strategy(new_strategy)

                self.update_strategy_protection()
                for strategy in self.get_strategies_by_timeframe(time_frame):
                    self.remove_strategy(strategy)

                self.update_strategy_weights()
                self.select_best_strategies(market_data)

                await self.api_call_manager.record_call()
            except Exception as e:
                self.logger.error(f"Error updating strategies: {str(e)}")
        else:
            wait_time = await self.api_call_manager.time_until_reset()
            self.logger.warning(f"API call limit reached. Waiting for {wait_time:.2f} seconds.")
            await asyncio.sleep(wait_time)
            return await self.update_strategies(market_data, time_frame, strategy_generator)

    def set_active_strategy(self, time_frame: TimeFrame, strategy: Strategy):
        try:
            if not isinstance(strategy, Strategy):
                raise TypeError("Invalid strategy type")
            if strategy.time_frame != time_frame:
                raise ValueError("Strategy time frame doesn't match the specified time frame")
            if strategy in self.strategies[time_frame].values():
                self.active_strategies[time_frame] = strategy
                strategy.protected_until = datetime.now() + self.protection_period
        except Exception as e:
            self.logger.error(f"Error setting active strategy: {str(e)}")

    def get_active_strategies(self) -> Dict[TimeFrame, Strategy]:
        return self.active_strategies

    def get_all_strategies(self) -> List[Strategy]:
        return [strategy for strategies in self.strategies.values() for strategy in strategies.values()]

    def get_strategies_by_timeframe(self, time_frame: TimeFrame) -> List[Strategy]:
        return list(self.strategies[time_frame].values())

    def update_strategy_performance(self, strategy: Strategy, performance_metrics: Dict):
        try:
            if strategy in self.get_all_strategies():
                strategy.update_performance(performance_metrics)
        except Exception as e:
            self.logger.error(f"Error updating strategy performance: {str(e)}")

    def calculate_strategy_weight(self, strategy: Strategy) -> float:
        try:
            performance = strategy.performance
        
            factor_weights = {
                'total_return': 0.2,
                'sharpe_ratio': 0.15,
                'max_drawdown': 0.1,
                'win_rate': 0.2,
                'profit_factor': 0.15,
                'calmar_ratio': 0.1,
                'recent_performance': 0.1
            }
        
            weighted_score = sum(
                factor_weights[factor] * performance.get(factor, 0)
                for factor in factor_weights
            )
        
            strategy_age = (self.current_timestamp - getattr(strategy, 'creation_timestamp', self.current_timestamp)) / (24 * 60 * 60)
        
            novelty_bonus = max(0, 0.2 - (strategy_age / 30) * 0.2) if strategy_age <= 30 else 0
        
            if strategy_age > 40:
                age_bonus = min(0.3, (strategy_age - 40) / 30 * 0.3)
            else:
                age_bonus = 0
        
            volatility_factor = 1 + (performance.get('volatility', 0) - self.average_market_volatility) / max(self.average_market_volatility, 0.0001)
        
            final_weight = (weighted_score + novelty_bonus + age_bonus) * volatility_factor
        
            return max(0, final_weight)
        except Exception as e:
            self.logger.error(f"Error calculating strategy weight: {str(e)}")
            return 0.0

    def select_best_strategies(self, market_data: pd.DataFrame):
        try:
            current_price = float(market_data['close'].iloc[-1])
            
            stop_loss_pct = self.config.ADAPTIVE_PARAMS['STOP_LOSS_PCT']
            take_profit_pct = self.config.ADAPTIVE_PARAMS['TAKE_PROFIT_PCT']
            
            stop_loss_price = current_price * (1 - stop_loss_pct)
            take_profit_price = current_price * (1 + take_profit_pct)
            
            position_size = self.risk_manager.calculate_position_size(
                self.config.get_portfolio_value(),
                current_price,
                stop_loss_price
            )
            
            all_strategies = {tf: self.get_strategies_by_timeframe(tf) for tf in TimeFrame}
            strategy_performance = {}
            
            for strategy in self.get_all_strategies():
                strategy_performance[strategy.name] = strategy.calculate_performance(market_data.copy(), position_size)

            suitable_strategies = self.strategy_selector.select_strategies(
                all_strategies,
                strategy_performance,
                market_data.copy(),
                position_size
            )

            for time_frame in TimeFrame:
                weighted_strategies = [(strategy, self.calculate_strategy_weight(strategy)) 
                                    for strategy in suitable_strategies.get(time_frame, [])]
                sorted_strategies = sorted(weighted_strategies, key=lambda x: x[1], reverse=True)
                if sorted_strategies:
                    selected_strategy = sorted_strategies[0][0]
                    self.set_active_strategy(time_frame, selected_strategy)
                    
        except Exception as e:
            self.logger.error(f"Error selecting best strategies: {str(e)}")

    def get_weighted_signal(self, time_frame: TimeFrame, market_data: pd.DataFrame) -> float:
        try:
            active_strategies = self.select_best_strategies(time_frame, market_data)
            if not active_strategies:
                return 0
            
            total_weight = sum(strategy.weight for strategy in active_strategies)
            weighted_signal = sum(strategy.generate_signal(market_data) * strategy.weight 
                                for strategy in active_strategies)
            
            return weighted_signal / total_weight if total_weight > 0 else 0
        except Exception as e:
            self.logger.error(f"Error calculating weighted signal: {str(e)}")
            return 0.0
    
    def update_strategy_weights(self):
        try:
            for time_frame in TimeFrame:
                strategies = self.get_strategies_by_timeframe(time_frame)
                for strategy in strategies:
                    strategy.weight = self.calculate_strategy_weight(strategy)
        except Exception as e:
            self.logger.error(f"Error updating strategy weights: {str(e)}")