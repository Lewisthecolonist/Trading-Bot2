import numpy as np
from scipy.optimize import differential_evolution
from joblib import Parallel, delayed
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Tuple, List, Union
from strategy import Strategy
from typing import Generator
from strategy import TimeFrame
import pandas as pd

class StrategyOptimizer:
    def __init__(self, config, market_simulator, strategies: Dict[TimeFrame, Dict[str, Strategy]]):
        self.config = config
        self.market_simulator = market_simulator
        self.strategies = strategies
    def optimize_strategy(self, strategy: Strategy) -> Tuple[Strategy, float]:
        param_ranges = self._get_strategy_param_ranges(strategy)
    
        def objective(params: np.ndarray) -> float:
            temp_strategy = strategy.clone()
            param_dict = self._convert_params_to_dict(params, param_ranges)
            temp_strategy.update_parameters(param_dict)
        
            performances = []
            timeframes = {
                TimeFrame.SHORT_TERM: 'min',     # Minutes (intraday)
                TimeFrame.MID_TERM: 'D',       # Daily
                TimeFrame.LONG_TERM: 'ME',      # Monthly
                TimeFrame.SEASONAL_TERM: 'A'    # Annual
            }

            # Evaluate each timeframe category
            for timeframe_category, periods in timeframes.items():
                category_performance = []
                for period in periods:
                    market_data = self.market_simulator.generate_market_data(
                        days=365 * 3,  # Extended historical data
                        timeframe=period
                    )
                    perf = self._evaluate_strategy(temp_strategy, market_data)
                    category_performance.append(perf)
                performances.append(np.mean(category_performance))
        
            weighted_performance = (
                0.4 * performances[0] +  # Short-term weight
                0.3 * performances[1] +  # Mid-term weight
                0.2 * performances[2] +  # Long-term weight
                0.1 * performances[3]    # Seasonal weight
            )
        
            return -weighted_performance

        result = differential_evolution(
            objective,
            bounds=list(param_ranges.values()),
            strategy='best1bin',
            maxiter=100,
            popsize=20,
            mutation=(0.5, 1.0),
            recombination=0.7,
            workers=-1
        )

        optimized_params = self._convert_params_to_dict(result.x, param_ranges)
        strategy.update_parameters(optimized_params)
        return strategy, -result.fun

    def _get_strategy_param_ranges(self, strategy: Strategy) -> Dict[str, Tuple[float, float]]:
        if 'trend_following' in strategy.favored_patterns:
            return {
                'MOVING_AVERAGE_SHORT': (5, 50),
                'MOVING_AVERAGE_LONG': (20, 200),
                'TREND_STRENGTH_THRESHOLD': (0.01, 0.1)
            }
        elif 'statistical_arbitrage' in strategy.favored_patterns:
            return {
                'LOOKBACK_PERIOD': (10, 100),
                'Z_SCORE_THRESHOLD': (1.0, 4.0),
                'CORRELATION_THRESHOLD': (0.5, 0.95)
            }
        elif 'momentum' in strategy.favored_patterns:
            return {
                'MOMENTUM_PERIOD': (5, 30),
                'ACCELERATION_FACTOR': (0.01, 0.05),
                'MAX_ACCELERATION': (0.1, 0.5)
            }
        elif 'options_strategy' in strategy.favored_patterns:
            return {
                'DELTA_THRESHOLD': (0.1, 0.5),
                'GAMMA_LIMIT': (0.05, 0.2),
                'VEGA_EXPOSURE_LIMIT': (500, 2000)
            }      
        elif 'grid_trading' in strategy.favored_patterns:
            return {
                'GRID_LEVELS': (5, 20),
                'GRID_SPACING': (0.005, 0.03),
                'PROFIT_PER_GRID': (0.002, 0.01)
            }   
        elif 'market_making' in strategy.favored_patterns:
            return {
                'BID_ASK_SPREAD': (0.001, 0.005),
                'INVENTORY_TARGET': (0.3, 0.7),
                'MAX_POSITION_DEVIATION': (0.1, 0.3)
            }

    def _evaluate_strategy(self, strategy: Strategy, market_data: pd.DataFrame) -> float:
        signals = strategy.generate_signal(market_data)
        returns = self._calculate_returns(signals, market_data)
        
        metrics = {
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'win_rate': self._calculate_win_rate(signals, returns)
        }
        
        # Custom scoring based on strategy type
        if 'trend_following' in strategy.favored_patterns:
            return 0.4 * metrics['sharpe_ratio'] + 0.3 * metrics['sortino_ratio'] + 0.3 * (1 - metrics['max_drawdown'])
        elif 'volatility_clustering' in strategy.favored_patterns:
            return 0.3 * metrics['sharpe_ratio'] + 0.4 * metrics['win_rate'] + 0.3 * (1 - metrics['max_drawdown'])
        
        return metrics['sharpe_ratio']  # Default scoring
    def optimize_all_strategies(self) -> Dict[TimeFrame, List[Tuple[Strategy, float]]]:
        optimized_strategies = {}
        for time_frame in TimeFrame:
            optimized_strategies[time_frame] = Parallel(n_jobs=-1)(
                delayed(self.optimize_strategy)(strategy) for strategy in self.strategies[time_frame].values()
            )
            optimized_strategies[time_frame] = sorted(optimized_strategies[time_frame], key=lambda x: x[1], reverse=True)
        return optimized_strategies
    
    def temporary_optimize(self, strategy: 'Strategy') -> Generator['Strategy', None, None]:
        original_params = strategy.get_parameters()
        original_capital = strategy.get_capital()
        
        optimized_strategy, _ = self.optimize_strategy(strategy)
        
        yield optimized_strategy
        
        strategy.update_parameters(original_params)
        strategy.set_capital(original_capital)
