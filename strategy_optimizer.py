import numpy as np
from scipy.optimize import differential_evolution
from joblib import Parallel, delayed
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Tuple, List, Union
from strategy import Strategy
from typing import Generator
from strategy import TimeFrame

class StrategyOptimizer:
    def __init__(self, config, market_simulator, strategies: Dict[TimeFrame, Dict[str, Strategy]]):
        self.config = config
        self.market_simulator = market_simulator
        self.strategies = strategies

    def optimize_strategy(self, strategy: Strategy) -> Tuple[Strategy, float]:
        strategy = self.strategies[strategy_name]

        def objective(params: np.ndarray) -> float:
            temp_strategy = strategy.clone()
            capital = params[0]
            adaptive_params = dict(zip(self.config.ADAPTIVE_PARAMS.keys(), params[1:]))
            temp_strategy.update_parameters(adaptive_params)
            temp_strategy.set_capital(capital)
            
            market_data = self.market_simulator.generate_market_data(days=self.config.SIMULATION_DAYS)
            
            tscv = TimeSeriesSplit(n_splits=5)
            performances = []
            for _, test_index in tscv.split(market_data):
                test_data = market_data.iloc[test_index]
                performance = self.market_simulator.run_simulation(temp_strategy, test_data)
                performances.append(performance['sharpe_ratio'])
            
            return -np.mean(performances)

        bounds = [(self.config.INITIAL_CAPITAL * 0.1, self.config.INITIAL_CAPITAL * 10)] + \
                 [(value * 0.1, value * 10) for value in self.config.ADAPTIVE_PARAMS.values()]

        result = differential_evolution(
            objective,
            bounds,
            popsize=self.config.POPULATION_SIZE,
            mutation=self.config.MUTATION_RANGE,
            recombination=self.config.RECOMBINATION_RATE,
            updating='deferred',
            workers=-1
        )

        optimized_capital = result.x[0]
        optimized_adaptive_params = dict(zip(self.config.ADAPTIVE_PARAMS.keys(), result.x[1:]))
        strategy.update_parameters(optimized_adaptive_params)
        strategy.set_capital(optimized_capital)

        return strategy, -result.fun

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
