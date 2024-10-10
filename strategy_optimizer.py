from scipy.optimize import differential_evolution
from joblib import Parallel, delayed
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Tuple, List

class StrategyOptimizer:
    def __init__(self, config, market_simulator, strategies: Dict[str, 'Strategy']):
        self.config = config
        self.market_simulator = market_simulator
        self.strategies = strategies

    def optimize_strategy(self, strategy_name: str) -> Tuple['Strategy', float]:
        if strategy_name not in self.strategies:
            raise KeyError(f"Strategy '{strategy_name}' not found")
        
        strategy = self.strategies[strategy_name]
        
        def objective(params: np.ndarray) -> Tuple[float, float]:
            capital = params[0]
            adaptive_params = dict(zip(self.config.ADAPTIVE_PARAMS.keys(), params[1:]))
            strategy.update_parameters(adaptive_params)
            strategy.set_capital(capital)
            
            market_data = self.market_simulator.generate_market_data(days=self.config.SIMULATION_DAYS)
            
            tscv = TimeSeriesSplit(n_splits=5)
            performances = []
            for _, test_index in tscv.split(market_data):
                test_data = market_data.iloc[test_index]
                performance = self.market_simulator.run_simulation(strategy, test_data)
                performances.append((performance['sharpe_ratio'], performance['max_drawdown']))
            
            mean_sharpe = np.mean([p[0] for p in performances])
            mean_drawdown = np.mean([p[1] for p in performances])
            return -mean_sharpe, mean_drawdown

        def adaptive_bounds(best_params: np.ndarray, iteration: int) -> List[Tuple[float, float]]:
            if iteration == 0:
                return [(self.config.INITIAL_CAPITAL * 0.1, self.config.INITIAL_CAPITAL * 10)] + \
                       [(value * 0.1, value * 10) for value in self.config.ADAPTIVE_PARAMS.values()]
            else:
                return [(max(p * 0.5, b[0]), min(p * 1.5, b[1])) for p, b in zip(best_params, bounds)]

        bounds = adaptive_bounds(None, 0)
        best_params = None
        best_fun = np.inf

        for i in range(5):
            result = differential_evolution(
                lambda x: objective(x)[0],
                bounds,
                constraints=({'type': 'ineq', 'fun': lambda x: -objective(x)[1] + self.config.MAX_DRAWDOWN}),
                popsize=self.config.POPULATION_SIZE,
                mutation=self.config.MUTATION_RANGE,
                recombination=self.config.RECOMBINATION_RATE,
                updating='deferred',
                workers=-1
            )

            if result.fun < best_fun:
                best_params = result.x
                best_fun = result.fun

            bounds = adaptive_bounds(best_params, i+1)

        optimized_capital = best_params[0]
        optimized_adaptive_params = dict(zip(self.config.ADAPTIVE_PARAMS.keys(), best_params[1:]))
        strategy.update_parameters(optimized_adaptive_params)
        strategy.set_capital(optimized_capital)

        return strategy, -best_fun

    def optimize_all_strategies(self) -> List[Tuple['Strategy', float]]:
        optimized_strategies = Parallel(n_jobs=-1)(
            delayed(self.optimize_strategy)(strategy_name) for strategy_name in self.strategies
        )
        return sorted(optimized_strategies, key=lambda x: x[1], reverse=True)
    
    def temporary_optimize(self, strategy: 'Strategy') -> 'Strategy':
        original_params = strategy.get_parameters()
        original_capital = strategy.get_capital()
        
        optimized_strategy, _ = self.optimize_strategy(strategy.name)
        
        yield optimized_strategy
        
        strategy.set_parameters(original_params)
        strategy.set_capital(original_capital)
        
        return strategy