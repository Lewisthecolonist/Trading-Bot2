from scipy.optimize import differential_evolution
from joblib import Parallel, delayed
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

class StrategyOptimizer:
    def __init__(self, config, market_simulator, strategies):
        self.config = config
        self.market_simulator = market_simulator
        self.strategies = strategies

    def optimize_strategy(self, strategy_name):
        strategy = self.strategies[strategy_name]
        
        def objective(params):
            capital = params[0]
            adaptive_params = dict(zip(self.config.ADAPTIVE_PARAMS.keys(), params[1:]))
            strategy.update_parameters(adaptive_params)
            strategy.set_capital(capital)
            
            # Generate market data once for all folds
            market_data = self.market_simulator.generate_market_data(days=self.config.SIMULATION_DAYS)
            
            # Implement time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            performances = []
            for train_index, test_index in tscv.split(market_data):
                train_data = market_data.iloc[train_index]
                test_data = market_data.iloc[test_index]
                
                # Run simulation on test data
                performance = self.market_simulator.run_simulation(strategy, test_data)
                performances.append((performance['sharpe_ratio'], performance['max_drawdown']))
            
            # Return mean performance across all folds
            mean_sharpe = np.mean([p[0] for p in performances])
            mean_drawdown = np.mean([p[1] for p in performances])
            return -mean_sharpe, mean_drawdown

        # Implement adaptive bounds
        def adaptive_bounds(best_params, iteration):
            if iteration == 0:
                return [(self.config.INITIAL_CAPITAL * 0.1, self.config.INITIAL_CAPITAL * 10)] + \
                       [(value * 0.1, value * 10) for value in self.config.ADAPTIVE_PARAMS.values()]
            else:
                return [(max(p * 0.5, b[0]), min(p * 1.5, b[1])) for p, b in zip(best_params, bounds)]

        bounds = adaptive_bounds(None, 0)
        best_params = None

        for i in range(5):  # Run optimization multiple times with adapting bounds 
            result = differential_evolution(
                lambda x: objective(x)[0],  # Optimize for Sharpe ratio
                bounds,
                constraints=({'type': 'ineq', 'fun': lambda x: -objective(x)[1] + 0.2}),  # Max drawdown constraint
                popsize=20,
                mutation=(0.5, 1),
                recombination=0.7,
                updating='deferred',
                workers=-1
            )

            if best_params is None or result.fun < best_fun:
                best_params = result.x
                best_fun = result.fun

            bounds = adaptive_bounds(best_params, i+1)

        optimized_capital = best_params[0]
        optimized_adaptive_params = dict(zip(self.config.ADAPTIVE_PARAMS.keys(), best_params[1:]))
        strategy.update_parameters(optimized_adaptive_params)
        strategy.set_capital(optimized_capital)

        return strategy, -best_fun

    def optimize_all_strategies(self):
        optimized_strategies = Parallel(n_jobs=-1)(
            delayed(self.optimize_strategy)(strategy_name) for strategy_name in self.strategies
        )
        return sorted(optimized_strategies, key=lambda x: x[1], reverse=True)
    
    def temporary_optimize(self, strategy, current_market_data):
        original_params = strategy.get_parameters()
        
        # Perform optimization on current market data
        optimized_strategy = self.optimize_strategy(strategy, current_market_data)
        
        # Use the optimized strategy for the current iteration
        yield optimized_strategy
        
        # Revert to original parameters
        strategy.set_parameters(original_params)
