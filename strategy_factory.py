from typing import Dict, Any, Callable
import json
import os
import pandas as pd
from strategy import Strategy
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import inspect

class StrategyConfigHandler(FileSystemEventHandler):
    def __init__(self, strategy_factory, config_file_path):
        self.strategy_factory = strategy_factory
        self.config_file_path = config_file_path

    def on_modified(self, event):
        if event.src_path == self.config_file_path:
            print("Strategy configuration file changed. Updating strategies...")
            self.strategy_factory.update_strategies()

class StrategyFactory:
    def __init__(self, market_maker, config, strategy_config_path):
        self.strategy_config_path = strategy_config_path
        self.market_maker = market_maker
        self.strategies: Dict[str, Strategy] = {}
        self.update_strategies()
        self.signal_methods = self.create_signal_methods()
        # Set up file watcher
        self.event_handler = StrategyConfigHandler(self, self.config_file_path)
        self.observer = Observer()
        self.observer.schedule(self.event_handler, os.path.dirname(self.config_file_path), recursive=False)
        self.observer.start()

    def update_strategies(self):
        new_configs = self.load_strategy_configs()
        current_strategy_names = set(self.strategies.keys())
        new_strategy_names = set(new_configs.keys())

        # Add or update strategies and signal functions
        for name, config in new_configs.items():
            if name not in self.strategies or self.strategies[name].parameters != config['parameters']:
                self.strategies[name] = self.create_strategy(config)
                self.market_maker.update_strategy(name, self.strategies[name])

        # Remove deleted strategies
        for name in current_strategy_names - new_strategy_names:
            del self.strategies[name]
            self.market_maker.remove_strategy(name)

    def load_strategy_configs(self) -> Dict[str, Dict[str, Any]]:
        with open(self.config_file_path, 'r') as f:
            configs = json.load(f)
        return {config['name']: config for config in configs}

    def create_strategy(self, config: Dict[str, Any]) -> Strategy:
        name = config['name']
        description = config['description']
        signal_function = self.get_signal_function(config['signal_type'])
        parameters = config['parameters']
        return Strategy(name, description, signal_function, parameters)

    def create_signal_methods(self) -> Dict[str, Callable]:
        signal_methods = {}
        for strategy_name, strategy in self.strategies.items():
            signal_method = self.create_signal_method(strategy)
            signal_methods[strategy_name] = signal_method
        return signal_methods

    def create_signal_method(self, strategy: Strategy) -> Callable:
        def signal_method(market_data: pd.DataFrame, params: Dict[str, Any]) -> float:
            # Get the strategy's signal generation logic
            signal_logic = strategy.generate_signal

            # Inspect the signal_logic function to get its parameters
            sig = inspect.signature(signal_logic)
            
            # Prepare the arguments for the signal_logic function
            args = {}
            for param in sig.parameters.values():
                if param.name == 'self':
                    continue
                elif param.name == 'market_data':
                    args[param.name] = market_data
                elif param.name in params:
                    args[param.name] = params[param.name]
                else:
                    # If a required parameter is missing, you might want to raise an exception
                    raise ValueError(f"Missing required parameter: {param.name}")

            # Call the strategy's signal generation logic with the prepared arguments
            return signal_logic(**args)

        return signal_method

    def get_signal_method(self, strategy_name: str) -> Callable:
        return self.signal_methods.get(strategy_name, self.default_signal)

    @staticmethod
    def default_signal(market_data: pd.DataFrame, params: Dict[str, Any]) -> float:
        return 0

    def stop(self):
        self.observer.stop()
        self.observer.join()
