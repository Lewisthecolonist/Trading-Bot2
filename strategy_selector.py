import google.generativeai as genai
import pandas as pd
import os
from typing import Dict, List
from strategy import Strategy, TimeFrame

class StrategySelector:
    def __init__(self, config):
        self.config = config
        genai.configure(api_key=os.environ['GOOGLE_AI_API_KEY'])
        self.model = genai.GenerativeModel('gemini-pro')

    def select_strategies(self, strategies: Dict[TimeFrame, List[Strategy]], strategy_performance: Dict[str, Dict], market_data: pd.DataFrame) -> Dict[TimeFrame, Strategy]:
        selected_strategies = {}
        for time_frame in TimeFrame:
            selected_strategies[time_frame] = self.select_strategy_for_timeframe(strategies[time_frame], strategy_performance, market_data, time_frame)
        return selected_strategies

    def select_strategy_for_timeframe(self, strategies: List[Strategy], strategy_performance: Dict[str, Dict], market_data: pd.DataFrame, time_frame: TimeFrame) -> Strategy:
        prompt = self._create_prompt(strategies, strategy_performance, market_data, time_frame)
        response = self.model.generate_content(prompt, generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=5,
        ))

        try:
            selected_strategy_index = int(response.text.strip()) - 1
            selected_strategy = strategies[selected_strategy_index]
            self.config.update_adaptive_params(selected_strategy)
            return selected_strategy
        except (ValueError, IndexError):
            print(f"Invalid strategy selection for {time_frame.value}. Defaulting to the first strategy.")
            return strategies[0] if strategies else None

    def _create_prompt(self, strategies: List[Strategy], strategy_performance: Dict[str, Dict], market_data: pd.DataFrame, time_frame: TimeFrame) -> str:
        prompt = f"Given the following market data for {time_frame.value} analysis:\n"
        prompt += f"Asset prices: {market_data['close'].tail().to_dict()}\n"
        prompt += f"Volume: {market_data['volume'].tail().to_dict()}\n"
        prompt += f"50-day moving average: {market_data['close'].rolling(50).mean().iloc[-1]}\n"
        prompt += f"14-day RSI: {self.calculate_rsi(market_data['close'], 14).iloc[-1]}\n"
        prompt += f"And the following trading strategies for {time_frame.value} with their performance:\n"

        for i, strategy in enumerate(strategies, 1):
            performance = strategy_performance.get(strategy.name, {})
            prompt += f"{i}. {strategy.name}: {strategy.description}\n"
            prompt += f"   Favored patterns: {', '.join(strategy.favored_patterns)}\n"
            prompt += f"   Sharpe Ratio: {performance.get('sharpe_ratio', 'N/A')}\n"
            prompt += f"   Profit Factor: {performance.get('profit_factor', 'N/A')}\n"
            prompt += f"   Win Rate: {performance.get('win_rate', 'N/A')}\n"

        prompt += f"Select the best strategy for the current {time_frame.value} market conditions, considering each strategy's favored patterns and performance. Respond with only the number of the strategy."

        return prompt

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
