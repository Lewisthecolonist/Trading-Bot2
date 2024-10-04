import google.generativeai as genai
import pandas as pd
import os

class StrategySelector:
    def __init__(self, config):
        self.config = config
        genai.configure(api_key=os.environ['GOOGLE_AI_API_KEY'])
        self.model = genai.GenerativeModel('gemini-pro')

    def select_strategy(self, strategies, strategy_performance, market_data):
        prompt = f"Given the following market data:\n"
        prompt += f"Asset prices: {market_data['close'].tail().to_dict()}\n"
        prompt += f"Volume: {market_data['volume'].tail().to_dict()}\n"
        prompt += f"50-day moving average: {market_data['close'].rolling(50).mean().iloc[-1]}\n"
        prompt += f"14-day RSI: {self.calculate_rsi(market_data['close'], 14).iloc[-1]}\n"
        prompt += "And the following trading strategies with their performance:\n"

        for i, (strategy_name, strategy) in enumerate(strategies.items(), 1):
            performance = strategy_performance.get(strategy_name, {})
            prompt += f"{i}. {strategy.name}: {strategy.description}\n"
            prompt += f"   Strength: {strategy.strength}\n"
            prompt += f"   Sharpe Ratio: {performance.get('sharpe_ratio', 'N/A')}\n"
            prompt += f"   Profit Factor: {performance.get('profit_factor', 'N/A')}\n"
            prompt += f"   Win Rate: {performance.get('win_rate', 'N/A')}\n"

        prompt += "Select the best strategy for the current market conditions, considering each strategy's strength and performance. Respond with only the number of the strategy."

        response = self.model.generate_content(prompt, generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=5,
        ))

        try:
            selected_strategy_index = int(response.text.strip()) - 1
            selected_strategy = list(strategies.values())[selected_strategy_index]
            self.config.update_adaptive_params(selected_strategy)
            return selected_strategy
        except (ValueError, IndexError):
            print("Invalid strategy selection. Defaulting to the first strategy.")
            return next(iter(strategies.values()))

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
