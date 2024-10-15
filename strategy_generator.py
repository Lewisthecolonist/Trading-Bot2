import google.generativeai as genai
import pandas as pd
import os
import json
from typing import Dict, List
from strategy import Strategy, TimeFrame
import time
from api_call_manager import APICallManager
import logging
import asyncio

class StrategyGenerator:
    def __init__(self, config):
        self.config = config
        genai.configure(api_key=os.environ['GOOGLE_AI_API_KEY'])
        self.model = genai.GenerativeModel('gemini-pro')
        self.api_call_manager = APICallManager()


    async def generate_strategies(self, market_data: pd.DataFrame) -> Dict[TimeFrame, List[Strategy]]:
        strategies = {}
        for time_frame in TimeFrame:
            try:
                if await self.api_call_manager.can_make_call():
                    prompt = self._create_prompt(market_data, time_frame)
                    response = self.model.generate_content(prompt)  # Remove await here
                    await self.api_call_manager.record_call()
                    strategies[time_frame] = self.parse_strategies(response.text, time_frame)
                    logging.info(f"Generated strategies for {time_frame}")
                else:
                    wait_time = await self.api_call_manager.time_until_reset()
                    logging.warning(f"API call limit reached. Waiting for {wait_time:.2f} seconds.")
                    await asyncio.sleep(wait_time)
                    return await self.generate_strategies(market_data)  # Retry after waiting
            except Exception as e:
                logging.error(f"Error generating strategies for {time_frame}: {str(e)}")
                strategies[time_frame] = []  # Set empty list for this time frame in case of error
        return strategies


    def _create_prompt(self, market_data: pd.DataFrame, time_frame: TimeFrame) -> str:
        prompt = f"Given the following market data for {time_frame.value} analysis:\n"
        prompt += f"Asset prices: {market_data['close'].tail().to_dict()}\n"
        prompt += f"Volume: {market_data['volume'].tail().to_dict()}\n"
        prompt += f"50-day moving average: {market_data['close'].rolling(50).mean().iloc[-1]}\n"
        prompt += f"14-day RSI: {self.calculate_rsi(market_data['close'], 14).iloc[-1]}\n"
        prompt += f"Generate {self.config.BASE_PARAMS['NUM_STRATEGIES_TO_GENERATE']} trading strategies suitable for {time_frame.value} trading. For each strategy, provide:\n"
        prompt += "1. A name for the strategy\n"
        prompt += "2. A brief description of how it works\n"
        prompt += "3. The key parameters it uses (e.g., moving average periods, RSI thresholds) in the format 'parameter_name: value'\n"
        prompt += "4. The strategy's favored patterns (e.g., ['trend_following', 'mean_reversion', 'volatility_clustering', 'momentum', 'breakout'])\n"
        prompt += "Provide the response in JSON format."
        return prompt

    def parse_strategies(self, strategies_text: str, time_frame: TimeFrame) -> List[Strategy]:
        generated_strategies = []
        print(f"Raw response for {time_frame.value}:\n{strategies_text}")  # Debug print

        try:
            strategies_data = json.loads(strategies_text)
            if not isinstance(strategies_data, list):
                raise ValueError("Expected a list of strategies")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing JSON response for {time_frame.value}: {str(e)}")
            print("Attempting to extract strategy information from raw text.")
            strategies_data = self.extract_strategy_info(strategies_text)

        for strategy_data in strategies_data:
            try:
                name = strategy_data['name']
                description = strategy_data['description']
                parameters = strategy_data['parameters']
                favored_patterns = strategy_data['favored_patterns']

                # Ensure required parameters are present
                parameters['INITIAL_CAPITAL'] = parameters.get('INITIAL_CAPITAL', 10000)
                parameters['MAX_POSITION_SIZE'] = parameters.get('MAX_POSITION_SIZE', 0.1)
                parameters['TRADING_FEE'] = parameters.get('TRADING_FEE', 0.001)

                # Add weight parameters for each favored pattern
                for pattern in favored_patterns:
                    parameters[f'{pattern}_weight'] = 1.0 / len(favored_patterns)

                generated_strategies.append(Strategy(name, description, parameters, favored_patterns, time_frame))
            except KeyError as e:
                print(f"Error processing strategy data: Missing key {e}")

        if not generated_strategies:
            print(f"No valid strategies found for {time_frame.value}. Falling back to default strategy.")
            return [Strategy("Default Strategy", "Simple moving average crossover", 
                         {"INITIAL_CAPITAL": 10000, "MAX_POSITION_SIZE": 0.1, "TRADING_FEE": 0.001, 
                          "short_ma_window": 10, "long_ma_window": 50},
                         ['trend_following'], time_frame)]

        return generated_strategies

    def extract_strategy_info(self, text: str) -> List[Dict]:
        strategies = []
        current_strategy = {}
        for line in text.split('\n'):
            if line.startswith('Name:'):
                if current_strategy:
                    strategies.append(current_strategy)
                current_strategy = {'name': line.split(':', 1)[1].strip()}
            elif line.startswith('Description:'):
                current_strategy['description'] = line.split(':', 1)[1].strip()
            elif line.startswith('Parameters:'):
                current_strategy['parameters'] = {}
            elif ':' in line and 'parameters' in current_strategy:
                key, value = line.split(':', 1)
                current_strategy['parameters'][key.strip()] = value.strip()
            elif line.startswith('Favored patterns:'):
                current_strategy['favored_patterns'] = [p.strip() for p in line.split(':', 1)[1].split(',')]
        if current_strategy:
            strategies.append(current_strategy)
        return strategies


    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))