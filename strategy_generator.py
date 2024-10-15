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
        try:
            if await self.api_call_manager.can_make_call():
                for time_frame in TimeFrame:
                    prompt = self._create_prompt(market_data, time_frame)
                    response = self.model.generate_content(prompt)
                    strategies[time_frame] = self.parse_strategies(response.text, time_frame)
                logging.info("Generated strategies for all time frames")
            else:
                wait_time = await self.api_call_manager.time_until_reset()
                logging.warning(f"API call limit reached. Waiting for {wait_time:.2f} seconds.")
                await asyncio.sleep(wait_time)
                return await self.generate_strategies(market_data)
        except Exception as e:
            logging.error(f"Error generating strategies: {str(e)}")
            for time_frame in TimeFrame:
                strategies[time_frame] = []
        return strategies

    def _create_prompt(self, market_data: pd.DataFrame, time_frame: TimeFrame) -> str:
        prompt = f"Given the following market data for {time_frame.value} analysis:\n"
        prompt += f"Asset prices: {market_data['close'].tail().to_dict()}\n"
        prompt += f"Volume: {market_data['volume'].tail().to_dict()}\n"
        prompt += f"50-day moving average: {market_data['close'].rolling(50).mean().iloc[-1]}\n"
        prompt += f"14-day RSI: {self.calculate_rsi(market_data['close'], 14).iloc[-1]}\n"
        prompt = f"Generate trading strategies for the {time_frame.value} time frame. Provide 2 strategies with the following information:\n"
        prompt += "1. A name for the strategy\n"
        prompt += "2. A brief description of how it works\n"
        prompt += "3. The key parameters it uses (e.g., moving average periods, RSI thresholds) in the format 'parameter_name: value'\n"
        prompt += "4. The strategy's favored patterns (e.g., ['trend_following', 'mean_reversion', 'volatility_clustering', 'momentum', 'breakout'])\n"
        prompt += "Provide the response in JSON format, with an array of strategies."

        return prompt

    def parse_strategies(self, strategies_text: str, time_frame: TimeFrame) -> List[Strategy]:
        # Clean the response
        cleaned_text = strategies_text.strip().replace('```json', '').replace('```', '')
        print(cleaned_text)
        try:
            strategies_data = json.loads(cleaned_text)
            if 'trading_strategies' in strategies_data:
                strategies_data = strategies_data['trading_strategies']
            elif 'strategies' in strategies_data:
                strategies_data = strategies_data['strategies']
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response for {time_frame.value}: {str(e)}")
            print("Attempting to extract strategy information from raw text.")
            strategies_data = self.extract_strategy_info(cleaned_text)

        generated_strategies = []
        for strategy_data in strategies_data:
            try:
                name = strategy_data['name']
                description = strategy_data['description']
                parameters = strategy_data['parameters']
                favored_patterns = strategy_data['favored_patterns' or 'patterns']

                # Ensure required parameters are present
                parameters['INITIAL_CAPITAL'] = parameters.get('INITIAL_CAPITAL', 10000)
                parameters['MAX_POSITION_SIZE'] = parameters.get('MAX_POSITION_SIZE', 0.1)
                parameters['TRADING_FEE'] = parameters.get('TRADING_FEE', 0.001)

                generated_strategies.append(Strategy(name, description, parameters, favored_patterns, time_frame))
            except KeyError as e:
                print(f"Error processing strategy data: Missing key {e}")

        if not generated_strategies:
            print(f"No valid strategies found for {time_frame.value}. Falling back to default strategy.")
            return [self.create_default_strategy(time_frame)]

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
    
    def create_default_strategy(self, time_frame):
        # Define different strategies for each time frame
        if time_frame == TimeFrame.SHORT_TERM:
            # Short-term strategy (e.g., day trading or swing trading)
            return Strategy(
                strategy_name="Short-Term Strategy",
                description="Strategy for short-term trading, focusing on quick gains",
                parameters={
                    "indicator": "RSI",
                    "threshold": 70,  # Sell when RSI is above 70
                    "stop_loss": 0.02  # Stop-loss at 2% below purchase price
                },
                favored_patterns=["Bullish Engulfing", "Bearish Reversal"],
                time_frame=TimeFrame.SHORT_TERM
            )

        elif time_frame == TimeFrame.MID_TERM:
            # Mid-term strategy (e.g., holding for weeks or months)
            return Strategy(
                strategy_name="Mid-Term Strategy",
                description="Strategy for mid-term trading, holding for weeks to months",
                parameters={
                    "indicator": "MACD",
                    "signal_line": 9,  # Signal line for MACD crossover
                    "take_profit": 0.10,  # Take profit at 10% gain
                    "stop_loss": 0.05  # Stop-loss at 5% below purchase price
                },
                favored_patterns=["Head and Shoulders", "Double Bottom"],
                time_frame=TimeFrame.MID_TERM
            )

        elif time_frame == TimeFrame.LONG_TERM:
            # Long-term strategy (e.g., buy-and-hold for years)
            return Strategy(
                strategy_name="Long-Term Strategy",
                description="Strategy for long-term investing, holding for years",
                parameters={
                    "indicator": "EMA",
                    "periods": [50, 200],  # Use 50-day and 200-day EMAs
                    "rebalancing_period": "Annually"  # Rebalance once a year
                },
                favored_patterns=["Golden Cross", "Ascending Triangle"],
                time_frame=TimeFrame.LONG_TERM
            )

        elif time_frame == TimeFrame.SEASONAL_TERM:
            # Seasonal-term strategy (e.g., based on seasonal trends)
            return Strategy(
                strategy_name="Seasonal-Term Strategy",
                description="Strategy based on seasonal trends, adjusted quarterly",
                parameters={
                    "indicator": "Seasonal Index",
                    "adjustment_period": "Quarterly",
                    "sectors": ["Technology", "Retail"]  # Favor sectors with seasonal growth
                },
                favored_patterns=["Seasonal Breakout", "Support Bounce"],
                time_frame=TimeFrame.SEASONAL_TERM
            )

        else:
            # Fallback default strategy if time frame doesn't match
            return Strategy(
                strategy_name="Default Strategy",
                description="Simple buy-and-hold strategy",
                parameters={},
                favored_patterns=[],
                time_frame=TimeFrame.SHORT_TERM  # Default to short-term
            )
