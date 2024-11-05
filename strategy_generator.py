import google.generativeai as genai
import pandas as pd
import os
import json
from typing import Dict, List
from strategy import Strategy, TimeFrame
import logging
import asyncio
from api_call_manager import APICallManager

class StrategyGenerator:
    def __init__(self, config):
        self.config = config
        genai.configure(api_key=os.environ['GOOGLE_AI_API_KEY'])
        self.model = genai.GenerativeModel('gemini-pro')
        self.api_call_manager = APICallManager()

    async def generate_strategies(self, market_data: pd.DataFrame) -> Dict[TimeFrame, List[Strategy]]:
        strategies = {}
        try:
            for time_frame in TimeFrame:
                prompt = self._create_prompt(market_data, time_frame)
                response = self.model.generate_content(prompt)
                
                # Parse the strategies as a dictionary
                strategies[time_frame] = self.parse_strategies(response.text, time_frame)

            logging.info("Generated strategies for all time frames")
        except Exception as e:
            logging.error(f"Error generating strategies: {str(e)}")
            for time_frame in TimeFrame:
                strategies[time_frame] = []
        return strategies

    def _create_prompt(self, market_data: pd.DataFrame, time_frame: TimeFrame) -> str:
        # Create a clear, precise prompt for the model
        prompt = f"Given the following market data for {time_frame.value} analysis:\n"
        prompt += f"Asset prices: {market_data['close'].tail().to_dict()}\n"
        prompt += f"Volume: {market_data['volume'].tail().to_dict()}\n"
        prompt += f"50-day moving average: {market_data['close'].rolling(50).mean().iloc[-1]}\n"
        prompt += f"14-day RSI: {self.calculate_rsi(market_data['close'], 14).iloc[-1]}\n"
        prompt += (
            f"Generate exactly 2 trading strategies for the {time_frame.value} time frame.\n"
            f"Make sure to only include the following patterns mentioned as they are the only ones currently supported:\n"
                "trend_following, "
                "mean_reversion, "
                "momentum, "
                "breakout, "
                "volatility_clustering, "
                "statistical_arbitrage, "
                "sentiment_analysis\n"
            f"Return a JSON array with this exact structure, but don't mind the parameters used this is just an example:\n"
            '[\n'
            '  {\n'
            '    "name": "Moving Average Strategy",\n'
            '    "description": "Moving average crossover strategy",\n'
            '    "parameters": {\n'
            '      "short_ma_window": 50,\n'
            '      "INITIAL_CAPITAL": 10000\n'
            '    },\n'
            '    "favored_patterns": ["trend_following"],\n'
            '    "time_frame": "1h"\n'
            '  },\n'
            '  {\n'
            '    "name": "RSI Strategy",\n'
            '    "description": "RSI-based momentum strategy",\n'
            '    "parameters": {\n'
            '      "rsi_period": 14,\n'
            '      "INITIAL_CAPITAL": 10000\n'
            '    },\n'
            '    "favored_patterns": ["momentum"],\n'
            '    "time_frame": "1h"\n'
            '  }\n'
            ']'
        )
        return prompt

    def parse_strategies(self, response_text: str, time_frame: TimeFrame) -> List[Strategy]:
        try:
            # Remove the  prefix and  suffix if present
            cleaned_response = response_text.replace('', '').replace('', '').strip()
            strategy_data = json.loads(cleaned_response)
            strategies = []
        
            for data in strategy_data:
                strategy = Strategy(
                    strategy_name=data.get('name', 'Default Strategy'),
                    description=data.get('description', 'Default Description'),
                    parameters=data.get('parameters', {'INITIAL_CAPITAL': 10000}),
                    favored_patterns=data.get('favored_patterns', ['trend_following']),
                    time_frame=time_frame
                )
                strategies.append(strategy)
            return strategies
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON response: {response_text}")
            return []

    def extract_strategy_info(self, text: str) -> List[Dict]:
        # Extract strategy information from raw text
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
        # RSI Calculation
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def create_default_strategy(self, time_frame):
        # Define a default strategy for fallback
        if time_frame == TimeFrame.SHORT_TERM:
            return Strategy(
                strategy_name="Short-Term Strategy",
                description="Strategy for short-term trading, focusing on quick gains",
                parameters={
                    "indicator": "RSI",
                    "threshold": 70,  
                    "stop_loss": 0.02
                },
                favored_patterns=["Bullish Engulfing", "Bearish Reversal"],
                time_frame=TimeFrame.SHORT_TERM
            )
        elif time_frame == TimeFrame.MID_TERM:
            return Strategy(
                strategy_name="Mid-Term Strategy",
                description="Strategy for mid-term trading, holding for weeks to months",
                parameters={
                    "indicator": "MACD",
                    "signal_line": 9,  
                    "take_profit": 0.10,
                    "stop_loss": 0.05
                },
                favored_patterns=["Head and Shoulders", "Double Bottom"],
                time_frame=TimeFrame.MID_TERM
            )
        elif time_frame == TimeFrame.LONG_TERM:
            return Strategy(
                strategy_name="Long-Term Strategy",
                description="Strategy for long-term investing, holding for years",
                parameters={
                    "indicator": "EMA",
                    "periods": [50, 200],
                    "rebalancing_period": "Annually"
                },
                favored_patterns=["Golden Cross", "Ascending Triangle"],
                time_frame=TimeFrame.LONG_TERM
            )
        elif time_frame == TimeFrame.SEASONAL_TERM:
            return Strategy(
                strategy_name="Seasonal-Term Strategy",
                description="Strategy based on seasonal trends, adjusted quarterly",
                parameters={
                    "indicator": "Seasonal Index",
                    "adjustment_period": "Quarterly",
                    "sectors": ["Technology", "Retail"]
                },
                favored_patterns=["Seasonal Breakout", "Support Bounce"],
                time_frame=TimeFrame.SEASONAL_TERM
            )
        else:
            return Strategy(
                strategy_name="Default Strategy",
                description="Simple buy-and-hold strategy",
                parameters={},
                favored_patterns=[],
                time_frame=TimeFrame.SHORT_TERM
            )