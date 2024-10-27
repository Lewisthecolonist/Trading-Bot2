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
            "Generate exactly 2 trading strategies for the {time_frame.value} time frame. "
            "Ensure the response is valid JSON with this structure:\n"
            '[{"name": "strategy_name", "description": "strategy_description", '
            '"parameters": {"param1": "value1", "param2": "value2"}, '
            '"favored_patterns": ["pattern1", "pattern2"]}]'
        )
        return prompt

    def parse_strategies(self, strategies_text: str, time_frame: TimeFrame) -> List[Strategy]:
        # Clean the response
        cleaned_text = strategies_text.strip().replace('```json', '').replace('```', '')
        
        if not cleaned_text:
            logging.error(f"Received empty response for {time_frame.value}")
            return [self.create_default_strategy(time_frame)]  # Fallback
        
        try:
            # Try to parse the cleaned JSON text
            strategies_data = json.loads(cleaned_text)
            
            # Support multiple possible key structures
            strategies_data = strategies_data.get('trading_strategies', strategies_data.get('strategies', []))

            # Ensure it's a list of strategies
            if not isinstance(strategies_data, list):
                logging.error(f"Unexpected format for {time_frame.value} strategies. Falling back to default.")
                return [self.create_default_strategy(time_frame)]
            
        except json.JSONDecodeError as e:
            # Log the error and try to extract strategies from raw text
            logging.error(f"Error parsing JSON response for {time_frame.value}: {str(e)}")
            logging.warning("Attempting to extract strategy information from raw text.")
            strategies_data = self.extract_strategy_info(cleaned_text)

        generated_strategies = []
        for strategy_data in strategies_data:
            try:
                generated_strategies.append(
                    Strategy(
                        strategy_name=strategy_data.get('name', 'Unknown Strategy'),
                        description=strategy_data.get('description', ''),
                        parameters=strategy_data.get('parameters', {}),
                        favored_patterns=strategy_data.get('favored_patterns', []),
                        time_frame=time_frame
                    )
                )
            except KeyError as e:
                logging.error(f"Error processing strategy data: Missing key {e}")

        # Fallback to default strategy if no valid strategies are found
        if not generated_strategies:
            logging.info(f"No valid strategies found for {time_frame.value}. Falling back to default strategy.")
            return [self.create_default_strategy(time_frame)]

        return generated_strategies

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
