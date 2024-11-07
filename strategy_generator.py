import google.generativeai as genai
import pandas as pd
import os
import json
from typing import Dict, List
from strategy import Strategy, TimeFrame
import logging
import asyncio
from api_call_manager import APICallManager

VALID_STRATEGY_PARAMETERS = {
    'trend_following': [
        'MOVING_AVERAGE_SHORT',
        'MOVING_AVERAGE_LONG',
        'TREND_STRENGTH_THRESHOLD',
        'TREND_CONFIRMATION_PERIOD',
        'MOMENTUM_FACTOR',
        'BREAKOUT_LEVEL',
        'TRAILING_STOP'
    ],
    'mean_reversion': [
        'MEAN_WINDOW',
        'STD_MULTIPLIER',
        'MEAN_REVERSION_THRESHOLD',
        'ENTRY_DEVIATION',
        'EXIT_DEVIATION',
        'BOLLINGER_PERIOD',
        'BOLLINGER_STD'
    ],
    'momentum': [
        'MOMENTUM_PERIOD',
        'MOMENTUM_THRESHOLD',
        'RSI_PERIOD',
        'RSI_OVERBOUGHT',
        'RSI_OVERSOLD',
        'ACCELERATION_FACTOR',
        'MAX_ACCELERATION',
        'MACD_FAST',
        'MACD_SLOW',
        'MACD_SIGNAL'
    ],
    'breakout': [
        'BREAKOUT_PERIOD',
        'BREAKOUT_THRESHOLD',
        'VOLUME_CONFIRMATION_MULT',
        'CONSOLIDATION_PERIOD',
        'SUPPORT_RESISTANCE_LOOKBACK',
        'BREAKOUT_CONFIRMATION_CANDLES',
        'ATR_PERIOD'
    ],
    'volatility_clustering': [
        'VOLATILITY_WINDOW',
        'HIGH_VOLATILITY_THRESHOLD',
        'LOW_VOLATILITY_THRESHOLD',
        'GARCH_LAG',
        'ATR_MULTIPLIER',
        'VOLATILITY_BREAKOUT_THRESHOLD',
        'VOLATILITY_MEAN_PERIOD'
    ],
    'statistical_arbitrage': [
        'LOOKBACK_PERIOD',
        'Z_SCORE_THRESHOLD',
        'CORRELATION_THRESHOLD',
        'HALF_LIFE',
        'HEDGE_RATIO',
        'ENTRY_THRESHOLD',
        'EXIT_THRESHOLD',
        'WINDOW_SIZE',
        'MIN_CORRELATION',
        'COINTEGRATION_THRESHOLD'
    ],
    'sentiment_analysis': [
        'POSITIVE_SENTIMENT_THRESHOLD',
        'NEGATIVE_SENTIMENT_THRESHOLD',
        'SENTIMENT_WINDOW',
        'SENTIMENT_IMPACT_WEIGHT',
        'NEWS_IMPACT_DECAY',
        'SENTIMENT_SMOOTHING_FACTOR',
        'SENTIMENT_VOLUME_THRESHOLD',
        'SENTIMENT_MOMENTUM_PERIOD'
    ]
}

class StrategyGenerator:
    def __init__(self, config):
        self.config = config
        genai.configure(api_key=os.environ['GOOGLE_AI_API_KEY'])
        self.model = genai.GenerativeModel('gemini-pro')
        self.api_call_manager = APICallManager()
        self.logger = logging.getLogger(__name__)

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
        prompt = f"Given the following market data for {time_frame.value} analysis:\n"
        prompt += f"Asset prices: {market_data['close'].tail().to_dict()}\n"
        prompt += f"Volume: {market_data['volume'].tail().to_dict()}\n"
        prompt += f"50-day moving average: {market_data['close'].rolling(50).mean().iloc[-1]}\n"
        prompt += f"14-day RSI: {self.calculate_rsi(market_data['close'], 14).iloc[-1]}\n"
        prompt += (
            f"Generate exactly 2 trading strategies for the {time_frame.value} time frame.\n"
            f"Each strategy must use one of these exact strategy types in favored_patterns:\n"
            "- trend_following\n"
            "- mean_reversion\n"
            "- momentum\n"
            "- breakout\n"
            "- volatility_clustering\n"
            "- statistical_arbitrage\n"
            "- sentiment_analysis\n\n"
            f"Each strategy must use between 2 and 5 parameters from these valid parameters for each strategy type:\n"
        )

        # Add valid parameters for each strategy type
        for strategy_type, params in VALID_STRATEGY_PARAMETERS.items():
            prompt += f"\n{strategy_type}: {', '.join(params)}\n"

        prompt += (
            f"Return a JSON array with this structure. Include between 2-5 parameters per strategy(Timeframe value is an example):\n"
            '[\n'
            '  {\n'
            '    "name": "Strategy Name",\n'
            '    "description": "Strategy description",\n'
            '    "parameters": {\n'
            '      "param1": value1,\n'
            '      "param2": value2,\n'
            '      "param3": value3  // Optional additional parameters up to 5\n'
            '    },\n'
            '    "favored_patterns": ["pattern1"],\n'
            '    "time_frame": "1h"\n'
            '  }\n'
            ']'
        )
        return prompt
    def parse_strategies(self, response_text: str, time_frame: TimeFrame) -> List[Strategy]:
        try:
            cleaned_response = response_text.replace('', '').replace('', '').strip()
            strategy_data = json.loads(cleaned_response)
            strategies = []
        
            valid_strategy_types = [
                'trend_following',
                'mean_reversion',
                'momentum',
                'breakout',
                'volatility_clustering',
                'statistical_arbitrage',
                'sentiment_analysis'
            ]
        
            for data in strategy_data:
                # Validate strategy type
                strategy_type = data.get('favored_patterns', [])[0] if data.get('favored_patterns') else None
                if not strategy_type or strategy_type not in valid_strategy_types:
                    continue
            
                valid_params = VALID_STRATEGY_PARAMETERS.get(strategy_type, [])
                filtered_parameters = {
                    k: v for k, v in data.get('parameters', {}).items() 
                    if k in valid_params or k == 'INITIAL_CAPITAL'
                }
            
                # Enforce parameter count requirements
                if len(filtered_parameters) < 2:
                    default_params = self.config.ADAPTIVE_PARAMS[f'{strategy_type.upper()}_PARAMS']
                    for param in valid_params[:2]:
                        if param not in filtered_parameters:
                            filtered_parameters[param] = default_params.get(param)
                elif len(filtered_parameters) > 5:
                    filtered_parameters = dict(list(filtered_parameters.items())[:5])
            
                strategy = Strategy(
                    strategy_name=data.get('name', 'Default Strategy'),
                    description=data.get('description', 'Default Description'),
                    parameters=filtered_parameters,
                    favored_patterns=[strategy_type],
                    time_frame=time_frame
                )
                strategies.append(strategy)
        
            return strategies
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON response: {response_text}")
            return []
        except Exception as e:
            logging.error(f"Error parsing strategies: {str(e)}")
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