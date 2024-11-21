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
        strategies = {tf: [] for tf in TimeFrame}

        for timeframe in TimeFrame:
            resampled_data = self._resample_data(market_data, timeframe)
            params = self._get_timeframe_parameters(timeframe)
            strategies[timeframe] = await self._generate_timeframe_strategies(
                resampled_data, 
                params
            )

        return strategies

    def _resample_data(self, data: pd.DataFrame, timeframe: TimeFrame):
        resample_rules = {
            'SHORT_TERM': "1min",     
            'MID_TERM': "1D",       
            'LONG_TERM': "1M",      
            'SEASONAL_TERM': "1Y"   
        }
    
        return data.resample(resample_rules[timeframe.name]).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
    def _create_prompt(self, market_data: pd.DataFrame, time_frame: TimeFrame) -> str:
        return self.config.get('timeframe_parameters', {}).get(TimeFrame.value, {})
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
        # Handle empty response
        if not response_text or response_text.isspace():
            logging.info("Empty response received, generating default strategies")
            return [self.create_default_strategy(time_frame)]
        
        # Clean and extract JSON
        cleaned_response = response_text.replace('', '').replace('', '').strip()
        if not cleaned_response:
            return [self.create_default_strategy(time_frame)]
        
        try:
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
                try:
                    # Validate strategy type
                    favored_patterns = data.get('favored_patterns', [])
                    if isinstance(favored_patterns, str):
                        favored_patterns = [favored_patterns]
                    strategy_type = favored_patterns[0] if favored_patterns else None
                    
                    if not strategy_type or strategy_type.lower() not in valid_strategy_types:
                        continue
                    
                    strategy_type = strategy_type.lower()
                    valid_params = VALID_STRATEGY_PARAMETERS.get(strategy_type, [])
                    
                    # Normalize and validate parameters
                    parameters = data.get('parameters', {})
                    if isinstance(parameters, str):
                        try:
                            parameters = json.loads(parameters)
                        except:
                            parameters = {}
                            
                    filtered_parameters = {
                        k.lower(): float(v) if isinstance(v, (int, float, str)) and str(v).replace('.','').isdigit() else v 
                        for k, v in parameters.items() 
                        if k.lower() in [p.lower() for p in valid_params] or k.lower() == 'initial_capital'
                    }
                
                    # Enforce parameter count requirements
                    if len(filtered_parameters) < 2:
                        default_params = self.config.ADAPTIVE_PARAMS[f'{strategy_type.upper()}_PARAMS']
                        for param in valid_params[:2]:
                            if param.lower() not in filtered_parameters:
                                filtered_parameters[param.lower()] = default_params.get(param)
                    elif len(filtered_parameters) > 5:
                        filtered_parameters = dict(list(filtered_parameters.items())[:5])
                
                    strategy = Strategy(
                        strategy_name=str(data.get('name', 'Default Strategy')),
                        description=str(data.get('description', 'Default Description')),
                        parameters=filtered_parameters,
                        favored_patterns=[strategy_type],
                        time_frame=time_frame
                    )
                    strategies.append(strategy)
                except Exception as strategy_error:
                    logging.warning(f"Error parsing individual strategy: {str(strategy_error)}")
                    continue
        
            return strategies if strategies else [self.create_default_strategy(time_frame)]
        
        except json.JSONDecodeError:
            logging.info("JSON parsing failed, using default strategy")
            return [self.create_default_strategy(time_frame)]
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
        if time_frame == TimeFrame.SHORT_TERM:
            return Strategy(
                strategy_name="Short-Term Momentum",
                description="Short-term momentum trading strategy",
                parameters={
                    'MOMENTUM_PERIOD': 14,
                    'MOMENTUM_THRESHOLD': 0.05,
                    'RSI_PERIOD': 14,
                    'RSI_OVERBOUGHT': 70,
                    'RSI_OVERSOLD': 30
                },
                favored_patterns=['momentum'],
                time_frame=TimeFrame.SHORT_TERM
            )
        elif time_frame == TimeFrame.MID_TERM:
            return Strategy(
                strategy_name="Mid-Term Mean Reversion",
                description="Mid-term mean reversion strategy",
                parameters={
                    'MEAN_WINDOW': 20,
                    'STD_MULTIPLIER': 2.0,
                    'MEAN_REVERSION_THRESHOLD': 0.05,
                    'ENTRY_DEVIATION': 0.02,
                    'EXIT_DEVIATION': 0.01
                },
                favored_patterns=['mean_reversion'],
                time_frame=TimeFrame.MID_TERM
            )
        elif time_frame == TimeFrame.LONG_TERM:
            return Strategy(
                strategy_name="Long-Term Trend Following",
                description="Long-term trend following strategy",
                parameters={
                    'MOVING_AVERAGE_SHORT': 50,
                    'MOVING_AVERAGE_LONG': 200,
                    'TREND_STRENGTH_THRESHOLD': 0.02,
                    'TREND_CONFIRMATION_PERIOD': 5
                },
                favored_patterns=['trend_following'],
                time_frame=TimeFrame.LONG_TERM
            )
        else:
            return Strategy(
                strategy_name="Volatility Clustering",
                description="Volatility-based trading strategy",
                parameters={
                    'VOLATILITY_WINDOW': 20,
                    'HIGH_VOLATILITY_THRESHOLD': 1.5,
                    'LOW_VOLATILITY_THRESHOLD': 0.5,
                    'ATR_MULTIPLIER': 2.0
                },
                favored_patterns=['volatility_clustering'],
                time_frame=time_frame
            )
        
    def _get_timeframe_parameters(self, timeframe: TimeFrame):
        return {
            TimeFrame.SHORT_TERM: {
                'data_points': 360,    # 6 hours in minutes
                'prediction_window': 60,  # 1 hour
                'sampling_interval': 'min'
            },
            TimeFrame.MID_TERM: {
                'data_points': 21,     # 3 weeks in days
                'prediction_window': 7,   # 1 week
                'sampling_interval': 'D'
            },
            TimeFrame.LONG_TERM: {
                'data_points': 12,     # 1 year in months
                'prediction_window': 1,    # 1 month
                'sampling_interval': 'ME'
            },
            TimeFrame.SEASONAL_TERM: {
                'data_points': 4,      # 3 years
                'prediction_window': 1,    # 1 year
                'sampling_interval': '2ME'
            }
        }[timeframe]

    def _generate_timeframe_strategies(self, market_data: pd.DataFrame, timeframe: TimeFrame) -> List[Strategy]:
        resampled_data = self._resample_data(market_data, timeframe)
        strategies = []
    
        # Generate strategies based on technical indicators
        for pattern in self.config.TRADING_PATTERNS:
            strategy_dict = self._get_strategy_parameters(timeframe)
            strategy_object = Strategy(
                name=strategy_dict["name"],
                parameters=strategy_dict["parameters"],
                favored_patterns=strategy_dict["favored_patterns"],
                time_frame=strategy_dict["time_frame"]
            )
            strategies.append(strategy_object)
    
        return strategies
