import google.generativeai as genai
import pandas as pd
import os
from typing import Dict, List
from strategy import Strategy, TimeFrame
import prophet as Prophet
from functools import lru_cache

class StrategySelector:
    def __init__(self, config):
        self.config = config
        genai.configure(api_key=os.environ['GOOGLE_AI_API_KEY'])
        self.model = genai.GenerativeModel('gemini-pro')
        self.prophet_model = Prophet.Prophet(daily_seasonality=True)
        self.prediction_horizon = {
            TimeFrame.SHORT_TERM: 1,
            TimeFrame.MID_TERM: 7,
            TimeFrame.LONG_TERM: 90,
            TimeFrame.SEASONAL_TERM: 365
        }

    @lru_cache(maxsize=100)
    def generate_predictions(self, time_frame: TimeFrame, market_data: pd.DataFrame):
        df = market_data[['close']].reset_index()
        df.columns = ['ds', 'y']
        self.prophet_model.fit(df)
        future = self.prophet_model.make_future_dataframe(periods=self.prediction_horizon[time_frame])
        forecast = self.prophet_model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    def select_strategies(self, strategies: Dict[TimeFrame, List[Strategy]], strategy_performance: Dict[str, Dict], market_data: Dict,asset_position_value: float) -> Dict[TimeFrame, List[Strategy]]:
        # Modified to focus on longer-term strategies
        prompt = self._create_prompt(strategies, strategy_performance, market_data, asset_position_value)
        response = self.model.generate_content(prompt)
        
        selected_strategies = {}
        for time_frame in [TimeFrame.MID_TERM, TimeFrame.LONG_TERM, TimeFrame.SEASONAL_TERM]:
            if time_frame in strategies:
                selected_indices = [int(idx) for idx in response.text.split('\n')[time_frame.value].split(',')]
                selected_strategies[time_frame] = [strategies[time_frame][i] for i in selected_indices if i < len(strategies[time_frame])][:10]

                # If we don't have enough suitable strategies, generate new ones
                if len(selected_strategies[time_frame]) < 3:
                    new_strategies = self.generate_new_strategies(time_frame, market_data, asset_position_value, 3 - len(selected_strategies[time_frame]))
                    selected_strategies[time_frame].extend(new_strategies)
                    strategies[time_frame].extend(new_strategies)  # Add new strategies to the existing list

        return selected_strategies
    def generate_new_strategies(self, time_frame: TimeFrame, market_data: pd.DataFrame, asset_position_value: float, num_strategies: int) -> List[Strategy]:
        prompt = self._create_new_strategy_prompt(time_frame, market_data, asset_position_value, num_strategies)
        response = self.model.generate_content(prompt, generation_config=genai.types.GenerationConfig(
            temperature=0.5,
            max_output_tokens=1000,
        ))

        new_strategies = []
        strategy_specs = response.text.split('\n\n')
        for spec in strategy_specs:
            strategy_dict = eval(spec)
            new_strategy = Strategy(
                strategy_name=strategy_dict['name'],
                description=strategy_dict['description'],
                parameters=strategy_dict['parameters'],
                favored_patterns=strategy_dict['favored_patterns'],
                time_frame=time_frame
            )
            new_strategies.append(new_strategy)

        return new_strategies

    def _create_new_strategy_prompt(self, time_frame: TimeFrame, market_data: pd.DataFrame, asset_position_value: float, num_strategies: int) -> str:
        prompt = f"Given the following market conditions for {time_frame.value} analysis:\n"
        prompt += f"Asset prices: {market_data['close'].tail().to_dict()}\n"
        prompt += f"Volume: {market_data['volume'].tail().to_dict()}\n"
        prompt += f"50-day moving average: {market_data['close'].rolling(50).mean().iloc[-1]}\n"
        prompt += f"14-day RSI: {self.calculate_rsi(market_data['close'], 14).iloc[-1]}\n"
        prompt += f"Current asset position value: ${asset_position_value:.2f}\n"
        prompt += f"\nCreate {num_strategies} new trading strategies optimized for these market conditions and the {time_frame.value} time frame."
        prompt += "\nRespond with a list of strategy dictionaries in the following format:"
        prompt += "\n{'name': 'Strategy Name', 'description': 'Strategy description', 'parameters': {'param1': value1, 'param2': value2}, 'favored_patterns': ['pattern1', 'pattern2']}"

        return prompt

    def _create_prompt(self, strategies: Dict[TimeFrame, List[Strategy]], strategy_performance: Dict[str, Dict], market_data: pd.DataFrame, asset_position_value: float) -> str:
        prompt = "Given the following market data:\n"
        prompt += f"Asset prices: {market_data['close'].tail().to_dict()}\n"
        prompt += f"Volume: {market_data['volume'].tail().to_dict()}\n"
        prompt += f"50-day moving average: {market_data['close'].rolling(50).mean().iloc[-1]}\n"
        prompt += f"14-day RSI: {self.calculate_rsi(market_data['close'], 14).iloc[-1]}\n"
        prompt += f"Bollinger Bands: {self.calculate_bollinger_bands(market_data['close']).tail().to_dict()}\n"
        prompt += f"MACD: {self.calculate_macd(market_data['close']).tail().to_dict()}\n"
        prompt += f"Current asset position value: ${asset_position_value:.2f}\n"

        for time_frame in TimeFrame:
            prompt += f"\nStrategies for {time_frame.value}:\n"
            predictions = self.generate_predictions(time_frame, market_data)
            last_prediction = predictions.iloc[-1]
            prompt += f"Predicted price in {self.prediction_horizon[time_frame]} days: {last_prediction['yhat']:.2f}\n"
            prompt += f"Prediction range: {last_prediction['yhat_lower']:.2f} - {last_prediction['yhat_upper']:.2f}\n"
            
            for i, strategy in enumerate(strategies[time_frame], 1):
                performance = strategy_performance.get(strategy.name, {})
                prompt += f"{i}. {strategy.name}: {strategy.description}\n"
                prompt += f"   Favored patterns: {', '.join(strategy.favored_patterns)}\n"
                prompt += f"   Parameters: {strategy.parameters}\n"
                prompt += f"   Sharpe Ratio: {performance.get('sharpe_ratio', 'N/A')}\n"
                prompt += f"   Profit Factor: {performance.get('profit_factor', 'N/A')}\n"
                prompt += f"   Win Rate: {performance.get('win_rate', 'N/A')}\n"

        prompt += "\nSelect up to 10 optimal strategies for each time frame considering current market conditions, predictions, and strategy performance. Don't select bad strategies because there are very litte or no optimal strategies. Respond with the indices of selected strategies for each time frame, separated by commas, in the following format:\n"
        prompt += "short_term: 1,3,5\nmid_term: 2,4,6\nlong_term: 1,2,3\nseasonal: 4,5,6"

        return prompt

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, num_std: int = 2) -> pd.DataFrame:
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return pd.DataFrame({'upper': upper_band, 'middle': sma, 'lower': lower_band})

    def calculate_macd(self, prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        histogram = macd - signal
        return pd.DataFrame({'macd': macd, 'signal': signal, 'histogram': histogram})

    def _get_timeframe_parameters(self, timeframe):
        return {
            TimeFrame.SHORT_TERM: {
                'data_points': 1440,  # Minutes in a day
                'prediction_window': 60,  # 1 hour
                'feature_importance_threshold': 0.8
            },
            TimeFrame.MID_TERM: {
                'data_points': 504,   # Hours in 3 weeks
                'prediction_window': 168,  # 1 week
                'feature_importance_threshold': 0.7
            },
            TimeFrame.LONG_TERM: {
                'data_points': 365,   # Days in a year
                'prediction_window': 30,   # 1 month
                'feature_importance_threshold': 0.6
            },
            TimeFrame.SEASONAL_TERM: {
                'data_points': 1095,  # Days in 3 years
                'prediction_window': 365,  # 1 year
                'feature_importance_threshold': 0.5
            }
        }[timeframe]
