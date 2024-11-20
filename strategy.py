from enum import Enum
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from config import Config
import itertools
from datetime import datetime, timedelta

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

class TimeFrame(Enum):
    SHORT_TERM = 'min'
    MID_TERM = "D"
    LONG_TERM = "ME"
    SEASONAL_TERM = "A"

class Strategy:
    def __init__(self, strategy_name: str, description: str, parameters: Dict[str, Any], favored_patterns: List[str], time_frame: TimeFrame):
        self.name = strategy_name
        self.description = description
        self.parameters = parameters
        self.favored_patterns = favored_patterns
        self.time_frame = time_frame
        self.capital = parameters.get('INITIAL_CAPITAL', 10000)
        self.performance = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
        }
        self.protection_period = None  # Add this line
        # Add strategy-specific parameter initialization
        if 'trend_following' in favored_patterns:
            self.parameters.update({k: parameters.get(k, Config().ADAPTIVE_PARAMS['TREND_FOLLOWING_PARAMS'][k]) 
                                 for k in VALID_STRATEGY_PARAMETERS['trend_following']})
        elif 'mean_reversion' in favored_patterns:
            self.parameters.update({k: parameters.get(k, Config().ADAPTIVE_PARAMS['MEAN_REVERSION_PARAMS'][k]) 
                                 for k in VALID_STRATEGY_PARAMETERS['mean_reversion']})
        elif 'momentum' in favored_patterns:
            self.parameters.update({k: parameters.get(k, Config().ADAPTIVE_PARAMS['MOMENTUM_PARAMS'][k]) 
                                 for k in VALID_STRATEGY_PARAMETERS['momentum']})
        elif 'breakout' in favored_patterns:
            self.parameters.update({k: parameters.get(k, Config().ADAPTIVE_PARAMS['BREAKOUT_PARAMS'][k]) 
                                 for k in VALID_STRATEGY_PARAMETERS['breakout']})
        elif 'volatility_clustering' in favored_patterns:
            self.parameters.update({k: parameters.get(k, Config().ADAPTIVE_PARAMS['VOLATILITY_CLUSTERING_PARAMS'][k]) 
                                 for k in VALID_STRATEGY_PARAMETERS['volatility_clustering']})
        elif 'statistical_arbitrage' in favored_patterns:
            self.parameters.update({k: parameters.get(k, Config().ADAPTIVE_PARAMS['STATISTICAL_ARBITRAGE_PARAMS'][k]) 
                                 for k in VALID_STRATEGY_PARAMETERS['statistical_arbitrage']})
        elif 'sentiment_analysis' in favored_patterns:
            self.parameters.update({k: parameters.get(k, Config().ADAPTIVE_PARAMS['SENTIMENT_ANALYSIS_PARAMS'][k]) 
                                 for k in VALID_STRATEGY_PARAMETERS['sentiment_analysis']})

    def set_capital(self, capital: float):
        self.capital = capital

    def update_parameters(self, new_parameters: Dict[str, Any]):
        self.parameters.update(new_parameters)

    def generate_signal(self, market_data: pd.DataFrame) -> float:
        if 'trend_following' in self.favored_patterns:
            return self._trend_following_signal(market_data)
        elif 'mean_reversion' in self.favored_patterns:
            return self._mean_reversion_signal(market_data)
        elif 'momentum' in self.favored_patterns:
            return self._momentum_signal(market_data)
        elif 'breakout' in self.favored_patterns:
            return self._breakout_signal(market_data)
        elif 'volatility_clustering' in self.favored_patterns:
            return self._volatility_clustering_signal(market_data)
        elif 'statistical_arbitrage' in self.favored_patterns:
            return self._statistical_arbitrage_signal(market_data)
        elif 'sentiment_analysis' in self.favored_patterns:
            return self._sentiment_analysis_signal(market_data)
        else:
            return 0  # No signal if no recognized pattern

    def _trend_following_signal(self, market_data: pd.DataFrame) -> float:
        short_ma = market_data['close'].rolling(window=self.parameters.get('short_ma_window', 10)).mean()
        long_ma = market_data['close'].rolling(window=self.parameters.get('long_ma_window', 50)).mean()
    
        # Add a trend strength indicator
        trend_strength = (short_ma.iloc[-1] - long_ma.iloc[-1]) / long_ma.iloc[-1]
        strength_threshold = self.parameters.get('trend_strength_threshold', 0.02)
    
        if short_ma.iloc[-1] > long_ma.iloc[-1] and trend_strength > strength_threshold:
            return 1
        elif short_ma.iloc[-1] < long_ma.iloc[-1] and trend_strength < -strength_threshold:
            return -1
        return 0


    def _mean_reversion_signal(self, market_data: pd.DataFrame) -> float:
        current_price = market_data['close'].iloc[-1]
        mean_price = market_data['close'].rolling(window=self.parameters.get('mean_window', 20)).mean().iloc[-1]
        threshold = self.parameters.get('mean_reversion_threshold', 0.05)
        if current_price > mean_price * (1 + threshold):
            return -1  # Sell signal
        elif current_price < mean_price * (1 - threshold):
            return 1  # Buy signal
        return 0

    def _momentum_signal(self, market_data: pd.DataFrame) -> float:
        momentum_period = self.parameters.get('momentum_period', 14)
        momentum = market_data['close'].pct_change(periods=momentum_period).iloc[-1]
        threshold = self.parameters.get('momentum_threshold', 0.05)
        return 1 if momentum > threshold else -1 if momentum < -threshold else 0

    def _breakout_signal(self, market_data: pd.DataFrame) -> float:
        breakout_period = self.parameters.get('breakout_period', 20)
        upper_breakout = market_data['close'].rolling(window=breakout_period).max().iloc[-1]
        lower_breakout = market_data['close'].rolling(window=breakout_period).min().iloc[-1]
        current_price = market_data['close'].iloc[-1]
        if current_price > upper_breakout:
            return 1  # Buy signal on upward breakout
        elif current_price < lower_breakout:
            return -1  # Sell signal on downward breakout
        return 0

    def _volatility_clustering_signal(self, market_data: pd.DataFrame) -> float:
        volatility_window = self.parameters.get('volatility_window', 20)
        current_volatility = market_data['close'].pct_change().rolling(window=volatility_window).std().iloc[-1]
        avg_volatility = market_data['close'].pct_change().rolling(window=volatility_window).std().mean()
    
        volatility_ratio = current_volatility / avg_volatility
        high_volatility_threshold = self.parameters.get('high_volatility_threshold', 1.5)
        low_volatility_threshold = self.parameters.get('low_volatility_threshold', 0.5)
    
        if volatility_ratio > high_volatility_threshold:
            return 1 if market_data['close'].pct_change().iloc[-1] > 0 else -1
        elif volatility_ratio < low_volatility_threshold:
            return 0.5 if market_data['close'].pct_change().iloc[-1] > 0 else -0.5
        return 0

    
    def _statistical_arbitrage_signal(self, market_data: pd.DataFrame) -> float:
        # Assuming we're comparing two related assets
        asset1 = market_data['asset1_close']
        asset2 = market_data['asset2_close']
    
        # Calculate the spread
        spread = asset1 - asset2
    
        # Calculate z-score of the spread
        spread_mean = spread.rolling(window=self.parameters.get('lookback_period', 20)).mean()
        spread_std = spread.rolling(window=self.parameters.get('lookback_period', 20)).std()
        z_score = (spread.iloc[-1] - spread_mean.iloc[-1]) / spread_std.iloc[-1]
    
        # Generate signal based on z-score
        threshold = self.parameters.get('z_score_threshold', 2)
        if z_score > threshold:
            return -1  # Sell asset1, buy asset2
        elif z_score < -threshold:
            return 1  # Buy asset1, sell asset2
        return 0

    def _sentiment_analysis_signal(self, market_data: pd.DataFrame) -> float:
        # Assuming we have a sentiment score in the market_data
        sentiment_score = market_data['sentiment_score'].iloc[-1]
    
        positive_threshold = self.parameters.get('positive_sentiment_threshold', 0.6)
        negative_threshold = self.parameters.get('negative_sentiment_threshold', 0.4)
    
        if sentiment_score > positive_threshold:
            return 1  # Bullish signal
        elif sentiment_score < negative_threshold:
            return -1  # Bearish signal
        return 0



    def calculate_performance(self, trades: List[Dict], initial_capital: float) -> Dict[str, float]:
        if not trades:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'average_trade_return': 0.0,
                'volatility': 0.0,
                'calmar_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0
            }

        # Calculate daily returns
        daily_returns = self._calculate_daily_returns(trades, initial_capital)

        # Calculate various metrics
        total_return = (daily_returns[-1] - initial_capital) / initial_capital
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        max_drawdown = self._calculate_max_drawdown(daily_returns)
        win_rate, profit_factor = self._calculate_win_rate_and_profit_factor(trades)
        average_trade_return = np.mean([trade['return'] for trade in trades])
        volatility = np.std(daily_returns)
        calmar_ratio = self._calculate_calmar_ratio(total_return, max_drawdown)
        sortino_ratio = self._calculate_sortino_ratio(daily_returns)
        max_consecutive_wins, max_consecutive_losses = self._calculate_consecutive_wins_losses(trades)

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'average_trade_return': average_trade_return,
            'volatility': volatility,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses
        }

    def _calculate_daily_returns(self, trades: List[Dict], initial_capital: float) -> np.ndarray:
        daily_values = [initial_capital]
        current_value = initial_capital
        for trade in trades:
            current_value += trade['profit']
            daily_values.append(current_value)
        return np.array(daily_values)

    def _calculate_sharpe_ratio(self, daily_returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        returns = np.diff(daily_returns) / daily_returns[:-1]
        excess_returns = returns - risk_free_rate / 252  # Assuming 252 trading days in a year
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

    def _calculate_max_drawdown(self, daily_returns: np.ndarray) -> float:
        peak = np.maximum.accumulate(daily_returns)
        drawdown = (peak - daily_returns) / peak
        return np.max(drawdown)

    def _calculate_win_rate_and_profit_factor(self, trades: List[Dict]) -> Tuple[float, float]:
        wins = sum(1 for trade in trades if trade['profit'] > 0)
        losses = sum(1 for trade in trades if trade['profit'] < 0)
        total_profit = sum(trade['profit'] for trade in trades if trade['profit'] > 0)
        total_loss = abs(sum(trade['profit'] for trade in trades if trade['profit'] < 0))
        
        win_rate = wins / len(trades) if trades else 0
        profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')
        
        return win_rate, profit_factor

    def _calculate_calmar_ratio(self, total_return: float, max_drawdown: float) -> float:
        return total_return / max_drawdown if max_drawdown != 0 else 0

    def _calculate_sortino_ratio(self, daily_returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        returns = np.diff(daily_returns) / daily_returns[:-1]
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.sqrt(np.mean(downside_returns**2))
        return np.sqrt(252) * np.mean(excess_returns) / downside_deviation if downside_deviation != 0 else 0

    def _calculate_consecutive_wins_losses(self, trades: List[Dict]) -> Tuple[int, int]:
        streaks = [len(list(group)) for key, group in itertools.groupby(trades, key=lambda x: x['profit'] > 0)]
        max_wins = max(streaks[::2]) if len(streaks) > 0 else 0
        max_losses = max(streaks[1::2]) if len(streaks) > 1 else 0
        return max_wins, max_losses
    
    def update_performance(self, new_performance: Dict[str, float]):
        self.performance.update(new_performance)

class TrendFollowingStrategy(Strategy):
    def __init__(self, config: Config, timestamp: float, time_frame: TimeFrame, parameters: Dict[str, Any]):
        super().__init__("Trend Following", "A strategy that follows market trends", parameters, ["trend_following"], time_frame)

class MeanReversionStrategy(Strategy):
    def __init__(self, config: Config, timestamp: float, time_frame: TimeFrame, parameters: Dict[str, Any]):
        super().__init__("Mean Reversion", "A strategy that assumes prices will revert to the mean", parameters, ["mean_reversion"], time_frame)

class MomentumStrategy(Strategy):
    def __init__(self, config: Config, timestamp: float, time_frame: TimeFrame, parameters: Dict[str, Any]):
        super().__init__("Momentum", "A strategy that follows price momentum", parameters, ["momentum"], time_frame)

class VolatilityStrategy(Strategy):
    def __init__(self, config: Config, timestamp: float, time_frame: TimeFrame, parameters: Dict[str, Any]):
        super().__init__("Volatility", "A strategy that trades based on market volatility", parameters, ["volatility_clustering"], time_frame)

class PatternRecognitionStrategy(Strategy):
    def __init__(self, config: Config, timestamp: float, time_frame: TimeFrame, parameters: Dict[str, Any]):
        super().__init__("Pattern Recognition", "A strategy that recognizes and trades on specific price patterns", parameters, ["breakout"], time_frame)

class StatisticalArbitrageStrategy(Strategy):
    def __init__(self, config: Config, timestamp: float, time_frame: TimeFrame, parameters: Dict[str, Any]):
        super().__init__("Statistical Arbitrage", "A strategy that exploits price differences between related assets", parameters, ["statistical_arbitrage"], time_frame)

class SentimentAnalysisStrategy(Strategy):
    def __init__(self, config: Config, timestamp: float, time_frame: TimeFrame, parameters: Dict[str, Any]):
        super().__init__("Sentiment Analysis", "A strategy that trades based on market sentiment", parameters, ["sentiment_analysis"], time_frame)
