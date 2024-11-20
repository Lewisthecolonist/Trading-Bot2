from typing import Dict, List, Tuple
from strategy import Strategy, TimeFrame
import pandas as pd
import numpy as np
from strategy_selector import StrategySelector
from datetime import datetime, timedelta
from strategy_generator import StrategyGenerator
from api_call_manager import APICallManager
import asyncio
import logging
from strategy import Strategy
from risk_manager import RiskManager

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

class StrategyManager:
    def __init__(self, config, use_ai_selection):
        self.strategies = {tf: {} for tf in TimeFrame}
        self.active_strategies = {}
        self.strategy_selector = StrategySelector(config)
        self.config = config
        self.protection_period = timedelta(hours=1)
        self.api_call_manager = APICallManager()
        self.logger = logging.getLogger(__name__)
        self.current_timestamp = datetime.now().timestamp()
        self.average_market_volatility = 0.0
        self.risk_manager = RiskManager(config)
        self.use_ai_selection = use_ai_selection
        self.last_ai_selection_time = datetime.now()
    
        # Add proper timeframe mapping
        self.timeframe_mapping = {
            TimeFrame.SHORT_TERM: min,   # Minutes
            TimeFrame.MID_TERM: 'D',       # Daily
            TimeFrame.LONG_TERM: 'ME',      # Monthly
            TimeFrame.SEASONAL_TERM: 'A'    # Annual
        }

    async def initialize_strategies(self, strategy_generator: StrategyGenerator, market_data: pd.DataFrame):
        if await self.api_call_manager.can_make_call():
            try:
                # Initialize with list-based storage instead of dictionaries
                self.strategies = {tf: [] for tf in TimeFrame}
        
                # Generate strategies
                generated = await strategy_generator.generate_strategies(market_data)
        
                # Process each timeframe with list storage
                for time_frame in TimeFrame:
                    if time_frame in generated:
                        strategy_list = generated[time_frame]
                        self.strategies[time_frame] = strategy_list
                
                        # Find best performing strategy
                        if strategy_list:
                            best_strat = max(strategy_list, key=lambda s: getattr(s, 'performance', {}).get('total_return', 0))
                            self.active_strategies[time_frame] = best_strat
        
                self.logger.info("Strategies initialized successfully")
        
            except Exception as e:
                self.logger.error(f"Error initializing strategies: {str(e)}")
                self.strategies = {tf: [] for tf in TimeFrame}

    def add_strategy(self, strategy: Strategy):
        try:
            # Generate unique name if duplicate exists
            base_name = strategy.name
            counter = 1
            while any(s.name == strategy.name for s in self.get_all_strategies()):
                strategy.name = f"{base_name}_{counter}"
                counter += 1

            # Check for duplicate parameter sets
            for existing_strategy in self.get_all_strategies():
                if (existing_strategy.favored_patterns == strategy.favored_patterns and
                    self._compare_parameters(existing_strategy.parameters, strategy.parameters)):
                    self.logger.info(f"Duplicate strategy parameters found, skipping: {strategy.name}")
                    return

            # Validate strategy type
            if not strategy.favored_patterns or strategy.favored_patterns[0] not in VALID_STRATEGY_PARAMETERS:
                self.logger.error(f"Invalid strategy type: {strategy.favored_patterns}")
                return

             # Store strategy using string name as key
            strategy_name = str(strategy.name)
            strategy.protected_until = datetime.now() + self.protection_period
            self.strategies[strategy.time_frame][strategy_name] = strategy
            self.logger.info(f"Successfully added strategy: {strategy_name}")

        except Exception as e:
            self.logger.error(f"Error adding strategy: {str(e)}")

    def _compare_parameters(self, params1: Dict, params2: Dict) -> bool:
        """Compare parameter sets for equality using hashable types"""
        return frozenset(sorted(params1.items())) == frozenset(sorted(params2.items()))

    def get_active_strategy(self, time_frame: TimeFrame) -> Strategy:
        # Create a hashable key using frozenset
        if time_frame in self.active_strategies:
            params = self.active_strategies[time_frame].get_parameters()
            strategy_key = frozenset((k, str(v)) for k, v in sorted(params.items()))
            return self.strategies[time_frame].get(strategy_key)
        return None

    def remove_strategy(self, strategy: Strategy):
        try:
            if (strategy not in self.active_strategies.values() and 
                strategy.protected_until and datetime.now() > strategy.protected_until):
                if strategy.name in self.strategies[strategy.time_frame]:
                    del self.strategies[strategy.time_frame][strategy.name]
        except Exception as e:
            self.logger.error(f"Error removing strategy: {str(e)}")

    def update_strategy_protection(self):
        current_time = datetime.now()
        for time_frame in TimeFrame:
            for strategy in self.strategies[time_frame].values():
                if hasattr(strategy, 'protected_until') and strategy.protected_until and current_time > strategy.protected_until:
                    strategy.protected_until = None

    async def update_strategies(self, market_data: pd.DataFrame, time_frame: TimeFrame, strategy_generator: StrategyGenerator):
        if await self.api_call_manager.can_make_call():
            try:
                # Update existing strategies
                strategies_list = self.get_strategies_by_timeframe(time_frame)
                for strategy in strategies_list:
                    self.update_strategy_performance(strategy, strategy.calculate_performance(market_data))

                # Select new strategies
                selected_strategies = await self.select_strategies(market_data)

                # Update active strategies
                for tf in TimeFrame:
                    if tf in selected_strategies:
                        self.set_active_strategy(tf, selected_strategies[tf])

                await self.api_call_manager.record_call()
            except Exception as e:
                self.logger.error(f"Error updating strategies: {str(e)}")

    def set_active_strategy(self, time_frame: TimeFrame, strategy: Strategy):
        try:
            if not isinstance(strategy, Strategy):
                raise TypeError("Invalid strategy type")
            if strategy.time_frame != time_frame:
                raise ValueError("Strategy time frame doesn't match the specified time frame")
            if strategy in self.strategies[time_frame].values():
                self.active_strategies[time_frame] = strategy
                strategy.protected_until = datetime.now() + self.protection_period
        except Exception as e:
            self.logger.error(f"Error setting active strategy: {str(e)}")


    def get_all_strategies(self) -> List[Strategy]:
        return [strategy for strategies in self.strategies.values() for strategy in strategies.values()]

    def get_strategies_by_timeframe(self, time_frame: TimeFrame) -> List[Strategy]:
        return list(self.strategies[time_frame].values())

    def update_strategy_performance(self, strategy: Strategy, performance_metrics: Dict):
        try:
            if strategy in self.get_all_strategies():
                strategy.update_performance(performance_metrics)
        except Exception as e:
            self.logger.error(f"Error updating strategy performance: {str(e)}")

    def calculate_strategy_weight(self, strategy: Strategy) -> float:
        try:
            performance = strategy.performance
        
            factor_weights = {
                'total_return': 0.2,
                'sharpe_ratio': 0.15,
                'max_drawdown': 0.1,
                'win_rate': 0.2,
                'profit_factor': 0.15,
                'calmar_ratio': 0.1,
                'recent_performance': 0.1
            }
        
            weighted_score = sum(
                factor_weights[factor] * performance.get(factor, 0)
                for factor in factor_weights
            )
        
            strategy_age = (self.current_timestamp - getattr(strategy, 'creation_timestamp', self.current_timestamp)) / (24 * 60 * 60)
        
            novelty_bonus = max(0, 0.2 - (strategy_age / 30) * 0.2) if strategy_age <= 30 else 0
        
            if strategy_age > 40:
                age_bonus = min(0.3, (strategy_age - 40) / 30 * 0.3)
            else:
                age_bonus = 0
        
            volatility_factor = 1 + (performance.get('volatility', 0) - self.average_market_volatility) / max(self.average_market_volatility, 0.0001)
        
            final_weight = (weighted_score + novelty_bonus + age_bonus) * volatility_factor
        
            return max(0, final_weight)
        except Exception as e:
            self.logger.error(f"Error calculating strategy weight: {str(e)}")
    
    async def select_strategies(self, market_data: pd.DataFrame, time_frame: TimeFrame = None) -> Dict[TimeFrame, Strategy]:
        current_time = datetime.now()
    
        # For short-term strategies, always use direct selection without AI
        if time_frame and time_frame == TimeFrame.SHORT_TERM:
            short_term_strategies = self.select_best_strategies(market_data)
            return {TimeFrame.SHORT_TERM: short_term_strategies.get(TimeFrame.SHORT_TERM)}
        
        # For real-time trading with AI selection enabled
        if self.use_ai_selection:
            # Check if it's time for AI selection (24-hour interval)
            if current_time - self.last_ai_selection_time >= self.ai_selection_interval:
                # Use AI to pre-filter strategies
                candidate_strategies = await self.strategy_selector.select_candidate_strategies(
                    self.strategies,
                    market_data
                )
                # Select best strategies from AI-filtered candidates
                selected_strategies = self.select_best_strategies(market_data, candidate_strategies)
                self.last_ai_selection_time = current_time
                return selected_strategies
            else:
                # Use existing selection until next AI update
                return self.get_active_strategies()
    
        # For backtesting or when AI selection is disabled
        return self.select_best_strategies(market_data)
    def select_best_strategies(self, market_data: pd.DataFrame, candidate_strategies=None) -> Dict[TimeFrame, Strategy]:
        # Convert strategies to list format while preserving structure
        strategies_to_evaluate = {}
        source_strategies = candidate_strategies if candidate_strategies is not None else self.strategies
    
        for timeframe in TimeFrame:
            strats = source_strategies.get(timeframe, {})
            if isinstance(strats, dict):
                strategies_to_evaluate[timeframe] = list(strats.values())
            else:
                strategies_to_evaluate[timeframe] = list(strats)

        strategy_scores = []
    
        # Market Analysis
        volatility = self.calculate_market_volatility(market_data)
        trend_strength = self.calculate_trend_strength(market_data)
        volume_profile = self.analyze_volume_profile(market_data)
        market_efficiency = self.calculate_market_efficiency_ratio(market_data)
        support_resistance = self.identify_support_resistance_levels(market_data)
        liquidity_score = self.calculate_liquidity_score(market_data)

        for time_frame in TimeFrame:
            momentum_metrics = self.calculate_timeframe_momentum(market_data, time_frame)
        
            for strategy in strategies_to_evaluate.get(time_frame, []):
                # Performance Metrics
                performance_metrics = {
                    'sharpe': self.calculate_sharpe_ratio(strategy, market_data),
                    'sortino': self.calculate_sortino_ratio(strategy, market_data),
                    'calmar': self.calculate_calmar_ratio(strategy, market_data),
                    'omega': self.calculate_omega_ratio(strategy, market_data),
                    'recovery': self.calculate_recovery_factor(strategy, market_data)
                }

                # Risk Metrics
                risk_metrics = {
                    'var': self.calculate_value_at_risk(strategy, market_data),
                    'max_drawdown': self.calculate_max_drawdown(strategy, market_data),
                    'tail_risk': self.calculate_tail_risk(strategy, market_data)
                }

                # Calculate all scores
                momentum_score = self.calculate_momentum_alignment(
                    strategy,
                    momentum_metrics['composite_momentum'],
                    momentum_metrics['breakout_probability'],
                    momentum_metrics['momentum_persistence']
                )

                market_alignment = self.calculate_market_condition_alignment(
                    strategy,
                    volatility,
                    trend_strength,
                    market_efficiency,
                    momentum_metrics,
                    support_resistance
                )

                type_specific_score = self.calculate_type_specific_score(
                    strategy.favored_patterns[0],
                    market_data,
                    volatility,
                    trend_strength,
                    volume_profile,
                    momentum_metrics
                )

                stability_score = self.calculate_performance_stability(strategy)

                execution_score = self.calculate_execution_quality_score(
                    strategy,
                    liquidity_score,
                    volume_profile,
                    market_data
                )

                weights = self.calculate_dynamic_weights(
                    momentum_metrics,
                    market_efficiency,
                    volatility,
                    strategy.favored_patterns[0]
                )

                # Final Score Calculation
                total_score = (
                    performance_metrics['sharpe'] * weights['performance'] +
                    performance_metrics['sortino'] * weights['performance'] * 0.5 +
                    performance_metrics['calmar'] * weights['performance'] * 0.3 +
                    performance_metrics['omega'] * weights['performance'] * 0.2 +
                    (1 - risk_metrics['var']) * weights['risk'] +
                    (1 - risk_metrics['max_drawdown']) * weights['risk'] * 0.7 +
                    (1 - risk_metrics['tail_risk']) * weights['risk'] * 0.3 +
                    momentum_score * weights['momentum'] +
                    market_alignment * weights['market_alignment'] +
                    type_specific_score * weights['strategy_specific'] +
                    stability_score * weights['stability'] +
                    execution_score * weights['execution']
                )

                strategy_scores.append((time_frame, strategy, total_score))

        # Select best strategies
        selected_strategies = {}
        for time_frame in TimeFrame:
            timeframe_scores = [(s, score) for (tf, s, score) in strategy_scores if tf == time_frame]
            if timeframe_scores:
                best_strategy = max(timeframe_scores, key=lambda x: x[1])[0]
                selected_strategies[time_frame] = best_strategy

        return selected_strategies
    def calculate_market_volatility(self, market_data: pd.DataFrame) -> float:
        returns = market_data['close'].pct_change().dropna()
        return returns.std() * np.sqrt(252)

    def calculate_trend_strength(self, market_data: pd.DataFrame) -> float:
        ma_short = market_data['close'].rolling(20).mean()
        ma_long = market_data['close'].rolling(50).mean()
        return (ma_short.iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1]

    def identify_market_regime(self, market_data: pd.DataFrame) -> str:
        volatility = self.calculate_market_volatility(market_data)
        trend = self.calculate_trend_strength(market_data)
    
        if volatility > self.config.ADAPTIVE_PARAMS['HIGH_VOLATILITY_THRESHOLD']:
            return 'high_volatility'
        elif abs(trend) > self.config.ADAPTIVE_PARAMS['TREND_STRENGTH_THRESHOLD']:
            return 'trending'
        else:
            return 'range_bound'

    def calculate_liquidity_score(self, market_data: pd.DataFrame) -> float:
        volume_ma = market_data['volume'].rolling(20).mean()
        spread = (market_data['high'] - market_data['low']) / market_data['close']
        return (volume_ma.iloc[-1] / volume_ma.mean()) * (1 - spread.mean())

    def analyze_market_sentiment(self, market_data: pd.DataFrame) -> float:
        if 'sentiment_score' in market_data.columns:
            return market_data['sentiment_score'].iloc[-1]
        return 0.5  # Neutral sentiment if no data available

    def calculate_performance_score(self, strategy: Strategy) -> float:
        perf = strategy.performance
        return (
            perf.get('total_return', 0) * 0.4 +
            perf.get('sharpe_ratio', 0) * 0.3 +
            (1 - abs(perf.get('max_drawdown', 1))) * 0.3
        )

    def calculate_risk_adjusted_score(self, strategy: Strategy) -> float:
        perf = strategy.performance
        return (
            perf.get('sharpe_ratio', 0) * 0.4 +
            perf.get('sortino_ratio', 0) * 0.3 +
            perf.get('calmar_ratio', 0) * 0.3
        )

    def calculate_consistency_score(self, strategy: Strategy) -> float:
        perf = strategy.performance
        return (
            perf.get('win_rate', 0) * 0.5 +
            perf.get('profit_factor', 1) * 0.5
        )

    def calculate_regime_compatibility(self, strategy: Strategy, market_regime: str) -> float:
        strategy_type = strategy.favored_patterns[0]
        regime_compatibility = {
            'trend_following': {'trending': 1.0, 'range_bound': 0.3, 'high_volatility': 0.5},
            'mean_reversion': {'trending': 0.3, 'range_bound': 1.0, 'high_volatility': 0.4},
            'momentum': {'trending': 0.8, 'range_bound': 0.4, 'high_volatility': 0.6},
            'volatility_clustering': {'trending': 0.5, 'range_bound': 0.6, 'high_volatility': 1.0}
        }
        return regime_compatibility.get(strategy_type, {}).get(market_regime, 0.5)

    def calculate_adaptation_score(self, strategy: Strategy) -> float:
        if not hasattr(strategy, 'performance_history'):
            return 0.5
    
        recent_performance = strategy.performance_history[-5:]
        improvement_rate = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
        return (1 + np.tanh(improvement_rate)) / 2

    def calculate_novelty_score(self, strategy: Strategy) -> float:
        strategy_age = (datetime.now() - strategy.creation_time).days
        return np.exp(-strategy_age / 30)  # Exponential decay over 30 days
    
    def get_weighted_signal(self, time_frame: TimeFrame, market_data: pd.DataFrame) -> float:
        try:
            active_strategies = self.select_best_strategies(time_frame, market_data)
            if not active_strategies:
                return 0
            total_weight = sum(strategy.weight for strategy in active_strategies)
            weighted_signal = sum(strategy.generate_signal(market_data) * strategy.weight) 
            for strategy in active_strategies:
                return weighted_signal / total_weight if total_weight > 0 else 0
        except Exception as e:
            self.logger.error(f"Error calculating weighted signal: {str(e)}")
            return 0.0
    
    def update_strategy_weights(self):
        try:
            for time_frame in TimeFrame:
                strategies = self.get_strategies_by_timeframe(time_frame)
                for strategy in strategies:
                    strategy.weight = self.calculate_strategy_weight(strategy)
        except Exception as e:
            self.logger.error(f"Error updating strategy weights: {str(e)}")

    def calculate_momentum_metrics(self, market_data: pd.DataFrame) -> Dict[str, float]:
        # Momentum strength calculation
        roc = market_data['close'].pct_change(periods=14)
        macd = self.calculate_macd(market_data['close'])
        rsi = self.calculate_rsi(market_data['close'])
    
        # Momentum duration estimation
        momentum_duration = self.estimate_momentum_duration(
            roc,
            macd,
            market_data['volume']
        )
    
        # Price deviation and breakout analysis
        atr = self.calculate_atr(market_data)
        bollinger_bands = self.calculate_bollinger_bands(market_data['close'])
        price_deviation = (market_data['close'] - bollinger_bands['middle']) / bollinger_bands['std']
    
        # Volume-weighted momentum
        volume_momentum = self.calculate_volume_momentum(
            market_data['close'],
            market_data['volume']
        )
    
        # Momentum persistence score
        persistence = self.calculate_momentum_persistence(
            roc,
            volume_momentum,
            price_deviation
        )
    
        return {
            'strength': self.normalize_momentum_strength(roc, macd, rsi),
            'duration': momentum_duration,
            'current_momentum': volume_momentum.iloc[-1],
            'price_deviation': price_deviation.iloc[-1],
            'breakout_probability': self.calculate_breakout_probability(
                price_deviation,
                volume_momentum,
                atr
            ),
            'persistence': persistence
        }

    def calculate_breakout_probability(self, price_deviation: pd.Series, volume_momentum: pd.Series, atr: pd.Series) -> float:
        # Combine price action, volume, and volatility for breakout analysis
        price_pressure = price_deviation.rolling(5).mean()
        volume_pressure = volume_momentum.rolling(5).mean()
        volatility_factor = atr.rolling(14).mean() / atr.rolling(14).std()
    
        breakout_score = (
            price_pressure * 0.4 +
            volume_pressure * 0.4 +
            volatility_factor * 0.2
        ).iloc[-1]
    
        return np.tanh(breakout_score)  # Normalize to [-1, 1]

    def estimate_momentum_duration(self, roc: pd.Series, macd: pd.Series, volume: pd.Series) -> float:
        # Estimate how long current momentum might persist
        momentum_strength = roc.rolling(14).mean()
        volume_trend = volume.rolling(14).mean() / volume.rolling(50).mean()
        macd_strength = macd['macd'] / macd['signal']
    
        duration_factor = (
            momentum_strength.abs() * 0.4 +
            volume_trend * 0.3 +
            macd_strength.abs() * 0.3
        ).iloc[-1]
    
        return duration_factor * 14  # Scale to approximate number of periods

    def calculate_momentum_persistence(self, roc: pd.Series, volume_momentum: pd.Series, price_deviation: pd.Series) -> float:
        # Calculate how likely momentum is to persist
        momentum_consistency = roc.rolling(14).std()
        volume_consistency = volume_momentum.rolling(14).std()
        price_resistance = price_deviation.abs().rolling(14).max()

        persistence_score = (
            (1 / momentum_consistency) * 0.4 +
            (1 / volume_consistency) * 0.3 +
            (1 / price_resistance) * 0.3
        ).iloc[-1]
    
        return np.clip(persistence_score, 0, 1)

    def calculate_momentum_alignment(self, strategy: Strategy, composite_momentum: float, breakout_probability: float, momentum_persistence: float) -> float:
        strategy_type = strategy.favored_patterns[0]
    
        # Different strategy types have different optimal momentum conditions
        momentum_alignment = {
            'trend_following': composite_momentum * breakout_probability,
            'momentum': composite_momentum * momentum_persistence,
            'mean_reversion': (1 - abs(composite_momentum)) * (1 - breakout_probability),
            'breakout': breakout_probability * momentum_persistence,
            'volatility_clustering': abs(composite_momentum) * (1 - momentum_persistence)
        }
        return momentum_alignment.get(strategy_type, 0.5)
    def calculate_timeframe_momentum(self, market_data: pd.DataFrame, time_frame: TimeFrame) -> Dict[str, float]:
        # Define lookback periods based on timeframe
        lookback_periods = {
            TimeFrame.SHORT_TERM: {'recent': 60, 'medium': 240, 'long': 480},      # Minutes
            TimeFrame.MID_TERM: {'recent': 7, 'medium': 14, 'long': 21},           # Days
            TimeFrame.LONG_TERM: {'recent': 30, 'medium': 180, 'long': 365},       # Days
            TimeFrame.SEASONAL_TERM: {'recent': 365, 'medium': 730, 'long': 1095}  # Days
        }

        # Adjust data sampling frequency based on timeframe
        data_sampling = {
            TimeFrame.SHORT_TERM: 'min',
            TimeFrame.MID_TERM: 'h',
            TimeFrame.LONG_TERM: 'W',
            TimeFrame.SEASONAL_TERM: 'ME'
        }

        periods = lookback_periods[time_frame]
    
        # Resample data according to timeframe
        resampled_data = market_data.resample(data_sampling[time_frame]).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        # Calculate momentum metrics for different timeframes
        recent_momentum = self.calculate_momentum_metrics(resampled_data.tail(periods['recent']))
        medium_momentum = self.calculate_momentum_metrics(resampled_data.tail(periods['medium']))
        long_momentum = self.calculate_momentum_metrics(resampled_data.tail(periods['long']))

        # Weighted momentum score based on timeframe
        momentum_weights = {
            TimeFrame.SHORT_TERM: {'recent': 0.6, 'medium': 0.3, 'long': 0.1},
            TimeFrame.MID_TERM: {'recent': 0.4, 'medium': 0.4, 'long': 0.2},
            TimeFrame.LONG_TERM: {'recent': 0.2, 'medium': 0.3, 'long': 0.5},
            TimeFrame.SEASONAL_TERM: {'recent': 0.3, 'medium': 0.3, 'long': 0.4}
        }

        weights = momentum_weights[time_frame]

        return {
            'composite_momentum': (
                recent_momentum['strength'] * weights['recent'] +
                medium_momentum['strength'] * weights['medium'] +
                long_momentum['strength'] * weights['long']
            ),
            'recent_momentum': recent_momentum,
            'medium_momentum': medium_momentum,
            'long_momentum': long_momentum,
            'breakout_probability': max(
                recent_momentum['breakout_probability'],
                medium_momentum['breakout_probability']
            ),
            'momentum_persistence': (
                recent_momentum['persistence'] * weights['recent'] +
                medium_momentum['persistence'] * weights['medium'] +
                long_momentum['persistence'] * weights['long']
            )
        }

    def analyze_volume_profile(self, market_data: pd.DataFrame) -> Dict[str, float]:
        volume = market_data['volume']
        price = market_data['close']
    
        # Calculate volume-weighted metrics
        vwap = (price * volume).cumsum() / volume.cumsum()
        volume_ma = volume.rolling(window=20).mean()
        relative_volume = volume / volume_ma
    
        # Identify key volume levels
        high_volume_threshold = volume_ma.mean() * 1.5
        low_volume_threshold = volume_ma.mean() * 0.5
    
        return {
            'vwap': vwap.iloc[-1],
            'relative_volume': relative_volume.iloc[-1],
            'high_volume_zones': (volume > high_volume_threshold).sum() / len(volume),
            'low_volume_zones': (volume < low_volume_threshold).sum() / len(volume),
            'volume_trend': (volume_ma.iloc[-1] / volume_ma.iloc[0]) - 1
        }
    def calculate_market_efficiency_ratio(self, market_data: pd.DataFrame) -> float:
        price = market_data['close']
        directional_movement = abs(price.iloc[-1] - price.iloc[0])
        path_movement = abs(price.diff()).sum()
        efficiency_ratio = directional_movement / path_movement if path_movement != 0 else 0
        return efficiency_ratio
    
    def identify_support_resistance_levels(self, market_data: pd.DataFrame) -> Dict[str, List[float]]:
        prices = market_data['close']
        highs = market_data['high']
        lows = market_data['low']
        volumes = market_data['volume']
    
        # Calculate price clusters
        price_clusters = pd.concat([highs, lows])
        hist, bins = np.histogram(price_clusters, bins=50)
    
        # Identify high-volume price levels
        volume_profile = pd.DataFrame({
            'price': (bins[:-1] + bins[1:]) / 2,
            'volume': hist
        })
    
        # Calculate moving averages for trend identification
        ma20 = prices.rolling(window=20).mean()
        ma50 = prices.rolling(window=50).mean()
    
        # Find support levels (price levels with high volume and bounces)
        support_levels = []
        for idx in range(1, len(prices)-1):
            if (lows.iloc[idx] < lows.iloc[idx-1] and 
                lows.iloc[idx] < lows.iloc[idx+1] and 
                volumes.iloc[idx] > volumes.mean()):
                support_levels.append(lows.iloc[idx])
    
        # Find resistance levels (price levels with high volume and rejections)
        resistance_levels = []
        for idx in range(1, len(prices)-1):
            if (highs.iloc[idx] > highs.iloc[idx-1] and 
                highs.iloc[idx] > highs.iloc[idx+1] and 
                volumes.iloc[idx] > volumes.mean()):
                resistance_levels.append(highs.iloc[idx])
    
        # Filter and consolidate levels
        support_levels = self._consolidate_levels(support_levels)
        resistance_levels = self._consolidate_levels(resistance_levels)
    
        return {
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'current_trend': 'uptrend' if ma20.iloc[-1] > ma50.iloc[-1] else 'downtrend',
            'strength_score': self._calculate_level_strength(support_levels, resistance_levels, prices.iloc[-1])
        }

    def _consolidate_levels(self, levels: List[float], tolerance: float = 0.02) -> List[float]:
        if not levels:
            return []
    
        levels = sorted(levels)
        consolidated = []
        current_group = [levels[0]]
    
        for level in levels[1:]:
            if abs(level - current_group[0]) / current_group[0] <= tolerance:
                current_group.append(level)
            else:
                consolidated.append(sum(current_group) / len(current_group))
                current_group = [level]
    
        consolidated.append(sum(current_group) / len(current_group))
        return consolidated

    def _calculate_level_strength(self, support_levels: List[float], resistance_levels: List[float], current_price: float) -> float:
        closest_support = min((abs(level - current_price), level) for level in support_levels)[1] if support_levels else current_price
        closest_resistance = min((abs(level - current_price), level) for level in resistance_levels)[1] if resistance_levels else current_price

        support_distance = abs(current_price - closest_support) / current_price
        resistance_distance = abs(current_price - closest_resistance) / current_price
    
        return 1 - min(support_distance, resistance_distance)

    def calculate_sharpe_ratio(self, strategy: Strategy, market_data: pd.DataFrame) -> float:
        returns = market_data['close'].pct_change().dropna()
        excess_returns = returns - 0.02/252  # Risk-free rate assumed 2% annually
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    def calculate_sortino_ratio(self, strategy: Strategy, market_data: pd.DataFrame) -> float:
        returns = market_data['close'].pct_change().dropna()
        excess_returns = returns - 0.02/252
        downside_returns = excess_returns[excess_returns < 0]
        return excess_returns.mean() / downside_returns.std() * np.sqrt(252)

    def calculate_calmar_ratio(self, strategy: Strategy, market_data: pd.DataFrame) -> float:
        returns = market_data['close'].pct_change().dropna()
        max_drawdown = self.calculate_max_drawdown(strategy, market_data)
        annual_return = returns.mean() * 252
        return annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    def calculate_omega_ratio(self, strategy: Strategy, market_data: pd.DataFrame) -> float:
        returns = market_data['close'].pct_change().dropna()
        threshold = 0
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns < threshold].sum())
        return gains / losses if losses != 0 else float('inf')

    def calculate_recovery_factor(self, strategy: Strategy, market_data: pd.DataFrame) -> float:
        returns = market_data['close'].pct_change().dropna()
        cumulative_return = (1 + returns).prod() - 1
        max_drawdown = self.calculate_max_drawdown(strategy, market_data)
        return cumulative_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')

    def calculate_value_at_risk(self, strategy: Strategy, market_data: pd.DataFrame) -> float:
        returns = market_data['close'].pct_change().dropna()
        return np.percentile(returns, 5)  # 95% VaR

    def calculate_tail_risk(self, strategy: Strategy, market_data: pd.DataFrame) -> float:
        returns = market_data['close'].pct_change().dropna()
        var_95 = np.percentile(returns, 5)
        tail_returns = returns[returns < var_95]
        return tail_returns.mean()

    def calculate_performance_stability(self, strategy: Strategy) -> float:
        if not hasattr(strategy, 'performance_history') or len(strategy.performance_history) < 2:
            return 0.5
    
        returns = pd.Series(strategy.performance_history)
        stability = 1 - returns.std() / (abs(returns.mean()) + 1e-6)
        return np.clip(stability, 0, 1)

    def calculate_execution_quality_score(self, strategy: Strategy, liquidity_score: float, 
                                   volume_profile: Dict[str, float], market_data: pd.DataFrame) -> float:
        spread = (market_data['high'] - market_data['low']) / market_data['close']
        avg_spread = spread.mean()
    
        execution_score = (
            liquidity_score * 0.4 +
            (1 - avg_spread) * 0.3 +
            volume_profile['relative_volume'] * 0.3
        )
        return np.clip(execution_score, 0, 1)

    def calculate_dynamic_weights(self, momentum_metrics: Dict[str, float], 
                            market_efficiency: float, volatility: float, 
                            strategy_type: str) -> Dict[str, float]:
        base_weights = {
            'performance': 0.25,
            'risk': 0.20,
            'momentum': 0.15,
            'market_alignment': 0.15,
            'strategy_specific': 0.10,
            'stability': 0.10,
            'execution': 0.05
        }
    
        # Adjust weights based on market conditions
        if volatility > 0.2:  # High volatility
            base_weights['risk'] += 0.05
            base_weights['performance'] -= 0.05
    
        if market_efficiency < 0.3:  # Low efficiency
            base_weights['execution'] += 0.05
            base_weights['momentum'] -= 0.05
    
        return base_weights

    def calculate_dynamic_threshold(self, strategy_scores: List[Tuple[Strategy, float]]) -> float:
        scores = [score for _, score in strategy_scores]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        return mean_score - 0.5 * std_score

    def calculate_macd(self, prices: pd.Series) -> Dict[str, pd.Series]:
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return {'macd': macd, 'signal': signal}

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_atr(self, market_data: pd.DataFrame, period: int = 14) -> pd.Series:
        high = market_data['high']
        low = market_data['low']
        close = market_data['close']
    
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def calculate_volume_momentum(self, prices: pd.Series, volume: pd.Series) -> pd.Series:
        price_change = prices.pct_change()
        volume_ratio = volume / volume.rolling(20).mean()
        return price_change * volume_ratio

    def normalize_momentum_strength(self, roc: pd.Series, macd: Dict[str, pd.Series], rsi: pd.Series) -> float:
        roc_norm = (roc.iloc[-1] - roc.mean()) / roc.std()
        macd_norm = (macd['macd'].iloc[-1] - macd['macd'].mean()) / macd['macd'].std()
        rsi_norm = (rsi.iloc[-1] - 50) / 25
    
        return np.tanh(0.4 * roc_norm + 0.3 * macd_norm + 0.3 * rsi_norm)
    
    def calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        middle_band = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
    
        return {
            'upper': middle_band + (std * num_std),
            'middle': middle_band,
            'lower': middle_band - (std * num_std),
            'std': std
        }