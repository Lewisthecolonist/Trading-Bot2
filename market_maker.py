import asyncio
import logging
from decimal import Decimal
import ccxt.async_support as ccxt
from wallet import Wallet
from order_book import OrderBook
from risk_manager import RiskManager
from inventory_manager import InventoryManager
from compliance import ComplianceChecker
from strategy_factory import StrategyFactory
from strategy_selector import StrategySelector
from strategy_generator import StrategyGenerator
from typing import Dict, Optional, Tuple
from strategy import Strategy, TimeFrame
from datetime import datetime, time, timedelta
from event import MarketEvent, SignalEvent
import pandas as pd
import json
import psutil
import aiofiles
from watchdog.observers import Observer
from strategy_manager import StrategyManager
import math

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

class MarketMakerError(Exception):
    pass

class OrderPlacementError(MarketMakerError):
    pass

class MarketMaker:
    def __init__(self, config, strategy_config_path):
        self.config = config
        self.strategy_config_path = strategy_config_path
        self.exchange = ccxt.kraken({
            'apiKey': config.BASE_PARAMS['KRAKEN_API_KEY'],
            'secret': config.BASE_PARAMS['KRAKEN_PRIVATE_KEY'],
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.wallet = None
        self.order_book = OrderBook(self.exchange, config.BASE_PARAMS['SYMBOL'])
        self.risk_manager = RiskManager(config)
        self.inventory_manager = InventoryManager(config, self.exchange)
        self.compliance_checker = ComplianceChecker(config)
        self.current_orders = {}
        self.strategy_selector = StrategySelector(config)
        self.strategy_performance = {}
        self.events = []
        self.observer = Observer()
        self.strategy_manager = StrategyManager(config, use_ai_selection=True)
        self.logger = logging.getLogger(__name__)
        self.strategy_generator = StrategyGenerator(config)
        self.timeframe_handlers = {
            TimeFrame.SHORT_TERM: self._handle_short_term,
            TimeFrame.MID_TERM: self._handle_mid_term,
            TimeFrame.LONG_TERM: self._handle_long_term,
            TimeFrame.SEASONAL_TERM: self._handle_seasonal_term
        }

    async def _handle_short_term(self, market_data):
        signal = await self.strategy_manager.get_weighted_signal(TimeFrame.SHORT_TERM, market_data)
        return self._execute_short_term_trades(signal, market_data)

    async def _handle_mid_term(self, market_data):
        signal = await self.strategy_manager.get_weighted_signal(TimeFrame.MID_TERM, market_data)
        return self._execute_mid_term_trades(signal, market_data)

    async def _handle_long_term(self, market_data):
        signal = await self.strategy_manager.get_weighted_signal(TimeFrame.LONG_TERM, market_data)
        return self._execute_long_term_trades(signal, market_data)

    async def _handle_seasonal_term(self, market_data):
        signal = await self.strategy_manager.get_weighted_signal(TimeFrame.SEASONAL_TERM, market_data)
        return self._execute_seasonal_trades(signal, market_data)

    async def log(self, message, level=logging.INFO):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.logger.log, level, message)

    async def initialize(self, market_data):
        await self.strategy_manager.initialize_strategies(self.strategy_generator, market_data)
        await self.exchange.load_markets()
        self.wallet = Wallet(self.exchange)
        await self.wallet.connect()
        self.strategy_factory = StrategyFactory(self, self.config, self.strategy_config_path)
        market_data = await self.update_market_data()
        await self.log(f"Connected to {self.exchange.name}")
        await self.log(f"Current balances: {self.wallet.balances}")

    async def run(self):
        await self.initialize()
        last_performance_update = asyncio.get_event_loop().time()
        last_rebalance = asyncio.get_event_loop().time()
        last_performance_report = asyncio.get_event_loop().time()
        last_health_check = asyncio.get_event_loop().time()
        last_backup = asyncio.get_event_loop().time()

        while True:
            try:
                current_time = asyncio.get_event_loop().time()

                # Get data for different timeframes
                short_term_data = await self.get_market_data('1m')
                mid_term_data = await self.get_market_data('1h') 
                long_term_data = await self.get_market_data('1w')
                seasonal_data = await self.get_market_data('1M')

                # Process each timeframe independently
                await self._process_timeframe(TimeFrame.SHORT_TERM, short_term_data)
                await self._process_timeframe(TimeFrame.MID_TERM, mid_term_data)
                await self._process_timeframe(TimeFrame.LONG_TERM, long_term_data)
                await self._process_timeframe(TimeFrame.SEASONAL_TERM, seasonal_data)

                # Update market data
                market_data = await self.update_market_data()

                # Generate and handle market events for each time frame
                for time_frame in TimeFrame:
                    active_strategy = self.strategy_manager.get_active_strategy(time_frame)
                    if active_strategy:
                        market_event = self.generate_market_event(time_frame, market_data)
                        await self.handle_market_event(market_event, active_strategy)

                # Process any pending events
                await self.process_events()

                # Adjust orders based on current market conditions
                await self.adjust_orders()

                # Periodic performance update and strategy selection
                if current_time - last_performance_update >= self.config.PERFORMANCE_UPDATE_INTERVAL:
                    strategy_performance = await self.update_strategy_performance()
                    await self.strategy_manager.select_strategies(market_data, strategy_performance)
                    self.strategy_manager.adjust_strategy_weights()
                    last_performance_update = current_time

                # Check and share profits at midnight
                await self.check_and_share_profits()

                # Periodic rebalancing
                if (current_time - last_rebalance) >= self.config.REBALANCE_INTERVAL:
                    await self.rebalance_inventory()
                    last_rebalance = current_time

                # Periodic performance reporting
                if (current_time - last_performance_report) >= 3600:  # Report every hour
                    await self.report_performance()
                    last_performance_report = current_time

                # Risk management
                inventory_risk = self.calculate_inventory_risk()
                if inventory_risk > self.config.MAX_INVENTORY_RISK:
                    await self.rebalance_inventory()

                # Compliance check
                await self.compliance_checker.check_compliance(self.wallet.balances, self.current_orders)

                # System health monitoring
                if (current_time - last_health_check) >= self.config.HEALTH_CHECK_INTERVAL:
                    await self.monitor_system_health()
                    last_health_check = current_time

                # Periodic data backup
                if (current_time - last_backup) >= self.config.BACKUP_INTERVAL:
                    await self.backup_data()
                    last_backup = current_time

                # Sleep for the configured interval
                await asyncio.sleep(self.config.ORDER_REFRESH_RATE)

            except Exception as e:
                retry = await self.handle_exchange_errors(e)
                if retry:
                    continue
                else:
                    await asyncio.sleep(self.config.ERROR_RETRY_INTERVAL)
    
    async def _update_trend_following_params(self, strategy: Strategy, market_data: pd.DataFrame):
        if not hasattr(strategy, 'parameters'):
            return
        
        volatility = self.calculate_current_volatility(market_data)
        trend_strength = (market_data['close'].iloc[-1] - market_data['close'].iloc[-20]) / market_data['close'].iloc[-20]
    
        trend_params = self.config.ADAPTIVE_PARAMS['TREND_FOLLOWING_PARAMS']
        for param, value in trend_params.items():
            if param in strategy.parameters:
                if param == 'MOVING_AVERAGE_SHORT':
                    strategy.parameters[param] = max(5, min(20, int(volatility * value)))
                elif param == 'MOVING_AVERAGE_LONG':
                    strategy.parameters[param] = max(20, min(100, int(volatility * value)))
                elif param == 'TREND_STRENGTH_THRESHOLD':
                    strategy.parameters[param] = max(0.01, min(0.05, value * abs(trend_strength)))
                elif param == 'TREND_CONFIRMATION_PERIOD':
                    strategy.parameters[param] = max(3, min(7, int(value * volatility)))
                elif param in ['MOMENTUM_FACTOR', 'BREAKOUT_LEVEL', 'TRAILING_STOP']:
                    strategy.parameters[param] = max(0.01, min(0.05, value * volatility))

    async def _update_mean_reversion_params(self, strategy: Strategy, market_data: pd.DataFrame):
        if not hasattr(strategy, 'parameters'):
            return

        volatility = self.calculate_current_volatility(market_data)
        mean_params = self.config.ADAPTIVE_PARAMS['MEAN_REVERSION_PARAMS']
    
        for param, value in mean_params.items():
            if param in strategy.parameters:
                if param == 'MEAN_WINDOW':
                    strategy.parameters[param] = max(10, min(30, int(value * volatility)))
                elif param == 'STD_MULTIPLIER':
                    strategy.parameters[param] = max(1.5, min(2.5, value * volatility))
                elif param in ['MEAN_REVERSION_THRESHOLD', 'ENTRY_DEVIATION', 'EXIT_DEVIATION']:
                    strategy.parameters[param] = max(0.01, min(0.1, value * volatility))
                elif param in ['BOLLINGER_PERIOD', 'BOLLINGER_STD']:
                    strategy.parameters[param] = max(10, min(30, int(value * volatility)))

    async def _update_breakout_params(self, strategy: Strategy, market_data: pd.DataFrame):
        if not hasattr(strategy, 'parameters'):
            return
        
        volatility = self.calculate_current_volatility(market_data)
        volume_trend = market_data['volume'].pct_change().mean()
        breakout_params = self.config.ADAPTIVE_PARAMS['BREAKOUT_PARAMS']
    
        for param, value in breakout_params.items():
            if param in strategy.parameters:
                if param == 'BREAKOUT_PERIOD':
                    strategy.parameters[param] = max(10, min(30, int(value * volatility)))
                elif param == 'BREAKOUT_THRESHOLD':
                    strategy.parameters[param] = max(0.01, min(0.05, value * volatility))
                elif param == 'VOLUME_CONFIRMATION_MULT':
                    strategy.parameters[param] = max(1.2, min(2.0, value * (1 + volume_trend)))
                elif param in ['CONSOLIDATION_PERIOD', 'ATR_PERIOD']:
                    strategy.parameters[param] = max(5, min(20, int(value * volatility)))

    async def _update_volatility_clustering_params(self, strategy: Strategy, market_data: pd.DataFrame):
        if not hasattr(strategy, 'parameters'):
            return
        
        current_volatility = self.calculate_current_volatility(market_data)
        vol_params = self.config.ADAPTIVE_PARAMS['VOLATILITY_CLUSTERING_PARAMS']
    
        for param, value in vol_params.items():
            if param in strategy.parameters:
                if param in ['VOLATILITY_WINDOW', 'VOLATILITY_MEAN_PERIOD']:
                    strategy.parameters[param] = max(10, min(50, int(value * current_volatility)))
                elif param in ['HIGH_VOLATILITY_THRESHOLD', 'LOW_VOLATILITY_THRESHOLD']:
                    strategy.parameters[param] = max(0.5, min(2.0, value * current_volatility))
                elif param == 'GARCH_LAG':
                    strategy.parameters[param] = max(3, min(7, int(value)))
                elif param == 'ATR_MULTIPLIER':
                    strategy.parameters[param] = max(1.5, min(3.0, value * current_volatility))

    async def _update_statistical_arbitrage_params(self, strategy: Strategy, market_data: pd.DataFrame):
        if not hasattr(strategy, 'parameters'):
            return
        
        correlation = market_data['asset1_close'].corr(market_data['asset2_close'])
        volatility = self.calculate_current_volatility(market_data)
        stat_arb_params = self.config.ADAPTIVE_PARAMS['STATISTICAL_ARBITRAGE_PARAMS']
    
        for param, value in stat_arb_params.items():
            if param in strategy.parameters:
                if param == 'LOOKBACK_PERIOD':
                    strategy.parameters[param] = max(10, min(30, int(value * volatility)))
                elif param == 'Z_SCORE_THRESHOLD':
                    strategy.parameters[param] = max(1.5, min(3.0, value * volatility))
                elif param == 'CORRELATION_THRESHOLD':
                    strategy.parameters[param] = max(0.6, min(0.9, value * correlation))
                elif param in ['ENTRY_THRESHOLD', 'EXIT_THRESHOLD']:
                    strategy.parameters[param] = max(0.5, min(2.5, value * volatility))

    async def _update_sentiment_params(self, strategy: Strategy, market_data: pd.DataFrame):
        if not hasattr(strategy, 'parameters'):
            return
        
        sentiment_volatility = market_data['sentiment_score'].std()
        volume_trend = market_data['volume'].pct_change().mean()
        sentiment_params = self.config.ADAPTIVE_PARAMS['SENTIMENT_ANALYSIS_PARAMS']
    
        for param, value in sentiment_params.items():
            if param in strategy.parameters:
                if param == 'SENTIMENT_WINDOW':
                    strategy.parameters[param] = max(12, min(36, int(value * sentiment_volatility)))
                elif param in ['POSITIVE_SENTIMENT_THRESHOLD', 'NEGATIVE_SENTIMENT_THRESHOLD']:
                    strategy.parameters[param] = max(0.3, min(0.7, value * (1 + sentiment_volatility)))
                elif param == 'SENTIMENT_IMPACT_WEIGHT':
                    strategy.parameters[param] = max(0.1, min(0.5, value * volume_trend))
                elif param == 'NEWS_IMPACT_DECAY':
                    strategy.parameters[param] = max(0.8, min(0.99, value))

    async def _update_options_params(self, strategy: Strategy, market_data: pd.DataFrame):
        implied_volatility = market_data['implied_volatility'].iloc[-1]
        delta = market_data['delta'].iloc[-1]
    
        strategy.parameters['DELTA_THRESHOLD'] = max(0.2, min(0.4, implied_volatility * 0.5))
        strategy.parameters['GAMMA_LIMIT'] = max(0.05, min(0.15, implied_volatility * 0.2))
        strategy.parameters['VEGA_EXPOSURE_LIMIT'] = max(500, min(1500, implied_volatility * 1000))

    async def _update_market_making_params(self, strategy: Strategy, market_data: pd.DataFrame):
        spread = market_data['ask'] - market_data['bid']
        volume = market_data['volume'].iloc[-1]
    
        strategy.parameters['BID_ASK_SPREAD'] = max(0.001, min(0.005, spread.mean() * 0.8))
        strategy.parameters['INVENTORY_TARGET'] = max(0.3, min(0.7, volume / market_data['volume'].mean()))
        strategy.parameters['ORDER_REFRESH_TIME'] = max(15, min(45, int(30 * spread.std() / spread.mean())))

    async def _update_grid_params(self, strategy: Strategy, market_data: pd.DataFrame):
        atr = self.calculate_atr(market_data)
        price_range = market_data['high'].max() - market_data['low'].min()
    
        strategy.parameters['GRID_LEVELS'] = max(5, min(15, int(price_range / atr)))
        strategy.parameters['GRID_SPACING'] = max(0.005, min(0.02, atr / market_data['close'].iloc[-1]))
        strategy.parameters['PROFIT_PER_GRID'] = max(0.002, min(0.008, atr / market_data['close'].iloc[-1] * 0.5))


    async def update_market_data(self):
        await self.order_book.update()
        await self.wallet.update_balances()
        self.inventory_manager.xbt_balance = self.wallet.get_balance('BTC')
        self.inventory_manager.usdt_balance = self.wallet.get_balance('USDT')
        market_data = await self.get_recent_data()
        return market_data

    def generate_market_event(self, time_frame: TimeFrame, market_data: pd.DataFrame) -> MarketEvent:
        return MarketEvent(datetime.now(), self.config.SYMBOL, time_frame, market_data)

    async def handle_market_event(self, event: MarketEvent, strategy: Strategy):
        signal = strategy.generate_signal(event.market_data)
        if signal != 0:
            self.events.append(SignalEvent(event.timestamp, self.config.SYMBOL, signal))

    async def process_events(self):
        for event in self.events:
            if isinstance(event, SignalEvent):
                await self.execute_signal(event)
        self.events.clear()

    async def execute_signal(self, signal_event: SignalEvent):
        if signal_event.signal > 0:
            await self.wallet.place_order(self.config.SYMBOL, 'market', 'buy', self.config.POSITION_SIZE)
        elif signal_event.signal < 0:
            await self.wallet.place_order(self.config.SYMBOL, 'market', 'sell', self.config.POSITION_SIZE)

    async def adjust_orders(self):
        adjusted_params = await self.adjust_order_parameters()
        if adjusted_params is None:
            await self.log("Failed to adjust order parameters. Skipping order placement.", logging.WARNING)
            return

        bid_price, ask_price, buy_amount, sell_amount = adjusted_params
        active_strategy = self.strategy_manager.get_active_strategy(TimeFrame.SHORT_TERM)

        # Apply strategy-specific adjustments using valid parameters
        if active_strategy:
            pattern = active_strategy.favored_patterns[0]
            if pattern in VALID_STRATEGY_PARAMETERS:
                valid_params = VALID_STRATEGY_PARAMETERS[pattern]
                for param in valid_params:
                    if param in active_strategy.parameters:
                        # Apply parameter adjustments based on strategy type
                        bid_price, ask_price, buy_amount, sell_amount = self._adjust_orders_by_strategy(
                            pattern, param, active_strategy.parameters[param],
                            bid_price, ask_price, buy_amount, sell_amount
                        )

        await self.cancel_existing_orders()
        await self.place_orders(bid_price, ask_price, buy_amount, sell_amount)


    async def place_orders(self, bid_price, ask_price, buy_amount, sell_amount):
        try:
            active_strategy = self.strategy_manager.get_active_strategy(TimeFrame.SHORT_TERM)
            market_data = await self.get_recent_data()
        
            # Base adjustments using market conditions
            volatility = self.calculate_current_volatility(market_data)
            volume_trend = market_data['volume'].pct_change().mean()
            price_momentum = market_data['close'].pct_change(periods=5).mean()
        
            if 'trend_following' in active_strategy.favored_patterns:
                trend_strength = active_strategy.parameters['TREND_STRENGTH_THRESHOLD']
                momentum_factor = active_strategy.parameters['MOMENTUM_FACTOR']
                breakout_level = active_strategy.parameters['BREAKOUT_LEVEL']
            
                # Enhanced trend-based adjustments
                trend_adjustment = trend_strength * (1 + momentum_factor * price_momentum)
                bid_price *= (1 - trend_adjustment)
                ask_price *= (1 + trend_adjustment)
            
                # Volume-based size adjustments
                size_multiplier = 1 + (volume_trend * breakout_level)
                buy_amount *= size_multiplier
                sell_amount *= size_multiplier
            
            elif 'statistical_arbitrage' in active_strategy.favored_patterns:
                z_score = active_strategy.parameters['Z_SCORE_THRESHOLD']
                correlation = active_strategy.parameters['CORRELATION_THRESHOLD']
                half_life = active_strategy.parameters['HALF_LIFE']
            
                # Spread adjustments based on statistical measures
                spread_volatility = market_data['spread'].std()
                spread_adjustment = z_score * spread_volatility * (1 - correlation)
                mean_reversion_factor = math.exp(-1/half_life)
            
                bid_price *= (1 - spread_adjustment * mean_reversion_factor)
                ask_price *= (1 + spread_adjustment * mean_reversion_factor)
            
            elif 'volatility_clustering' in active_strategy.favored_patterns:
                vol_threshold = active_strategy.parameters['HIGH_VOLATILITY_THRESHOLD']
                garch_lag = active_strategy.parameters['GARCH_LAG']
                atr_multiplier = active_strategy.parameters['ATR_MULTIPLIER']
            
                # Volatility-based adjustments
                vol_adjustment = vol_threshold * volatility * atr_multiplier
                recent_volatility = market_data['close'].pct_change().rolling(garch_lag).std().iloc[-1]
            
                # Adjust sizes inversely to volatility
                size_scalar = 1 / (1 + recent_volatility)
                buy_amount *= size_scalar
                sell_amount *= size_scalar
            
            elif 'sentiment_analysis' in active_strategy.favored_patterns:
                sentiment_impact = active_strategy.parameters['SENTIMENT_IMPACT_WEIGHT']
                sentiment_score = market_data['sentiment_score'].iloc[-1]
                news_decay = active_strategy.parameters['NEWS_IMPACT_DECAY']
            
                # Sentiment-based price adjustments
                sentiment_adjustment = sentiment_impact * sentiment_score * news_decay
                bid_price *= (1 + sentiment_adjustment)
                ask_price *= (1 + sentiment_adjustment)
            
            elif 'momentum' in active_strategy.favored_patterns:
                acceleration = active_strategy.parameters['ACCELERATION_FACTOR']
                max_acceleration = active_strategy.parameters['MAX_ACCELERATION']
                rsi_value = self.calculate_rsi(market_data, active_strategy.parameters['RSI_PERIOD'])
            
                # Momentum-based adjustments
                momentum_adjustment = min(acceleration * abs(price_momentum), max_acceleration)
                rsi_factor = (rsi_value - 50) / 50  # Normalize RSI impact
            
                bid_price *= (1 + momentum_adjustment * rsi_factor)
                ask_price *= (1 + momentum_adjustment * rsi_factor)
            
            elif 'market_making' in active_strategy.favored_patterns:
                spread = active_strategy.parameters['BID_ASK_SPREAD']
                inventory_target = active_strategy.parameters['INVENTORY_TARGET']
                current_inventory = self.inventory_manager.get_inventory_ratio()
            
                # Inventory-based adjustments
                inventory_skew = (current_inventory - inventory_target) * spread
                bid_price *= (1 - spread - inventory_skew)
                ask_price *= (1 + spread - inventory_skew)
            
            elif 'grid_trading' in active_strategy.favored_patterns:
                grid_spacing = active_strategy.parameters['GRID_SPACING']
                grid_levels = active_strategy.parameters['GRID_LEVELS']
                profit_per_grid = active_strategy.parameters['PROFIT_PER_GRID']
            
                # Place multiple orders at grid levels
                for i in range(grid_levels):
                    grid_bid = bid_price * (1 - i * grid_spacing)
                    grid_ask = ask_price * (1 + i * grid_spacing)
                    grid_amount = buy_amount / grid_levels * (1 + profit_per_grid * i)
                
                    await self._place_grid_orders(grid_bid, grid_ask, grid_amount, grid_amount)
                return

            # Final safety checks and limits
            bid_price, ask_price = self.apply_price_limits(bid_price, ask_price)
            buy_amount, sell_amount = self.apply_size_limits(buy_amount, sell_amount)

            # Place the final orders
            buy_order = await self.wallet.place_order(
                symbol=self.config.SYMBOL,
                order_type='limit',
                side='buy',
                amount=float(buy_amount),
                price=float(bid_price)
            )
            self.current_orders['bid'] = buy_order['id']

            sell_order = await self.wallet.place_order(
                symbol=self.config.SYMBOL,
                order_type='limit',
                side='sell',
                amount=float(sell_amount),
                price=float(ask_price)
            )
            self.current_orders['ask'] = sell_order['id']

        except Exception as e:
            await self.log(f"Error placing orders: {e}", logging.ERROR)
            raise OrderPlacementError(f"Error placing orders: {e}")

    async def cancel_existing_orders(self):
        for order_id in self.current_orders.values():
            try:
                await self.wallet.cancel_order(order_id, self.config.SYMBOL)
            except Exception as e:
                await self.log(f"Error cancelling order {order_id}: {e}", logging.ERROR)
        self.current_orders = {}

    async def rebalance_inventory(self):
        btc_price = await self.get_btc_price()
        optimal_capital = self.strategy_manager.get_optimal_capital()
        current_xbt_balance = self.wallet.get_balance('XBT')
        current_usdt_balance = self.wallet.get_balance('USDT')
        target_value = optimal_capital / 2
        target_xbt_balance = target_value / btc_price
        target_usdt_balance = target_value

        # Calculate extra capital outside the position
        extra_capital = max(0, self.wallet.get_total_balance() - optimal_capital)

        # Adjust XBT balance
        xbt_difference = target_xbt_balance - current_xbt_balance
        if xbt_difference > 0:
            if extra_capital > 0:
                xbt_to_buy = min(xbt_difference, extra_capital / btc_price)
                await self.wallet.place_order(self.config.SYMBOL, 'market', 'buy', xbt_to_buy)
                extra_capital -= xbt_to_buy * btc_price
            else:
                usdt_to_convert = min(current_usdt_balance, xbt_difference * btc_price)
                xbt_to_buy = usdt_to_convert / btc_price
                await self.wallet.place_order(self.config.SYMBOL, 'market', 'buy', xbt_to_buy)
        elif xbt_difference < 0:
            xbt_to_sell = abs(xbt_difference)
            await self.wallet.place_order(self.config.SYMBOL, 'market', 'sell', xbt_to_sell)

        # Adjust USDT balance
        usdt_difference = target_usdt_balance - current_usdt_balance
        if usdt_difference > 0:
            if extra_capital > 0:
                usdt_to_add = min(usdt_difference, extra_capital)
                await self.wallet.transfer_to_trading_account('USDT', usdt_to_add)
                extra_capital -= usdt_to_add
            else:
                xbt_to_convert = min(current_xbt_balance, usdt_difference / btc_price)
                await self.wallet.place_order(self.config.SYMBOL, 'market', 'sell', xbt_to_convert)
        elif usdt_difference < 0:
            usdt_to_remove = abs(usdt_difference)
            await self.wallet.transfer_from_trading_account('USDT', usdt_to_remove)

        await self.wallet.update_balances()
        await self.log(f"Rebalanced inventory: XBT={self.wallet.get_balance('XBT')}, USDT={self.wallet.get_balance('USDT')}", logging.INFO)

    async def get_recent_data(self, timestamp: Optional[datetime] = None) -> pd.DataFrame:
        if timestamp is None:
            timestamp = datetime.now()
        
        end_time = timestamp
        start_time = end_time - timedelta(hours=24)  # Get last 24 hours of data
        
        ohlcv = await self.exchange.fetch_ohlcv(
            self.config.BASE_PARAMS['SYMBOL'],
            timeframe='1h',
            since=int(start_time.timestamp() * 1000),
            limit=24
        )
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df

    async def calculate_performance_metrics(self):
        performance = {}
    
        # Calculate PnL
        initial_balance = self.inventory_manager.initial_xbt_balance
        current_balance = self.wallet.get_balance('XBT')
        pnl = (current_balance - initial_balance) / initial_balance
        performance['pnl'] = pnl
    
        # Calculate Sharpe Ratio
        returns = (await self.get_recent_data())['close'].pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5)  # Annualized Sharpe Ratio
        performance['sharpe_ratio'] = sharpe_ratio
    
        # Calculate maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        max_drawdown = (cumulative_returns.cummax() - cumulative_returns) / cumulative_returns.cummax()
        performance['max_drawdown'] = max_drawdown.max()
    
        # Calculate win rate
        trades = await self.wallet.get_my_trades(self.config.SYMBOL)
        winning_trades = sum(1 for trade in trades if trade['profit'] > 0)
        total_trades = len(trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        performance['win_rate'] = win_rate
    
        return performance

    async def adjust_order_parameters(self):
        try:
            ticker = await self.exchange.fetch_ticker(self.config.SYMBOL)
            current_price = Decimal(str(ticker['last']))

            bid_price = current_price * (Decimal('1') - self.config.MAX_SPREAD)
            ask_price = current_price * (Decimal('1') + self.config.MAX_SPREAD)

            bid_price = max(bid_price, current_price * (Decimal('1') - self.config.MAX_PRICE_DEVIATION))
            ask_price = min(ask_price, current_price * (Decimal('1') + self.config.MAX_PRICE_DEVIATION))

            tick_size = Decimal(str(self.exchange.markets[self.config.SYMBOL]['precision']['price']))
            bid_price = round(bid_price / tick_size) * tick_size
            ask_price = round(ask_price / tick_size) * tick_size

            base_balance, quote_balance = await self.get_available_balances()
            max_buy_amount = quote_balance / bid_price
            max_sell_amount = base_balance

            min_order_size = Decimal(str(self.exchange.markets[self.config.SYMBOL]['limits']['amount']['min']))
            max_order_size = Decimal(str(self.exchange.markets[self.config.SYMBOL]['limits']['amount']['max']))

            buy_amount = min(max(min_order_size, max_buy_amount), max_order_size)
            sell_amount = min(max(min_order_size, max_sell_amount), max_order_size)

            lot_size = Decimal(str(self.exchange.markets[self.config.SYMBOL]['precision']['amount']))
            buy_amount = round(buy_amount / lot_size) * lot_size
            sell_amount = round(sell_amount / lot_size) * lot_size

            await self.log(f"Adjusted order parameters - Bid: {bid_price}, Ask: {ask_price}, Buy Amount: {buy_amount}, Sell Amount: {sell_amount}", logging.INFO)

            return bid_price, ask_price, buy_amount, sell_amount

        except Exception as e:
            await self.log(f"Error adjusting order parameters: {e}", logging.ERROR)
            return None

    async def handle_exchange_errors(self, error):
        await self.log(f"Exchange error occurred: {error}", logging.ERROR)
        
        if isinstance(error, ccxt.NetworkError):
            await self.log("Network error detected. Waiting before retry...", logging.WARNING)
            await asyncio.sleep(self.config.NETWORK_ERROR_RETRY_WAIT)
            return True
        
        elif isinstance(error, ccxt.ExchangeError):
            if "Insufficient funds" in str(error):
                await self.log("Insufficient funds. Adjusting order sizes...", logging.WARNING)
                await self.adjust_order_sizes()
            else:
                await self.log("Unhandled exchange error. Notifying administrator...", logging.ERROR)
                await self.notify_administrator(error)
        
        elif isinstance(error, ccxt.InvalidOrder):
            await self.log("Invalid order parameters. Adjusting and retrying...", logging.WARNING)
            await self.adjust_order_parameters()
            return True
        
        else:
            await self.log("Unhandled error. Notifying administrator...", logging.ERROR)
            await self.notify_administrator(error)
        
        return False

    async def monitor_system_health(self):
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        if cpu_percent > self.config.MAX_CPU_USAGE:
            await self.log(f"High CPU usage detected: {cpu_percent}%", logging.WARNING)
        
        if memory_percent > self.config.MAX_MEMORY_USAGE:
            await self.log(f"High memory usage detected: {memory_percent}%", logging.WARNING)
        
        if disk_percent > self.config.MAX_DISK_USAGE:
            await self.log(f"High disk usage detected: {disk_percent}%", logging.WARNING)
        
        try:
            start_time = datetime.now()
            await self.exchange.fetch_ticker(self.config.SYMBOL)
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            if latency > self.config.MAX_NETWORK_LATENCY:
                await self.log(f"High network latency detected: {latency}ms", logging.WARNING)
        
        except Exception as e:
            await self.log(f"Error checking network latency: {e}", logging.ERROR)

    async def backup_data(self):
        backup_data = {
            'timestamp': datetime.now().isoformat(),
            'balances': self.wallet.balances,
            'open_orders': self.current_orders,
            'performance_metrics': await self.calculate_performance_metrics(),
            'active_strategies': self.strategy_manager.get_active_strategies_dict(),
        }
        
        filename = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            async with aiofiles.open(filename, mode='w') as f:
                await f.write(json.dumps(backup_data, indent=2))
            await self.log(f"Backup created successfully: {filename}", logging.INFO)
        except Exception as e:
            await self.log(f"Error creating backup: {e}", logging.ERROR)

    async def get_available_balances(self):
        balances = await self.wallet.get_balances()
        base_currency, quote_currency = self.config.SYMBOL.split('/')
        base_balance = Decimal(str(balances.get(base_currency, 0)))
        quote_balance = Decimal(str(balances.get(quote_currency, 0)))
        return base_balance, quote_balance

    async def get_btc_price(self):
        ticker = await self.exchange.fetch_ticker(self.config.SYMBOL)
        return Decimal(str(ticker['last']))

    async def check_and_share_profits(self):
        now = datetime.now().time()
        if now == time(hour=0, minute=0):  # At midnight
            await self.wallet.share_profits()

    async def report_performance(self):
        performance = await self.calculate_performance_metrics()
        await self.log(f"Performance metrics: {performance}", logging.INFO)

    def calculate_inventory_risk(self):
        net_position = self.inventory_manager.xbt_balance - self.inventory_manager.initial_xbt_balance
        return abs(net_position) / self.inventory_manager.initial_xbt_balance

    async def notify_administrator(self, error):
        # Implement logic to send notification to administrator (e.g., email, SMS)
        pass

    async def update_strategy_performance(self):
        performance = {}
        for time_frame in TimeFrame:
            for strategy in self.strategy_manager.get_strategies(time_frame):
                strategy_performance = await self.calculate_strategy_performance(strategy)
                self.strategy_manager.update_strategy_performance(strategy, strategy_performance)
                performance[strategy.name] = strategy_performance
        return performance

    async def calculate_strategy_performance(self, strategy: Strategy) -> Dict[str, float]:
        # Implement performance calculation logic here
        # This should return a dictionary with performance metrics
        # For example:
        return {
            'sharpe_ratio': 1.5,
            'profit_factor': 1.2,
            'win_rate': 0.6,
            # Add more metrics as needed
        }

    def set_wallet(self, wallet):
        self.wallet = wallet

    def update_strategy(self, strategy_name: str, strategy: Strategy):
        self.strategy_manager.update_strategy(strategy_name, strategy)
        self.logger.info(f"Strategy '{strategy_name}' has been updated.")

    def remove_strategy(self, strategy_name: str):
        self.strategy_manager.remove_strategy(strategy_name)
        self.logger.info(f"Strategy '{strategy_name}' has been removed.")

    def stop(self):
        if hasattr(self, 'strategy_factory'):
            self.strategy_factory.stop()
        self.observer.stop()
        self.observer.join()

    async def close(self):
        if hasattr(self, 'wallet') and self.wallet:
            await self.wallet.close()
        await self.exchange.close()

    def __del__(self):
        asyncio.create_task(self.close())
        self.stop()

    async def adjust_order_sizes(self):
        try:
            base_balance, quote_balance = await self.get_available_balances()
            ticker = await self.exchange.fetch_ticker(self.config.SYMBOL)
            current_price = Decimal(str(ticker['last']))

            # Calculate the total portfolio value in quote currency
            total_value = base_balance * current_price + quote_balance

            # Determine the target position size based on the risk percentage
            target_position_value = total_value * self.config.POSITION_RISK_PERCENTAGE

            # Calculate new order sizes
            new_base_order_size = target_position_value / current_price
            new_quote_order_size = target_position_value

            # Ensure order sizes are within allowed limits
            min_order_size = Decimal(str(self.exchange.markets[self.config.SYMBOL]['limits']['amount']['min']))
            max_order_size = Decimal(str(self.exchange.markets[self.config.SYMBOL]['limits']['amount']['max']))

            new_base_order_size = min(max(min_order_size, new_base_order_size), max_order_size)
            new_quote_order_size = min(max(min_order_size * current_price, new_quote_order_size), max_order_size * current_price)

            # Round amounts to the nearest lot size
            lot_size = Decimal(str(self.exchange.markets[self.config.SYMBOL]['precision']['amount']))
            new_base_order_size = round(new_base_order_size / lot_size) * lot_size
            new_quote_order_size = round(new_quote_order_size / (lot_size * current_price)) * (lot_size * current_price)

            await self.log(f"Adjusted order sizes - Base: {new_base_order_size}, Quote: {new_quote_order_size}", logging.INFO)

            return new_base_order_size, new_quote_order_size

        except Exception as e:
            await self.log(f"Error adjusting order sizes: {e}", logging.ERROR)
            return None

    def is_volatility_sufficient(self, current_market_data):
        volatility = self.calculate_current_volatility(current_market_data)
        return volatility > self.config.ADAPTIVE_PARAMS['VOLATILITY_THRESHOLD']

    def calculate_current_volatility(self, market_data):
        returns = market_data['close'].pct_change().dropna()
        return float(returns.std() * (252 ** 0.5))  # Annualized volatility

    async def update_adaptive_parameters(self):
        current_market_data = await self.get_recent_data()
        
        # Update volatility threshold
        current_volatility = self.calculate_current_volatility(current_market_data)
        self.config.ADAPTIVE_PARAMS['VOLATILITY_THRESHOLD'] = current_volatility * 0.8  # 80% of current volatility
        
        # Update position size based on current volatility
        if current_volatility > self.config.ADAPTIVE_PARAMS['HIGH_VOLATILITY_THRESHOLD']:
            self.config.POSITION_SIZE *= 0.8  # Reduce position size in high volatility
        elif current_volatility < self.config.ADAPTIVE_PARAMS['LOW_VOLATILITY_THRESHOLD']:
            self.config.POSITION_SIZE *= 1.2  # Increase position size in low volatility
        
        # Update other adaptive parameters as needed
        # ...

        await self.log(f"Updated adaptive parameters: {self.config.ADAPTIVE_PARAMS}", logging.INFO)

    async def load_backup(self, backup_file):
        try:
            async with aiofiles.open(backup_file, mode='r') as f:
                backup_data = json.loads(await f.read())
            
            # Restore balances
            self.wallet.balances = backup_data['balances']
            
            # Restore open orders
            self.current_orders = backup_data['open_orders']
            
            # Restore active strategies
            active_strategies = backup_data['active_strategies']
            for time_frame, strategy_name in active_strategies.items():
                strategy = self.strategy_manager.get_strategy_by_name(strategy_name)
                if strategy:
                    self.strategy_manager.set_active_strategy(TimeFrame[time_frame], strategy)
            
            await self.log(f"Backup loaded successfully from {backup_file}", logging.INFO)
        except Exception as e:
            await self.log(f"Error loading backup: {e}", logging.ERROR)

    async def handle_market_event(self, event: MarketEvent, strategy: Strategy):
        if self.is_volatility_sufficient(event.market_data):
            for temp_optimized_strategy in self.strategy_manager.temporary_optimize(strategy, event.market_data):
                signal = temp_optimized_strategy.generate_signal(event.market_data)
                if signal != 0:
                    self.events.append(SignalEvent(event.timestamp, self.config.SYMBOL, signal))
        else:
            signal = strategy.generate_signal(event.market_data)
            if signal != 0:
                self.events.append(SignalEvent(event.timestamp, self.config.SYMBOL, signal))
        
        await self.update_adaptive_parameters()

    def adjust_spread(self, base_spread, market_volatility):
        return base_spread * (1 + market_volatility)

    def apply_price_limits(self, bid_price: Decimal, ask_price: Decimal) -> Tuple[Decimal, Decimal]:
        current_price = self.get_current_price()
        max_deviation = current_price * self.config.BASE_PARAMS['MAX_PRICE_DEVIATION']
    
        bid_price = max(bid_price, current_price - max_deviation)
        ask_price = min(ask_price, current_price + max_deviation)
    
        # Round to exchange tick size
        tick_size = Decimal(str(self.exchange.markets[self.config.SYMBOL]['precision']['price']))
        bid_price = round(bid_price / tick_size) * tick_size
        ask_price = round(ask_price / tick_size) * tick_size
    
        return bid_price, ask_price

    def apply_size_limits(self, buy_amount: Decimal, sell_amount: Decimal) -> Tuple[Decimal, Decimal]:
        min_size = self.config.ADAPTIVE_PARAMS['MIN_ORDER_SIZE']
        max_size = self.config.ADAPTIVE_PARAMS['MAX_ORDER_SIZE']
    
        buy_amount = max(min(buy_amount, max_size), min_size)
        sell_amount = max(min(sell_amount, max_size), min_size)
    
        # Round to exchange lot size
        lot_size = Decimal(str(self.exchange.markets[self.config.SYMBOL]['precision']['amount']))
        buy_amount = round(buy_amount / lot_size) * lot_size
        sell_amount = round(sell_amount / lot_size) * lot_size
    
        return buy_amount, sell_amount

    def calculate_rsi(self, market_data: pd.DataFrame, period: int) -> float:
        delta = market_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
    
        return rsi.iloc[-1]

    def get_current_price(self) -> Decimal:
        return Decimal(str(self.order_book.get_mid_price()))
    
    async def _process_timeframe(self, timeframe: TimeFrame, market_data: pd.DataFrame):
        signal = await self.strategy_manager.get_weighted_signal(timeframe, market_data)
        position_size = self.risk_manager.calculate_position_size(timeframe, signal)
        await self.execute_trades(timeframe, signal, position_size)