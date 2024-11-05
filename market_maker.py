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
from typing import Dict, Optional
from strategy import Strategy, TimeFrame
from datetime import datetime, time, timedelta
from event import MarketEvent, SignalEvent
import pandas as pd
import json
import psutil
import aiofiles
from watchdog.observers import Observer
from strategy_manager import StrategyManager

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
        self.strategy_manager = StrategyManager(config)
        self.logger = logging.getLogger(__name__)
        self.strategy_generator = StrategyGenerator(config)

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
        last_strategy_generation = asyncio.get_event_loop().time()
        last_rebalance = asyncio.get_event_loop().time()
        last_performance_report = asyncio.get_event_loop().time()
        last_health_check = asyncio.get_event_loop().time()
        last_backup = asyncio.get_event_loop().time()

        while True:
            try:
                current_time = asyncio.get_event_loop().time()

                # Update market data
                market_data = await self.update_market_data()

                # Periodically generate new strategies
                if current_time - last_strategy_generation >= self.config.STRATEGY_GENERATION_INTERVAL:
                    await self.strategy_manager.generate_strategies(market_data)
                    last_strategy_generation = current_time

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
        volatility = self.calculate_current_volatility(market_data)
        trend_strength = (market_data['close'].iloc[-1] - market_data['close'].iloc[-20]) / market_data['close'].iloc[-20]
    
    # Dynamically adjust moving average windows based on volatility
        if volatility > self.config.ADAPTIVE_PARAMS['HIGH_VOLATILITY_THRESHOLD']:
            strategy.parameters['MOVING_AVERAGE_SHORT'] = max(5, strategy.parameters.get('MOVING_AVERAGE_SHORT', 10) - 2)
            strategy.parameters['MOVING_AVERAGE_LONG'] = max(20, strategy.parameters.get('MOVING_AVERAGE_LONG', 50) - 5)
        else:
            strategy.parameters['MOVING_AVERAGE_SHORT'] = min(20, strategy.parameters.get('MOVING_AVERAGE_SHORT', 10) + 2)
            strategy.parameters['MOVING_AVERAGE_LONG'] = min(100, strategy.parameters.get('MOVING_AVERAGE_LONG', 50) + 5)
    
        strategy.parameters['TREND_STRENGTH_THRESHOLD'] = max(0.01, min(0.05, abs(trend_strength) * 0.8))

    async def _update_statistical_arbitrage_params(self, strategy: Strategy, market_data: pd.DataFrame):
        # Adjust z-score threshold based on recent spread volatility
        spread_volatility = market_data['asset1_close'].sub(market_data['asset2_close']).std()
        strategy.parameters['Z_SCORE_THRESHOLD'] = max(1.5, min(3.0, spread_volatility * 1.5))
    
        # Adjust lookback period based on market regime
        correlation = market_data['asset1_close'].corr(market_data['asset2_close'])
        if correlation > 0.8:
            strategy.parameters['LOOKBACK_PERIOD'] = max(10, strategy.parameters.get('LOOKBACK_PERIOD', 20) - 2)
        else:
            strategy.parameters['LOOKBACK_PERIOD'] = min(40, strategy.parameters.get('LOOKBACK_PERIOD', 20) + 2)

    async def _update_volatility_params(self, strategy: Strategy, market_data: pd.DataFrame):
        current_volatility = self.calculate_current_volatility(market_data)
        avg_volatility = market_data['close'].pct_change().rolling(window=20).std().mean()
    
        strategy.parameters['HIGH_VOLATILITY_THRESHOLD'] = max(1.2, min(2.0, current_volatility / avg_volatility * 1.5))
        strategy.parameters['LOW_VOLATILITY_THRESHOLD'] = max(0.3, min(0.7, current_volatility / avg_volatility * 0.5))
        strategy.parameters['VOLATILITY_WINDOW'] = max(10, min(30, int(20 * avg_volatility / current_volatility)))

    async def _update_sentiment_params(self, strategy: Strategy, market_data: pd.DataFrame):
        recent_sentiment = market_data['sentiment_score'].iloc[-10:].mean()
        sentiment_volatility = market_data['sentiment_score'].std()
    
        strategy.parameters['POSITIVE_SENTIMENT_THRESHOLD'] = min(0.8, max(0.6, recent_sentiment + sentiment_volatility))
        strategy.parameters['NEGATIVE_SENTIMENT_THRESHOLD'] = max(0.2, min(0.4, recent_sentiment - sentiment_volatility))
        strategy.parameters['SENTIMENT_IMPACT_WEIGHT'] = max(0.1, min(0.5, sentiment_volatility * 2))

    async def _update_momentum_params(self, strategy: Strategy, market_data: pd.DataFrame):
        returns_volatility = market_data['close'].pct_change().std()
        momentum = market_data['close'].pct_change(periods=14).iloc[-1]
    
        strategy.parameters['MOMENTUM_THRESHOLD'] = max(0.02, min(0.08, returns_volatility * 2))
        strategy.parameters['ACCELERATION_FACTOR'] = max(0.01, min(0.03, abs(momentum) * 0.5))
        strategy.parameters['MAX_ACCELERATION'] = max(0.1, min(0.3, returns_volatility * 4))

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

        await self.cancel_existing_orders()
        await self.place_orders(bid_price, ask_price, buy_amount, sell_amount)

    async def place_orders(self, bid_price, ask_price, buy_amount, sell_amount):
        try:
            active_strategy = self.strategy_manager.get_active_strategy(TimeFrame.SHORT_TERM)
        
            if 'trend_following' in active_strategy.favored_patterns:
                trend_strength = active_strategy.parameters['TREND_STRENGTH_THRESHOLD']
                bid_price *= (1 - trend_strength)
                ask_price *= (1 + trend_strength)
            
            elif 'statistical_arbitrage' in active_strategy.favored_patterns:
                z_score = active_strategy.parameters['Z_SCORE_THRESHOLD']
                spread_adjustment = z_score * market_data['spread'].std()
                bid_price *= (1 - spread_adjustment)
                ask_price *= (1 + spread_adjustment)
            
            elif 'volatility_clustering' in active_strategy.favored_patterns:
                vol_threshold = active_strategy.parameters['HIGH_VOLATILITY_THRESHOLD']
                vol_adjustment = vol_threshold * market_data['volatility'].iloc[-1]
                buy_amount *= (1 - vol_adjustment)
                sell_amount *= (1 - vol_adjustment)
            
            elif 'sentiment_analysis' in active_strategy.favored_patterns:
                sentiment_impact = active_strategy.parameters['SENTIMENT_IMPACT_WEIGHT']
                sentiment_score = market_data['sentiment_score'].iloc[-1]
                bid_price *= (1 + sentiment_impact * sentiment_score)
                ask_price *= (1 + sentiment_impact * sentiment_score)
            
            elif 'momentum' in active_strategy.favored_patterns:
                momentum_factor = active_strategy.parameters['ACCELERATION_FACTOR']
                momentum = market_data['close'].pct_change(periods=active_strategy.parameters['MOMENTUM_PERIOD']).iloc[-1]
                bid_price *= (1 + momentum_factor * momentum)
                ask_price *= (1 + momentum_factor * momentum)
            
            elif 'market_making' in active_strategy.favored_patterns:
                spread = active_strategy.parameters['BID_ASK_SPREAD']
                bid_price *= (1 - spread)
                ask_price *= (1 + spread)
            
            elif 'grid_trading' in active_strategy.favored_patterns:
                grid_spacing = active_strategy.parameters['GRID_SPACING']
                grid_levels = active_strategy.parameters['GRID_LEVELS']
                for i in range(grid_levels):
                    grid_bid = bid_price * (1 - i * grid_spacing)
                    grid_ask = ask_price * (1 + i * grid_spacing)
                    await self._place_grid_orders(grid_bid, grid_ask, buy_amount/grid_levels, sell_amount/grid_levels)
                return

            # Place the orders with adjusted parameters
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

    # Add any additional methods or modifications here as needed