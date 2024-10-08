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
from typing import Dict, Optional
from strategy import Strategy
from datetime import datetime, time, timedelta
from event import MarketEvent, SignalEvent
import pandas as pd
import json
import psutil
import aiofiles
from decimal import Decimal

class MarketMakerError(Exception):
    pass

class OrderPlacementError(MarketMakerError):
    pass

class MarketMaker:
    def __init__(self, config, strategy_config_path):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.strategies: Dict[str, Strategy] = {}
        self.exchange = ccxt.kraken({
            'apiKey': config.KRAKEN_API_KEY,
            'secret': config.KRAKEN_SECRET_KEY,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.wallet = Wallet(self.exchange)
        self.order_book = OrderBook(self.exchange, config.SYMBOL)
        self.risk_manager = RiskManager(config)
        self.inventory_manager = InventoryManager(config, self.exchange)
        self.compliance_checker = ComplianceChecker(config)
        self.current_orders = {}
        self.strategy_factory = StrategyFactory(strategy_config_path, self)
        self.strategies = self.strategy_factory.strategies
        self.strategy_selector = StrategySelector(config)
        self.strategy_performance = {}
        self.events = []
        self.current_strategy = None

    async def log(self, message, level=logging.INFO):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.logger.log, level, message)

    async def report_performance(self):
        performance = await self.calculate_performance_metrics()
        await self.log(f"Performance metrics: {performance}", logging.INFO)

    def calculate_inventory_risk(self):
        net_position = self.inventory_manager.xbt_balance - self.inventory_manager.initial_xbt_balance
        return abs(net_position) / self.inventory_manager.initial_xbt_balance

    def adjust_spread(self, base_spread, market_volatility):
        return base_spread * (1 + market_volatility)

    def update_strategy(self, strategy_name: str, strategy: Strategy):
        self.strategies[strategy_name] = strategy
        self.logger.info(f"Strategy '{strategy_name}' has been updated.")

    def remove_strategy(self, strategy_name: str):
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            self.logger.info(f"Strategy '{strategy_name}' has been removed.")

    def stop(self):
        self.strategy_factory.stop()

    async def initialize(self):
        await self.exchange.load_markets()
        await self.wallet.connect()
        self.logger.info(f"Connected to {self.exchange.name}")
        self.logger.info(f"Current balances: {self.wallet.balances}")

    async def check_and_share_profits(self):
        now = datetime.now().time()
        if now == time(hour=0, minute=0):  # At midnight
            await self.wallet.share_profits()

    async def update_market_data(self):
        await self.order_book.update()
        await self.wallet.update_balances()
        self.inventory_manager.xbt_balance = self.wallet.get_balance('XBT')
        self.inventory_manager.usdt_balance = self.wallet.get_balance('USDT')

    async def run(self):
        await self.initialize()
        last_performance_report = datetime.now()
        last_rebalance = datetime.now()

        while True:
            try:
                current_time = datetime.now()

                # Update market data
                await self.update_market_data()

                # Check and update strategy
                new_strategy = self.strategy_selector.select_strategy(self.order_book.get_current_price())
                if new_strategy != self.current_strategy:
                    self.current_strategy = new_strategy
                    await self.rebalance_inventory()

                # Generate and handle market events
                market_event = MarketEvent(current_time, self.config.SYMBOL, self.order_book.get_current_price())
                await self.handle_market_event(market_event)

                # Process any pending events
                for event in self.events:
                    if isinstance(event, SignalEvent):
                        await self.execute_signal(event)

                # Adjust orders based on current market conditions
                await self.adjust_orders()

                # Check and share profits at midnight
                await self.check_and_share_profits()

                # Periodic rebalancing
                if (current_time - last_rebalance).total_seconds() >= self.config.REBALANCE_INTERVAL:
                    await self.rebalance_inventory()
                    last_rebalance = current_time

                # Periodic performance reporting
                if (current_time - last_performance_report).total_seconds() >= 3600:  # Report every hour
                    await self.report_performance()
                    last_performance_report = current_time

                # Risk management
                inventory_risk = self.calculate_inventory_risk()
                if inventory_risk > self.config.MAX_INVENTORY_RISK:
                    await self.rebalance_inventory()

                # Compliance check
                await self.compliance_checker.check_compliance(self.wallet.balances, self.current_orders)

                # Sleep for the configured interval
                await asyncio.sleep(self.config.ORDER_REFRESH_RATE)

                if (current_time - last_health_check).total_seconds() >= self.config.HEALTH_CHECK_INTERVAL:
                    await self.monitor_system_health()
                    last_health_check = current_time

                # Periodic data backup
                if (current_time - last_backup).total_seconds() >= self.config.BACKUP_INTERVAL:
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

            except Exception as e:
                await self.log(f"Error in market maker main loop: {e}", logging.ERROR)
                await asyncio.sleep(10)

    async def place_orders(self):
        adjusted_params = await self.adjust_order_parameters()
        if adjusted_params is None:
            await self.log("Failed to adjust order parameters. Skipping order placement.", logging.WARNING)
            return

        bid_price, ask_price, buy_amount, sell_amount = adjusted_params

        try:
            # Place buy order
            buy_order = await self.wallet.place_order(
                symbol=self.config.SYMBOL,
                order_type='limit',
                side='buy',
                amount=float(buy_amount),
                price=float(bid_price)
            )
            self.current_orders['bid'] = buy_order['id']

            # Place sell order
            sell_order = await self.wallet.place_order(
                symbol=self.config.SYMBOL,
                order_type='limit',
                side='sell',
                amount=float(sell_amount),
                price=float(ask_price)
            )
            self.current_orders['ask'] = sell_order['id']

            await self.log(f"Orders placed - Buy: {buy_amount} @ {bid_price}, Sell: {sell_amount} @ {ask_price}", logging.INFO)

        except Exception as e:
            await self.log(f"Error placing orders: {e}", logging.ERROR)
            raise OrderPlacementError(f"Error placing orders: {e}")

    def is_volatility_sufficient(self, current_market_data):
        volatility = self.calculate_current_volatility(current_market_data)
        return volatility > self.config.ADAPTIVE_PARAMS['VOLATILITY_THRESHOLD']

    def calculate_current_volatility(self, market_data):
        returns = market_data['close'].pct_change().dropna()
        return float(returns.std() * (252 ** 0.5))  # Annualized volatility

    async def handle_market_event(self, event: MarketEvent):
        current_market_data = await self.get_recent_data(event.timestamp)
        
        if self.is_volatility_sufficient(current_market_data):
            for temp_optimized_strategy in self.strategy_optimizer.temporary_optimize(self.current_strategy, current_market_data):
                signal = temp_optimized_strategy.generate_signal(current_market_data)
                if signal != 0:
                    self.events.append(SignalEvent(event.timestamp, self.config.SYMBOL, signal))
        else:
            signal = self.current_strategy.generate_signal(current_market_data)
            if signal != 0:
                self.events.append(SignalEvent(event.timestamp, self.config.SYMBOL, signal))
        
        self.update_strategy(event.timestamp)

    async def cancel_existing_orders(self):
        for order_id in self.current_orders.values():
            try:
                await self.wallet.cancel_order(order_id, self.config.SYMBOL)
            except Exception as e:
                await self.log(f"Error cancelling order {order_id}: {e}", logging.ERROR)
        self.current_orders = {}

    async def rebalance_inventory(self):
        btc_price = self.config.get_btc_price()
        optimal_capital = self.current_strategy.get_optimal_capital()
        current_xbt_balance = self.wallet.get_balance('XBT')
        current_usdt_balance = self.wallet.get_balance('USDT')
        target_value = optimal_capital / 2
        target_xbt_value = target_value
        target_xbt_balance = target_xbt_value / btc_price
        target_usdt_balance = optimal_capital / 2

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

        # Add any remaining extra capital
        if extra_capital > 0:
            xbt_to_buy = extra_capital / (2 * btc_price)
            await self.wallet.place_order(self.config.SYMBOL, 'market', 'buy', xbt_to_buy)
            await self.wallet.transfer_to_trading_account('USDT', extra_capital / 2)

        await self.wallet.update_balances()
        await self.log(f"Rebalanced inventory: XBT={self.wallet.get_balance('XBT')}, USDT={self.wallet.get_balance('USDT')}", logging.INFO)

    async def get_recent_data(self, timestamp: Optional[datetime] = None) -> pd.DataFrame:
        if timestamp is None:
            timestamp = datetime.now()
        
        end_time = timestamp
        start_time = end_time - timedelta(hours=24)  # Get last 24 hours of data
        
        ohlcv = await self.exchange.fetch_ohlcv(
            self.config.SYMBOL,
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

    async def adjust_orders(self):
        current_price = self.order_book.get_current_price()
        bid_price = current_price * (1 - self.config.SPREAD)
        ask_price = current_price * (1 + self.config.SPREAD)
        position_size = self.config.POSITION_SIZE

        await self.place_orders(bid_price, ask_price, position_size)

    async def execute_signal(self, signal_event: SignalEvent):
        if signal_event.signal > 0:
            # Buy signal
            await self.wallet.place_order(self.config.SYMBOL, 'market', 'buy', self.config.POSITION_SIZE)
        elif signal_event.signal < 0:
            # Sell signal
            await self.wallet.place_order(self.config.SYMBOL, 'market', 'sell', self.config.POSITION_SIZE)

    async def handle_exchange_errors(self, error):
        await self.log(f"Exchange error occurred: {error}", logging.ERROR)
        
        if isinstance(error, ccxt.NetworkError):
            await self.log("Network error detected. Waiting before retry...", logging.WARNING)
            await asyncio.sleep(self.config.NETWORK_ERROR_RETRY_WAIT)
            return True  # Indicate that a retry should be attempted
        
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
            return True  # Indicate that a retry should be attempted
        
        else:
            await self.log("Unhandled error. Notifying administrator...", logging.ERROR)
            await self.notify_administrator(error)
        
        return False  # Indicate that no retry should be attempted

    async def adjust_order_parameters(self):
        try:
            # Fetch the current market price
            ticker = await self.exchange.fetch_ticker(self.config.SYMBOL)
            current_price = Decimal(str(ticker['last']))

            # Adjust bid and ask prices
            bid_price = current_price * (Decimal('1') - self.config.MAX_SPREAD)
            ask_price = current_price * (Decimal('1') + self.config.MAX_SPREAD)

            # Ensure prices are within allowed limits
            bid_price = max(bid_price, current_price * (Decimal('1') - self.config.MAX_PRICE_DEVIATION))
            ask_price = min(ask_price, current_price * (Decimal('1') + self.config.MAX_PRICE_DEVIATION))

            # Round prices to the nearest tick size
            tick_size = Decimal(str(self.exchange.markets[self.config.SYMBOL]['precision']['price']))
            bid_price = round(bid_price / tick_size) * tick_size
            ask_price = round(ask_price / tick_size) * tick_size

            # Adjust order sizes
            base_balance, quote_balance = await self.get_available_balances()
            max_buy_amount = quote_balance / bid_price
            max_sell_amount = base_balance

            # Ensure order sizes are within allowed limits
            min_order_size = Decimal(str(self.exchange.markets[self.config.SYMBOL]['limits']['amount']['min']))
            max_order_size = Decimal(str(self.exchange.markets[self.config.SYMBOL]['limits']['amount']['max']))

            buy_amount = min(max(min_order_size, max_buy_amount), max_order_size)
            sell_amount = min(max(min_order_size, max_sell_amount), max_order_size)

            # Round amounts to the nearest lot size
            lot_size = Decimal(str(self.exchange.markets[self.config.SYMBOL]['precision']['amount']))
            buy_amount = round(buy_amount / lot_size) * lot_size
            sell_amount = round(sell_amount / lot_size) * lot_size

            await self.log(f"Adjusted order parameters - Bid: {bid_price}, Ask: {ask_price}, Buy Amount: {buy_amount}, Sell Amount: {sell_amount}", logging.INFO)

            return bid_price, ask_price, buy_amount, sell_amount

        except Exception as e:
            await self.log(f"Error adjusting order parameters: {e}", logging.ERROR)
            return None

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

    async def notify_administrator(self, error):
        # Implement logic to send notification to administrator (e.g., email, SMS)
        pass

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
        
        # Check network latency
        try:
            start_time = datetime.now()
            await self.exchange.fetch_ticker(self.config.SYMBOL)
            latency = (datetime.now() - start_time).total_seconds() * 1000  # in milliseconds
            
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
            'current_strategy': self.current_strategy.__dict__ if self.current_strategy else None,
        }
        
        filename = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            async with aiofiles.open(filename, mode='w') as f:
                await f.write(json.dumps(backup_data, indent=2))
            await self.log(f"Backup created successfully: {filename}", logging.INFO)
        except Exception as e:
            await self.log(f"Error creating backup: {e}", logging.ERROR)

    def set_wallet(self, wallet):
        self.wallet = wallet
    
    def update_strategy(self, strategy_name: str, strategy: Strategy):
        self.strategies[strategy_name] = strategy
        self.logger.info(f"Strategy '{strategy_name}' has been updated.")

    async def get_available_balances(self):
        balances = await self.wallet.get_balances()
        base_currency, quote_currency = self.config.SYMBOL.split('/')
        base_balance = Decimal(str(balances.get(base_currency, 0)))
        quote_balance = Decimal(str(balances.get(quote_currency, 0)))
        return base_balance, quote_balance

    def __del__(self):
        if hasattr(self, 'wallet') and self.wallet:
            asyncio.create_task(self.wallet.close())
        self.stop()
    
    async def initialize(self):
        if self.wallet is None:
            self.wallet = Wallet(self.exchange)