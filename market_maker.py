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
from typing import Dict
from strategy import Strategy
from datetime import datetime, time
from event import MarketEvent, SignalEvent

class MarketMakerError(Exception):
    pass

class OrderPlacementError(MarketMakerError):
    pass

class MarketMaker:
    def __init__(self, config, strategy_config_path: str):
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
        
        while True:
            try:
                await self.update_market_data()
                await self.adjust_orders()
                await self.check_and_share_profits()
                
                inventory_risk = self.calculate_inventory_risk()
                if inventory_risk > self.config.MAX_INVENTORY_RISK:
                    await self.rebalance_inventory()
                
                current_time = datetime.now()
                if (current_time - last_performance_report).total_seconds() >= 3600:  # Report every hour
                    await self.report_performance()
                    last_performance_report = current_time
                
                await asyncio.sleep(self.config.ORDER_REFRESH_RATE)
            except Exception as e:
                await self.log(f"Error in market maker main loop: {e}", logging.ERROR)
                await asyncio.sleep(10)

    async def place_orders(self, bid_price, ask_price, position_size):
        await self.cancel_existing_orders()

        try:
            xbt_balance = await self.wallet.get_balance('XBT')
            usdt_balance = await self.wallet.get_balance('USDT')

            if usdt_balance < bid_price * position_size:
                await self.log(f"Insufficient USDT balance to place buy order. Required: {bid_price * position_size}, Available: {usdt_balance}", logging.WARNING)
                return
            if xbt_balance < position_size:
                await self.log(f"Insufficient XBT balance to place sell order. Required: {position_size}, Available: {xbt_balance}", logging.WARNING)
                return

            current_volatility = self.calculate_current_volatility(await self.get_recent_data())
            spread = self.adjust_spread(self.config.BASE_SPREAD, current_volatility)
            adjusted_bid_price = bid_price * (1 - spread)
            adjusted_ask_price = ask_price * (1 + spread)

            bid_order = await self.exchange.create_limit_buy_order(self.config.SYMBOL, position_size, adjusted_bid_price)
            ask_order = await self.exchange.create_limit_sell_order(self.config.SYMBOL, position_size, adjusted_ask_price)
            
            self.current_orders['bid'] = bid_order['id']
            self.current_orders['ask'] = ask_order['id']

            stop_loss = await self.risk_manager.set_stop_loss(adjusted_bid_price, position_size)
            take_profit = await self.risk_manager.set_take_profit(adjusted_ask_price, position_size)

            await self.exchange.create_stop_market_order(self.config.SYMBOL, 'sell', position_size, stop_loss)
            await self.exchange.create_take_profit_market_order(self.config.SYMBOL, 'sell', position_size, take_profit)

            await self.log(f"Orders placed - Bid: {adjusted_bid_price}, Ask: {adjusted_ask_price}, Size: {position_size}", logging.INFO)
        except Exception as e:
            raise OrderPlacementError(f"Error placing orders: {e}")

    def is_volatility_sufficient(self, current_market_data):
        volatility = self.calculate_current_volatility(current_market_data)
        return volatility > self.config.ADAPTIVE_PARAMS['VOLATILITY_THRESHOLD']

    def calculate_current_volatility(self, market_data):
        returns = market_data['close'].pct_change().dropna()
        return float(returns.std() * (252 ** 0.5))  # Annualized volatility

    async def handle_market_event(self, event: MarketEvent):
        current_market_data = self.get_recent_data(event.timestamp)
        
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
                await self.exchange.cancel_order(order_id, self.config.SYMBOL)
            except Exception as e:
                await self.log(f"Error cancelling order {order_id}: {e}", logging.ERROR)
        self.current_orders = {}

    async def rebalance_inventory(self):
        # Implement inventory rebalancing logic here
        pass

    async def calculate_performance_metrics(self):
        # Implement performance calculation logic here
        pass

    async def get_recent_data(self, timestamp=None):
        # Implement logic to fetch recent market data
        pass
