import os
import ccxt
import threading
import time
from decimal import Decimal
import google.generativeai as genai
import itertools

class Config:
    def __init__(self):
        self.nonce_counter = itertools.count(int(time.time() * 1000))
        self.nonce_lock = threading.Lock()
        self.BASE_PARAMS = {
            # API Keys
            'CRYPTO_COMPARE_API_KEY': os.getenv("CRYPTO_COMPARE_API_KEY"),
            'MORALIS_API_KEY': os.getenv('MORALIS_API_KEY'),
            'KRAKEN_API_KEY': os.getenv('API_KEY'),
            'KRAKEN_PRIVATE_KEY': os.getenv('KRAKEN_PRIVATE_KEY'),
            'GOOGLE_AI_API_KEY': genai.configure(api_key=os.environ['GOOGLE_AI_API_KEY']),
            'PROFIT_SHARING_ADDRESS': '0x9A252E29eB31d76BcC3408E5F98694a0f7A764D6',  # Replace with the actual MetaMask address
            'PROFIT_SHARING_PERCENTAGE': 0.05,  # 5%
            # Trading parameters
            'SYMBOL': 'BTC/USDT',
            'INITIAL_CAPITAL': 10000,
            'SENTIMENT_THRESHOLD_LOW': 30,
            'SENTIMENT_THRESHOLD_HIGH': 70,
            #Risk manager bases
            'BASE_STOP_LOSS_PCT': 0.05,
            'BASE_TAKE_PROFIT_PCT': 0.1,
            # Strategy parameters
            'MAX_STRATEGIES_PER_TIMEFRAME': 50,
            'STRATEGY_UPDATE_INTERVAL': 100,
            'MONTE_CARLO_SIMULATIONS': 1000,
            'GENETIC_ALGORITHM_GENERATIONS': 20,
            'GENETIC_ALGORITHM_POPULATION': 50,
            'MUTATION_RATE': 0.2,
            'CROSSOVER_RATE': 0.7,
            'NUM_STRATEGIES_TO_GENERATE': 2,
            'OPTIMIZATION_INTERVAL': 1000,

            # Dynamic hedging parameters
            'HEDGE_ACTIVATION_THRESHOLD': .75,  # Activate hedging when volatility > 75%
            'HEDGE_DEACTIVATION_THRESHOLD': 0.45,  # Deactivate hedging when volatility < 45% 
            'MIN_HEDGE_RATIO': 0.1,
            'MAX_HEDGE_RATIO': 0.9,
            'BACKTEST_DURATION': 0,
            'BACKTEST_UPDATE_INTERVAL': 600,
            # MarketMaker specific parameters
            'MAX_SPREAD': Decimal('0.01'),  # 1% maximum spread
            'MAX_PRICE_DEVIATION': Decimal('0.05'),  # 5% maximum deviation from current price
            'POSITION_RISK_PERCENTAGE': Decimal('0.1'),  # 10% of total portfolio value for each position
            'ORDER_REFRESH_RATE': 60,  # Refresh orders every 60 seconds
            'MARKET_MAKER_DURATION': 0,
            # Error handling
            'NETWORK_ERROR_RETRY_WAIT': 60,  # seconds
            'ERROR_RETRY_INTERVAL': 300,  # seconds

            # System health
            'MAX_CPU_USAGE': 80,  # percent
            'MAX_MEMORY_USAGE': 80,  # percent
            'MAX_DISK_USAGE': 80,  # percent
            'MAX_NETWORK_LATENCY': 1000,  # milliseconds
            'HEALTH_CHECK_INTERVAL': 300,  # seconds

            # Backup
            'BACKUP_INTERVAL': 3600,  # seconds (1 hour)

            # Performance metrics
            'PERFORMANCE_METRICS': ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate'],

            # Logging
            'LOG_LEVEL': 'INFO',
            'LOG_FILE': 'trading_system.log',

            # API rate limiting
            'API_RATE_LIMIT': 10,  # maximum number of API calls per minute

            # Backtesting parameters
            'BACKTESTING_START_DATE': '2022-01-01',
            'BACKTESTING_END_DATE': '2023-01-01',
            'INITIAL_BALANCE': 10000,  # in quote currency (e.g., USD)
            'RISK_FREE_RATE': 0.02,  # 2% annual risk-free rate for Sharpe ratio calculation
            'MAX_DRAWDOWN_THRESHOLD': 0.2,  # 20% maximum allowable drawdown
            'LEVERAGE': 1,  # 1 means no leverage, 2 means 2x leverage, etc.
        }
        
        self.ADAPTIVE_PARAMS = {
            # Strategy parameters (you might want to move these to strategies.json)
            'MOVING_AVERAGE_SHORT': 10,
            'MOVING_AVERAGE_LONG': 30,
            'RSI_PERIOD': 14,
            'RSI_OVERBOUGHT': 70,
            'RSI_OVERSOLD': 30,
            'LOOKBACK_PERIOD': 7,
            'TRADING_FEE': 0.001,
            'MAX_POSITION_SIZE': 0.1,
            'STOP_LOSS_PCT': 0.05,
            'TAKE_PROFIT_PCT': 0.1,
            'MAX_DRAWDOWN': 0.2,
            'VOLATILITY_WINDOW': 20,
            'LIQUIDITY_THRESHOLD': 100,  # BTC
            'VOLATILITY_THRESHOLD': 0.05,  # 5% daily volatility
            'IMPERMANENT_LOSS_THRESHOLD': 0.05,
            'MARKET_IMPACT_FACTOR': 0.1,
            'STRATEGY_STRENGTH_WEIGHTS': {
                'sharpe_ratio': 0.4,
                'profit_factor': 0.3,
                'win_rate': 0.3
            },
            'ANNUAL_BTC_VOLATILITY': 0.46,  # 80% annual volatility for BTC
            'USDT_INTEREST_RATE': 0.05,  # 5% annual interest rate for USDT
            'RISK_PER_TRADE': 0.02,  # 2% risk per trade
            'TREND_FOLLOWING_PARAMS': {
            'MOMENTUM_PERIOD': 14,
            'MOMENTUM_THRESHOLD': 0.05,
            'TREND_STRENGTH_THRESHOLD': 0.02
            },

            'MEAN_REVERSION_PARAMS': {
                'MEAN_WINDOW': 20,
                'MEAN_REVERSION_THRESHOLD': 0.05
            },

            'BREAKOUT_PARAMS': {
                'BREAKOUT_PERIOD': 20,
                'BREAKOUT_THRESHOLD': 0.02
            },

            'VOLATILITY_PARAMS': {
                'VOLATILITY_WINDOW': 20,
                'HIGH_VOLATILITY_THRESHOLD': 1.5,
                'LOW_VOLATILITY_THRESHOLD': 0.5
            },
            # Extreme market condition parameters
            'BULL_MARKET_THRESHOLD': 2.0,  # 100% cumulative return for bull market
            'BEAR_MARKET_THRESHOLD': 0.5,  # -50% cumulative return for bear market
            'EXTREME_MARKET_MULTIPLIER': 2.0, #Either divides or multiplies based on the market
            # New risk management parameters
            'LIQUIDITY_RISK_MULTIPLIER': 0.5,
            'VOLATILITY_RISK_MULTIPLIER': 0.7,
            'DRAWDOWN_RISK_MULTIPLIER': 0.6,
            'SENTIMENT_LOW_MULTIPLIER': 0.8,
            'SENTIMENT_HIGH_MULTIPLIER': 1.2,
            'ON_CHAIN_BEARISH_MULTIPLIER': 0.8,
            'ON_CHAIN_BULLISH_MULTIPLIER': 1.2,
            'FUNDING_RATE_MULTIPLIER': 0.9,
            'IL_HIGH_RISK_MULTIPLIER': 0.7,
            'IL_MEDIUM_RISK_MULTIPLIER': 0.9,

            # Impermanent Loss parameters
            'IL_RISK_TOLERANCE': 0.1,  # 10% tolerance for Impermanent Loss at Risk (ILaR)
            'FUNDING_RATE_THRESHOLD': 0.001,  # 0.1% funding rate
            'SIGNAL_IMPACT_FACTOR': 0.2,  # How much the signal affects prices
            'INVENTORY_IMPACT_FACTOR': 0.1,  # How much inventory affects market impact
            'MIN_SPREAD': 0.0005,  # Minimum 0.05% spread
            'PRICE_PRECISION': 2,  # Round prices to 2 decimal places
            'PRICE_ADJUSTMENT_THRESHOLD': 0.005,  # Adjust prices if they've moved by 0.5%
            'MIN_ORDER_SIZE': Decimal('0.001'),  # Minimum order size in base currency
            'MAX_ORDER_SIZE': Decimal('1.0'),  # Maximum order size in base currency
            'TICK_SIZE': Decimal('0.1'),  # Price tick size
            'LOT_SIZE': Decimal('0.001'),  # Amount lot size

            'TREND_FOLLOWING_PARAMS': {
                'MOVING_AVERAGE_SHORT': 10,
                'MOVING_AVERAGE_LONG': 30,
                'TREND_STRENGTH_THRESHOLD': 0.02,
                'TREND_CONFIRMATION_PERIOD': 5,
                'MOMENTUM_FACTOR': 0.1,
                'BREAKOUT_LEVEL': 0.03,
                'TRAILING_STOP': 0.02
            },

            'MEAN_REVERSION_PARAMS': {
                'MEAN_WINDOW': 20,
                'STD_MULTIPLIER': 2.0,
                'MEAN_REVERSION_THRESHOLD': 0.05,
                'ENTRY_DEVIATION': 0.02,
                'EXIT_DEVIATION': 0.01,
                'BOLLINGER_PERIOD': 20,
                'BOLLINGER_STD': 2.0
            },

            'MOMENTUM_PARAMS': {
                'MOMENTUM_PERIOD': 14,
                'MOMENTUM_THRESHOLD': 0.05,
                'RSI_PERIOD': 14,
                'RSI_OVERBOUGHT': 70,
                'RSI_OVERSOLD': 30,
                'ACCELERATION_FACTOR': 0.02,
                'MAX_ACCELERATION': 0.2,
                'MACD_FAST': 12,
                'MACD_SLOW': 26,
                'MACD_SIGNAL': 9
            },

            'BREAKOUT_PARAMS': {
                'BREAKOUT_PERIOD': 20,
                'BREAKOUT_THRESHOLD': 0.02,
                'VOLUME_CONFIRMATION_MULT': 1.5,
                'CONSOLIDATION_PERIOD': 10,
                'SUPPORT_RESISTANCE_LOOKBACK': 50,
                'BREAKOUT_CONFIRMATION_CANDLES': 3,
                'ATR_PERIOD': 14
            },

            'VOLATILITY_CLUSTERING_PARAMS': {
                'VOLATILITY_WINDOW': 20,
                'HIGH_VOLATILITY_THRESHOLD': 1.5,
                'LOW_VOLATILITY_THRESHOLD': 0.5,
                'GARCH_LAG': 5,
                'ATR_MULTIPLIER': 2.0,
                'VOLATILITY_BREAKOUT_THRESHOLD': 2.0,
                'VOLATILITY_MEAN_PERIOD': 50
            },

            'STATISTICAL_ARBITRAGE_PARAMS': {
                'LOOKBACK_PERIOD': 20,
                'Z_SCORE_THRESHOLD': 2.0,
                'CORRELATION_THRESHOLD': 0.8,
                'HALF_LIFE': 10,
                'HEDGE_RATIO': 1.0,
                'ENTRY_THRESHOLD': 2.0,
                'EXIT_THRESHOLD': 0.5,
                'WINDOW_SIZE': 60,
                'MIN_CORRELATION': 0.7,
                'COINTEGRATION_THRESHOLD': 0.05
            },

            'SENTIMENT_ANALYSIS_PARAMS': {
                'POSITIVE_SENTIMENT_THRESHOLD': 0.6,
                'NEGATIVE_SENTIMENT_THRESHOLD': 0.4,
                'SENTIMENT_WINDOW': 24,
                'SENTIMENT_IMPACT_WEIGHT': 0.3,
                'NEWS_IMPACT_DECAY': 0.95,
                'SENTIMENT_SMOOTHING_FACTOR': 0.1,
                'SENTIMENT_VOLUME_THRESHOLD': 1000,
                'SENTIMENT_MOMENTUM_PERIOD': 12
            }

        }
        self.current_strategy = None
        
        def get_nonce(self):
            with self.nonce_lock:
                return next(self.nonce_counter)
        
        # Initialize exchange
        self.exchange = ccxt.kraken({
            'apiKey': self.BASE_PARAMS['KRAKEN_API_KEY'],
            'secret': self.BASE_PARAMS['KRAKEN_PRIVATE_KEY'],
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
            'nonce': lambda: self.get_nonce()
        })

        # Portfolio tracking
        self.portfolio_value = self.BASE_PARAMS['INITIAL_CAPITAL']
        self.btc_quantity = 0
        self.usdt_quantity = self.BASE_PARAMS['INITIAL_CAPITAL']

        # Dynamic hedging state
        self.dynamic_hedging_active = False
        self.hedge_ratio = 0.0

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor_portfolio)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    def update_adaptive_params(self, strategy):
        self.current_strategy = strategy
        self.ADAPTIVE_PARAMS = {param: strategy.parameters[param] for param in strategy.optimize_params}

    def get_all_params(self):
        return {**self.BASE_PARAMS, **self.ADAPTIVE_PARAMS}

    def monitor_portfolio(self):
        while True:
            try:
                self.update_portfolio()
                self.adjust_parameters()
                time.sleep(60)  # Update every minute
            except Exception as e:
                print(f"Error in portfolio monitoring: {e}")
                time.sleep(60)  # Wait a minute before trying again

    def update_portfolio(self):
        try:
            balance = self.exchange.fetch_balance()
            self.btc_quantity = Decimal(str(balance['BTC']['total']))
            self.usdt_quantity = Decimal(str(balance['USDT']['total']))

            ticker = self.exchange.fetch_ticker(self.BASE_PARAMS['SYMBOL'])
            btc_value_in_usdt = self.btc_quantity * Decimal(str(ticker['last']))

            self.portfolio_value = float(btc_value_in_usdt + self.usdt_quantity)
        except Exception as e:
            print(f"Error updating portfolio: {e}")

    def adjust_parameters(self):
        btc_price = self.get_btc_price()
        volatility = self.get_current_volatility()

        self.ADAPTIVE_PARAMS['MAX_POSITION_SIZE'] = min(
            self.ADAPTIVE_PARAMS['MAX_POSITION_SIZE'], 
            self.portfolio_value * 0.1 / btc_price
        )

        if self.portfolio_value > 100000:
            self.ADAPTIVE_PARAMS['RISK_PER_TRADE'] = self.ADAPTIVE_PARAMS['RISK_PER_TRADE'] * 0.8
        elif self.portfolio_value < 10000:
            self.ADAPTIVE_PARAMS['RISK_PER_TRADE'] = self.ADAPTIVE_PARAMS['RISK_PER_TRADE'] * 1.2
        else:
            self.ADAPTIVE_PARAMS['RISK_PER_TRADE'] = self.ADAPTIVE_PARAMS['RISK_PER_TRADE']

        self.ADAPTIVE_PARAMS['STOP_LOSS_PCT'] = max(self.ADAPTIVE_PARAMS['STOP_LOSS_PCT'], volatility * 2)
        self.ADAPTIVE_PARAMS['TAKE_PROFIT_PCT'] = max(self.ADAPTIVE_PARAMS['TAKE_PROFIT_PCT'], volatility * 3)
        self.ADAPTIVE_PARAMS['LIQUIDITY_THRESHOLD'] = self.ADAPTIVE_PARAMS['LIQUIDITY_THRESHOLD'] * btc_price / 10000

        # Adjust dynamic hedging parameters
        if volatility > self.BASE_PARAMS['HEDGE_ACTIVATION_THRESHOLD'] and not self.dynamic_hedging_active:
            self.dynamic_hedging_active = True
            self.hedge_ratio = self.BASE_PARAMS['MIN_HEDGE_RATIO']
        elif volatility < self.BASE_PARAMS['HEDGE_DEACTIVATION_THRESHOLD'] and self.dynamic_hedging_active:
            self.dynamic_hedging_active = False
            self.hedge_ratio = 0.0
        elif self.dynamic_hedging_active:
            self.hedge_ratio = min(max(volatility, self.BASE_PARAMS['MIN_HEDGE_RATIO']), self.BASE_PARAMS['MAX_HEDGE_RATIO'])

    def get_btc_price(self):
        try:
            ticker = self.exchange.fetch_ticker(self.BASE_PARAMS['SYMBOL'])
            return ticker['last']
        except Exception as e:
            print(f"Error fetching BTC price: {e}")
            return None

    def get_current_volatility(self):
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.BASE_PARAMS['SYMBOL'], '1d', limit=self.ADAPTIVE_PARAMS['VOLATILITY_WINDOW'])
            closes = [x[4] for x in ohlcv]
            returns = [closes[i]/closes[i-1] - 1 for i in range(1, len(closes))]
            return float(Decimal(str(sum(returns) / len(returns))).quantize(Decimal('0.0001')))
        except Exception as e:
            print(f"Error calculating volatility: {e}")
            return self.ADAPTIVE_PARAMS['VOLATILITY_THRESHOLD']

    def get_portfolio_value(self):
        return self.portfolio_value

    def get_btc_quantity(self):
        return float(self.btc_quantity)

    def get_usdt_quantity(self):
        return float(self.usdt_quantity)

    def get_hedge_ratio(self):
        return self.hedge_ratio

    def is_dynamic_hedging_active(self):
        return self.dynamic_hedging_active