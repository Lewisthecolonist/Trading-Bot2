from config import Config
import ccxt  # For fetching market data
import ta  # For technical indicators
import requests  # For API calls to sentiment and on-chain data providers
import san
from glassnode import GlassnodeClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class RiskManager:
    def __init__(self, config):
        self.config = config

        # Initialize APIs
        san.ApiConfig.api_key = config.CRYPTO_COMPARE_API_KEY
        self.glassnode = GlassnodeClient(api_key=config.GLASSNODE_API_KEY)
        self.exchange = ccxt.binance({
            'apiKey': config.BINANCE_API_KEY,
            'secret': config.BINANCE_SECRET_KEY,
        })

        # Risk parameters
        self.max_position_size = config.MAX_POSITION_SIZE
        self.base_stop_loss_pct = config.BASE_STOP_LOSS_PCT
        self.base_take_profit_pct = config.BASE_TAKE_PROFIT_PCT
        self.max_drawdown = config.MAX_DRAWDOWN
        self.volatility_window = config.VOLATILITY_WINDOW
        self.liquidity_threshold = config.LIQUIDITY_THRESHOLD

    def fetch_order_book(self, symbol='BTC/USDT', limit=20):
        return self.exchange.fetch_order_book(symbol, limit)

    def check_liquidity(self, order_book):
        bid_liquidity = sum(bid[1] for bid in order_book['bids'][:10])
        ask_liquidity = sum(ask[1] for ask in order_book['asks'][:10])
        return min(bid_liquidity, ask_liquidity) > self.liquidity_threshold

    def calculate_position_size(self, portfolio_value, entry_price, stop_loss_price):
        risk_per_trade = portfolio_value * self.config.RISK_PER_TRADE
        price_risk = abs(entry_price - stop_loss_price)
        position_size = risk_per_trade / price_risk
        max_allowed = portfolio_value * self.max_position_size
        return min(position_size, max_allowed)

    def set_dynamic_stop_loss(self, entry_price, position_type, atr):
        atr_multiplier = 2  # Adjust based on risk tolerance
        if position_type == "long":
            return entry_price - (atr * atr_multiplier)
        elif position_type == "short":
            return entry_price + (atr * atr_multiplier)
        else:
            raise ValueError("Invalid position type. Must be 'long' or 'short'.")

    def set_dynamic_take_profit(self, entry_price, position_type, atr):
        atr_multiplier = 3  # Adjust based on desired risk-reward ratio
        if position_type == "long":
            return entry_price + (atr * atr_multiplier)
        elif position_type == "short":
            return entry_price - (atr * atr_multiplier)
        else:
            raise ValueError("Invalid position type. Must be 'long' or 'short'.")

    def calculate_atr(self, historical_data, period=14):
        high = historical_data['high']
        low = historical_data['low']
        close = historical_data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr.iloc[-1]

    def check_extreme_volatility(self, historical_data):
        returns = historical_data['close'].pct_change()
        volatility = returns.std() * np.sqrt(365)  # Annualized volatility
        return volatility > self.config.VOLATILITY_THRESHOLD

    def check_max_drawdown(self, current_value, peak_value):
        drawdown = (peak_value - current_value) / peak_value
        return drawdown > self.max_drawdown

    def get_market_sentiment(self):
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        try:
            social_volume = san.get(
                "social_volume_total",
                slug="bitcoin",
                from_date=start_date,
                to_date=end_date,
                interval="1d"
            )

            sentiment_balance = san.get(
                "sentiment_balance_total",
                slug="bitcoin",
                from_date=start_date,
                to_date=end_date,
                interval="1d"
            )

            social_volume_score = self.normalize_data(social_volume['value'])
            sentiment_balance_score = self.normalize_data(sentiment_balance['value'])

            composite_score = (social_volume_score * 0.4 + sentiment_balance_score * 0.6)
            return int(composite_score * 100)  # Return score on a 0-100 scale

        except Exception as e:
            print(f"Error fetching sentiment data: {e}")
            return 50  # Return neutral sentiment if there's an error

    def get_on_chain_metrics(self):
        try:
            nvt = self.glassnode.get_metric('nvt', 'BTC')
            sopr = self.glassnode.get_metric('sopr', 'BTC')

            nvt_score = self.normalize_data(nvt['data'][-30:])
            sopr_score = self.normalize_data(sopr['data'][-30:])

            composite_score = (nvt_score * 0.5 + sopr_score * 0.5)
            return composite_score
        except Exception as e:
            print(f"Error fetching on-chain metrics: {e}")
            return 0.5  # Return neutral score if there's an error

    def normalize_data(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def get_funding_rate(self):
        try:
            funding_rate = self.exchange.fetch_funding_rate('BTC/USDT')
            return funding_rate['fundingRate']
        except:
            return 0  # Return 0 if unable to fetch funding rate

    def apply_risk_management(self, signal, portfolio_value, current_price, historical_data):
        order_book = self.fetch_order_book()
        if not self.check_liquidity(order_book):
            return 0, None, None  # No trade due to low liquidity

        if self.check_extreme_volatility(historical_data):
            return 0, None, None  # No trade due to extreme volatility

        position_type = "long" if signal > 0 else "short" if signal < 0 else None
        if position_type is None:
            return 0, None, None  # No trade

        atr = self.calculate_atr(historical_data)
        stop_loss = self.set_dynamic_stop_loss(current_price, position_type, atr)
        take_profit = self.set_dynamic_take_profit(current_price, position_type, atr)

        position_size = self.calculate_position_size(portfolio_value, current_price, stop_loss)

        peak_value = historical_data['close'].max()
        if self.check_max_drawdown(current_price, peak_value):
            return 0, None, None  # No trade due to max drawdown

        sentiment = self.get_market_sentiment()
        if sentiment < self.config.SENTIMENT_THRESHOLD_LOW:
            position_size *= 0.5  # Reduce position size if sentiment is bearish
        elif sentiment > self.config.SENTIMENT_THRESHOLD_HIGH:
            position_size *= 1.2  # Increase position size if sentiment is bullish

        on_chain_score = self.get_on_chain_metrics()
        if on_chain_score < 0.3:
            position_size *= 0.8  # Reduce position size if on-chain metrics are bearish
        elif on_chain_score > 0.7:
            position_size *= 1.1  # Increase position size if on-chain metrics are bullish

        funding_rate = self.get_funding_rate()
        if abs(funding_rate) > self.config.FUNDING_RATE_THRESHOLD:
            if (position_type == "long" and funding_rate > 0) or (position_type == "short" and funding_rate < 0):
                position_size *= 0.9  # Reduce position size if funding rate is unfavorable

        return position_size, stop_loss, take_profit