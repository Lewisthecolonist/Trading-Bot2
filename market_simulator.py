import numpy as np
import pandas as pd
from scipy.stats import skewnorm, t, norm, levy_stable, beta, multivariate_normal
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
from numba import jit
from copulas.multivariate import GaussianMultivariate
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from scipy.stats import gamma
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from collections import deque
import torch
import torch.nn as nn
from pykalman import KalmanFilter
from prophet import Prophet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from scipy.integrate import odeint

class MarketSimulator:
    def __init__(self, config, strategies):
        self.config = config
        self.strategies = strategies
        self.endog_data = np.array([])  # Initialized with an empty array to avoid NoneType issues
        self.rng = np.random.default_rng()
        self.copula = GaussianMultivariate()
        self.price_scaler = MinMaxScaler(feature_range=(1, 1000))
        self.market_graph = self._create_market_graph()
        self.regime_model = None
        self.volatility_model = self._create_volatility_model()
        self.sentiment_model = None  # Updated initialization
        self.order_book = self._create_order_book()
        self.market_maker = self._create_market_maker()
        self.liquidity_pool = self._create_liquidity_pool()
        self.neural_network = self._create_neural_network()
        self.kalman_filter = self._create_kalman_filter()
        self.prophet_model = self._create_prophet_model()
        self.gp_regressor = self._create_gp_regressor()
        
    def generate_market_data(self, days=365):
        # Initial price generation with regime switching
        prices, regimes = self._generate_complex_regime_switching_prices(days)
        returns = np.diff(np.log(prices))
        self.endog_data = returns
        self._update_regime_model()

        # Generate base market factors
        factors = self._generate_correlated_factors(days)
        on_chain_metrics = self._generate_on_chain_metrics(days)
        sentiment = self._generate_sentiment(days)

        # Apply strategy impacts with stress multipliers
        strategy_weights = {
            'trend_following': 0.15,
            'mean_reversion': 0.15,
            'momentum': 0.15,
            'breakout': 0.15,
            'volatility_clustering': 0.15,
            'statistical_arbitrage': 0.15,
            'sentiment_analysis': 0.10
        }
    
        stress_multiplier = np.random.choice([1.5, 2.0, 2.5], size=days)
    
        # Apply weighted strategy impacts
        for strategy in self.strategies.values():
            for pattern in strategy.favored_patterns:
                if pattern in strategy_weights:
                    weight = strategy_weights[pattern]
                    strategy_prices = getattr(self, f'_simulate_{pattern}')(prices, strategy.parameters)
                    prices = prices + (strategy_prices - prices) * weight * stress_multiplier

        # Apply existing market effects
        volumes = self._generate_volume_series(days, prices, regimes)
        volatility = self._generate_volatility_series(days, prices, regimes)
        liquidity = self._generate_liquidity(volumes, prices)
        order_book = self._simulate_order_book(days, prices, volumes, volatility, liquidity)
    
        # Apply sophisticated market dynamics
        prices, volumes = self._apply_network_effects(prices, volumes)
        prices = self._apply_long_term_cycles(prices)
        prices = self._apply_jump_diffusion(prices)
        prices = self._simulate_flash_crash(prices)
        prices = self.price_scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        prices = self._add_microstructure_noise(prices)
    
        # Generate additional market features
        funding_rates = self._simulate_funding_rate(days, prices, volumes)
        exchange_prices = self._simulate_multiple_exchanges(prices)
        prices, volumes = self._apply_time_of_day_effects(prices, volumes)
        prices = self._simulate_news_events(prices, sentiment)
        prices = self._simulate_regulatory_events(prices)
    
        # Apply advanced market mechanics
        order_book = self._simulate_spoofing(order_book)
        liquidity_pool_data = self._simulate_liquidity_pool(days, prices)
    
        # Apply ML predictions and forecasts
        prices = self._apply_neural_network_prediction(prices)
        prices = self._apply_kalman_filter(prices)
        prophet_forecast = self._generate_prophet_forecast(prices)
        gp_forecast = self._generate_gp_forecast(prices)
        lotka_volterra = self._simulate_lotka_volterra(days)
        fractional_brownian = self._generate_fractional_brownian_motion(days)

        # Return the complete market dataset
        return pd.DataFrame({
            'price': prices,
            'volume': volumes,
            'volatility': volatility,
            'regime': regimes,
            'sentiment': sentiment,
            'on_chain_metric': on_chain_metrics[:, 0],
            'funding_rate': funding_rates,
            'order_book_imbalance': self._calculate_order_book_imbalance(order_book),
            'liquidity': liquidity,
            'bid': order_book['bids'][0],
            'ask': order_book['asks'][0],
            'bid_volume': order_book['bid_volumes'][0],
            'ask_volume': order_book['ask_volumes'][0],
            'exchange_1_price': exchange_prices[0],
            'exchange_2_price': exchange_prices[1],
            'exchange_3_price': exchange_prices[2],
            'active_addresses': on_chain_metrics[:, 0],
            'transaction_volume': on_chain_metrics[:, 1],
            'liquidity_pool_reserve_0': liquidity_pool_data['reserve_0'],
            'liquidity_pool_reserve_1': liquidity_pool_data['reserve_1'],
            'liquidity_pool_k': liquidity_pool_data['k'],
            'prophet_forecast': prophet_forecast,
            'gp_forecast': gp_forecast,
            'lotka_volterra': lotka_volterra,
            'fractional_brownian': fractional_brownian
        })

    def _create_market_graph(self):
        G = nx.barabasi_albert_graph(n=100, m=2)
        for (u, v) in G.edges():
            G.edges[u,v]['weight'] = self.rng.uniform(0.1, 1.0)
        return G

    def _create_regime_model(self):
        if self.endog_data.size == 0:  # Check if endog_data is populated
            raise ValueError("endog_data is not defined. Generate market data first.")
        return MarkovRegression(endog=self.endog_data, k_regimes=4, trend='c', switching_variance=True, switching_exog=True)

    def _update_regime_model(self):
        if self.regime_model is None:
            self.regime_model = self._create_regime_model()
        self.regime_model = self.regime_model.fit()

    def _create_volatility_model(self):
        return arch_model(y=None, vol='Garch', p=1, o=1, q=1, dist='skewt')

    def _create_sentiment_model(self):
        if self.endog_data.size == 0:
            raise ValueError("endog_data is not defined. Generate market data first.")
        return ARIMA(order=(1, 1, 1), endog=self.endog_data)

    def _create_order_book(self):
        return {'bids': deque(), 'asks': deque()}

    def _create_market_maker(self):
        return {'inventory': 0, 'cash': 1000000}

    def _create_liquidity_pool(self):
        return {'reserve_0': 1000000, 'reserve_1': 1000000, 'k': 1000000 * 1000000}

    @jit(nopython=True)
    def _generate_complex_regime_switching_prices(self, days):
        e = np.random.normal(0, 1, days)
        beta = np.array([0.002, -0.002, 0.0001, 0.005])
        sigma = np.array([0.01, 0.03, 0.005, 0.02])
        P = np.array([[0.95, 0.02, 0.02, 0.01],
                      [0.03, 0.93, 0.02, 0.02],
                      [0.02, 0.03, 0.94, 0.01],
                      [0.01, 0.02, 0.02, 0.95]])
        regimes = np.zeros(days, dtype=np.int32)
        for t in range(1, days):
            regimes[t] = np.random.choice(4, p=P[regimes[t-1]])
        returns = np.zeros(days)
        h = np.zeros(days)
        omega, alpha, beta_garch = 0.00001, 0.1, 0.8
        for t in range(1, days):
            h[t] = omega + alpha * returns[t-1]**2 + beta_garch * h[t-1]
            returns[t] = beta[regimes[t]] + np.sqrt(h[t]) * sigma[regimes[t]] * e[t]
        prices = 100 * np.exp(np.cumsum(returns))
        return prices, regimes

    def _generate_correlated_factors(self, days):
        data = np.random.normal(0, 1, size=(days, 5))
        self.copula.fit(data)
        factors = self.copula.sample(days)
        factors[:, 0] = skewnorm.ppf(norm.cdf(factors[:, 0]), a=-2, loc=50, scale=10)
        factors[:, 1] = gamma.ppf(norm.cdf(factors[:, 1]), a=2, scale=0.5)
        factors[:, 2] = t.ppf(norm.cdf(factors[:, 2]), df=3, loc=0, scale=0.001)
        factors[:, 3] = levy_stable.ppf(norm.cdf(factors[:, 3]), alpha=1.5, beta=0)
        factors[:, 4] = beta.ppf(norm.cdf(factors[:, 4]), a=2, b=5)
        return factors

    def _generate_on_chain_metrics(self, days):
        active_addresses = np.cumsum(np.random.normal(1000, 100, days))
        transaction_volume = np.exp(np.random.normal(10, 1, days))
        unique_senders = np.cumsum(np.random.poisson(500, days))
        gas_used = np.cumsum(np.random.gamma(2, 1000000, days))
        return np.column_stack((active_addresses, transaction_volume, unique_senders, gas_used))

    def _generate_sentiment(self, days):
        sentiment = np.zeros(days)
        model = self.sentiment_model.fit(np.random.normal(0, 1, 100))
        for i in range(days):
            sentiment[i] = model.simulate(1)[0]
        return np.tanh(sentiment)  # Normalize to [-1, 1]

    def _simulate_trend_following(self, prices, params):
        df = pd.DataFrame({'price': prices})
        short_ma = df['price'].rolling(window=params.get('MOVING_AVERAGE_SHORT', 10)).mean()
        long_ma = df['price'].rolling(window=params.get('MOVING_AVERAGE_LONG', 30)).mean()
        trend_strength = (short_ma - long_ma) / long_ma
        return prices * (1 + trend_strength * params.get('TREND_STRENGTH_THRESHOLD', 0.1))

    def _simulate_mean_reversion(self, prices, params):
        df = pd.DataFrame({'price': prices})
        mean_price = df['price'].rolling(window=params.get('MEAN_WINDOW', 20)).mean()
        deviation = (df['price'] - mean_price) / mean_price
        return prices * (1 - deviation * params.get('MEAN_REVERSION_THRESHOLD', 0.05))

    def _simulate_momentum(self, prices, params):
        df = pd.DataFrame({'price': prices})
        momentum = df['price'].pct_change(periods=params.get('MOMENTUM_PERIOD', 14))
        return prices * (1 + momentum * params.get('MOMENTUM_THRESHOLD', 0.05))

    def _simulate_breakout(self, prices, params):
        df = pd.DataFrame({'price': prices})
        upper_band = df['price'].rolling(window=params.get('BREAKOUT_PERIOD', 20)).max()
        lower_band = df['price'].rolling(window=params.get('BREAKOUT_PERIOD', 20)).min()
        breakout = np.where(prices > upper_band, 1.01, np.where(prices < lower_band, 0.99, 1))
        return prices * breakout

    def _simulate_volatility_clustering(self, prices, params):
        df = pd.DataFrame({'price': prices})
        volatility = df['price'].pct_change().rolling(params.get('VOLATILITY_WINDOW', 20)).std()
        vol_ratio = volatility / volatility.rolling(100).mean()
        return prices * np.where(vol_ratio > params.get('HIGH_VOLATILITY_THRESHOLD', 1.5), 1.1, 0.9)

    def _simulate_statistical_arbitrage(self, prices, params):
        asset2 = prices * (1 + np.random.normal(0, 0.01, len(prices)))
        df = pd.DataFrame({'asset1': prices, 'asset2': asset2})
        spread = df['asset1'] - df['asset2']
        z_score = (spread - spread.rolling(params.get('LOOKBACK_PERIOD', 20)).mean()) / \
        spread.rolling(params.get('LOOKBACK_PERIOD', 20)).std()
        return prices * (1 + z_score / params.get('Z_SCORE_THRESHOLD', 2))

    def _simulate_sentiment_based(self, prices, params):
        sentiment_impact = self._generate_sentiment(len(prices)) * params.get('SENTIMENT_IMPACT_WEIGHT', 0.3)
        return prices * (1 + sentiment_impact)

    def _apply_strategy_patterns(self, prices, regimes):
        original_prices = prices.copy()
    
        for strategy in self.strategies.values():
            if 'trend_following' in strategy.favored_patterns:
                prices = self._simulate_trend_following(prices, strategy.parameters)
            elif 'mean_reversion' in strategy.favored_patterns:
                prices = self._simulate_mean_reversion(prices, strategy.parameters)
            elif 'momentum' in strategy.favored_patterns:
                prices = self._simulate_momentum(prices, strategy.parameters)
            elif 'breakout' in strategy.favored_patterns:
                prices = self._simulate_breakout(prices, strategy.parameters)
            elif 'volatility_clustering' in strategy.favored_patterns:
                prices = self._simulate_volatility_clustering(prices, strategy.parameters)
            elif 'statistical_arbitrage' in strategy.favored_patterns:
                prices = self._simulate_statistical_arbitrage(prices, strategy.parameters)
            elif 'sentiment_analysis' in strategy.favored_patterns:
                prices = self._simulate_sentiment_based(prices, strategy.parameters)
    
        return prices

    @jit(nopython=True)
    def _generate_volume_series(self, days, prices, regimes):
        base_volume = np.random.lognormal(mean=np.log(1e6), sigma=0.5, size=days)
        regime_factor = np.where(regimes == 1, 1.5, np.where(regimes == 0, 1.2, 0.8))
        price_changes = np.abs(np.diff(np.log(prices), prepend=0))
        volume_factor = np.exp(price_changes * 10)
        return base_volume * regime_factor * volume_factor

    def _generate_volatility_series(self, days, prices, regimes):
        returns = np.diff(np.log(prices))
        model = self.volatility_model.fit(returns)
        forecasts = model.forecast(horizon=days)
        return np.sqrt(forecasts.variance.values[-1, :])

    @jit(nopython=True)
    def _generate_liquidity(self, volumes, prices):
        base_liquidity = volumes * prices
        noise = np.random.normal(1, 0.1, len(volumes))
        return base_liquidity * noise

    def _simulate_order_book(self, days, prices, volumes, volatility, liquidity):
        spreads = np.maximum(0.001, np.random.normal(0.002, 0.001, size=days)) * prices
        bids = prices - spreads / 2
        asks = prices + spreads / 2
        
        depth_levels = 10
        bid_depths = [bids * (1 - i * 0.001) for i in range(depth_levels)]
        ask_depths = [asks * (1 + i * 0.001) for i in range(depth_levels)]
        
        bid_volumes = volumes[:, np.newaxis] * np.random.beta(2, 2, size=(days, depth_levels))
        ask_volumes = volumes[:, np.newaxis] * np.random.beta(2, 2, size=(days, depth_levels))
        
        volume_adjust = np.exp(volatility * 10) * (liquidity / np.mean(liquidity))
        bid_volumes *= volume_adjust[:, np.newaxis]
        ask_volumes *= volume_adjust[:, np.newaxis]
        
        return {
            'bids': bid_depths,
            'asks': ask_depths,
            'bid_volumes': bid_volumes.T,
            'ask_volumes': ask_volumes.T
        }

    def _apply_network_effects(self, prices, volumes):
        for node in self.market_graph.nodes():
            if self.rng.random() < 0.01:
                impact = self.rng.normal(0, 0.02)
                neighbors = list(self.market_graph.neighbors(node))
                for neighbor in neighbors:
                    start = neighbor * (len(prices) // 100)
                    end = (neighbor + 1) * (len(prices) // 100)
                    edge_weight = self.market_graph.edges[node, neighbor]['weight']
                    prices[start:end] *= (1 + impact * edge_weight)
                    volumes[start:end] *= (1 + abs(impact) * edge_weight)
        return prices, volumes

    @jit(nopython=True)
    def _apply_long_term_cycles(self, prices):
        cycle_length = 4 * 365
        cycle = np.sin(np.linspace(0, 2*np.pi, cycle_length))
        cycle = np.tile(cycle, len(prices)//cycle_length + 1)[:len(prices)]
        prices *= (1 + 0.2 * cycle)
        return prices

    def _apply_jump_diffusion(self, prices):
        jump_times = self.rng.poisson(lam=5, size=len(prices))
        jump_sizes = self.rng.normal(0, 0.05, size=len(prices))
        prices *= np.exp(jump_times * jump_sizes)
        return prices

    def _add_microstructure_noise(self, prices):
        noise = self.rng.normal(0, 0.0001, size=len(prices))
        return prices * (1 + noise)

    def _simulate_flash_crash(self, prices, probability=0.001, crash_severity=0.1):
        crash_mask = np.random.random(len(prices)) < probability
        crash_factors = np.where(crash_mask, 1 - crash_severity, 1)
        return prices * crash_factors

    def _simulate_multiple_exchanges(self, base_prices, num_exchanges=3):
        exchange_prices = []
        for _ in range(num_exchanges):
            noise = np.random.normal(1, 0.001, len(base_prices))
            exchange_prices.append(base_prices * noise)
        return np.array(exchange_prices)

    def _simulate_spoofing(self, order_book, spoof_probability=0.01, spoof_size_factor=10):
        if np.random.random() < spoof_probability:
            spoof_side = np.random.choice(['bid', 'ask'])
            spoof_level = np.random.randint(0, len(order_book[f'{spoof_side}s']))
            order_book[f'{spoof_side}_volumes'][spoof_level] *= spoof_size_factor
        return order_book

    def _simulate_funding_rate(self, days, prices, volumes):
        base_rate = 0.0001
        price_volatility = np.std(np.diff(np.log(prices)))
        volume_factor = np.log(volumes) / np.mean(np.log(volumes))
        funding_rates = np.cumsum(np.random.normal(base_rate, price_volatility * 0.1, days))
        return funding_rates * volume_factor

    def _apply_time_of_day_effects(self, prices, volumes):
        hours = np.arange(len(prices)) % 24
        asian_session = (hours >= 1) & (hours < 9)
        european_session = (hours >= 8) & (hours < 16)
        us_session = (hours >= 14) & (hours < 22)
        
        session_factors = np.ones(len(prices))
        session_factors[asian_session] *= 1.1
        session_factors[european_session] *= 1.2
        session_factors[us_session] *= 1.3
        
        return prices * session_factors, volumes * session_factors

    def _simulate_news_events(self, prices, sentiment, probability=0.01, impact_range=(-0.05, 0.05)):
        news_mask = np.random.random(len(prices)) < probability
        news_impacts = np.random.uniform(*impact_range, size=len(prices))
        sentiment_factor = (sentiment + 1) / 2  # Normalize sentiment to [0, 1]
        return prices * (1 + news_mask * news_impacts * sentiment_factor)

    def _simulate_regulatory_events(self, prices, probability=0.001, impact_range=(-0.1, 0.1)):
        regulatory_mask = np.random.random(len(prices)) < probability
        regulatory_impacts = np.random.uniform(*impact_range, size=len(prices))
        return prices * (1 + regulatory_mask * regulatory_impacts)

    def _create_neural_network(self):
        model = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        return model

    def _create_kalman_filter(self):
        return KalmanFilter(transition_matrices=[1],
                            observation_matrices=[1],
                            initial_state_mean=0,
                            initial_state_covariance=1,
                            observation_covariance=1,
                            transition_covariance=.01)

    def _create_prophet_model(self):
        return Prophet()

    def _create_gp_regressor(self):
        kernel = RBF() + WhiteKernel() + Matern(nu=1.5)
        return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    def _apply_neural_network_prediction(self, prices):
        input_data = torch.tensor(prices[-10:]).float().unsqueeze(0)
        with torch.no_grad():
            prediction = self.neural_network(input_data).item()
        return np.append(prices, prediction)

    def _apply_kalman_filter(self, prices):
        filtered_state_means, _ = self.kalman_filter.filter(prices)
        return filtered_state_means.flatten()

    def _generate_prophet_forecast(self, prices):
        df = pd.DataFrame({'ds': pd.date_range(start='2020-01-01', periods=len(prices)), 'y': prices})
        self.prophet_model.fit(df)
        future = self.prophet_model.make_future_dataframe(periods=30)
        forecast = self.prophet_model.predict(future)
        return forecast['yhat'].values[-30:]

    def _generate_gp_forecast(self, prices):
        X = np.arange(len(prices)).reshape(-1, 1)
        self.gp_regressor.fit(X, prices)
        X_pred = np.arange(len(prices), len(prices) + 30).reshape(-1, 1)
        y_pred, _ = self.gp_regressor.predict(X_pred, return_std=True)
        return y_pred

    def _simulate_lotka_volterra(self, days):
        def lotka_volterra(X, t, a, b, c, d):
            x, y = X
            dxdt = a*x - b*x*y
            dydt = -c*y + d*x*y
            return [dxdt, dydt]

        t = np.linspace(0, days, days)
        X0 = [1, 1]
        a, b, c, d = 1, 0.1, 1.5, 0.75
        solution = odeint(lotka_volterra, X0, t, args=(a, b, c, d))
        return solution[:, 0]  # Return prey population

    def _generate_fractional_brownian_motion(self, days, H=0.7):
        def fbm(n, H):
            r = np.zeros(n)
            r[0] = np.random.randn()
            for i in range(1, n):
                r[i] = sum([np.random.randn() * (j+1)**(H-0.5) - j**(H-0.5) for j in range(i)])
            return r

        return fbm(days, H)

    def _simulate_liquidity_pool(self, days, prices):
        reserve_0 = np.zeros(days)
        reserve_1 = np.zeros(days)
        k = np.zeros(days)
        
        reserve_0[0] = self.liquidity_pool['reserve_0']
        reserve_1[0] = self.liquidity_pool['reserve_1']
        k[0] = self.liquidity_pool['k']
        
        for i in range(1, days):
            swap_amount = np.random.normal(0, 0.01) * reserve_0[i-1]
            if swap_amount > 0:
                reserve_0[i] = reserve_0[i-1] + swap_amount
                reserve_1[i] = k[i-1] / reserve_0[i]
            else:
                reserve_1[i] = reserve_1[i-1] - abs(swap_amount)
                reserve_0[i] = k[i-1] / reserve_1[i]
            k[i] = reserve_0[i] * reserve_1[i]
        
        return {'reserve_0': reserve_0, 'reserve_1': reserve_1, 'k': k}

    def _calculate_order_book_imbalance(self, order_book):
        total_bid_volume = np.sum(order_book['bid_volumes'])
        total_ask_volume = np.sum(order_book['ask_volumes'])
        return (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)

    def _calculate_transaction_cost(self, shares, price, volume, volatility, liquidity):
        base_cost = 0.001 * shares * price
        slippage = 0.1 * (shares / volume) * price * volatility
        liquidity_cost = 0.01 * (shares * price) / liquidity
        market_impact = self._calculate_market_impact(shares, price, volume)
        return base_cost + slippage + liquidity_cost + market_impact

    def _calculate_market_impact(self, shares, price, volume):
        impact_factor = 0.1 * (shares / volume) ** 0.5
        return impact_factor * price

    def _apply_execution_delay(self, order_time, max_delay=5):
        return order_time + np.random.randint(0, max_delay)

    def run_simulation(self, strategy):
        market_data = self.generate_market_data(365)  # Generate a year of data
        portfolio_value = strategy.get_capital()
        positions = []
        daily_returns = []
        trade_history = []

        for i in range(len(market_data)):
            signal = strategy.generate_signal(market_data.iloc[i])
            if signal != 0:
                trade_size = abs(signal) * portfolio_value * strategy.parameters['MAX_POSITION_SIZE']
                shares = trade_size / market_data['price'].iloc[i]
                execution_time = self._apply_execution_delay(i)
                
                if execution_time < len(market_data):
                    execution_price = market_data['price'].iloc[execution_time]
                    transaction_cost = self._calculate_transaction_cost(
                        shares, 
                        execution_price, 
                        market_data['volume'].iloc[execution_time], 
                        market_data['volatility'].iloc[execution_time],
                        market_data['liquidity'].iloc[execution_time]
                    )
                    
                    if signal > 0:
                        positions.append({'shares': shares, 'entry_price': execution_price})
                        portfolio_value -= trade_size + transaction_cost
                        trade_history.append(('buy', market_data.index[execution_time], shares, execution_price))
                    elif signal < 0 and positions:
                        for position in positions:
                            portfolio_value += position['shares'] * execution_price - transaction_cost
                            trade_history.append(('sell', market_data.index[execution_time], position['shares'], execution_price))
                        positions = []

            daily_value = portfolio_value + sum(position['shares'] * market_data['price'].iloc[i] for position in positions)
            daily_returns.append((daily_value / strategy.get_capital()) - 1)

        return self._calculate_performance_metrics(strategy.get_capital(), portfolio_value, daily_returns, trade_history)

    def _calculate_performance_metrics(self, initial_capital, final_value, daily_returns, trade_history):
        total_return = (final_value - initial_capital) / initial_capital
        sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
        max_drawdown = np.min(np.minimum.accumulate(daily_returns))
        volatility = np.std(daily_returns) * np.sqrt(252)
        sortino_ratio = np.sqrt(252) * np.mean(daily_returns) / np.std([r for r in daily_returns if r < 0])
        win_rate = sum(1 for r in daily_returns if r > 0) / len(daily_returns)

        total_trades = len(trade_history)
        profitable_trades = sum(1 for trade in trade_history if trade[0] == 'sell' and trade[3] > trade[1])
        profit_factor = sum(trade[3] - trade[1] for trade in trade_history if trade[0] == 'sell' and trade[3] > trade[1]) / \
                        abs(sum(trade[3] - trade[1] for trade in trade_history if trade[0] == 'sell' and trade[3] <= trade[1]))

        calmar_ratio = self._calculate_calmar_ratio(daily_returns)
        omega_ratio = self._calculate_omega_ratio(daily_returns)
        tail_ratio = self._calculate_tail_ratio(daily_returns)

        return {
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio,
            'omega_ratio': omega_ratio,
            'tail_ratio': tail_ratio,
            'trade_history': trade_history
        }

    def _calculate_calmar_ratio(self, returns, periods=252):
        max_drawdown = self._calculate_max_drawdown(returns)
        if max_drawdown == 0:
            return np.nan
        return (np.mean(returns) * periods) / abs(max_drawdown)

    def _calculate_omega_ratio(self, returns, threshold=0):
        return np.sum(np.maximum(returns - threshold, 0)) / np.sum(np.maximum(threshold - returns, 0))

    def _calculate_tail_ratio(self, returns):
        return abs(np.percentile(returns, 95)) / abs(np.percentile(returns, 5))

    def _calculate_max_drawdown(self, returns):
        wealth_index = (1 + returns).cumprod()
        previous_peaks = np.maximum.accumulate(wealth_index)
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        return drawdowns.min()

    def optimize_strategy(self, strategy, param_ranges):
        def objective(params):
            for i, param in enumerate(strategy.parameters):
                strategy.parameters[param] = params[i]
            results = self.run_simulation(strategy)
            return -results['sharpe_ratio']  # Minimize negative Sharpe ratio

        bounds = [param_ranges[param] for param in strategy.parameters]
        result = minimize(objective, x0=[np.mean(b) for b in bounds], bounds=bounds, method='L-BFGS-B')
        
        for i, param in enumerate(strategy.parameters):
            strategy.parameters[param] = result.x[i]
        
        return strategy, -result.fun  # Return optimized strategy and its Sharpe ratio

    def cluster_market_regimes(self, n_clusters=3):
        market_data = self.generate_market_data(None)  # Generate some sample data
        features = market_data[['volatility', 'volume', 'sentiment', 'on_chain_metric']].values
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
        return clusters
    
    def _apply_timeframe_effects(self, prices, timeframe):
        TimeFrame = timeframe
        effects = {
            TimeFrame.SHORT_TERM: {
                'volatility': 0.002,
                'mean_reversion': 0.3,
                'trend_strength': 0.1
            },
            TimeFrame.MID_TERM: {
                'volatility': 0.005,
                'mean_reversion': 0.5,
                'trend_strength': 0.3
            },
            TimeFrame.LONG_TERM: {
                'volatility': 0.01,
                'mean_reversion': 0.7,
                'trend_strength': 0.6
            },
            TimeFrame.SEASONAL_TERM: {
                'volatility': 0.015,
                'mean_reversion': 0.9,
                'trend_strength': 0.8
            }
        }
