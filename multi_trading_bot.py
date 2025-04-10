import time
import datetime
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import pytz
from datetime import datetime, timedelta
import logging
import talib
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TradingBot")

class TradingBot:
    def __init__(self, symbols=["USDJPY", "EURUSD"], risk_percent=0.02, 
                 sl_pips_default=30, tp_multiplier_default=2, fast_period=12, slow_period=26, 
                 signal_period=9, rsi_period=14, rsi_overbought=70, rsi_oversold=30):
        """
        Initialize the trading bot with parameters
        
        Args:
            symbols (list): List of forex pairs to trade
            risk_percent (float): Percentage of account to risk per trade
            sl_pips_default (int): Default stop loss in pips
            tp_multiplier_default (float): Default multiplier for take profit (relative to stop loss)
            fast_period (int): Fast EMA period for MACD
            slow_period (int): Slow EMA period for MACD
            signal_period (int): Signal line period for MACD
            rsi_period (int): RSI period
            rsi_overbought (int): RSI overbought threshold
            rsi_oversold (int): RSI oversold threshold
        """
        self.symbols = symbols
        self.risk_percent = risk_percent
        self.sl_pips_default = sl_pips_default
        self.tp_multiplier_default = tp_multiplier_default
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.timeframes = {
            'long_term': mt5.TIMEFRAME_H4,  # 4 hours for long-term trend
            'medium_term': mt5.TIMEFRAME_H1,  # 1 hour for medium-term trend
            'short_term': mt5.TIMEFRAME_M15,  # 15 minutes for short-term trend
            'execution': mt5.TIMEFRAME_M2   # 2 minutes for trade execution
        }
        self.atr_period = 14
        self.adx_period = 14
        self.adx_threshold = 25  # ADX above this value indicates a trend
        self.volatility_threshold = 1.5  # Multiplier for average ATR to detect high volatility
        self.max_spread_pips = 5.0  # Maximum spread allowed for trading
        
        # Cache for historical data and indicators
        self.data_cache = {}
        self.last_update_time = {}
        
    def initialize_mt5(self):
        """Initialize connection to MetaTrader 5"""
        if not mt5.initialize():
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        
        logger.info(f"MT5 initialized successfully. Version: {mt5.version()}")
        
        # Check if symbols exist
        for symbol in self.symbols:
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Symbol {symbol} not found")
                return False
        
        return True
    
    def get_account_info(self):
        """Get account info and balance"""
        account_info = mt5.account_info()
        if account_info is None:
            logger.error(f"Failed to get account info: {mt5.last_error()}")
            return None
        
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'leverage': account_info.leverage
        }
    
    def fetch_historical_data(self, symbol, timeframe, bars=1000):
        """
        Fetch historical OHLCV data from MT5
        
        Args:
            symbol (str): The forex pair
            timeframe (int): MT5 timeframe constant
            bars (int): Number of candles to fetch
            
        Returns:
            pandas.DataFrame: DataFrame with OHLCV data
        """
        # Define timezone
        timezone = pytz.timezone("UTC")
        
        # Calculate the end time (current time)
        to_date = datetime.now(timezone)
        
        # Fetch data
        rates = mt5.copy_rates_from(symbol, timeframe, to_date, bars)
        
        if rates is None or len(rates) == 0:
            logger.error(f"Failed to fetch data for {symbol}: {mt5.last_error()}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        
        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Set time as index
        df.set_index('time', inplace=True)
        
        return df
    
    def calculate_indicators_streaming(self, df):
        """
        Calculate technical indicators using TA-Lib streaming functions
        for improved performance
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
                
        Returns:
            pandas.DataFrame: DataFrame with added indicators
        """
        # Get input arrays once to avoid repeated lookups
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['tick_volume'].values
        
        # Use TA-Lib streaming functions for better performance
        
        # MACD - streaming version
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.stream.MACD(
            close, 
            fastperiod=self.fast_period, 
            slowperiod=self.slow_period, 
            signalperiod=self.signal_period
        )
        
        # RSI - streaming version
        df['rsi'] = talib.stream.RSI(close, timeperiod=self.rsi_period)
        
        # Moving Averages - streaming versions
        df['ema_fast'] = talib.stream.EMA(close, timeperiod=self.fast_period)
        df['ema_slow'] = talib.stream.EMA(close, timeperiod=self.slow_period)
        df['sma_50'] = talib.stream.SMA(close, timeperiod=50)
        df['sma_200'] = talib.stream.SMA(close, timeperiod=200)
        
        # Bollinger Bands - streaming version
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.stream.BBANDS(
            close, timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # ATR for volatility - streaming version
        df['atr'] = talib.stream.ATR(high, low, close, timeperiod=self.atr_period)
        
        # ADX and Directional Movement - streaming versions
        adx_period = self.adx_period
        df['adx'] = talib.stream.ADX(high, low, close, timeperiod=adx_period)
        df['plus_di'] = talib.stream.PLUS_DI(high, low, close, timeperiod=adx_period)
        df['minus_di'] = talib.stream.MINUS_DI(high, low, close, timeperiod=adx_period)
        
        # Stochastic - streaming version
        df['slowk'], df['slowd'] = talib.stream.STOCH(
            high, low, close, 
            fastk_period=14, slowk_period=3, slowk_matype=0, 
            slowd_period=3, slowd_matype=0
        )
        
        # CCI (Commodity Channel Index) - streaming version
        df['cci'] = talib.stream.CCI(high, low, close, timeperiod=14)
        
        
        return df
    
    def update_indicators_incremental(self, symbol, new_data):
        """
        Update indicators incrementally using TA-Lib streaming functions
        
        Args:
            symbol (str): The forex pair
            new_data (dict): New price data by timeframe
                
        Returns:
            dict: Updated dataframes with indicators
        """
        updated_dfs = {}
        
        # Process each timeframe
        for timeframe in ['long_term', 'medium_term', 'short_term', 'execution']:
            if timeframe not in new_data or new_data[timeframe] is None:
                continue
                
            # Get the existing dataframe from cache
            existing_df = self.data_cache[symbol][timeframe]
            
            if existing_df is None:
                # If no existing data, calculate all indicators
                updated_df = self.calculate_indicators_streaming(new_data[timeframe])
            else:
                # If we have existing data, append the new data
                updated_df = pd.concat([existing_df, new_data[timeframe]])
                
                # Remove any duplicates (by index)
                updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
                
                # Sort by index (time)
                updated_df = updated_df.sort_index()
                
                # Use streaming functions to update only the new data points
                # Get the arrays for the full dataset
                close = updated_df['close'].values
                high = updated_df['high'].values
                low = updated_df['low'].values
                volume = updated_df['tick_volume'].values
                
                # Calculate indicators using streaming functions
                # This will be much faster as it reuses the state from previous calculations
                updated_df['macd'], updated_df['macd_signal'], updated_df['macd_hist'] = talib.stream.MACD(
                    close, 
                    fastperiod=self.fast_period, 
                    slowperiod=self.slow_period, 
                    signalperiod=self.signal_period
                )
                
                updated_df['rsi'] = talib.stream.RSI(close, timeperiod=self.rsi_period)
                updated_df['ema_fast'] = talib.stream.EMA(close, timeperiod=self.fast_period)
                updated_df['ema_slow'] = talib.stream.EMA(close, timeperiod=self.slow_period)
                updated_df['sma_50'] = talib.stream.SMA(close, timeperiod=50)
                updated_df['sma_200'] = talib.stream.SMA(close, timeperiod=200)
                
                updated_df['bb_upper'], updated_df['bb_middle'], updated_df['bb_lower'] = talib.stream.BBANDS(
                    close, timeperiod=20, nbdevup=2, nbdevdn=2
                )
                
                updated_df['atr'] = talib.stream.ATR(high, low, close, timeperiod=self.atr_period)
                updated_df['adx'] = talib.stream.ADX(high, low, close, timeperiod=self.adx_period)
                updated_df['plus_di'] = talib.stream.PLUS_DI(high, low, close, timeperiod=self.adx_period)
                updated_df['minus_di'] = talib.stream.MINUS_DI(high, low, close, timeperiod=self.adx_period)
                
                updated_df['slowk'], updated_df['slowd'] = talib.stream.STOCH(
                    high, low, close, 
                    fastk_period=14, slowk_period=3, slowk_matype=0, 
                    slowd_period=3, slowd_matype=0
                )
                
                updated_df['cci'] = talib.stream.CCI(high, low, close, timeperiod=14)
            
            # Store the updated dataframe
            updated_dfs[timeframe] = updated_df
        
        return updated_dfs

    def get_market_data(self, symbol, force_update=False):
        """
        Efficiently get market data with caching and incremental updates
        using TA-Lib streaming functions
        
        Args:
            symbol (str): The forex pair
            force_update (bool): Force update data regardless of cache
                
        Returns:
            tuple: (df_long, df_medium, df_short, df_execution)
        """
        current_time = time.time()
        
        # Initialize cache entry if doesn't exist
        if symbol not in self.data_cache:
            self.data_cache[symbol] = {
                'long_term': None, 
                'medium_term': None, 
                'short_term': None, 
                'execution': None
            }
            self.last_update_time[symbol] = 0
        
        # Define update intervals based on timeframe (in seconds)
        update_intervals = {
            'long_term': 14400,    # 4 hours
            'medium_term': 3600,   # 1 hour
            'short_term': 900,     # 15 minutes
            'execution': 120       # 2 minutes
        }
        
        # Check if we need to update data
        need_update = force_update or (current_time - self.last_update_time[symbol] > update_intervals['execution'])
        
        if need_update:
            logger.debug(f"Updating market data for {symbol}")
            
            # Determine if we need a full refresh or just incremental updates
            is_initial_load = self.last_update_time[symbol] == 0
            
            if is_initial_load:
                # Initial load - fetch all data
                with ThreadPoolExecutor(max_workers=4) as executor:
                    # Submit fetch tasks
                    futures = {
                        'long_term': executor.submit(self.fetch_historical_data, symbol, self.timeframes['long_term']),
                        'medium_term': executor.submit(self.fetch_historical_data, symbol, self.timeframes['medium_term']),
                        'short_term': executor.submit(self.fetch_historical_data, symbol, self.timeframes['short_term']),
                        'execution': executor.submit(self.fetch_historical_data, symbol, self.timeframes['execution'])
                    }
                    
                    # Get results
                    new_data = {}
                    for timeframe, future in futures.items():
                        try:
                            result = future.result()
                            if result is not None:
                                new_data[timeframe] = result
                        except Exception as e:
                            logger.error(f"Error fetching {timeframe} data for {symbol}: {e}")
                
                # Calculate initial indicators using streaming functions
                updated_dfs = {}
                for timeframe, df in new_data.items():
                    if df is not None:
                        updated_dfs[timeframe] = self.calculate_indicators_streaming(df)
            else:
                # Incremental update - fetch only new bars
                # Determine how many new bars to fetch for each timeframe
                bars_to_fetch = {
                    'long_term': 5,      # Fetch 5 latest bars for long-term
                    'medium_term': 10,    # Fetch 10 latest bars for medium-term
                    'short_term': 20,     # Fetch 20 latest bars for short-term
                    'execution': 30       # Fetch 30 latest bars for execution timeframe
                }
                
                with ThreadPoolExecutor(max_workers=4) as executor:
                    # Submit fetch tasks for incremental updates
                    futures = {}
                    for timeframe, tf_constant in self.timeframes.items():
                        futures[timeframe] = executor.submit(
                            self.fetch_historical_data, 
                            symbol, 
                            tf_constant,
                            bars=bars_to_fetch[timeframe]
                        )
                    
                    # Get results
                    new_data = {}
                    for timeframe, future in futures.items():
                        try:
                            result = future.result()
                            if result is not None:
                                new_data[timeframe] = result
                        except Exception as e:
                            logger.error(f"Error fetching {timeframe} data for {symbol}: {e}")
                
                # Update indicators incrementally
                updated_dfs = self.update_indicators_incremental(symbol, new_data)
            
            # Update cache with new data
            for timeframe, df in updated_dfs.items():
                self.data_cache[symbol][timeframe] = df
            
            # Update timestamp
            self.last_update_time[symbol] = current_time
        
        # Return the data
        return (
            self.data_cache[symbol]['long_term'],
            self.data_cache[symbol]['medium_term'],
            self.data_cache[symbol]['short_term'],
            self.data_cache[symbol]['execution']
        )
    
    def analyze_market_condition(self, df_long, df_medium, df_short):
        """
        Analyze market conditions using multiple timeframes
        
        Args:
            df_long (pandas.DataFrame): Long-term data with indicators
            df_medium (pandas.DataFrame): Medium-term data with indicators
            df_short (pandas.DataFrame): Short-term data with indicators
            
        Returns:
            dict: Market condition analysis
        """
        # Get latest values - use iloc only once per DataFrame to improve performance
        long_latest = df_long.iloc[-1]
        medium_latest = df_medium.iloc[-1]
        short_latest = df_short.iloc[-1]
        
        # Extract values from the latest rows
        long_close = long_latest['close']
        long_sma50 = long_latest['sma_50']
        long_sma200 = long_latest['sma_200']
        long_adx = long_latest['adx']
        long_plus_di = long_latest['plus_di']
        long_minus_di = long_latest['minus_di']
        long_atr = long_latest['atr']
        long_cci = long_latest['cci']
        
        medium_close = medium_latest['close']
        medium_sma50 = medium_latest['sma_50']
        medium_adx = medium_latest['adx']
        medium_plus_di = medium_latest['plus_di']
        medium_minus_di = medium_latest['minus_di']
        
        short_close = short_latest['close']
        short_adx = short_latest['adx']
        short_plus_di = short_latest['plus_di']
        short_minus_di = short_latest['minus_di']
        short_rsi = short_latest['rsi']
        short_cci = short_latest['cci']
        
        # Calculate ATR average only once
        long_atr_avg = df_long['atr'].mean()
        
        # Check for divergence between price and momentum indicators
        # RSI divergence (short term)
        price_direction = 1 if short_close > df_short['close'].iloc[-5] else -1
        rsi_direction = 1 if short_rsi > df_short['rsi'].iloc[-5] else -1
        rsi_divergence = price_direction != rsi_direction
        
        # Analyze long-term trend
        if long_close > long_sma200 and long_sma50 > long_sma200:
            long_trend = "bullish"
        elif long_close < long_sma200 and long_sma50 < long_sma200:
            long_trend = "bearish"
        else:
            # Check using ADX and DI for trend determination
            if long_adx > self.adx_threshold:
                if long_plus_di > long_minus_di:
                    long_trend = "bullish"
                else:
                    long_trend = "bearish"
            else:
                long_trend = "ranging"
        
        # Analyze medium-term trend
        if medium_close > medium_sma50:
            if medium_adx > self.adx_threshold and medium_plus_di > medium_minus_di:
                medium_trend = "bullish"
            elif medium_adx > self.adx_threshold and medium_plus_di < medium_minus_di:
                medium_trend = "bearish"
            else:
                medium_trend = "neutral"
        else:
            if medium_adx > self.adx_threshold and medium_plus_di < medium_minus_di:
                medium_trend = "bearish"
            elif medium_adx > self.adx_threshold and medium_plus_di > medium_minus_di:
                medium_trend = "bullish"
            else:
                medium_trend = "neutral"
        
        # Analyze short-term trend
        if short_adx > self.adx_threshold:
            if short_plus_di > short_minus_di:
                short_trend = "bullish"
            else:
                short_trend = "bearish"
        else:
            if short_rsi > 60:
                short_trend = "bullish"
            elif short_rsi < 40:
                short_trend = "bearish"
            else:
                short_trend = "ranging"
        
        # Detect strong trends
        strong_uptrend = (
            long_trend == "bullish" and
            medium_trend == "bullish" and
            short_trend == "bullish" and
            long_adx > 30 and
            long_plus_di > long_minus_di * 1.5
        )
        
        strong_downtrend = (
            long_trend == "bearish" and
            medium_trend == "bearish" and
            short_trend == "bearish" and
            long_adx > 30 and
            long_minus_di > long_plus_di * 1.5
        )
        
        # Check for sideways/ranging market
        sideways = (
            long_adx < 20 and
            abs(long_close - long_sma50) / long_sma50 < 0.01 and
            df_long['bb_upper'].iloc[-1] - df_long['bb_lower'].iloc[-1] < long_atr * 3
        )
        
        # Check for high volatility
        high_volatility = long_atr > long_atr_avg * self.volatility_threshold
        
        # Analyze for potential reversals
        oversold = short_rsi < self.rsi_oversold and short_cci < -100
        overbought = short_rsi > self.rsi_overbought and short_cci > 100
        
        return {
            'long_term_trend': long_trend,
            'medium_term_trend': medium_trend,
            'short_term_trend': short_trend,
            'strong_uptrend': strong_uptrend,
            'strong_downtrend': strong_downtrend,
            'sideways': sideways,
            'high_volatility': high_volatility,
            'rsi_divergence': rsi_divergence,
            'oversold': oversold,
            'overbought': overbought,
            'adx': long_adx  # Overall trend strength
        }
    
    def determine_trade_parameters(self, market_condition, symbol):
        """
        Determine optimal stop loss and take profit based on market conditions
        
        Args:
            market_condition (dict): Market condition analysis
            symbol (str): The forex pair
            
        Returns:
            tuple: (sl_pips, tp_pips, trailing_stop)
        """
        # Get data from cache rather than fetching again
        _, _, _, df_execution = self.get_market_data(symbol)
        
        # Get ATR for dynamic position sizing
        current_atr = df_execution['atr'].iloc[-1]
        
        # Base values
        sl_pips = self.sl_pips_default
        tp_multiplier = self.tp_multiplier_default
        trailing_stop = False
        
        # Adjust based on market conditions
        if market_condition['high_volatility']:
            # In high volatility, widen stops and use ATR for dynamic sizing
            atr_multiplier = 1.5
            sl_pips = max(self.sl_pips_default, int(current_atr * atr_multiplier * 100))
            tp_multiplier = 2.0  # Higher reward potential in volatile markets
            trailing_stop = True  # Use trailing stop in volatile markets
        
        elif market_condition['strong_uptrend'] or market_condition['strong_downtrend']:
            # In strong trends, use tighter stops and higher take profits
            sl_pips = int(self.sl_pips_default * 0.8)  # Tighter stop loss
            tp_multiplier = 2.5  # Higher profit target in trends
            trailing_stop = True  # Use trailing stop in strong trends
        
        elif market_condition['sideways']:
            # In sideways markets, use wider stops but smaller targets
            sl_pips = int(self.sl_pips_default * 1.2)  # Wider stop loss
            tp_multiplier = 1.5  # Lower profit target in ranging markets
            trailing_stop = False  # No trailing stop in sideways markets
        
        # Calculate take profit in pips
        tp_pips = int(sl_pips * tp_multiplier)
        
        return sl_pips, tp_pips, trailing_stop
    
    def calculate_position_size(self, symbol, stop_loss_pips):
        """
        Calculate position size based on risk percentage
        
        Args:
            symbol (str): The forex pair
            stop_loss_pips (float): Stop loss in pips
            
        Returns:
            float: Position size in lots
        """
        account_info = self.get_account_info()
        if account_info is None:
            return 0.01  # Default minimum
        
        balance = account_info['balance']
        risk_amount = balance * self.risk_percent
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {symbol}")
            return 0.01
        
        # Calculate pip value
        contract_size = symbol_info.trade_contract_size
        point_value = symbol_info.point
        
        # Convert pips to points (usually 1 pip = 10 points for most pairs)
        points = stop_loss_pips * 10
        
        # For JPY pairs, 1 pip is often 0.01
        if 'JPY' in symbol:
            points = stop_loss_pips * 100
        
        # Calculate pip value
        pip_value = (point_value * contract_size) / point_value
        
        # Calculate position size in standard lots (100,000 units)
        position_size = risk_amount / (points * pip_value)
        
        # Convert to MT5 lot size and ensure it's within limits
        symbol_info = mt5.symbol_info(symbol)
        min_lot = symbol_info.volume_min
        max_lot = symbol_info.volume_max
        lot_step = symbol_info.volume_step
        
        # Round to the nearest lot step
        position_size = round(position_size / lot_step) * lot_step
        
        # Ensure within limits
        position_size = max(min_lot, min(position_size, max_lot))
        
        return position_size
    
    def determine_trade_strategy(self, market_condition, symbol):
        """
        Determine the best trading strategy based on market conditions
        
        Args:
            market_condition (dict): Market condition analysis
            symbol (str): The forex pair
            
        Returns:
            tuple: (should_trade, order_type, strategy_name)
        """
        should_trade = False
        order_type = ""
        strategy_name = ""
        
        # Get the short-term data from cache
        _, _, df_short, _ = self.get_market_data(symbol)
        
        latest_rsi = df_short['rsi'].iloc[-1]
        latest_cci = df_short['cci'].iloc[-1]
        latest_macd = df_short['macd'].iloc[-1]
        latest_macd_signal = df_short['macd_signal'].iloc[-1]
        latest_macd_hist = df_short['macd_hist'].iloc[-1]
        latest_close = df_short['close'].iloc[-1]
        latest_bb_upper = df_short['bb_upper'].iloc[-1]
        latest_bb_lower = df_short['bb_lower'].iloc[-1]
        
        # Strategy 1: Trend following in strong trends
        if market_condition['strong_uptrend']:
            # Buy on pullbacks in strong uptrends
            if latest_rsi < 45 and latest_macd > latest_macd_signal:
                should_trade = True
                order_type = "BUY"
                strategy_name = "strong_uptrend_pullback"
        
        elif market_condition['strong_downtrend']:
            # Sell on rallies in strong downtrends
            if latest_rsi > 55 and latest_macd < latest_macd_signal:
                should_trade = True
                order_type = "SELL"
                strategy_name = "strong_downtrend_rally"
        
        # Strategy 2: Range trading in sideways markets
        elif market_condition['sideways']:
            # Buy near support in range
            if latest_close < latest_bb_lower * 1.01 and latest_rsi < 30:
                should_trade = True
                order_type = "BUY"
                strategy_name = "range_support"
            
            # Sell near resistance in range
            elif latest_close > latest_bb_upper * 0.99 and latest_rsi > 70:
                should_trade = True
                order_type = "SELL"
                strategy_name = "range_resistance"
        
        # Strategy 3: Momentum trading in uncertain long-term but clear short-term trends
        elif market_condition['long_term_trend'] == "ranging" and market_condition['short_term_trend'] != "ranging":
            if market_condition['short_term_trend'] == "bullish" and latest_macd_hist > 0 and latest_macd_hist > df_short['macd_hist'].iloc[-2]:
                should_trade = True
                order_type = "BUY"
                strategy_name = "short_term_momentum_bullish"
            
            elif market_condition['short_term_trend'] == "bearish" and latest_macd_hist < 0 and latest_macd_hist < df_short['macd_hist'].iloc[-2]:
                should_trade = True
                order_type = "SELL"
                strategy_name = "short_term_momentum_bearish"
        
        # Strategy 4: Counter-trend trading on divergences
        elif market_condition['rsi_divergence']:
            # Bullish divergence: price makes lower low but RSI makes higher low
            price_lower_low = df_short['close'].iloc[-1] < min(df_short['close'].iloc[-5:-1])
            rsi_higher_low = df_short['rsi'].iloc[-1] > min(df_short['rsi'].iloc[-5:-1])
            
            if price_lower_low and rsi_higher_low and latest_rsi < 40:
                should_trade = True
                order_type = "BUY"
                strategy_name = "bullish_divergence"
            
            # Bearish divergence: price makes higher high but RSI makes lower high
            price_higher_high = df_short['close'].iloc[-1] > max(df_short['close'].iloc[-5:-1])
            rsi_lower_high = df_short['rsi'].iloc[-1] < max(df_short['rsi'].iloc[-5:-1])
            
            if price_higher_high and rsi_lower_high and latest_rsi > 60:
                should_trade = True
                order_type = "SELL"
                strategy_name = "bearish_divergence"
        
        # Strategy 5: Trend alignment across timeframes
        elif all(trend in ["bullish", "neutral"] for trend in [
            market_condition['long_term_trend'], 
            market_condition['medium_term_trend'], 
            market_condition['short_term_trend']
        ]):
            if latest_macd > latest_macd_signal and latest_macd_hist > 0:
                should_trade = True
                order_type = "BUY"
                strategy_name = "aligned_bull_trends"
        
        elif all(trend in ["bearish", "neutral"] for trend in [
            market_condition['long_term_trend'], 
            market_condition['medium_term_trend'], 
            market_condition['short_term_trend']
        ]):
            if latest_macd < latest_macd_signal and latest_macd_hist < 0:
                should_trade = True
                order_type = "SELL"
                strategy_name = "aligned_bear_trends"
        
        # Strategy 6: Extreme oversold/overbought conditions
        elif market_condition['oversold'] and market_condition['long_term_trend'] != "bearish":
            should_trade = True
            order_type = "BUY"
            strategy_name = "oversold_reversal"
        
        elif market_condition['overbought'] and market_condition['long_term_trend'] != "bullish":
            should_trade = True
            order_type = "SELL"
            strategy_name = "overbought_reversal"
        
        return should_trade, order_type, strategy_name
    
    def check_trading_conditions(self, symbol):
        """
        Check if trading conditions are met for a symbol
        
        Args:
            symbol (str): The forex pair
            
        Returns:
            tuple: (should_trade, order_type, strategy_name)
        """
        # Get market data using the optimized method
        df_long, df_medium, df_short, _ = self.get_market_data(symbol)
        
        if df_long is None or df_medium is None or df_short is None:
            return False, "", ""
        
        # Analyze market conditions
        market_condition = self.analyze_market_condition(df_long, df_medium, df_short)
        
        # Determine trading strategy
        should_trade, order_type, strategy_name = self.determine_trade_strategy(market_condition, symbol)
        
        # Log market condition and strategy
        logger.info(f"Market condition for {symbol}: {market_condition}")
        
        if should_trade:
            logger.info(f"Trading signal for {symbol}: {order_type} using strategy: {strategy_name}")
        
        return should_trade, order_type, strategy_name
    
    def place_order(self, symbol, order_type, lot_size, sl_pips, tp_pips, trailing_stop=False, comment=""):
        """
        Place an order in MT5
        
        Args:
            symbol (str): The forex pair
            order_type (str): "BUY" or "SELL"
            lot_size (float): Position size in lots
            sl_pips (float): Stop loss in pips
            tp_pips (float): Take profit in pips
            trailing_stop (bool): Whether to use trailing stop
            comment (str): Order comment
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {symbol}")
            return False
        
        # Check if spread is acceptable
        current_spread = symbol_info.spread / 10  # Convert to pips
        if current_spread > self.max_spread_pips:
            logger.warning(f"Spread for {symbol} is too high: {current_spread} pips. Maximum allowed: {self.max_spread_pips} pips")
            return False
        
        # Calculate points per pip
        points_per_pip = 10
        if 'JPY' in symbol:
            points_per_pip = 100
        
        # Current price
        if order_type == "BUY":
            price = mt5.symbol_info_tick(symbol).ask
            sl_price = price - (sl_pips * points_per_pip * symbol_info.point)
            tp_price = price + (tp_pips * points_per_pip * symbol_info.point)
            mt5_order_type = mt5.ORDER_TYPE_BUY
        else:  # SELL
            price = mt5.symbol_info_tick(symbol).bid
            sl_price = price + (sl_pips * points_per_pip * symbol_info.point)
            tp_price = price - (tp_pips * points_per_pip * symbol_info.point)
            mt5_order_type = mt5.ORDER_TYPE_SELL
        
        # Prepare the request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5_order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 10,
            "magic": 123456,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        # Send the order
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode}. Comment: {result.comment}")
            return False
        
        logger.info(f"Order placed successfully: Ticket={result.order}, Type={order_type}, Size={lot_size}, Symbol={symbol}")
        
        # Set up trailing stop if requested
        if trailing_stop and result.order > 0:
            self.setup_trailing_stop(result.order, symbol, sl_pips)
        
        return True
    
    def setup_trailing_stop(self, ticket, symbol, sl_pips):
        """
        Setup a trailing stop for an existing position
        
        Args:
            ticket (int): Position ticket number
            symbol (str): The forex pair
            sl_pips (float): Initial stop loss in pips
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {symbol}")
            return False
        
        # Calculate points per pip
        points_per_pip = 10
        if 'JPY' in symbol:
            points_per_pip = 100
        
        # Convert pips to points
        trailing_points = int(sl_pips * points_per_pip)
        
        # Apply trailing stop
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": 0,  # Will be set by the trailing stop
            "tp": 0,  # Will be set by the trailing stop
            "trailing_stop": trailing_points
        }
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to set trailing stop: {result.retcode}. Comment: {result.comment}")
            return False
        
        logger.info(f"Trailing stop set for position {ticket} with {sl_pips} pips")
        return True
    
    def get_dynamic_exit_conditions(self, strategy, symbol, order_type):
        """
        Determine dynamic exit conditions based on strategy and market
        
        Args:
            strategy (str): Trading strategy name
            symbol (str): The forex pair
            order_type (str): "BUY" or "SELL"
            
        Returns:
            dict: Exit conditions
        """
        # Get the latest data from cache
        _, _, df_short, _ = self.get_market_data(symbol)
        
        # Base exit conditions
        exit_conditions = {
            'trailing_stop': False,
            'time_based_exit': False,
            'exit_hours': 24,
            'indicator_exit': False,
            'exit_indicator': None,
            'exit_threshold': None,
            'exit_comparison': None
        }
        
        # Customize based on strategy
        if strategy in ["strong_uptrend_pullback", "strong_downtrend_rally"]:
            # For trend following, use trailing stops
            exit_conditions['trailing_stop'] = True
        
        elif strategy in ["range_support", "range_resistance"]:
            # For range trading, use tight profit targets and quick exits
            exit_conditions['time_based_exit'] = True
            exit_conditions['exit_hours'] = 4  # Exit within 4 hours
            exit_conditions['indicator_exit'] = True
            
            if order_type == "BUY":
                exit_conditions['exit_indicator'] = 'rsi'
                exit_conditions['exit_threshold'] = 60
                exit_conditions['exit_comparison'] = 'above'
            else:  # SELL
                exit_conditions['exit_indicator'] = 'rsi'
                exit_conditions['exit_threshold'] = 40
                exit_conditions['exit_comparison'] = 'below'
        
        elif strategy in ["short_term_momentum_bullish", "short_term_momentum_bearish"]:
            # For momentum trades, exit on reversal signals
            exit_conditions['indicator_exit'] = True
            
            if order_type == "BUY":
                exit_conditions['exit_indicator'] = 'macd_hist'
                exit_conditions['exit_threshold'] = 0
                exit_conditions['exit_comparison'] = 'below'
            else:  # SELL
                exit_conditions['exit_indicator'] = 'macd_hist'
                exit_conditions['exit_threshold'] = 0
                exit_conditions['exit_comparison'] = 'above'
        
        elif strategy in ["bullish_divergence", "bearish_divergence"]:
            # For divergence trades, use both time-based and indicator-based exits
            exit_conditions['time_based_exit'] = True
            exit_conditions['exit_hours'] = 12
            exit_conditions['indicator_exit'] = True
            
            if order_type == "BUY":
                exit_conditions['exit_indicator'] = 'cci'
                exit_conditions['exit_threshold'] = 100
                exit_conditions['exit_comparison'] = 'above'
            else:  # SELL
                exit_conditions['exit_indicator'] = 'cci'
                exit_conditions['exit_threshold'] = -100
                exit_conditions['exit_comparison'] = 'below'
        
        elif strategy in ["aligned_bull_trends", "aligned_bear_trends"]:
            # For aligned trend trades, use trailing stops and monitor ADX
            exit_conditions['trailing_stop'] = True
            exit_conditions['indicator_exit'] = True
            
            if order_type == "BUY":
                exit_conditions['exit_indicator'] = 'adx'
                exit_conditions['exit_threshold'] = 20
                exit_conditions['exit_comparison'] = 'below'
            else:  # SELL
                exit_conditions['exit_indicator'] = 'adx'
                exit_conditions['exit_threshold'] = 20
                exit_conditions['exit_comparison'] = 'below'
        
        elif strategy in ["oversold_reversal", "overbought_reversal"]:
            # For reversal trades, use time-based exits and profit targets
            exit_conditions['time_based_exit'] = True
            exit_conditions['exit_hours'] = 8
            
        return exit_conditions
    
    def check_exit_conditions(self, position, exit_conditions):
        """
        Check if exit conditions are met for a position
        
        Args:
            position (dict): Position information
            exit_conditions (dict): Exit conditions to check
            
        Returns:
            bool: True if should exit, False otherwise
        """
        # Get position details
        symbol = position.symbol
        position_type = 'BUY' if position.type == mt5.POSITION_TYPE_BUY else 'SELL'
        open_time = datetime.fromtimestamp(position.time)
        ticket = position.ticket
        
        # Get market data from cache instead of fetching again
        _, _, df_short, _ = self.get_market_data(symbol, force_update=True)
        
        # Check time-based exit
        if exit_conditions['time_based_exit']:
            current_time = datetime.now()
            time_open = (current_time - open_time).total_seconds() / 3600  # hours
            
            if time_open >= exit_conditions['exit_hours']:
                logger.info(f"Time-based exit triggered for {symbol} position {ticket}")
                return True
        
        # Check indicator-based exit
        if exit_conditions['indicator_exit']:
            if exit_conditions['exit_indicator'] not in df_short.columns:
                logger.warning(f"Exit indicator {exit_conditions['exit_indicator']} not found in DataFrame")
                return False
            
            indicator_value = df_short[exit_conditions['exit_indicator']].iloc[-1]
            threshold = exit_conditions['exit_threshold']
            comparison = exit_conditions['exit_comparison']
            
            should_exit = False
            
            if comparison == 'above' and indicator_value > threshold:
                should_exit = True
            elif comparison == 'below' and indicator_value < threshold:
                should_exit = True
            
            if should_exit:
                logger.info(f"Indicator-based exit triggered for {symbol} position {ticket}. "
                            f"{exit_conditions['exit_indicator']} = {indicator_value}, "
                            f"threshold = {threshold}, comparison = {comparison}")
                return True
        
        return False
    
    def close_position(self, ticket):
        """
        Close an open position
        
        Args:
            ticket (int): Position ticket number
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Get position info
        position = mt5.positions_get(ticket=ticket)
        
        if not position:
            logger.error(f"Position {ticket} not found")
            return False
        
        position = position[0]
        
        # Prepare close request
        symbol = position.symbol
        lot_size = position.volume
        
        # Determine order type for closing
        if position.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 10,
            "magic": 123456,
            "comment": "Position closed by bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        # Send the request
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to close position {ticket}: {result.retcode}. Comment: {result.comment}")
            return False
        
        logger.info(f"Position {ticket} closed successfully")
        return True
    
    def manage_existing_positions(self):
        """
        Manage existing positions based on exit conditions
        
        Returns:
            int: Number of positions closed
        """
        positions = mt5.positions_get()
        
        if positions is None:
            logger.error(f"Failed to get positions: {mt5.last_error()}")
            return 0
        
        closed_count = 0
        
        for position in positions:
            # Only manage positions opened by this bot
            if position.magic != 123456:
                continue
            
            symbol = position.symbol
            position_type = 'BUY' if position.type == mt5.POSITION_TYPE_BUY else 'SELL'
            
            # Get the strategy from the comment
            strategy = position.comment.split(':')[-1].strip() if ':' in position.comment else "unknown"
            
            # Get exit conditions for this strategy
            exit_conditions = self.get_dynamic_exit_conditions(strategy, symbol, position_type)
            
            # Check if we should exit
            if self.check_exit_conditions(position, exit_conditions):
                if self.close_position(position.ticket):
                    closed_count += 1
        
        return closed_count
    
    def check_open_positions(self, symbol):
        """
        Check if there are already open positions for the symbol
        
        Args:
            symbol (str): The forex pair
            
        Returns:
            bool: True if positions exist, False otherwise
        """
        positions = mt5.positions_get(symbol=symbol)
        
        if positions is None:
            logger.error(f"Failed to get positions: {mt5.last_error()}")
            return False
        
        return len(positions) > 0
    
    def run(self, interval=120):
        """
        Run the trading bot in a loop
        
        Args:
            interval (int): Seconds between each check
        """
        if not self.initialize_mt5():
            logger.error("Failed to initialize MT5. Exiting.")
            return
        
        logger.info(f"Trading bot started. Monitoring {self.symbols}")
        
        try:
            while True:
                # First, manage existing positions
                closed_positions = self.manage_existing_positions()
                if closed_positions > 0:
                    logger.info(f"Closed {closed_positions} positions based on exit conditions")
                
                # Then check for new trade opportunities
                for symbol in self.symbols:
                    # Skip if already have open positions for this symbol
                    # if self.check_open_positions(symbol):
                    #     logger.info(f"Already have open positions for {symbol}. Skipping.")
                    #     continue
                    
                    # Check if we should trade - uses optimized data retrieval
                    should_trade, order_type, strategy_name = self.check_trading_conditions(symbol)
                    
                    if should_trade:
                        # Get market data - reuse cached data
                        df_long, df_medium, df_short, df_execution = self.get_market_data(symbol)
                        
                        # Analyze market condition
                        market_condition = self.analyze_market_condition(df_long, df_medium, df_short)
                        
                        # Determine optimal trade parameters
                        sl_pips, tp_pips, trailing_stop = self.determine_trade_parameters(market_condition, symbol)
                        
                        # Calculate position size
                        lot_size = self.calculate_position_size(symbol, sl_pips)
                        
                        # Place the order
                        comment = f"Bot trade: {order_type} strategy: {strategy_name}"
                        
                        success = self.place_order(
                            symbol=symbol,
                            order_type=order_type,
                            lot_size=lot_size,
                            sl_pips=sl_pips,
                            tp_pips=tp_pips,
                            trailing_stop=trailing_stop,
                            comment=comment
                        )
                        
                        if success:
                            logger.info(f"Placed {order_type} order for {symbol} with lot size {lot_size}, "
                                    f"SL {sl_pips} pips, TP {tp_pips} pips, strategy: {strategy_name}")
                        else:
                            logger.error(f"Failed to place {order_type} order for {symbol}")
                    
                # Sleep until next check
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # Shutdown MT5
            mt5.shutdown()
            logger.info("MT5 connection closed")

if __name__ == "__main__":
    # Create and run the trading bot with optimized code
    bot = TradingBot(
        symbols=["USDJPY", "EURUSD"],
        risk_percent=0.02,  # Risk 2% per trade
        sl_pips_default=30,  # 30 pips default stop loss
        tp_multiplier_default=2,  # Take profit is 2x stop loss by default
        fast_period=12,
        slow_period=26,
        signal_period=9,
        rsi_period=14,
        rsi_overbought=70,
        rsi_oversold=30
    )
    
    # Run the bot with 120 seconds interval
    bot.run(interval=120)  # Check every 2 minutes