import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import pytz

# Function to initialize MT5 connection
def initialize_mt5(username, password, server, path):
    """Initialize MT5 connection"""
    if not mt5.initialize(path=path):
        print(f"initialize() failed, error code = {mt5.last_error()}")
        return False
    
    # Login to the MT5 account
    authorized = mt5.login(username, password, server)
    if not authorized:
        print(f"login failed, error code = {mt5.last_error()}")
        mt5.shutdown()
        return False
    
    print("MT5 connection established successfully")
    return True

# Function to get historical data from MT5
def get_historical_data(symbol, timeframe, num_bars):
    """Get historical data from MT5"""
    # Define the timeframe mapping
    timeframe_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M2": mt5.TIMEFRAME_M2,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1
    }
    
    # Get the current time in UTC timezone
    timezone = pytz.timezone("UTC")
    utc_from = datetime.now(timezone) - timedelta(days=num_bars/24)
    
    # Get the historical data
    rates = mt5.copy_rates_from(symbol, timeframe_map[timeframe], utc_from, num_bars)
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(rates)
    # Convert timestamp to datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    return df

# Calculate technical indicators
def calculate_indicators(df):
    """Calculate various technical indicators on the DataFrame"""
    # Moving Averages (Short, Medium, Long)
    df['ma_short'] = df['close'].rolling(window=9).mean()  # 9-period MA
    df['ma_medium'] = df['close'].rolling(window=21).mean()  # 21-period MA
    df['ma_long'] = df['close'].rolling(window=50).mean()  # 50-period MA
    df['ma_200'] = df['close'].rolling(window=200).mean()  # 200-period MA for major trend
    
    # MA Crossovers and Slopes
    df['ma_short_prev'] = df['ma_short'].shift(1)
    df['ma_medium_prev'] = df['ma_medium'].shift(1)
    df['ma_short_slope'] = (df['ma_short'] - df['ma_short_prev']) / df['ma_short_prev'] * 100
    df['ma_medium_slope'] = (df['ma_medium'] - df['ma_medium_prev']) / df['ma_medium_prev'] * 100
    df['ma_cross'] = np.where(
        (df['ma_short'] > df['ma_medium']) & (df['ma_short_prev'] <= df['ma_medium_prev']), 1,  # Golden Cross
        np.where(
            (df['ma_short'] < df['ma_medium']) & (df['ma_short_prev'] >= df['ma_medium_prev']), -1,  # Death Cross
            0
        )
    )
    
    # Enhanced RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()  # Using EMA instead of SMA for more responsiveness
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # RSI Divergence Detection
    df['price_higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
    df['price_lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
    df['rsi_higher_high'] = (df['rsi'] > df['rsi'].shift(1)) & (df['rsi'].shift(1) > df['rsi'].shift(2))
    df['rsi_lower_low'] = (df['rsi'] < df['rsi'].shift(1)) & (df['rsi'].shift(1) < df['rsi'].shift(2))
    
    # Bearish divergence: Price makes higher high but RSI makes lower high
    df['bearish_divergence'] = df['price_higher_high'] & ~df['rsi_higher_high']
    # Bullish divergence: Price makes lower low but RSI makes higher low
    df['bullish_divergence'] = df['price_lower_low'] & ~df['rsi_lower_low']
    
    # MACD (Moving Average Convergence Divergence) with histogram momentum
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['signal']
    df['macd_hist_prev'] = df['macd_histogram'].shift(1)
    df['macd_hist_prev2'] = df['macd_histogram'].shift(2)
    
    # MACD Histogram momentum and divergence
    df['macd_hist_momentum'] = df['macd_histogram'] - df['macd_hist_prev']
    df['macd_hist_acceleration'] = df['macd_hist_momentum'] - (df['macd_hist_prev'] - df['macd_hist_prev2'])
    
    # MACD Zero Line Crossover
    df['macd_cross_zero'] = np.where(
        (df['macd'] > 0) & (df['macd'].shift(1) <= 0), 1,  # Bullish
        np.where(
            (df['macd'] < 0) & (df['macd'].shift(1) >= 0), -1,  # Bearish
            0
        )
    )
    
    # MACD Signal Line Crossover
    df['macd_cross_signal'] = np.where(
        (df['macd'] > df['signal']) & (df['macd'].shift(1) <= df['signal'].shift(1)), 1,  # Bullish
        np.where(
            (df['macd'] < df['signal']) & (df['macd'].shift(1) >= df['signal'].shift(1)), -1,  # Bearish
            0
        )
    )
    
    # ATR (Average True Range) and Normalized ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.DataFrame({'hl': high_low, 'hc': high_close, 'lc': low_close}).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    df['natr'] = 100 * df['atr'] / df['close']  # Normalized ATR as percentage of price
    df['atr_ratio'] = df['atr'] / df['atr'].rolling(window=100).mean()  # ATR expansion/contraction
    
    # Chandelier Exit - A trailing stop based on ATR
    df['chandelier_long'] = df['high'].rolling(window=22).max() - df['atr'] * 3
    df['chandelier_short'] = df['low'].rolling(window=22).min() + df['atr'] * 3
    
    # Bollinger Bands
    std_dev = df['close'].rolling(window=20).std()
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_upper'] = df['bb_middle'] + (std_dev * 2)
    df['bb_lower'] = df['bb_middle'] - (std_dev * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ADX (Average Directional Index) and Directional Indicators (DI+, DI-)
    # +DM and -DM
    df['plus_dm'] = np.where(
        (df['high'] - df['high'].shift() > df['low'].shift() - df['low']) & 
        (df['high'] - df['high'].shift() > 0), 
        df['high'] - df['high'].shift(), 
        0
    )
    df['minus_dm'] = np.where(
        (df['low'].shift() - df['low'] > df['high'] - df['high'].shift()) & 
        (df['low'].shift() - df['low'] > 0), 
        df['low'].shift() - df['low'], 
        0
    )
    
    # Smoothed +DM, -DM, and TR using Wilder's smoothing
    period = 14
    df['tr_smoothed'] = true_range.ewm(alpha=1/period, adjust=False).mean()
    df['plus_dm_smoothed'] = df['plus_dm'].ewm(alpha=1/period, adjust=False).mean()
    df['minus_dm_smoothed'] = df['minus_dm'].ewm(alpha=1/period, adjust=False).mean()
    
    # +DI and -DI
    df['di_plus'] = 100 * df['plus_dm_smoothed'] / df['tr_smoothed']
    df['di_minus'] = 100 * df['minus_dm_smoothed'] / df['tr_smoothed']
    
    # DI Crossover
    df['di_cross'] = np.where(
        (df['di_plus'] > df['di_minus']) & (df['di_plus'].shift(1) <= df['di_minus'].shift(1)), 1,  # Bullish
        np.where(
            (df['di_plus'] < df['di_minus']) & (df['di_plus'].shift(1) >= df['di_minus'].shift(1)), -1,  # Bearish
            0
        )
    )
    
    # DX and ADX with trend strength classification
    df['dx'] = 100 * np.abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])
    df['adx'] = df['dx'].ewm(alpha=1/period, adjust=False).mean()
    df['adx_prev'] = df['adx'].shift(1)
    df['adx_trend'] = np.where(df['adx'] > 25, 
                              np.where(df['adx'] > df['adx_prev'], 'Strengthening', 'Weakening'),
                              'Ranging')
    
    # Vortex Indicator - For trend confirmation
    df['vm_plus'] = np.abs(df['high'] - df['low'].shift(1))
    df['vm_minus'] = np.abs(df['low'] - df['high'].shift(1))
    df['vi_plus'] = df['vm_plus'].rolling(window=14).sum() / df['tr_smoothed']
    df['vi_minus'] = df['vm_minus'].rolling(window=14).sum() / df['tr_smoothed']
    df['vi_cross'] = np.where(
        (df['vi_plus'] > df['vi_minus']) & (df['vi_plus'].shift(1) <= df['vi_minus'].shift(1)), 1,  # Bullish
        np.where(
            (df['vi_plus'] < df['vi_minus']) & (df['vi_plus'].shift(1) >= df['vi_minus'].shift(1)), -1,  # Bearish
            0
        )
    )
    
    # Fibonacci Retracement - Track pullbacks in trends
    window = 20
    df['local_high'] = df['high'].rolling(window=window).max()
    df['local_low'] = df['low'].rolling(window=window).min()
    df['fib_range'] = df['local_high'] - df['local_low']
    df['fib_38.2'] = df['local_high'] - df['fib_range'] * 0.382
    df['fib_50.0'] = df['local_high'] - df['fib_range'] * 0.5
    df['fib_61.8'] = df['local_high'] - df['fib_range'] * 0.618
    
    # Volume analysis - Volume weighted indicators
    if 'volume' in df.columns:
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    return df

# Define the advanced trading strategy
def trading_strategy(df, current_positions):
    """Implement an advanced trading strategy using calculated indicators
    focusing on trend identification, confirmation, and reversal detection"""
    
    # We'll use multiple timeframes for analysis
    # The main dataframe should have at least 10 rows for lookback
    if len(df) < 10:
        return False, False, False, False
    
    # Get recent data points for analysis
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Initialize signals
    buy_signal = False
    sell_signal = False
    close_buy = False
    close_sell = False
    
    # ---------- TREND IDENTIFICATION ----------
    # Identify the current market regime
    major_trend = "bullish" if latest['close'] > latest['ma_200'] else "bearish"
    intermediate_trend = "bullish" if latest['ma_short'] > latest['ma_medium'] else "bearish"
    
    # Trend strength confirmation
    strong_trend = latest['adx'] > 25 and latest['adx'] > latest['adx_prev']
    trend_alignment = (major_trend == "bullish" and intermediate_trend == "bullish") or \
                      (major_trend == "bearish" and intermediate_trend == "bearish")
    
    # ---------- ENTRY SIGNALS ----------
    # Complex Buy Signal (Multiple confirmations required)
    buy_conditions = [
        # Primary trend conditions
        intermediate_trend == "bullish",
        latest['ma_short_slope'] > 0.1,  # Rising short MA
        
        # Momentum conditions (need at least 2)
        [
            latest['rsi'] < 40 and latest['rsi'] > latest['rsi'].shift(1),  # RSI oversold but rising
            latest['macd_histogram'] > 0 and latest['macd_hist_momentum'] > 0,  # Positive and rising histogram
            latest['macd_cross_signal'] == 1,  # MACD crossed above signal line
            latest['bullish_divergence'],  # Bullish RSI divergence
            latest['di_cross'] == 1,  # DI+ crossed above DI-
            latest['vi_cross'] == 1,  # Vortex indicator bullish crossover
        ],
        
        # Volatility conditions
        latest['atr_ratio'] > 0.8,  # Adequate volatility
        latest['bb_pct'] < 0.3,  # Price in lower Bollinger Band range
        
        # Confirmation filters
        latest['close'] > latest['fib_61.8'],  # Price above key Fibonacci level
        not latest['bearish_divergence'],  # No bearish divergence
    ]
    
    # Count momentum confirmations
    momentum_confirmations = sum(1 for cond in buy_conditions[2] if cond)
    
    # Buy signal if primary conditions met and at least 2 momentum conditions
    if all(buy_conditions[:2]) and momentum_confirmations >= 2 and all(buy_conditions[3:]):
        # Extra filter - volume confirmation if available
        if 'volume' in df.columns:
            if latest['volume_ratio'] > 1.2:  # Above average volume
                buy_signal = True
        else:
            buy_signal = True
    
    # Complex Sell Signal (Multiple confirmations required)
    sell_conditions = [
        # Primary trend conditions
        intermediate_trend == "bearish",
        latest['ma_short_slope'] < -0.1,  # Falling short MA
        
        # Momentum conditions (need at least 2)
        [
            latest['rsi'] > 60 and latest['rsi'] < latest['rsi'].shift(1),  # RSI overbought but falling
            latest['macd_histogram'] < 0 and latest['macd_hist_momentum'] < 0,  # Negative and falling histogram
            latest['macd_cross_signal'] == -1,  # MACD crossed below signal line
            latest['bearish_divergence'],  # Bearish RSI divergence
            latest['di_cross'] == -1,  # DI- crossed above DI+
            latest['vi_cross'] == -1,  # Vortex indicator bearish crossover
        ],
        
        # Volatility conditions
        latest['atr_ratio'] > 0.8,  # Adequate volatility
        latest['bb_pct'] > 0.7,  # Price in upper Bollinger Band range
        
        # Confirmation filters
        latest['close'] < latest['fib_38.2'],  # Price below key Fibonacci level
        not latest['bullish_divergence'],  # No bullish divergence
    ]
    
    # Count momentum confirmations
    momentum_confirmations = sum(1 for cond in sell_conditions[2] if cond)
    
    # Sell signal if primary conditions met and at least 2 momentum conditions
    if all(sell_conditions[:2]) and momentum_confirmations >= 2 and all(sell_conditions[3:]):
        # Extra filter - volume confirmation if available
        if 'volume' in df.columns:
            if latest['volume_ratio'] > 1.2:  # Above average volume
                sell_signal = True
        else:
            sell_signal = True
    
    # ---------- EXIT SIGNALS (TREND REVERSAL DETECTION) ----------
    # Advanced exit strategy focusing on trend reversal confirmation
    
    # Close Buy Position (Multiple reversal confirmations)
    if "buy" in current_positions:
        # Strong reversal signals (require multiple confirmations)
        reversal_count = 0
        
        # 1. Death Cross
        if latest['ma_cross'] == -1:
            reversal_count += 2  # This is a strong signal, counts double
        
        # 2. MACD Signal Line Bearish Cross
        if latest['macd_cross_signal'] == -1:
            reversal_count += 1
        
        # 3. ADX showing weakening trend and DI- crossing above DI+
        if latest['adx_trend'] == 'Weakening' and latest['di_cross'] == -1:
            reversal_count += 1
        
        # 4. RSI Bearish Divergence
        if latest['bearish_divergence']:
            reversal_count += 1
        
        # 5. Break below Chandelier Exit stop
        if latest['close'] < latest['chandelier_long']:
            reversal_count += 1
        
        # 6. Vortex Indicator reversal
        if latest['vi_cross'] == -1:
            reversal_count += 1
        
        # 7. Price closes below key moving average
        if latest['close'] < latest['ma_medium'] and prev['close'] > prev['ma_medium']:
            reversal_count += 1
        
        # 8. Bollinger Band squeeze followed by downside break
        bb_squeeze = df['bb_width'].iloc[-3:-1].mean() < df['bb_width'].iloc[-7:-3].mean() * 0.8
        if bb_squeeze and latest['close'] < latest['bb_middle'] and prev['close'] > prev['bb_middle']:
            reversal_count += 1
        
        # Close buy position if enough reversal signals
        if reversal_count >= 3:
            close_buy = True
    
    # Close Sell Position (Multiple reversal confirmations)
    if "sell" in current_positions:
        # Strong reversal signals (require multiple confirmations)
        reversal_count = 0
        
        # 1. Golden Cross
        if latest['ma_cross'] == 1:
            reversal_count += 2  # This is a strong signal, counts double
        
        # 2. MACD Signal Line Bullish Cross
        if latest['macd_cross_signal'] == 1:
            reversal_count += 1
        
        # 3. ADX showing weakening trend and DI+ crossing above DI-
        if latest['adx_trend'] == 'Weakening' and latest['di_cross'] == 1:
            reversal_count += 1
        
        # 4. RSI Bullish Divergence
        if latest['bullish_divergence']:
            reversal_count += 1
        
        # 5. Break above Chandelier Exit stop
        if latest['close'] > latest['chandelier_short']:
            reversal_count += 1
        
        # 6. Vortex Indicator reversal
        if latest['vi_cross'] == 1:
            reversal_count += 1
        
        # 7. Price closes above key moving average
        if latest['close'] > latest['ma_medium'] and prev['close'] < prev['ma_medium']:
            reversal_count += 1
        
        # 8. Bollinger Band squeeze followed by upside break
        bb_squeeze = df['bb_width'].iloc[-3:-1].mean() < df['bb_width'].iloc[-7:-3].mean() * 0.8
        if bb_squeeze and latest['close'] > latest['bb_middle'] and prev['close'] < prev['bb_middle']:
            reversal_count += 1
        
        # Close sell position if enough reversal signals
        if reversal_count >= 3:
            close_sell = True
    
    # Additional trade management - trailing stop logic
    if "buy" in current_positions:
        # Dynamic trailing stop based on ATR for buy positions
        trailing_stop = latest['close'] - (latest['atr'] * 3)
        # If the price has moved significantly in our favor, use a tighter stop
        entry_price = current_positions.get("buy_price", 0)
        if entry_price > 0 and latest['close'] > entry_price + (latest['atr'] * 6):
            trailing_stop = max(trailing_stop, latest['close'] - (latest['atr'] * 1.5))
            
        # Update the trailing stop if it's higher than the previous one
        prev_stop = current_positions.get("trailing_stop_buy", 0)
        if trailing_stop > prev_stop:
            current_positions["trailing_stop_buy"] = trailing_stop
        
        # Check if price has hit the trailing stop
        if latest['close'] < current_positions.get("trailing_stop_buy", 0):
            close_buy = True
    
    if "sell" in current_positions:
        # Dynamic trailing stop based on ATR for sell positions
        trailing_stop = latest['close'] + (latest['atr'] * 3)
        # If the price has moved significantly in our favor, use a tighter stop
        entry_price = current_positions.get("sell_price", 0)
        if entry_price > 0 and latest['close'] < entry_price - (latest['atr'] * 6):
            trailing_stop = min(trailing_stop, latest['close'] + (latest['atr'] * 1.5))
            
        # Update the trailing stop if it's lower than the previous one
        prev_stop = current_positions.get("trailing_stop_sell", float('inf'))
        if prev_stop == 0 or trailing_stop < prev_stop:
            current_positions["trailing_stop_sell"] = trailing_stop
        
        # Check if price has hit the trailing stop
        if latest['close'] > current_positions.get("trailing_stop_sell", float('inf')):
            close_sell = True
    
    # Print detailed analysis for debugging
    print(f"--- Strategy Analysis ---")
    print(f"Major Trend: {major_trend}, Intermediate Trend: {intermediate_trend}")
    print(f"ADX: {latest['adx']:.2f}, Trend: {latest['adx_trend']}")
    print(f"RSI: {latest['rsi']:.2f}, MACD Hist: {latest['macd_histogram']:.6f}, Momentum: {latest['macd_hist_momentum']:.6f}")
    print(f"Buy Signal: {buy_signal}, Sell Signal: {sell_signal}")
    print(f"Close Buy: {close_buy}, Close Sell: {close_sell}")
    
    return buy_signal, sell_signal, close_buy, close_sell

# Function to place orders in MT5
def place_order(symbol, order_type, volume, price=None, sl=None, tp=None, comment="Python order"):
    """Place a trade order in MT5"""
    # Prepare the request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    # Add price, SL, TP if provided
    if price:
        request["price"] = price
    if sl:
        request["sl"] = sl
    if tp:
        request["tp"] = tp
    
    # Send the order
    result = mt5.order_send(request)
    
    # Process the result
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed, retcode={result.retcode}")
        return None
    
    print(f"Order placed successfully: {result.order}")
    return result.order

# Function to calculate position size based on risk percentage
def calculate_position_size(symbol, risk_percent, stop_loss_pips):
    """Calculate position size based on risk percentage and stop loss"""
    # Get account info
    account_info = mt5.account_info()
    if account_info is None:
        raise Exception("Failed to get account info")
    
    # Get symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        raise Exception(f"Failed to get symbol info for {symbol}")
    
    # Get daily account performance to limit daily risk
    from_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    to_date = datetime.now()
    deals = mt5.history_deals_get(from_date, to_date)
    
    daily_profit_loss = 0
    if deals is not None:
        for deal in deals:
            daily_profit_loss += deal.profit
    
    # Calculate daily risk already taken as a percentage of equity
    equity = account_info.equity
    daily_risk_pct = abs(daily_profit_loss) / equity * 100
    
    # Adjust risk percent if daily risk is approaching limit
    if daily_risk_pct > max_daily_risk_pct * 0.5:  # If we've used more than 50% of daily risk
        # Scale down risk as we approach the daily limit
        remaining_risk_pct = max(0, max_daily_risk_pct - daily_risk_pct)
        adjusted_risk_percent = min(risk_percent, remaining_risk_pct)
        print(f"Risk adjusted from {risk_percent}% to {adjusted_risk_percent}% due to daily risk limit")
        risk_percent = adjusted_risk_percent
    
    # Calculate risk amount
    risk_amount = equity * (risk_percent / 100)
    
    # Get current market volatility
    volatility_factor = 1.0
    try:
        # Check recent ATR vs historical ATR to adjust for abnormal volatility
        orders = mt5.orders_get(symbol=symbol)
        if orders is not None and len(orders) > 0:
            recent_atr = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_H1, datetime.now(), 24)
            if recent_atr is not None:
                recent_atr_df = pd.DataFrame(recent_atr)
                high_low = recent_atr_df['high'] - recent_atr_df['low']
                current_atr = high_low.mean()
                
                # Get historical average
                historical_atr = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_H1, datetime.now() - timedelta(days=30), 720)
                if historical_atr is not None:
                    historical_atr_df = pd.DataFrame(historical_atr)
                    historical_high_low = historical_atr_df['high'] - historical_atr_df['low']
                    avg_historical_atr = historical_high_low.mean()
                    
                    # If current volatility is much higher than normal, reduce position size
                    if current_atr > avg_historical_atr * 1.5:
                        volatility_factor = avg_historical_atr / current_atr
                        print(f"Adjusting position size due to high volatility. Factor: {volatility_factor:.2f}")
    except Exception as e:
        print(f"Error checking volatility: {e}")
    
    # Calculate pip value
    point_value = symbol_info.point
    pip_value = 10 * point_value  # Assuming 1 pip = 10 points (standard for 4-digit brokers)
    contract_size = symbol_info.trade_contract_size
    tick_value = symbol_info.trade_tick_value
    
    # Calculate position size with volatility adjustment
    tick_size = symbol_info.trade_tick_size
    ticks_per_pip = pip_value / tick_size
    loss_in_ticks = stop_loss_pips * ticks_per_pip
    position_size = risk_amount / (loss_in_ticks * tick_value / contract_size) * volatility_factor
    
    # Apply Kelly Criterion for position sizing based on win rate
    kelly_factor = 1.0
    try:
        # Get historical trades for this symbol
        deals = mt5.history_deals_get(datetime.now() - timedelta(days=90), datetime.now(), symbol=symbol)
        if deals is not None and len(deals) >= 10:  # Only apply if we have enough trade history
            wins = sum(1 for deal in deals if deal.profit > 0)
            total_trades = len(deals)
            win_rate = wins / total_trades if total_trades > 0 else 0
            
            # Calculate average win/loss ratio
            avg_win = sum(deal.profit for deal in deals if deal.profit > 0) / wins if wins > 0 else 0
            avg_loss = abs(sum(deal.profit for deal in deals if deal.profit <= 0)) / (total_trades - wins) if (total_trades - wins) > 0 else 0
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1
            
            # Kelly Criterion: f* = (p*b - (1-p))/b where p=win probability, b=win/loss ratio
            kelly_percent = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            
            # Use half-Kelly for more conservative sizing
            kelly_factor = max(0.1, min(1.0, kelly_percent / 2))
            print(f"Applying Kelly factor: {kelly_factor:.2f} (Win rate: {win_rate:.2f}, W/L ratio: {win_loss_ratio:.2f})")
            
            position_size *= kelly_factor
    except Exception as e:
        print(f"Error applying Kelly criterion: {e}")
    
    # Round down to nearest lot step
    lot_step = symbol_info.volume_step
    position_size = int(position_size / lot_step) * lot_step
    
    # Ensure position size is within limits
    if position_size < symbol_info.volume_min:
        position_size = symbol_info.volume_min
    if position_size > symbol_info.volume_max:
        position_size = symbol_info.volume_max
    
    # Calculate potential loss at this position size to double-check
    potential_loss_currency = (stop_loss_pips * pip_value * position_size * contract_size) / tick_size
    potential_loss_percent = (potential_loss_currency / equity) * 100
    
    print(f"Position size: {position_size:.2f} lots")
    print(f"Risk amount: {risk_amount:.2f} ({risk_percent:.2f}% of equity)")
    print(f"Potential loss: {potential_loss_currency:.2f} ({potential_loss_percent:.2f}% of equity)")
    
    return position_size


def run_trading_bot(
                symbol="EURUSD",
                timeframe="H1",
                risk_percent=1,  # Risk 1% of account per trade
                stop_loss_atr_mult=2.5,  # Use 2.5 x ATR for stop loss
                profit_atr_mult=5,  # Use 5 x ATR for take profit
                check_interval=300  # Check every 5 minutes
            ):
 "sell"
                    current_positions[position_type] = position.ticket
            
            # Get trading signals
            buy_signal, sell_signal, close_buy, close_sell = trading_strategy(df, current_positions)
            
            # Process signals
            symbol_info = mt5.symbol_info(symbol)
            ask = symbol_info.ask
            bid = symbol_info.bid
            
            # Close buy positions if signal
            if close_buy and "buy" in current_positions:
                print(f"Closing buy position for {symbol}")
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": positions[0].volume,
                    "type": mt5.ORDER_TYPE_SELL,
                    "position": current_positions["buy"],
                    "price": bid,
                    "comment": "Close buy position",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                mt5.order_send(close_request)
            
            # Close sell positions if signal
            if close_sell and "sell" in current_positions:
                print(f"Closing sell position for {symbol}")
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": positions[0].volume,
                    "type": mt5.ORDER_TYPE_BUY,
                    "position": current_positions["sell"],
                    "price": ask,
                    "comment": "Close sell position",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                mt5.order_send(close_request)
            
            # Execute buy signal
            if buy_signal and "buy" not in current_positions:
                print(f"Buy signal for {symbol}")
                # Calculate position size
                position_size = calculate_position_size(symbol, risk_percent, stop_loss_pips)
                
                # Calculate SL and TP
                sl = ask - (stop_loss_pips * symbol_info.point * 10)
                tp = ask + (take_profit_pips * symbol_info.point * 10)
                
                # Place buy order
                place_order(
                    symbol=symbol,
                    order_type=mt5.ORDER_TYPE_BUY,
                    volume=position_size,
                    price=ask,
                    sl=sl,
                    tp=tp,
                    comment="MT5 Python Bot Buy"
                )
            
            # Execute sell signal
            if sell_signal and "sell" not in current_positions:
                print(f"Sell signal for {symbol}")
                # Calculate position size
                position_size = calculate_position_size(symbol, risk_percent, stop_loss_pips)
                
                # Calculate SL and TP
                sl = bid + (stop_loss_pips * symbol_info.point * 10)
                tp = bid - (take_profit_pips * symbol_info.point * 10)
                
                # Place sell order
                place_order(
                    symbol=symbol,
                    order_type=mt5.ORDER_TYPE_SELL,
                    volume=position_size,
                    price=bid,
                    sl=sl,
                    tp=tp,
                    comment="MT5 Python Bot Sell"
                )
            
            # Wait for next check
            print(f"Finished checking, sleeping for {check_interval} seconds...")
            time.sleep(check_interval)
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(check_interval)

# Risk management system
class RiskManager:
    """Advanced risk management system to protect account"""
    
    def __init__(self, max_daily_risk_pct=5, max_drawdown_pct=15, cooldown_after_losses=3):
        self.max_daily_risk_pct = max_daily_risk_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.cooldown_after_losses = cooldown_after_losses
        self.consecutive_losses = 0
        self.in_cooldown = False
        self.cooldown_end_time = None
        self.peak_equity = 0
        self.current_drawdown_pct = 0
        
    def update_equity_stats(self, equity):
        """Update equity high watermark and drawdown"""
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        if self.peak_equity > 0:
            self.current_drawdown_pct = (self.peak_equity - equity) / self.peak_equity * 100
        
    def should_trade(self, equity, daily_loss_pct=0):
        """Determine if trading should continue based on risk parameters"""
        # Update stats
        self.update_equity_stats(equity)
        
        # Check cooldown status
        if self.in_cooldown:
            if datetime.now() >= self.cooldown_end_time:
                self.in_cooldown = False
                self.consecutive_losses = 0
                print("Cooldown period ended, resuming trading")
            else:
                remaining = (self.cooldown_end_time - datetime.now()).total_seconds() / 60
                print(f"In cooldown period, {remaining:.1f} minutes remaining")
                return False
        
        # Check drawdown limit
        if self.current_drawdown_pct >= self.max_drawdown_pct:
            print(f"Trading halted: Max drawdown reached ({self.current_drawdown_pct:.2f}% > {self.max_drawdown_pct}%)")
            return False
        
        # Check daily loss limit
        if daily_loss_pct >= self.max_daily_risk_pct:
            print(f"Trading halted: Daily loss limit reached ({daily_loss_pct:.2f}% > {self.max_daily_risk_pct}%)")
            return False
        
        return True
    
    def record_trade_result(self, profit):
        """Record the result of a trade and update consecutive loss counter"""
        if profit <= 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.cooldown_after_losses:
                self.in_cooldown = True
                self.cooldown_end_time = datetime.now() + timedelta(hours=1)
                print(f"Entered cooldown period after {self.consecutive_losses} consecutive losses")
        else:
            self.consecutive_losses = 0

# Backtest functionality
def backtest_strategy(symbol, timeframe, start_date, end_date, initial_deposit=10000, risk_percent=1, commission=0.0):
    """Backtest the strategy on historical data"""
    print(f"Backtesting strategy on {symbol}, {timeframe} from {start_date} to {end_date}")
    
    # Get historical data
    timezone = pytz.timezone("UTC")
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone)
    
    # Define the timeframe mapping
    timeframe_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1
    }
    
    # Get the data
    rates = mt5.copy_rates_range(symbol, timeframe_map[timeframe], start, end)
    if rates is None or len(rates) == 0:
        print("Failed to get historical data for backtesting")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Calculate indicators
    df = calculate_indicators(df)
    df = df.dropna()
    
    # Backtest variables
    balance = initial_deposit
    equity = initial_deposit
    trades = []
    open_position = None
    max_drawdown = 0
    peak_equity = initial_deposit
    
    # Point value
    symbol_info = mt5.symbol_info(symbol)
    point = symbol_info.point
    
    # Process each bar
    for i in range(1, len(df) - 1):
        # Skip the first few bars to ensure indicators are properly calculated
        if i < 50:
            continue
        
        # Get the current bar data
        current = df.iloc[i]
        next_bar = df.iloc[i + 1]  # For simulating next bar's open for entry
        
        # Current equity calculation
        if open_position:
            if open_position['type'] == 'buy':
                current_price = current['close']
                position_pnl = (current_price - open_position['entry_price']) * open_position['size']
            else:  # sell
                current_price = current['close']
                position_pnl = (open_position['entry_price'] - current_price) * open_position['size']
            
            equity = balance + position_pnl
            
            # Track maximum drawdown
            if equity > peak_equity:
                peak_equity = equity
            drawdown = (peak_equity - equity) / peak_equity * 100
            max_drawdown = max(max_drawdown, drawdown)
        else:
            equity = balance
        
        # Create a subset of data for analysis
        analysis_df = df.iloc[:i+1].copy()
        
        # Create a temporary current_positions dict for the trading strategy
        current_positions = {}
        if open_position:
            current_positions[open_position['type']] = 1
            if open_position['type'] == 'buy':
                current_positions['buy_price'] = open_position['entry_price']
            else:
                current_positions['sell_price'] = open_position['entry_price']
        
        # Get trading signals
        buy_signal, sell_signal, close_buy, close_sell = trading_strategy(analysis_df, current_positions)
        
        # Process close signals
        if open_position and ((open_position['type'] == 'buy' and close_buy) or 
                             (open_position['type'] == 'sell' and close_sell)):
            # Close the position
            exit_price = next_bar['open']  # Simulate execution at next bar open
            
            # Calculate profit/loss
            if open_position['type'] == 'buy':
                profit = (exit_price - open_position['entry_price']) * open_position['size']
                profit_pips = (exit_price - open_position['entry_price']) / point / 10
            else:
                profit = (open_position['entry_price'] - exit_price) * open_position['size']
                profit_pips = (open_position['entry_price'] - exit_price) / point / 10
            
            # Deduct commission
            profit -= commission * open_position['size']
            
            # Update balance
            balance += profit
            equity = balance
            
            # Record the trade
            trade_result = {
                'type': open_position['type'],
                'entry_date': open_position['entry_date'],
                'entry_price': open_position['entry_price'],
                'exit_date': current['time'],
                'exit_price': exit_price,
                'profit': profit,
                'profit_pips': profit_pips,
                'balance': balance
            }
            trades.append(trade_result)
            
            print(f"Closed {open_position['type']} position at {exit_price}, Profit: {profit:.2f} ({profit_pips:.1f} pips), Balance: {balance:.2f}")
            
            # Reset position
            open_position = None
        
        # Process entry signals
        if not open_position:
            if buy_signal:
                # Calculate position size (simplified for backtest)
                atr_value = current['atr']
                stop_loss_distance = atr_value * 2.5  # Using ATR for dynamic SL
                stop_loss_pips = stop_loss_distance / point / 10
                
                # For backtest, we'll use a simplified position sizing
                risk_amount = balance * risk_percent / 100
                position_size = risk_amount / stop_loss_distance
                
                # Open buy position
                entry_price = next_bar['open']  # Simulate execution at next bar open
                stop_loss = entry_price - stop_loss_distance
                take_profit = entry_price + (stop_loss_distance * 2)  # 2:1 reward-to-risk
                
                open_position = {
                    'type': 'buy',
                    'entry_date': current['time'],
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'size': position_size
                }
                
                print(f"Opened Buy at {entry_price}, SL: {stop_loss}, TP: {take_profit}, Size: {position_size:.2f}")
                
            elif sell_signal:
                # Calculate position size (simplified for backtest)
                atr_value = current['atr']
                stop_loss_distance = atr_value * 2.5  # Using ATR for dynamic SL
                stop_loss_pips = stop_loss_distance / point / 10
                
                # For backtest, we'll use a simplified position sizing
                risk_amount = balance * risk_percent / 100
                position_size = risk_amount / stop_loss_distance
                
                # Open sell position
                entry_price = next_bar['open']  # Simulate execution at next bar open
                stop_loss = entry_price + stop_loss_distance
                take_profit = entry_price - (stop_loss_distance * 2)  # 2:1 reward-to-risk
                
                open_position = {
                    'type': 'sell',
                    'entry_date': current['time'],
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'size': position_size
                }
                
                print(f"Opened Sell at {entry_price}, SL: {stop_loss}, TP: {take_profit}, Size: {position_size:.2f}")
        
        # Check for stop loss or take profit hit
        if open_position:
            # For buy positions
            if open_position['type'] == 'buy':
                # Check if low price hit stop loss
                if current['low'] <= open_position['stop_loss']:
                    exit_price = open_position['stop_loss']
                    profit = (exit_price - open_position['entry_price']) * open_position['size']
                    profit_pips = (exit_price - open_position['entry_price']) / point / 10
                    
                    # Deduct commission
                    profit -= commission * open_position['size']
                    
                    # Update balance
                    balance += profit
                    equity = balance
                    
                    # Record the trade
                    trade_result = {
                        'type': open_position['type'],
                        'entry_date': open_position['entry_date'],
                        'entry_price': open_position['entry_price'],
                        'exit_date': current['time'],
                        'exit_price': exit_price,
                        'profit': profit,
                        'profit_pips': profit_pips,
                        'balance': balance,
                        'exit_type': 'stop_loss'
                    }
                    trades.append(trade_result)
                    
                    print(f"Stop Loss hit for Buy at {exit_price}, Loss: {profit:.2f} ({profit_pips:.1f} pips), Balance: {balance:.2f}")
                    
                    # Reset position
                    open_position = None
                
                # Check if high price hit take profit
                elif current['high'] >= open_position['take_profit']:
                    exit_price = open_position['take_profit']
                    profit = (exit_price - open_position['entry_price']) * open_position['size']
                    profit_pips = (exit_price - open_position['entry_price']) / point / 10
                    
                    # Deduct commission
                    profit -= commission * open_position['size']
                    
                    # Update balance
                    balance += profit
                    equity = balance
                    
                    # Record the trade
                    trade_result = {
                        'type': open_position['type'],
                        'entry_date': open_position['entry_date'],
                        'entry_price': open_position['entry_price'],
                        'exit_date': current['time'],
                        'exit_price': exit_price,
                        'profit': profit,
                        'profit_pips': profit_pips,
                        'balance': balance,
                        'exit_type': 'take_profit'
                    }
                    trades.append(trade_result)
                    
                    print(f"Take Profit hit for Buy at {exit_price}, Profit: {profit:.2f} ({profit_pips:.1f} pips), Balance: {balance:.2f}")
                    
                    # Reset position
                    open_position = None
            
            # For sell positions
            elif open_position['type'] == 'sell':
                # Check if high price hit stop loss
                if current['high'] >= open_position['stop_loss']:
                    exit_price = open_position['stop_loss']
                    profit = (open_position['entry_price'] - exit_price) * open_position['size']
                    profit_pips = (open_position['entry_price'] - exit_price) / point / 10
                    
                    # Deduct commission
                    profit -= commission * open_position['size']
                    
                    # Update balance
                    balance += profit
                    equity = balance
                    
                    # Record the trade
                    trade_result = {
                        'type': open_position['type'],
                        'entry_date': open_position['entry_date'],
                        'entry_price': open_position['entry_price'],
                        'exit_date': current['time'],
                        'exit_price': exit_price,
                        'profit': profit,
                        'profit_pips': profit_pips,
                        'balance': balance,
                        'exit_type': 'stop_loss'
                    }
                    trades.append(trade_result)
                    
                    print(f"Stop Loss hit for Sell at {exit_price}, Loss: {profit:.2f} ({profit_pips:.1f} pips), Balance: {balance:.2f}")
                    
                    # Reset position
                    open_position = None
                
                # Check if low price hit take profit
                elif current['low'] <= open_position['take_profit']:
                    exit_price = open_position['take_profit']
                    profit = (open_position['entry_price'] - exit_price) * open_position['size']
                    profit_pips = (open_position['entry_price'] - exit_price) / point / 10
                    
                    # Deduct commission
                    profit -= commission * open_position['size']
                    
                    # Update balance
                    balance += profit
                    equity = balance
                    
                    # Record the trade
                    trade_result = {
                        'type': open_position['type'],
                        'entry_date': open_position['entry_date'],
                        'entry_price': open_position['entry_price'],
                        'exit_date': current['time'],
                        'exit_price': exit_price,
                        'profit': profit,
                        'profit_pips': profit_pips,
                        'balance': balance,
                        'exit_type': 'take_profit'
                    }
                    trades.append(trade_result)
                    
                    print(f"Take Profit hit for Sell at {exit_price}, Profit: {profit:.2f} ({profit_pips:.1f} pips), Balance: {balance:.2f}")
                    
                    # Reset position
                    open_position = None
    
    # Close any open position at the end of the backtest
    if open_position:
        exit_price = df.iloc[-1]['close']
        
        # Calculate profit/loss
        if open_position['type'] == 'buy':
            profit = (exit_price - open_position['entry_price']) * open_position['size']
            profit_pips = (exit_price - open_position['entry_price']) / point / 10
        else:
            profit = (open_position['entry_price'] - exit_price) * open_position['size']
            profit_pips = (open_position['entry_price'] - exit_price) / point / 10
        
        # Deduct commission
        profit -= commission * open_position['size']
        
        # Update balance
        balance += profit
        
        # Record the trade
        trade_result = {
            'type': open_position['type'],
            'entry_date': open_position['entry_date'],
            'entry_price': open_position['entry_price'],
            'exit_date': df.iloc[-1]['time'],
            'exit_price': exit_price,
            'profit': profit,
            'profit_pips': profit_pips,
            'balance': balance,
            'exit_type': 'end_of_test'
        }
        trades.append(trade_result)
    
    # Calculate backtest statistics
    if trades:
        # Convert trades to DataFrame for analysis
        trades_df = pd.DataFrame(trades)
        
        # Total profit
        total_profit = trades_df['profit'].sum()
        total_profit_pips = trades_df['profit_pips'].sum()
        
        # Win rate
        wins = trades_df[trades_df['profit'] > 0].shape[0]
        losses = trades_df[trades_df['profit'] <= 0].shape[0]
        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        
        # Profit factor
        gross_profit = trades_df[trades_df['profit'] > 0]['profit'].sum()
        gross_loss = abs(trades_df[trades_df['profit'] <= 0]['profit'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Average profit/loss
        avg_profit = trades_df[trades_df['profit'] > 0]['profit'].mean() if wins > 0 else 0
        avg_loss = trades_df[trades_df['profit'] <= 0]['profit'].mean() if losses > 0 else 0
        
        # Sharpe ratio (approximation)
        if len(trades_df) > 1:
            returns = trades_df['profit'] / initial_deposit * 100  # Convert to percentage returns
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Return on investment
        roi = (balance - initial_deposit) / initial_deposit * 100
        
        # Print results
        print("\n--- Backtest Results ---")
        print(f"Initial Deposit: {initial_deposit}")
        print(f"Final Balance: {balance:.2f}")
        print(f"Total Return: {roi:.2f}%")
        print(f"Total Profit: {total_profit:.2f} ({total_profit_pips:.1f} pips)")
        print(f"Total Trades: {len(trades)}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Average Profit: {avg_profit:.2f}")
        print(f"Average Loss: {avg_loss:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        
        # Additional statistics
        avg_win_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
        print(f"Average Win/Loss Ratio: {avg_win_loss_ratio:.2f}")
        
        # Expectancy
        expectancy = (win_rate/100 * avg_profit) + ((1-win_rate/100) * avg_loss)
        print(f"Expectancy: {expectancy:.2f}")
        
        # System quality number
        if expectancy != 0 and returns.std() > 0:
            sqn = np.sqrt(len(trades)) * (expectancy / returns.std())
            print(f"System Quality Number (SQN): {sqn:.2f}")
        
        # Return the results as a dictionary
        return {
            'trades': trades_df,
            'balance': balance,
            'roi': roi,
            'total_profit': total_profit,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
    else:
        print("No trades executed during backtest period")
        return None

# Advanced parameter optimization
def optimize_parameters(symbol, timeframe, start_date, end_date, params_to_optimize):
    """Run parameter optimization using grid search"""
    print(f"Starting parameter optimization for {symbol} on {timeframe}")
    
    # Define parameter grid
    param_grid = []
    param_names = []
    
    for param_name, param_values in params_to_optimize.items():
        param_names.append(param_name)
        param_grid.append(param_values)
    
    # Generate all parameter combinations
    param_combinations = list(itertools.product(*param_grid))
    
    # Store results
    optimization_results = []
    
    # Run backtest for each parameter combination
    for combo in param_combinations:
        params = dict(zip(param_names, combo))
        param_str = ", ".join([f"{name}={value}" for name, value in params.items()])
        print(f"\nTesting parameters: {param_str}")
        
        # Apply parameters to strategy (this would need to be implemented)
        # For example, you could have global variables that the strategy uses
        
        # Run backtest with these parameters
        result = backtest_strategy(symbol, timeframe, start_date, end_date)
        
        if result:
            # Store results along with parameters
            result_summary = {
                'params': params,
                'roi': result['roi'],
                'profit_factor': result['profit_factor'],
                'win_rate': result['win_rate'],
                'sharpe_ratio': result['sharpe_ratio'],
                'max_drawdown': result['max_drawdown']
            }
            optimization_results.append(result_summary)
    
    # Sort results by various metrics
    if optimization_results:
        # By ROI
        best_roi = sorted(optimization_results, key=lambda x: x['roi'], reverse=True)[0]
        # By profit factor
        best_pf = sorted(optimization_results, key=lambda x: x['profit_factor'], reverse=True)[0]
        # By Sharpe ratio
        best_sharpe = sorted(optimization_results, key=lambda x: x['sharpe_ratio'], reverse=True)[0]
        # By combined score (normalized ranking)
        for result in optimization_results:
            result['combined_score'] = (
                result['roi'] / best_roi['roi'] * 0.3 +
                result['profit_factor'] / best_pf['profit_factor'] * 0.3 +
                result['sharpe_ratio'] / best_sharpe['sharpe_ratio'] * 0.3 -
                result['max_drawdown'] / 100 * 0.1  # Penalize drawdown
            )
        
        best_combined = sorted(optimization_results, key=lambda x: x['combined_score'], reverse=True)[0]
        
        # Print best results
        print("\n--- Optimization Results ---")
        print(f"Best ROI: {best_roi['params']} - ROI: {best_roi['roi']:.2f}%, PF: {best_roi['profit_factor']:.2f}")
        print(f"Best Profit Factor: {best_pf['params']} - PF: {best_pf['profit_factor']:.2f}, ROI: {best_pf['roi']:.2f}%")
        print(f"Best Sharpe: {best_sharpe['params']} - Sharpe: {best_sharpe['sharpe_ratio']:.2f}, ROI: {best_sharpe['roi']:.2f}%")
        print(f"Best Combined Score: {best_combined['params']} - Score: {best_combined['combined_score']:.2f}, ROI: {best_combined['roi']:.2f}%, PF: {best_combined['profit_factor']:.2f}")
        
        return optimization_results
    else:
        print("No valid optimization results")
        return None

# Example usage
if __name__ == "__main__":
    # Import libraries needed for optimization
    import itertools
    
    # MT5 account credentials
    USERNAME = 12345678  # Your MT5 account number
    PASSWORD = "your_password"  # Your MT5 account password
    SERVER = "MetaQuotes-Demo"  # Your MT5 server name
    PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"  # Path to your MT5 installation
    
    # Initialize connection
    if initialize_mt5(USERNAME, PASSWORD, SERVER, PATH):
        # Choose whether to run the bot in live mode or backtest mode
        mode = "live"  # Options: "live", "backtest", "optimize"
        
        if mode == "live":
            # Create risk manager
            risk_mgr = RiskManager(
                max_daily_risk_pct=3,
                max_drawdown_pct=10,
                cooldown_after_losses=3
            )
            
            # Run the trading bot with advanced parameters
            run_trading_bot(
                symbol="EURUSD",
                timeframe="H1",
                risk_percent=1,  # Risk 1% of account per trade
                stop_loss_atr_mult=2.5,  # Use 2.5 x ATR for stop loss
                profit_atr_mult=5,  # Use 5 x ATR for take profit
                check_interval=300  # Check every 5 minutes
            )
        
        elif mode == "backtest":
            # Run a backtest
            backtest_results = backtest_strategy(
                symbol="EURUSD",
                timeframe="2M",
                start_date="2023-01-01",
                end_date="2023-12-31",
                initial_deposit=10000,
                risk_percent=1,
                commission=7  # $7 per standard lot round turn
            )
        
        elif mode == "optimize":
            # Define parameters to optimize
            params = {
                'rsi_buy_threshold': [30, 35, 40],
                'rsi_sell_threshold': [60, 65, 70],
                'adx_threshold': [20, 25, 30],
                'profit_atr_mult': [3, 4, 5],
                'stop_loss_atr_mult': [2, 2.5, 3]
            }
            
            # Run optimization
            opt_results = optimize_parameters(
                symbol="EURUSD",
                timeframe="2M",
                start_date="2023-01-01",
                end_date="2023-12-31",
                params_to_optimize=params
            )
    else:
        print("Failed to initialize MT5, exiting.")
    
    # Shutdown MT5 when done
    mt5.shutdown()