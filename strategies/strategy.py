import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from datetime import datetime, timedelta
import time
import pytz
import talib  # Technical Analysis Library
import warnings
warnings.filterwarnings('ignore')

# Connect to MetaTrader 5
def initialize_mt5(account=None, server=None):
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return False
    
    # Login if credentials are provided
    if account is not None:
        authorized = mt5.login(account, server=server)
        if not authorized:
            print("login failed, error code =", mt5.last_error())
            return False
    print("login success")
    return True

# Function to get multi-timeframe data
def get_multi_timeframe_data(symbol, timeframes, bars=300):
    data = {}
    timezone = pytz.timezone("UTC")
    utc_from = datetime.now() + timedelta(hours=3)  # Extra days for indicators
    
    for tf in timeframes:
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
        if rates is None or len(rates) == 0:
            print(f"Failed to get data for {symbol} on timeframe {tf}, error: {mt5.last_error()}")
            continue
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        print(df['time'].iloc[-1])
        # Add timeframe info
        tf_names = {
            mt5.TIMEFRAME_M1: "M1", mt5.TIMEFRAME_M2: "M2", mt5.TIMEFRAME_M5: "M5", mt5.TIMEFRAME_M15: "M15",
            mt5.TIMEFRAME_M30: "M30", mt5.TIMEFRAME_H1: "H1", mt5.TIMEFRAME_H4: "H4",
            mt5.TIMEFRAME_D1: "D1", mt5.TIMEFRAME_W1: "W1", mt5.TIMEFRAME_MN1: "MN1"
        }
        df['timeframe'] = tf_names.get(tf, str(tf))
        
        data[tf] = df
    
    return data

# Function to calculate advanced technical indicators
def add_technical_indicators(df):
    # Basic price data
    df['hl2'] = (df['high'] + df['low']) / 2
    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    # Trend indicators
    df['sma20'] = df['close'].rolling(window=20).mean()
    df['sma50'] = df['close'].rolling(window=50).mean()
    df['sma100'] = df['close'].rolling(window=100).mean()
    df['sma200'] = df['close'].rolling(window=200).mean()
    
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema100'] = df['close'].ewm(span=100, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # Use TALib for more accurate calculations if available
    try:
        # Trend-following indicators
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Oscillators for momentum and reversals
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['tick_volume'], timeperiod=14)
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
        
        # MACD for trend and momentum
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # Bollinger Bands for volatility and ranges
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # ATR for volatility measurement
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Ichimoku Cloud for trend analysis
        high_values = df['high'].values
        low_values = df['low'].values
        close_values = df['close'].values
        
        df['ichimoku_tenkan'] = talib.SMA(high_values + low_values / 2, 9)
        df['ichimoku_kijun'] = talib.SMA(high_values + low_values / 2, 26)
        df['ichimoku_senkou_a'] = (df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2
        df['ichimoku_senkou_b'] = talib.SMA(high_values + low_values / 2, 52)
    except:
        # Fallback to pandas calculations if TALib is not available
        # Calculate RSI manually
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD manually
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Calculate Bollinger Bands manually
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # Calculate ATR manually
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
    
    # Calculate high and low of recent periods for support/resistance
    df['highest_20'] = df['high'].rolling(window=20).max()
    df['lowest_20'] = df['low'].rolling(window=20).min()
    df['highest_50'] = df['high'].rolling(window=50).max()
    df['lowest_50'] = df['low'].rolling(window=50).min()
    
    # Range indicators
    df['daily_range'] = df['high'] - df['low']
    df['avg_range_20'] = df['daily_range'].rolling(window=20).mean()
    
    # Volume indicators
    if 'tick_volume' in df.columns:
        df['volume_sma20'] = df['tick_volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['tick_volume'] / df['volume_sma20']
    
    return df

# Function to detect divergences
def detect_divergences(df):
    divergences = []
    
    # We need a sufficient length of data to detect divergences
    if len(df) < 50:
        return divergences
    
    # Regular Bearish Divergence: Price makes higher high but indicator makes lower high
    # Regular Bullish Divergence: Price makes lower low but indicator makes higher low
    # Hidden Bearish Divergence: Price makes lower high but indicator makes higher high
    # Hidden Bullish Divergence: Price makes higher low but indicator makes lower low
    
    # Use RSI for divergence detection
    for i in range(20, len(df)-5):
        window = df.iloc[i-20:i+5]
        
        # Find local price highs and lows
        price_highs = []
        price_lows = []
        
        for j in range(1, len(window)-1):
            if window.iloc[j]['close'] > window.iloc[j-1]['close'] and window.iloc[j]['close'] > window.iloc[j+1]['close']:
                price_highs.append((j, window.iloc[j]['close'], window.iloc[j]['rsi']))
            if window.iloc[j]['close'] < window.iloc[j-1]['close'] and window.iloc[j]['close'] < window.iloc[j+1]['close']:
                price_lows.append((j, window.iloc[j]['close'], window.iloc[j]['rsi']))
        
        # Need at least two highs or two lows to find divergence
        if len(price_highs) >= 2:
            # Check for regular bearish divergence
            if price_highs[-1][1] > price_highs[-2][1] and price_highs[-1][2] < price_highs[-2][2]:
                divergences.append({
                    'type': 'Regular Bearish',
                    'time': df.iloc[i]['time'],
                    'price': df.iloc[i]['close'],
                    'indicator': 'RSI'
                })
            
            # Check for hidden bearish divergence
            if price_highs[-1][1] < price_highs[-2][1] and price_highs[-1][2] > price_highs[-2][2]:
                divergences.append({
                    'type': 'Hidden Bearish',
                    'time': df.iloc[i]['time'],
                    'price': df.iloc[i]['close'],
                    'indicator': 'RSI'
                })
        
        if len(price_lows) >= 2:
            # Check for regular bullish divergence
            if price_lows[-1][1] < price_lows[-2][1] and price_lows[-1][2] > price_lows[-2][2]:
                divergences.append({
                    'type': 'Regular Bullish',
                    'time': df.iloc[i]['time'],
                    'price': df.iloc[i]['close'],
                    'indicator': 'RSI'
                })
            
            # Check for hidden bullish divergence
            if price_lows[-1][1] > price_lows[-2][1] and price_lows[-1][2] < price_lows[-2][2]:
                divergences.append({
                    'type': 'Hidden Bullish',
                    'time': df.iloc[i]['time'],
                    'price': df.iloc[i]['close'],
                    'indicator': 'RSI'
                })
    
    return divergences

# Function to detect trend reversals
def detect_trend_reversals(df):
    reversals = []
    
    # Need sufficient data
    if len(df) < 50:
        return reversals
    
    for i in range(30, len(df)-1):
        # Check for trend change using moving averages crossovers
        if (df.iloc[i-1]['sma20'] < df.iloc[i-1]['sma50'] and 
            df.iloc[i]['sma20'] > df.iloc[i]['sma50']):
            reversals.append({
                'type': 'Bullish MA Crossover',
                'time': df.iloc[i]['time'],
                'price': df.iloc[i]['close']
            })
        
        elif (df.iloc[i-1]['sma20'] > df.iloc[i-1]['sma50'] and 
              df.iloc[i]['sma20'] < df.iloc[i]['sma50']):
            reversals.append({
                'type': 'Bearish MA Crossover',
                'time': df.iloc[i]['time'],
                'price': df.iloc[i]['close']
            })
        
        # Check for reversal based on RSI extremes
        if df.iloc[i-1]['rsi'] < 30 and df.iloc[i]['rsi'] > 30:
            reversals.append({
                'type': 'Bullish RSI Reversal',
                'time': df.iloc[i]['time'],
                'price': df.iloc[i]['close']
            })
        
        elif df.iloc[i-1]['rsi'] > 70 and df.iloc[i]['rsi'] < 70:
            reversals.append({
                'type': 'Bearish RSI Reversal',
                'time': df.iloc[i]['time'],
                'price': df.iloc[i]['close']
            })
        
        # Check for price action reversals (engulfing patterns)
        if (df.iloc[i-1]['close'] < df.iloc[i-1]['open'] and  # Previous candle is bearish
            df.iloc[i]['close'] > df.iloc[i]['open'] and      # Current candle is bullish
            df.iloc[i]['close'] > df.iloc[i-1]['open'] and    # Current close above previous open
            df.iloc[i]['open'] < df.iloc[i-1]['close']):      # Current open below previous close
            reversals.append({
                'type': 'Bullish Engulfing',
                'time': df.iloc[i]['time'],
                'price': df.iloc[i]['close']
            })
        
        elif (df.iloc[i-1]['close'] > df.iloc[i-1]['open'] and  # Previous candle is bullish
              df.iloc[i]['close'] < df.iloc[i]['open'] and      # Current candle is bearish
              df.iloc[i]['close'] < df.iloc[i-1]['open'] and    # Current close below previous open
              df.iloc[i]['open'] > df.iloc[i-1]['close']):      # Current open above previous close
            reversals.append({
                'type': 'Bearish Engulfing',
                'time': df.iloc[i]['time'],
                'price': df.iloc[i]['close']
            })
    
    return reversals

# Function to analyze ranges and volatility
def analyze_ranges(df):
    analysis = {}
    
    # Calculate average daily range
    avg_daily_range = df['daily_range'].mean()
    
    # Calculate ATR as percentage of price
    current_price = df.iloc[-1]['close']
    atr_percentage = (df.iloc[-1]['atr'] / current_price) * 100
    
    # Calculate historical volatility
    returns = np.log(df['close'] / df['close'].shift(1))
    historical_volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility in percentage
    
    # Determine if currently in a range or trending
    recent_df = df.iloc[-20:]
    range_high = recent_df['high'].max()
    range_low = recent_df['low'].min()
    range_width = (range_high - range_low) / current_price * 100
    
    # Check if price is bouncing between support and resistance
    touches_high = sum(1 for h in recent_df['high'] if h > range_high * 0.995)
    touches_low = sum(1 for l in recent_df['low'] if l < range_low * 1.005)
    
    # Linear regression to determine trend strength
    x = np.arange(len(recent_df))
    slope, intercept, r_value, p_value, std_err = linregress(x, recent_df['close'])
    r_squared = r_value ** 2
    market_condition = ""
    if r_squared < 0.3 and touches_high >= 2 and touches_low >= 2:
        market_condition = "Range-bound"
    elif r_squared > 0.7:
        if slope > 0:
            market_condition = "Strong Uptrend"
        else:
            market_condition = "Strong Downtrend"
    elif r_squared > 0.3:
        if slope > 0:
            market_condition = "Weak Uptrend"
        else:
            market_condition = "Weak Downtrend"
    else:
        market_condition = "Choppy/Undefined"
    
    analysis['avg_daily_range'] = avg_daily_range
    analysis['atr'] = df.iloc[-1]['atr']
    analysis['atr_percentage'] = atr_percentage
    analysis['historical_volatility'] = historical_volatility
    analysis['range_high'] = range_high
    analysis['range_low'] = range_low
    analysis['range_width_percent'] = range_width
    analysis['market_condition'] = market_condition
    analysis['trend_strength'] = r_squared
    return analysis

# Function to estimate potential slippage
def analyze_slippage(symbol, account_volume):
    """
    Estimate potential slippage based on order book depth and historical spread data
    """
    # Get current symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return {"error": f"Symbol {symbol} not found"}
    
    # Get current spread
    current_spread_points = symbol_info.spread
    point_value = symbol_info.point
    spread_pips = current_spread_points * point_value
    
    # Get order book (market depth)
    order_book = mt5.market_book_get(symbol)
    if order_book is None:
        return {
            "spread_points": current_spread_points,
            "spread_pips": spread_pips,
            "estimated_slippage": "Unknown - No market depth data"
        }
    
    # Calculate order book liquidity
    buy_volume = sum(entry.volume for entry in order_book if entry.type == mt5.BOOK_TYPE_BUY)
    sell_volume = sum(entry.volume for entry in order_book if entry.type == mt5.BOOK_TYPE_SELL)
    
    # Basic slippage estimation based on account volume vs. available liquidity
    liquidity_ratio = min(buy_volume, sell_volume) / max(account_volume, 0.01)
    
    estimated_slippage = 0
    slippage_risk = ""
    
    if liquidity_ratio > 100:  # Very liquid
        estimated_slippage = spread_pips * 0.1
        slippage_risk = "Very Low"
    elif liquidity_ratio > 50:  # Liquid
        estimated_slippage = spread_pips * 0.25
        slippage_risk = "Low"
    elif liquidity_ratio > 10:  # Moderate
        estimated_slippage = spread_pips * 0.5
        slippage_risk = "Moderate"
    elif liquidity_ratio > 5:  # Low
        estimated_slippage = spread_pips * 1
        slippage_risk = "High"
    else:  # Very low
        estimated_slippage = spread_pips * 2
        slippage_risk = "Very High"
    
    return {
        "spread_points": current_spread_points,
        "spread_pips": spread_pips,
        "buy_liquidity": buy_volume,
        "sell_liquidity": sell_volume,
        "liquidity_ratio": liquidity_ratio,
        "estimated_slippage": estimated_slippage,
        "slippage_risk": slippage_risk
    }

# Comprehensive market analysis function
def analyze_market_conditions(symbol, timeframes, account_volume=1.0):
    """
    Comprehensive analysis of market conditions across multiple timeframes
    """
    # Get multi-timeframe data
    data = get_multi_timeframe_data(symbol, timeframes)
    if not data:
        return {"error": "Failed to retrieve market data"}
    analysis_results = {}
    
    # Analyze each timeframe
    for tf, df in data.items():
        # Add technical indicators
        df = add_technical_indicators(df)
        # Get recent data for analysis
        latest = df.iloc[-1]
        
        # Detect divergences
        divergences = detect_divergences(df)
        # Detect trend reversals
        reversals = detect_trend_reversals(df)
        # Analyze ranges and volatility
        range_analysis = analyze_ranges(df)
        # Determine trend direction
        print(latest['close'], round(latest['sma200'], 4), round(latest['sma50'], 4), round(latest['sma20'], 4))
        if latest['close'] > latest['sma200'] and latest['sma50'] > latest['sma200']:
            long_term_trend = "Bullish"
        elif latest['close'] < latest['sma200'] and latest['sma50'] < latest['sma200']:
            long_term_trend = "Bearish"
        else:
            long_term_trend = "Neutral"
        
        if latest['close'] > latest['sma50'] and latest['sma20'] > latest['sma50']:
            medium_term_trend = "Bullish"
        elif latest['close'] < latest['sma50'] and latest['sma20'] < latest['sma50']:
            medium_term_trend = "Bearish"
        else:
            medium_term_trend = "Neutral"
        
        if latest['close'] > latest['sma20'] and latest['ema20'] > latest['sma20']:
            short_term_trend = "Bullish"
        elif latest['close'] < latest['sma20'] and latest['ema20'] < latest['sma20']:
            short_term_trend = "Bearish"
        else:
            short_term_trend = "Neutral"
        
        # Momentum status
        if latest['rsi'] > 70:
            momentum = "Overbought"
        elif latest['rsi'] < 30:
            momentum = "Oversold"
        elif latest['rsi'] > 50 and latest['rsi'] < 70:
            momentum = "Bullish"
        elif latest['rsi'] > 30 and latest['rsi'] < 50:
            momentum = "Bearish"
        else:
            momentum = "Neutral"
        
        # Recent candle patterns
        recent_candles = []
        
        # Check last 3 candles for specific patterns
        for i in range(-3, 0):
            candle = df.iloc[i]
            prev_candle = df.iloc[i-1] if i > -3 else None
            
            body_size = abs(candle['close'] - candle['open'])
            range_size = candle['high'] - candle['low']
            body_ratio = body_size / range_size if range_size > 0 else 0
            
            if body_ratio < 0.2:
                recent_candles.append("Doji")
            elif candle['close'] > candle['open'] and body_ratio > 0.6:
                recent_candles.append("Strong Bullish")
            elif candle['close'] < candle['open'] and body_ratio > 0.6:
                recent_candles.append("Strong Bearish")
            elif candle['close'] > candle['open']:
                recent_candles.append("Bullish")
            elif candle['close'] < candle['open']:
                recent_candles.append("Bearish")
            
            # Check for hammer and shooting star
            if body_ratio < 0.5:
                upper_wick = candle['high'] - max(candle['open'], candle['close'])
                lower_wick = min(candle['open'], candle['close']) - candle['low']
                
                if lower_wick > 2 * body_size and upper_wick < 0.2 * lower_wick:
                    recent_candles[-1] = "Hammer"
                elif upper_wick > 2 * body_size and lower_wick < 0.2 * upper_wick:
                    recent_candles[-1] = "Shooting Star"
        
        # Determine support and resistance levels
        supports = [
            latest['lowest_20'],
            latest['lowest_50'],
            latest['bb_lower']
        ]
        
        resistances = [
            latest['highest_20'],
            latest['highest_50'],
            latest['bb_upper']
        ]
        
    # Store the analysis results for this timeframe
    analysis_results[tf] = {
        "timestamp": latest['time'],
        "price": latest['close'],
        "long_term_trend": long_term_trend,
        "medium_term_trend": medium_term_trend,
        "short_term_trend": short_term_trend,
        "momentum": momentum,
        "range_analysis": range_analysis,
        "recent_candles": recent_candles,
        "support_levels": supports,
        "resistance_levels": resistances,
        "divergences": divergences,
        "reversals": reversals,
        "indicators": {
            "rsi": latest['rsi'],
            "macd": latest['macd'],
            "macd_signal": latest['macd_signal'],
            "atr": latest['atr'],
            "atr_percentage": (latest['atr'] / latest['close']) * 100
        }
    }
    
    # Analyze slippage
    slippage_analysis = analyze_slippage(symbol, account_volume)
    
    # Combine all analysis into a comprehensive result
    comprehensive_analysis = {
        "symbol": symbol,
        "timeframe_analysis": analysis_results,
        "slippage_analysis": slippage_analysis,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return comprehensive_analysis

# Trading decision function based on market conditions
def generate_trading_strategy(market_analysis, risk_percentage=1.0, max_slippage=5.0):
    """
    Generate trading decisions based on comprehensive market analysis
    """
    symbol = market_analysis['symbol']
    
    # Extract analysis for different timeframes
    m15_analysis = None
    m5_analysis = None
    m2_analysis = None
    
    for tf, analysis in market_analysis['timeframe_analysis'].items():
        if tf == mt5.TIMEFRAME_M15:
            m15_analysis = analysis
        elif tf == mt5.TIMEFRAME_M5:
            m5_analysis = analysis
        elif tf == mt5.TIMEFRAME_M2:
            m2_analysis = analysis
    
    if not (m15_analysis and m5_analysis and m2_analysis):
        return {"error": "Missing required timeframe analysis"}
    
    # Check if slippage risk is acceptable
    slippage_risk = market_analysis['slippage_analysis'].get('slippage_risk', 'Unknown')
    estimated_slippage = 2 #market_analysis['slippage_analysis'].get('estimated_slippage', max_slippage + 1)
    if estimated_slippage > max_slippage:
        return {
            "decision": "No Trade",
            "reason": f"Slippage risk too high: {slippage_risk} with estimated {estimated_slippage} pips slippage"
        }
    
    # Look for multi-timeframe confluence
    long_confluence = (
        m15_analysis['long_term_trend'] == "Bullish" and
        m5_analysis['medium_term_trend'] == "Bullish" and
        m2_analysis['short_term_trend'] == "Bullish"
    )
    
    short_confluence = (
        m15_analysis['long_term_trend'] == "Bearish" and
        m5_analysis['medium_term_trend'] == "Bearish" and
        m2_analysis['short_term_trend'] == "Bearish"
    )
    
    # Check for reversals that might invalidate trend
    recent_reversals = m2_analysis['reversals']
    has_bearish_reversal = any(r['type'].startswith('Bearish') for r in recent_reversals[-5:] if r)
    has_bullish_reversal = any(r['type'].startswith('Bullish') for r in recent_reversals[-5:] if r)
    
    # Check for divergences that might signal reversal
    recent_divergences = m2_analysis['divergences']
    has_bearish_divergence = any(d['type'].startswith('Regular Bearish') for d in recent_divergences[-5:] if d)
    has_bullish_divergence = any(d['type'].startswith('Regular Bullish') for d in recent_divergences[-5:] if d)
    
    # Current price and market condition
    current_price = m2_analysis['price']
    market_condition = m2_analysis['range_analysis']['market_condition']
    
    # Determine if we should trade based on conditions
    if market_condition == "Range-bound":
        # Trading strategy for range-bound markets
        range_high = m2_analysis['range_analysis']['range_high']
        range_low = m2_analysis['range_analysis']['range_low']
        
        distance_to_high = (range_high - current_price) / current_price * 100
        distance_to_low = (current_price - range_low) / current_price * 100
        
        if distance_to_high < 0.2:  # Very close to range high
            decision = "Sell"
            reason = "Price at top of established range"
            risk_reward = 2.0  # Risk:Reward ratio
        elif distance_to_low < 0.2:  # Very close to range low
            decision = "Buy"
            reason = "Price at bottom of established range"
            risk_reward = 2.0
        else:
            decision = "No Trade"
            reason = "Price in middle of range - wait for edge"
            risk_reward = 0
    elif long_confluence and not has_bearish_reversal and not has_bearish_divergence:
        # Strong uptrend strategy
        if m2_analysis['momentum'] == "Oversold" or m2_analysis['indicators']['rsi'] < 40:
            decision = "Buy"
            reason = "Strong uptrend with pullback to oversold levels"
            risk_reward = 3.0
        elif m5_analysis['momentum'] == "Bullish" and m2_analysis['momentum'] == "Bullish":
            decision = "Buy"
            reason = "Strong bullish momentum across timeframes"
            risk_reward = 2.5
        else:
            decision = "Wait"
            reason = "Uptrend intact but waiting for better entry"
            risk_reward = 0
    elif short_confluence and not has_bullish_reversal and not has_bullish_divergence:
        # Strong downtrend strategy
        if m2_analysis['momentum'] == "Overbought" or m2_analysis['indicators']['rsi'] > 60:
            decision = "Sell"
            reason = "Strong downtrend with rally to overbought levels"
            risk_reward = 3.0
        elif m5_analysis['momentum'] == "Bearish" and m2_analysis['momentum'] == "Bearish":
            decision = "Sell"
            reason = "Strong bearish momentum across timeframes"
            risk_reward = 2.5
        else:
            decision = "Wait"
            reason = "Downtrend intact but waiting for better entry"
            risk_reward = 0
    else:
        # Mixed or unclear conditions
        if has_bullish_reversal and m15_analysis['long_term_trend'] != "Bearish":
            decision = "Consider Buy"
            reason = "Potential trend reversal to upside"
            risk_reward = 1.5
        elif has_bearish_reversal and m15_analysis['long_term_trend'] != "Bullish":
            decision = "Consider Sell"
            reason = "Potential trend reversal to downside"
            risk_reward = 1.5
        else:
            decision = "No Trade"
            reason = "Conflicting signals across timeframes"
            risk_reward = 0
    
    # Calculate position size based on risk
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return {"error": f"Symbol info for {symbol} not found"}
    
    account_info = mt5.account_info()
    if account_info is None:
        return {"error": "Account info not available"}
    
    account_balance = account_info.balance
    risk_amount = account_balance * (risk_percentage / 100)
    
    # Calculate stop loss distance based on ATR
    atr = m2_analysis['indicators']['atr']
    sl_multiplier = 2.0  # Multiple of ATR for stop loss
    
    sl_distance_price = atr * sl_multiplier
    sl_distance_pips = sl_distance_price / symbol_info.point
    
    # Calculate position size
    if decision == "Buy" or decision == "Consider Buy":
        stop_loss = current_price - sl_distance_price
        take_profit = current_price + (sl_distance_price * risk_reward)
    elif decision == "Sell" or decision == "Consider Sell":
        stop_loss = current_price + sl_distance_price
        take_profit = current_price - (sl_distance_price * risk_reward)
    else:
        stop_loss = 0
        take_profit = 0
    
    pip_value = symbol_info.trade_tick_value * (sl_distance_pips / symbol_info.trade_tick_size)
    position_size = risk_amount / pip_value if pip_value > 0 else 0
    
    # Calculate max position size based on available margin
    max_leverage = account_info.leverage
    margin_required = current_price * symbol_info.volume_min * 100000 / max_leverage
    max_position_size = account_balance * 0.2 / margin_required  # Using 20% of balance max
    
    # Adjust position size if necessary
    position_size = min(position_size, max_position_size)
    position_size = max(position_size, symbol_info.volume_min)
    position_size = round(position_size / symbol_info.volume_step) * symbol_info.volume_step
    
    return {
        "symbol": symbol,
        "decision": decision,
        "reason": reason,
        "current_price": current_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "position_size": position_size,
        "risk_amount": risk_amount,
        "risk_reward_ratio": risk_reward,
        "estimated_slippage": estimated_slippage,
        "long_term_trend": m15_analysis['long_term_trend'],
        "medium_term_trend": m5_analysis['medium_term_trend'],
        "short_term_trend": m2_analysis['short_term_trend'],
        "market_condition": market_condition
    }

# Function to execute trades based on the strategy
def execute_trade(strategy, sl_buffer=5, tp_buffer=5, max_attempts=3):
    """
    Execute a trade based on the strategy with retry logic for slippage
    
    Parameters:
    - strategy: Trading strategy dictionary from generate_trading_strategy
    - sl_buffer: Additional buffer for stop loss in points to prevent immediate triggering
    - tp_buffer: Additional buffer for take profit in points
    - max_attempts: Maximum number of attempts to execute the trade
    
    Returns:
    - Trade result dictionary
    """
    if strategy.get('decision') not in ['Buy', 'Sell']:
        return {"status": "No Trade", "reason": strategy.get('reason', "No valid trading decision")}
    
    symbol = strategy['symbol']
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return {"status": "Error", "reason": f"Symbol {symbol} not found"}
    
    # Ensure the symbol is selected in Market Watch
    if not symbol_info.visible:
        mt5.symbol_select(symbol, True)
    
    # Prepare trade request
    action = mt5.TRADE_ACTION_DEAL
    order_type = mt5.ORDER_TYPE_BUY if strategy['decision'] == 'Buy' else mt5.ORDER_TYPE_SELL
    price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
    
    sl = strategy['stop_loss']
    tp = strategy['take_profit']
    
    # Apply buffer to stop loss and take profit
    point = symbol_info.point
    if order_type == mt5.ORDER_TYPE_BUY:
        sl -= sl_buffer * point
        tp += tp_buffer * point
    else:
        sl += sl_buffer * point
        tp -= tp_buffer * point
    
    volume = 0.01
    deviation = 5  # Max slippage in points
    request = {
        "action": action,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": deviation,
        "magic": 123456,  # Magic number for identifying trades
        # "comment": f"Python Trade - {strategy['reason']}",
        "comment": "Python Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    
    # Attempt to execute the trade with retry logic
    for attempt in range(1, max_attempts + 1):
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return {
                "status": "Success",
                "order_id": result.order,
                "volume": volume,
                "price": price,
                "stop_loss": sl,
                "take_profit": tp,
                "attempt": attempt
            }
        
        elif result.retcode == mt5.TRADE_RETCODE_REQUOTE:
            # Update price and try again
            new_price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
            request["price"] = new_price
            print(f"Requote, attempt {attempt}/{max_attempts} with new price {new_price}")
        
        elif result.retcode == mt5.TRADE_RETCODE_INVALID_VOLUME:
            # Adjust volume and try again
            new_volume = symbol_info.volume_min
            request["volume"] = new_volume
            print(f"Invalid volume, attempt {attempt}/{max_attempts} with minimum volume {new_volume}")
        
        else:
            return {
                "status": "Error",
                "reason": f"Order failed, retcode: {result.retcode}",
                "attempt": attempt
            }
    
    return {"status": "Error", "reason": f"Failed after {max_attempts} attempts"}

# Function to monitor open positions and adjust them as needed
def monitor_positions(symbol, trailing_stop_pips=20, break_even_pips=30):
    """
    Monitor open positions and adjust stop losses based on market conditions
    
    Parameters:
    - symbol: Trading symbol
    - trailing_stop_pips: Distance in pips for trailing stop
    - break_even_pips: Profit in pips at which to move stop loss to break even
    
    Returns:
    - Dictionary of position updates
    """
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return {"status": "No open positions"}
    
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return {"status": "Error", "reason": f"Symbol {symbol} not found"}
    
    point = symbol_info.point
    digits = symbol_info.digits
    
    position_updates = []
    
    for position in positions:
        position_id = position.ticket
        position_type = position.type  # 0 for buy, 1 for sell
        open_price = position.price_open
        current_price = position.price_current
        current_sl = position.sl
        current_tp = position.tp
        
        # Calculate profit in pips
        profit_points = (current_price - open_price) / point if position_type == 0 else (open_price - current_price) / point
        
        update_needed = False
        new_sl = current_sl
        
        # Check if we should move to break even
        if profit_points >= break_even_pips and (
            (position_type == 0 and current_sl < open_price) or
            (position_type == 1 and (current_sl == 0 or current_sl > open_price))
        ):
            # Move stop loss to break even + small buffer
            new_sl = open_price + (2 * point) if position_type == 0 else open_price - (2 * point)
            update_needed = True
        
        # Check if we should apply trailing stop
        if profit_points >= trailing_stop_pips:
            # Calculate trailing stop level
            if position_type == 0:  # Buy position
                trail_sl = current_price - (trailing_stop_pips * point)
                if current_sl < trail_sl:
                    new_sl = trail_sl
                    update_needed = True
            else:  # Sell position
                trail_sl = current_price + (trailing_stop_pips * point)
                if current_sl == 0 or current_sl > trail_sl:
                    new_sl = trail_sl
                    update_needed = True
        
        # Update stop loss if needed
        if update_needed:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": position_id,
                "sl": round(new_sl, digits),
                "tp": current_tp
            }
            
            result = mt5.order_send(request)
            
            position_updates.append({
                "position_id": position_id,
                "type": "Buy" if position_type == 0 else "Sell",
                "open_price": open_price,
                "current_price": current_price,
                "old_sl": current_sl,
                "new_sl": new_sl,
                "profit_pips": profit_points,
                "status": "Updated" if result.retcode == mt5.TRADE_RETCODE_DONE else "Failed",
                "retcode": result.retcode
            })
    
    return {"status": "Processed", "updates": position_updates}

# Function to close positions based on analysis
def close_positions_on_reversal(symbol, market_analysis):
    """
    Close positions if market analysis shows trend reversal against position direction
    
    Parameters:
    - symbol: Trading symbol
    - market_analysis: Market analysis dictionary
    
    Returns:
    - Dictionary of closed positions
    """
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return {"status": "No open positions"}
    
    # Extract analysis for H1 timeframe
    m2_analysis = None
    for tf, analysis in market_analysis['timeframe_analysis'].items():
        if tf == mt5.TIMEFRAME_H1:
            m2_analysis = analysis
            break
    
    if m2_analysis is None:
        return {"status": "Error", "reason": "H1 timeframe analysis not available"}
    
    # Check for reversal signals
    recent_reversals = m2_analysis['reversals']
    has_bearish_reversal = any(r['type'].startswith('Bearish') for r in recent_reversals[-3:] if r)
    has_bullish_reversal = any(r['type'].startswith('Bullish') for r in recent_reversals[-3:] if r)
    
    # Check for divergences that might signal reversal
    recent_divergences = m2_analysis['divergences']
    has_bearish_divergence = any(d['type'].startswith('Regular Bearish') for d in recent_divergences[-3:] if d)
    has_bullish_divergence = any(d['type'].startswith('Regular Bullish') for d in recent_divergences[-3:] if d)
    
    closed_positions = []
    
    for position in positions:
        position_id = position.ticket
        position_type = position.type  # 0 for buy, 1 for sell
        
        # Determine if we should close the position
        should_close = False
        reason = ""
        
        if position_type == 0:  # Buy position
            if has_bearish_reversal:
                should_close = True
                reason = "Bearish reversal signal"
            elif has_bearish_divergence:
                should_close = True
                reason = "Bearish divergence signal"
            elif m2_analysis['momentum'] == "Overbought" and m2_analysis['indicators']['rsi'] > 75:
                should_close = True
                reason = "Extreme overbought conditions"
        else:  # Sell position
            if has_bullish_reversal:
                should_close = True
                reason = "Bullish reversal signal"
            elif has_bullish_divergence:
                should_close = True
                reason = "Bullish divergence signal"
            elif m2_analysis['momentum'] == "Oversold" and m2_analysis['indicators']['rsi'] < 25:
                should_close = True
                reason = "Extreme oversold conditions"
        
        # Close position if needed
        if should_close:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": position_id,
                "symbol": symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position_type == 0 else mt5.ORDER_TYPE_BUY,
                "price": mt5.symbol_info_tick(symbol).bid if position_type == 0 else mt5.symbol_info_tick(symbol).ask,
                "deviation": 20,
                "magic": 123456,
                "comment": f"Close on {reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            closed_positions.append({
                "position_id": position_id,
                "type": "Buy" if position_type == 0 else "Sell",
                "reason": reason,
                "status": "Closed" if result.retcode == mt5.TRADE_RETCODE_DONE else "Failed",
                "retcode": result.retcode
            })
    
    return {"status": "Processed", "closed": closed_positions}

# Main function to run the entire system
def run_trading_system(symbols, timeframes, risk_percentage=1.0, max_slippage=5.0, 
                      trailing_stop_pips=20, break_even_pips=30, 
                      account_volume=1.0, monitor_interval_minutes=60):
    """
    Run the complete trading system with market analysis, trade execution and position management
    
    Parameters:
    - symbols: List of symbols to analyze and trade
    - timeframes: List of timeframes to analyze
    - risk_percentage: Risk per trade as percentage of account balance
    - max_slippage: Maximum allowed slippage in pips
    - trailing_stop_pips: Pips for trailing stop
    - break_even_pips: Pips of profit at which to move stop loss to break even
    - account_volume: Standard lot size for account
    - monitor_interval_minutes: How often to run position monitoring in minutes
    
    Returns:
    - None (prints status updates)
    """
    if not initialize_mt5(account=91362231, server="MetaQuotes-Demo"):
        print("Failed to initialize MT5. Exiting.")
        return
    
    print(f"Trading system initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Analyzing symbols: {', '.join(symbols)}")
    print(f"Using timeframes: {', '.join([str(tf) for tf in timeframes])}")
    print(f"Risk per trade: {risk_percentage}% of balance")
    
    # Main trading loop
    last_monitor_time = datetime.now()
    
    # try:
    
    while True: 
        # while True:
        #     time.sleep(1)
        #     if datetime.now().second == 50:break
        for symbol in symbols:
            print(f"\nAnalyzing {symbol} at {datetime.now().strftime('%H:%M:%S')}")
            
            # Comprehensive market analysis
            market_analysis = analyze_market_conditions(symbol, timeframes, account_volume)
            print(market_analysis)
            if "error" in market_analysis:
                print(f"Error in market analysis: {market_analysis['error']}")
                continue
            
            # Generate trading strategy
            strategy = generate_trading_strategy(market_analysis, risk_percentage, max_slippage)
            
            if "error" in strategy:
                print(f"Error in strategy generation: {strategy['error']}")
                continue
            print(f"\nTrading decision for {symbol}: {strategy['decision']}")
            print(f"Reason: {strategy['reason']}")
            print(f"Long-term trend: {strategy['long_term_trend']}")
            print(f"Medium-term trend: {strategy['medium_term_trend']}")
            print(f"Short-term trend: {strategy['short_term_trend']}")
            print(f"Market condition: {strategy['market_condition']}")
            
            # Execute trade if decision is to Buy or Sell
            if strategy['decision'] in ['Buy', 'Sell']:
                print(f"Executing {strategy['decision']} trade for {symbol}...")
                result = execute_trade(strategy)
                print(f"Trade execution result: {result['status']}")
                if result['status'] == 'Success':
                    print(f"Order ID: {result['order_id']}, Entry: {result['price']}")
                    print(f"Stop Loss: {result['stop_loss']}, Take Profit: {result['take_profit']}")
            
            # Check if we need to monitor and update positions
            current_time = datetime.now()
            time_diff = (current_time - last_monitor_time).total_seconds() / 60
            if time_diff >= monitor_interval_minutes:
                for symbol in symbols:
                    print(f"\nMonitoring positions for {symbol}...")
                    
                    # Update stop losses
                    monitoring_result = monitor_positions(symbol, trailing_stop_pips, break_even_pips)
                    if monitoring_result['status'] == 'Processed' and len(monitoring_result['updates']) > 0:
                        print(f"Updated stop losses for {len(monitoring_result['updates'])} positions")
                    
                    # Close positions based on market reversals
                    close_result = close_positions_on_reversal(symbol, market_analysis)
                    if close_result['status'] == 'Processed' and len(close_result['closed']) > 0:
                        print(f"Closed {len(close_result['closed'])} positions due to market reversals")
                
                last_monitor_time = current_time
        
        # Wait before next analysis cycle
        print(f"\nWaiting for 2 minutes before next analysis cycle...")
        count = 0
        while True:
            time.sleep(117)
            break

    # except KeyboardInterrupt:
    #     print("\nTrading system stopped by user.")
    # except Exception as e:
    #     print(f"\nError in trading system: {e}")
    # finally:
    #     mt5.shutdown()
    #     print("MT5 connection closed.")

# Example usage
if __name__ == "__main__":
    # Define symbols and timeframes to analyze
    symbols_to_trade = ["EURUSD"]
    timeframes_to_analyze = [mt5.TIMEFRAME_M2, mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15]
    
    # Run the trading system
    run_trading_system(
        symbols=symbols_to_trade,
        timeframes=timeframes_to_analyze,
        risk_percentage=1.0,  # Risk 1% per trade
        max_slippage=5.0,     # Maximum 5 pips slippage
        trailing_stop_pips=25, # 25 pips trailing stop
        break_even_pips=30,   # Move to break even after 30 pips profit
        account_volume=0.1,   # 0.1 lot standard position
        monitor_interval_minutes=30  # Monitor positions every 30 minutes
    )