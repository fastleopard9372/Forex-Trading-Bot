import logging
import pandas as pd
import numpy as np
from datetime import datetime
import talib as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .base_strategy import BaseStrategy
from order import Order, OrderType, OrderSide, OrderStatus

bot_logger = logging.getLogger("bot_logger")


class RsiMacd(BaseStrategy):

    def __init__(self, name, params, tfs):
        super().__init__(name, params, tfs)
        self.tf = self.tfs["tf"]  # Primary timeframe
        
        # Market-specific defaults based on typical volatility
        symbol = self.params.get("symbol", "generic").upper()
        
        # Initialize with symbol-specific defaults
        if "JPY" in symbol:  # Higher volatility for JPY pairs
            self.ema_short = self.params.get("ema_short", 20)
            self.ema_medium = self.params.get("ema_medium", 100)
            self.ema_long = self.params.get("ema_long", 200)
            self.atr_multiplier_sl = self.params.get("atr_multiplier_sl", 2.5)
            self.dynamic_tp_multiplier = self.params.get("dynamic_tp_multiplier", 2.25)
        elif "EUR" in symbol or "GBP" in symbol:  # Medium volatility
            self.ema_short = self.params.get("ema_short", 30)
            self.ema_medium = self.params.get("ema_medium", 120)
            self.ema_long = self.params.get("ema_long", 240)
            self.atr_multiplier_sl = self.params.get("atr_multiplier_sl", 2.0)
            self.dynamic_tp_multiplier = self.params.get("dynamic_tp_multiplier", 2.0)
        else:  # Default/lower volatility pairs
            self.ema_short = self.params.get("ema_short", 25)
            self.ema_medium = self.params.get("ema_medium", 55)
            self.ema_long = self.params.get("ema_long", 110)
            self.atr_multiplier_sl = self.params.get("atr_multiplier_sl", 1.75)
            self.dynamic_tp_multiplier = self.params.get("dynamic_tp_multiplier", 1.75)
        
        # RSI parameters
        self.rsi_period = self.params.get("rsi_period", 14)
        self.rsi_overbought = self.params.get("rsi_overbought", 70)
        self.rsi_oversold = self.params.get("rsi_oversold", 30)
        
        # Volatility parameters
        self.atr_period = self.params.get("atr_period", 7)
        self.bb_period = self.params.get("bb_period", 20)
        self.bb_dev = self.params.get("bb_dev", 2.0)
        
        # Market condition parameters
        self.adx_period = self.params.get("adx_period", 14)
        self.adx_threshold = self.params.get("adx_threshold", 18)
        
        # Market disorder detection
        self.market_disorder = False
        self.market_disorder_threshold = self.params.get("market_disorder_threshold", 2.0)
        
        # Trading parameters
        self.min_trades_interval = self.params.get("min_trades_interval", 1)
        self.last_trade_idx = 0
        self.trade_history = []
        self.max_history_size = 100  # Increased for better optimization
        
        # Dynamic TP/SL
        self.use_dynamic_tpsl = self.params.get("use_dynamic_tpsl", True)
        
        # Partial profit parameters
        self.partial_profit_enabled = self.params.get("partial_profit_enabled", True)
        self.partial_profit_size = self.params.get("partial_profit_size", 0.5)  # 50% by default
        self.partial_profit_min_pips = self.params.get("partial_profit_min_pips", 5.0)
        
        # Optimal exit detection parameters
        self.exit_detection_enabled = self.params.get("exit_detection_enabled", True)
        self.exit_optimization_lookback = self.params.get("exit_optimization_lookback", 20)
        
        # Auto-optimization parameters
        self.auto_optimization_enabled = self.params.get("auto_optimization_enabled", True)
        self.min_trades_for_optimization = self.params.get("min_trades_for_optimization", 30)
        self.optimization_interval = self.params.get("optimization_interval", 10)
        self.trades_since_optimization = 0
        
        # Trading state
        self.trading_flag = 0
        
        # Session filter (time-based)
        self.session_filter = self.params.get("session_filter", True)
        
        # Trend tracking
        self.current_trend = 0  # -1=downtrend, 0=neutral, 1=uptrend
        self.trend_start_idx = 0
        self.trend_start_price = 0.0
        
        # Statistics storage for optimization
        self.market_stats = {
            "volatility_history": [],
            "trend_strength_history": [],
            "optimal_exits": {
                "trending": [],
                "ranging": [],
                "volatile": [],
                "normal": []
            }
        }

    def attach(self, tfs_chart):
        self.tfs_chart = tfs_chart
        self.init_indicators()

    def init_indicators(self):
        # Get chart data
        chart = self.tfs_chart[self.tf]
        
        # Basic trend indicators - EMAs
        self.ema_s = ta.EMA(chart["Close"], self.ema_short)
        self.ema_m = ta.EMA(chart["Close"], self.ema_medium)
        self.ema_l = ta.EMA(chart["Close"], self.ema_long)
        
        # Momentum indicator - RSI
        self.rsi = ta.RSI(chart["Close"], self.rsi_period)
        self.rsi_slope = pd.Series(np.gradient(self.rsi), index=self.rsi.index)
        
        # Volatility indicators
        self.atr = ta.ATR(chart["High"], chart["Low"], chart["Close"], self.atr_period)
        
        # Bollinger Bands
        self.upper_bb, self.middle_bb, self.lower_bb = ta.BBANDS(
            chart["Close"], 
            timeperiod=self.bb_period, 
            nbdevup=self.bb_dev, 
            nbdevdn=self.bb_dev
        )
        self.bb_width = (self.upper_bb - self.lower_bb) / self.middle_bb
        
        # Trend strength indicator
        self.adx = ta.ADX(chart["High"], chart["Low"], chart["Close"], self.adx_period)
        
        # MACD for additional confirmation
        self.macd, self.macdsignal, self.macdhist = ta.MACD(
            chart["Close"], 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        
        # Stochastic for momentum confirmation and divergence
        self.slowk, self.slowd = ta.STOCH(
            chart["High"], 
            chart["Low"], 
            chart["Close"],
            fastk_period=5,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0
        )
        
        # Trend direction and price location indicators (no copy needed)
        self.trend_direction = pd.Series(
            np.where(self.ema_s > self.ema_l, 1, np.where(self.ema_s < self.ema_l, -1, 0)),
            index=chart.index
        )
        
        self.price_location = pd.Series(
            (chart["Close"] - self.lower_bb) / (self.upper_bb - self.lower_bb),
            index=chart.index
        )
        
        # Is the market trending?
        self.is_trending = self.adx > self.adx_threshold
        
        # Record when we start trading
        self.start_trading_time = chart.iloc[-1]["Open time"]

        # Initialize market condition for all historical data
        self.market_condition_series = self.detect_market_condition(full_data=True)
        
        # Initialize trend tracking
        if len(self.trend_direction) > 0:
            self.current_trend = self.trend_direction.iloc[-1]
            self.trend_start_idx = self._find_trend_start_idx()
            if self.trend_start_idx < len(chart):
                self.trend_start_price = chart.iloc[self.trend_start_idx]["Close"]
        
        # Initialize market statistics for optimization
        self._initialize_market_statistics()
        
        # Update market disorder detection
        self.update_market_disorder()

    def _initialize_market_statistics(self):
        """Initialize historical market statistics for optimization"""
        chart = self.tfs_chart[self.tf]
        
        # Skip if not enough data
        if len(chart) < 100:
            return
            
        # Calculate volatility history (ATR as % of price)
        self.market_stats["volatility_history"] = [
            atr / price * 100 if price > 0 and not pd.isna(atr) else 0 
            for atr, price in zip(self.atr, chart["Close"])
        ]
        
        # Calculate trend strength history (ADX values)
        self.market_stats["trend_strength_history"] = [
            adx if not pd.isna(adx) else 0 
            for adx in self.adx
        ]
        
        # Calculate optimal exit points retrospectively
        for i in range(50, len(chart)):
            # Look for peaks and valleys to identify optimal exits
            current_condition = self.market_condition_series.iloc[i] if i < len(self.market_condition_series) else 1
            
            # Skip disorderly market conditions
            if current_condition == 5:
                continue
                
            # For uptrends, look for local peaks
            if self.trend_direction.iloc[i] == 1:
                # Check if this is a local peak (price higher than next 3 and previous 3 bars)
                surrounding_prices = chart.iloc[max(0, i-3):min(len(chart), i+4)]["Close"]
                if len(surrounding_prices) >= 4 and chart.iloc[i]["Close"] == max(surrounding_prices):
                    # Record optimal exit for this market condition
                    condition_name = {1: "normal", 2: "trending", 3: "volatile", 4: "ranging"}.get(current_condition, "normal")
                    self.market_stats["optimal_exits"][condition_name].append({
                        "idx": i,
                        "price": chart.iloc[i]["Close"],
                        "rsi": self.rsi.iloc[i] if i < len(self.rsi) else 50,
                        "adx": self.adx.iloc[i] if i < len(self.adx) else 25,
                        "trend_duration": i - self._find_trend_start_idx(i),
                        "bb_width": self.bb_width.iloc[i] if i < len(self.bb_width) else 0.02
                    })
            
            # For downtrends, look for local valleys
            elif self.trend_direction.iloc[i] == -1:
                # Check if this is a local valley (price lower than next 3 and previous 3 bars)
                surrounding_prices = chart.iloc[max(0, i-3):min(len(chart), i+4)]["Close"]
                if len(surrounding_prices) >= 4 and chart.iloc[i]["Close"] == min(surrounding_prices):
                    # Record optimal exit for this market condition
                    condition_name = {1: "normal", 2: "trending", 3: "volatile", 4: "ranging"}.get(current_condition, "normal")
                    self.market_stats["optimal_exits"][condition_name].append({
                        "idx": i,
                        "price": chart.iloc[i]["Close"],
                        "rsi": self.rsi.iloc[i] if i < len(self.rsi) else 50,
                        "adx": self.adx.iloc[i] if i < len(self.adx) else 25,
                        "trend_duration": i - self._find_trend_start_idx(i),
                        "bb_width": self.bb_width.iloc[i] if i < len(self.bb_width) else 0.02
                    })

    def _find_trend_start_idx(self, end_idx=None):
        """Find the index where the current trend started"""
        if end_idx is None:
            end_idx = len(self.trend_direction) - 1
            
        if end_idx < 0 or end_idx >= len(self.trend_direction):
            return 0
            
        current_direction = self.trend_direction.iloc[end_idx]
        
        # Walk backwards to find where the trend started
        start_idx = end_idx
        for i in range(end_idx, 0, -1):
            if self.trend_direction.iloc[i] != current_direction:
                start_idx = i + 1
                break
                
        return start_idx

    def update_indicators(self, tf):
        if tf != self.tf:
            return
            
        chart = self.tfs_chart[self.tf]
        last_idx = len(chart) - 1
        
        # Update EMAs
        self.ema_s.loc[last_idx] = ta.stream.EMA(chart["Close"], self.ema_short)
        self.ema_m.loc[last_idx] = ta.stream.EMA(chart["Close"], self.ema_medium)
        self.ema_l.loc[last_idx] = ta.stream.EMA(chart["Close"], self.ema_long)
        
        # Update RSI
        self.rsi.loc[last_idx] = ta.stream.RSI(chart["Close"], self.rsi_period)
        if last_idx > 0:
            self.rsi_slope.loc[last_idx] = self.rsi.iloc[last_idx] - self.rsi.iloc[last_idx-1]
        
        # Update ATR
        self.atr.loc[last_idx] = ta.stream.ATR(
            chart["High"], chart["Low"], chart["Close"], self.atr_period
        )
        
        # Update Bollinger Bands
        upper, middle, lower = ta.stream.BBANDS(
            chart["Close"], 
            timeperiod=self.bb_period, 
            nbdevup=self.bb_dev, 
            nbdevdn=self.bb_dev
        )
        self.upper_bb.loc[last_idx] = upper
        self.middle_bb.loc[last_idx] = middle
        self.lower_bb.loc[last_idx] = lower
        self.bb_width.loc[last_idx] = (upper - lower) / middle
        
        # Update ADX
        self.adx.loc[last_idx] = ta.stream.ADX(
            chart["High"], chart["Low"], chart["Close"], self.adx_period
        )
        
        # Update MACD
        macd, macdsignal, macdhist = ta.stream.MACD(
            chart["Close"], 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        self.macd.loc[last_idx] = macd
        self.macdsignal.loc[last_idx] = macdsignal
        self.macdhist.loc[last_idx] = macdhist
        
        # Update Stochastic
        slowk, slowd = ta.stream.STOCH(
            chart["High"],
            chart["Low"],
            chart["Close"],
            fastk_period=5,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0
        )
        self.slowk.loc[last_idx] = slowk
        self.slowd.loc[last_idx] = slowd
        
        # Update trend direction and price location
        current_trend_dir = 1 if self.ema_s.iloc[last_idx] > self.ema_l.iloc[last_idx] else (
            -1 if self.ema_s.iloc[last_idx] < self.ema_l.iloc[last_idx] else 0
        )
        self.trend_direction.loc[last_idx] = current_trend_dir
        
        # Check if trend direction changed
        if current_trend_dir != self.current_trend:
            self.current_trend = current_trend_dir
            self.trend_start_idx = last_idx
            self.trend_start_price = chart.iloc[last_idx]["Close"]
        
        self.price_location.loc[last_idx] = (
            (chart["Close"].iloc[last_idx] - self.lower_bb.iloc[last_idx]) / 
            (self.upper_bb.iloc[last_idx] - self.lower_bb.iloc[last_idx])
        )
        
        # Update is_trending
        self.is_trending.loc[last_idx] = self.adx.iloc[last_idx] > self.adx_threshold
        
        # Get market condition for just the latest candle
        new_condition = self.detect_market_condition(full_data=False)
        
        # Add to the series
        self.market_condition_series.loc[chart.index[last_idx]] = new_condition
        
        # Update market statistics for optimization
        self._update_market_statistics(last_idx)
        
        # Update market disorder detection
        self.update_market_disorder()

    def _update_market_statistics(self, idx):
        """Update market statistics with latest data"""
        chart = self.tfs_chart[self.tf]
        
        # Update volatility history
        if idx < len(self.atr) and idx < len(chart["Close"]):
            atr_value = self.atr.iloc[idx]
            price = chart["Close"].iloc[idx]
            if price > 0 and not pd.isna(atr_value):
                self.market_stats["volatility_history"].append(atr_value / price * 100)
            else:
                self.market_stats["volatility_history"].append(0)
        
        # Update trend strength history
        if idx < len(self.adx):
            adx_value = self.adx.iloc[idx]
            if not pd.isna(adx_value):
                self.market_stats["trend_strength_history"].append(adx_value)
            else:
                self.market_stats["trend_strength_history"].append(0)
    
    def detect_market_condition(self, full_data=False, start_idx=None, end_idx=None):
        """
        Detect market conditions based on indicators.
        
        Parameters:
        -----------
        full_data : bool
            If True, process all candles in chart. If False, only process latest candle.
        start_idx : int, optional
            Starting index for processing (default is 0 or len-1 depending on full_data)
        end_idx : int, optional
            Ending index for processing (default is len-1)
            
        Returns:
        --------
        pandas.Series or int
            Series with market conditions if full_data=True, otherwise single condition value
        """
        chart = self.tfs_chart[self.tf]
        
        # Determine processing range
        if full_data:
            start_idx = 0 if start_idx is None else max(0, start_idx)
            end_idx = len(chart) - 1 if end_idx is None else min(len(chart) - 1, end_idx)
            process_range = range(start_idx, end_idx + 1)
            conditions = [1] * start_idx  # Fill with default value for any skipped indices
        else:
            # Process only the latest candle by default
            end_idx = len(chart) - 1 if end_idx is None else min(len(chart) - 1, end_idx)
            process_range = [end_idx]
            conditions = None  # We'll return a single value, not a series
        
        # Define minimum lookback for indicator validity
        min_lookback = max(self.ema_long, self.bb_period, self.atr_period)
        
        # Process each candle in the specified range
        for i in process_range:
            # Skip early candles where indicators aren't fully formed
            if i < min_lookback:
                if full_data:
                    conditions.append(1)  # Default to Normal for early candles
                else:
                    return 1  # Return Normal for early candles
                continue
            
            # Extract relevant indicator values for this candle
            # Using direct indexing for efficiency
            current_price = chart["Close"].iloc[i]
            
            # Calculate percentage price change
            price_change_pct = 0
            if i > 0:
                prev_price = chart["Close"].iloc[i-1]
                price_change_pct = abs(current_price - prev_price) / prev_price * 100
            
            # Get indicator values
            atr_val = self.atr.iloc[i] if not pd.isna(self.atr.iloc[i]) else 0
            atr_pct = atr_val / current_price * 100
            
            bb_width_val = self.bb_width.iloc[i] if not pd.isna(self.bb_width.iloc[i]) else 0
            
            ema_short = self.ema_s.iloc[i]
            ema_medium = self.ema_m.iloc[i]
            ema_long = self.ema_l.iloc[i]
            
            rsi_val = self.rsi.iloc[i]
            adx_val = self.adx.iloc[i] if not pd.isna(self.adx.iloc[i]) else 0
            
            # Get price location within Bollinger Bands
            if not pd.isna(self.lower_bb.iloc[i]) and not pd.isna(self.upper_bb.iloc[i]):
                bb_range = self.upper_bb.iloc[i] - self.lower_bb.iloc[i]
                if bb_range > 0:
                    price_loc = (current_price - self.lower_bb.iloc[i]) / bb_range
                else:
                    price_loc = 0.5
            else:
                price_loc = 0.5
            
            # Determine market condition using hierarchical checks
            
            condition = 1  # Normal    
            
                        
            # 1. Check for trending market
            if ((ema_short > ema_medium > ema_long and rsi_val > 55 and adx_val > self.adx_threshold) or
                 (ema_short < ema_medium < ema_long and rsi_val < 45 and adx_val > self.adx_threshold)):
                condition = 2  # Trending
            
            # 2. Check for ranging market
            elif (0.4 < price_loc < 0.6 and 
                 40 < rsi_val < 60 and
                 bb_width_val < 0.025 and
                 adx_val < self.adx_threshold):
                condition = 4  # Ranging 
            
            # 3. Check for volatile market
            if (atr_pct > 0.08 or bb_width_val > 0.04):
                condition = 3  # Volatile

            # 4. Check for disorderly market (highest priority)
            elif (price_change_pct > 0.2 or  # Sudden large price change
                atr_pct > 0.15 or          # Very high volatility
                bb_width_val > 0.06):      # Very wide Bollinger Bands
                condition = 5  # Disorderly
            
            # Store or return the condition
            if full_data:
                conditions.append(condition)
            else:
                return condition
        
        # Return series if processing full data
        if full_data:
            return pd.Series(conditions, index=chart.index[start_idx:end_idx+1])
        
        # This should never be reached if not full_data
        return 1

    def update_market_disorder(self):
        """Detect disorderly market conditions"""
        chart = self.tfs_chart[self.tf]
        if len(chart) < 10:
            self.market_disorder = False
            return
            
        # Method 1: Check BB width (volatility)
        current_volatility = self.bb_width.iloc[-1]
        avg_volatility = self.bb_width.iloc[-10:].mean()
        volatility_ratio = current_volatility / avg_volatility if avg_volatility > 0 else 1.0
        
        # Method 2: Check price vs EMA distance
        ema_distance = abs(chart["Close"].iloc[-1] - self.ema_m.iloc[-1]) / self.ema_m.iloc[-1]
        ema_disorder = ema_distance > 0.002  # 0.2% deviation
        
        # Method 3: Check if current market condition is classified as disorderly
        condition_disorder = self.market_condition_series.iloc[-1] == 5
        
        # Combine all methods
        self.market_disorder = (
            volatility_ratio > self.market_disorder_threshold or 
            ema_disorder or
            condition_disorder
        )
        
        # Log market disorder detection
        if self.market_disorder:
            bot_logger.info(f"Market disorder detected: volatility_ratio={volatility_ratio:.2f}, "
                          f"ema_distance={ema_distance:.4f}")

    def adapt_parameters_to_market(self):
        """Dynamically adjust strategy parameters based on current market conditions"""
        chart = self.tfs_chart[self.tf]
        
        # Skip if not enough data
        if len(chart) < 200:
            return
        
        # Get current market stats
        current_condition = self.market_condition_series.iloc[-1]
        recent_volatility = self.atr.iloc[-20:].mean() / chart["Close"].iloc[-1] * 100  # ATR as % of price
        adx_strength = self.adx.iloc[-1]
        bb_width_current = self.bb_width.iloc[-1]
        
        # Calculate market characteristics over last 200 bars
        volatility_percentile = pd.Series(self.atr.iloc[-200:] / chart["Close"].iloc[-200:] * 100).rank(pct=True).iloc[-1]
        trending_percentile = pd.Series(self.adx.iloc[-200:]).rank(pct=True).iloc[-1]
        ranging_score = 1 - abs(self.rsi.iloc[-1] - 50) / 50  # 1 = neutral RSI, 0 = extreme RSI
        
        # 1. Adapt ATR multipliers for stop loss
        if current_condition == 5:  # Disorderly
            self.atr_multiplier_sl = 3.5
        elif current_condition == 3:  # Volatile
            self.atr_multiplier_sl = 2.5 + (volatility_percentile * 0.5)  # 2.5-3.0 range
        elif current_condition == 2:  # Trending
            self.atr_multiplier_sl = 2.0 + (trending_percentile * 0.5)  # 2.0-2.5 range
        elif current_condition == 4:  # Ranging
            self.atr_multiplier_sl = 1.5 + (volatility_percentile * 0.5)  # 1.5-2.0 range
        else:  # Normal
            self.atr_multiplier_sl = 2.0
        
        # 2. Adapt dynamic TP multiplier (risk-reward ratio)
        if current_condition == 2:  # Trending
            # Higher reward targets in trending markets
            self.dynamic_tp_multiplier = 2.5 + (trending_percentile * 1.0)  # 2.5-3.5 range
        elif current_condition == 4:  # Ranging
            # Lower targets in ranging markets
            self.dynamic_tp_multiplier = 1.5 + (volatility_percentile * 0.5)  # 1.5-2.0 range
        else:
            self.dynamic_tp_multiplier = 2.0
        
        # 3. Adapt RSI thresholds based on market conditions
        if current_condition == 2:  # Trending
            # Wider RSI thresholds in trending markets - trends can persist in overbought/oversold
            trending_direction = 1 if self.ema_s.iloc[-1] > self.ema_l.iloc[-1] else -1
            if trending_direction == 1:  # Uptrend
                self.rsi_overbought = 80  # Higher overbought threshold
                self.rsi_oversold = 30    # Standard oversold
            else:  # Downtrend
                self.rsi_overbought = 70  # Standard overbought
                self.rsi_oversold = 20    # Lower oversold threshold
        else:
            # Standard thresholds for other conditions
            self.rsi_overbought = 70
            self.rsi_oversold = 30
        
        # 4. Adapt trade frequency parameters
        if current_condition == 5:  # Disorderly
            # Less frequent trading in disorderly markets
            self.min_trades_interval = max(5, self.params.get("min_trades_interval", 1) * 2)
        elif current_condition == 2 and trending_percentile > 0.7:  # Strong trend
            # More frequent trading in strong trends
            self.min_trades_interval = max(1, self.params.get("min_trades_interval", 1) // 2)
        else:
            # Default frequency
            self.min_trades_interval = self.params.get("min_trades_interval", 1)
        
        # 5. Adapt partial profit parameters
        if current_condition == 2:  # Trending
            # In trending markets, take smaller partial positions
            self.partial_profit_size = 0.3  # 30% of position
            self.partial_profit_min_pips = 8.0  # Require more profit before partial exit
        elif current_condition == 4:  # Ranging
            # In ranging markets, take larger partial positions
            self.partial_profit_size = 0.6  # 60% of position
            self.partial_profit_min_pips = 3.0  # Take profits earlier
        else:
            # Default partial profit settings
            self.partial_profit_size = self.params.get("partial_profit_size", 0.5)
            self.partial_profit_min_pips = self.params.get("partial_profit_min_pips", 5.0)
        
        # 6. Adapt indicators for optimal trend detection
        # Based on volatility, adjust EMA periods
        if volatility_percentile > 0.8:  # High volatility
            # Use shorter EMAs for faster response
            self.ema_short = max(5, int(self.params.get("ema_short", 20) * 0.8))
            self.ema_medium = max(15, int(self.params.get("ema_medium", 50) * 0.8))
        elif volatility_percentile < 0.2:  # Low volatility
            # Use longer EMAs for less noise
            self.ema_short = int(self.params.get("ema_short", 20) * 1.2)
            self.ema_medium = int(self.params.get("ema_medium", 50) * 1.2)
        
        # Log parameter adjustments
        bot_logger.info(f"Adapted parameters - Market condition: {current_condition}, " 
                      f"SL mult: {self.atr_multiplier_sl:.2f}, TP mult: {self.dynamic_tp_multiplier:.2f}, "
                      f"Trade interval: {self.min_trades_interval}")

    def auto_optimize_parameters(self):
        """Automatically optimize parameters based on historical performance"""
        if len(self.trade_history) < self.min_trades_for_optimization:
            return False
        
        # Reset counter
        self.trades_since_optimization = 0
        
        # Get parameter analysis
        analysis_result = self.analyze_parameter_performance()
        if not isinstance(analysis_result, dict) or "optimal_settings" not in analysis_result:
            return False
        
        # Get current market condition
        current_condition = {
            1: "Normal", 2: "Trending", 3: "Volatile", 
            4: "Ranging", 5: "Disorderly"
        }.get(self.market_condition_series.iloc[-1], "Normal")
        
        # Check if we have optimal settings for current condition
        if current_condition not in analysis_result["optimal_settings"]:
            return False
        
        optimal = analysis_result["optimal_settings"][current_condition]
        
        # Apply optimal ATR multiplier if available
        if "best_atr_multiplier" in optimal:
            atr_range = optimal["best_atr_multiplier"]
            if atr_range == "1.5-2.0":
                self.atr_multiplier_sl = 1.75
            elif atr_range == "2.0-2.5":
                self.atr_multiplier_sl = 2.25
            elif atr_range == "2.5-3.0":
                self.atr_multiplier_sl = 2.75
            else:  # 3.0+
                self.atr_multiplier_sl = 3.25
        
        # Apply optimal TP multiplier if available
        if "best_tp_multiplier" in optimal:
            tp_range = optimal["best_tp_multiplier"]
            if tp_range == "1.5-2.0":
                self.dynamic_tp_multiplier = 1.75
            elif tp_range == "2.0-2.5":
                self.dynamic_tp_multiplier = 2.25
            elif tp_range == "2.5-3.0":
                self.dynamic_tp_multiplier = 2.75
            else:  # 3.0+
                self.dynamic_tp_multiplier = 3.25
        
        # Apply optimal partial profit settings if available
        if "best_partial_size" in optimal:
            self.partial_profit_size = optimal["best_partial_size"]
        
        if "best_partial_min_pips" in optimal:
            self.partial_profit_min_pips = optimal["best_partial_min_pips"]
        
        bot_logger.info(f"Auto-optimized parameters for {current_condition} market: "
                      f"ATR_mult={self.atr_multiplier_sl:.2f}, TP_mult={self.dynamic_tp_multiplier:.2f}, "
                      f"Partial_size={self.partial_profit_size:.2f}")
        
        return True

    def analyze_parameter_performance(self):
        """Analyze which parameter settings perform best in different market conditions"""
        if len(self.trade_history) < 20:
            return "Not enough trades for parameter analysis"
        
        parameter_analysis = {}
        
        # Analyze performance by market condition
        for condition in range(1, 6):
            condition_name = {
                1: "Normal", 2: "Trending", 3: "Volatile", 
                4: "Ranging", 5: "Disorderly"
            }.get(condition, f"Unknown ({condition})")
            
            # Get trades for this market condition
            condition_trades = [t for t in self.trade_history 
                              if t["market_context"]["market_condition"] == condition
                              and "parameters" in t]
            
            if len(condition_trades) < 5:
                continue
                
            # Group trades by parameter ranges
            atr_mult_groups = {"1.5-2.0": [], "2.0-2.5": [], "2.5-3.0": [], "3.0+": []}
            tp_mult_groups = {"1.5-2.0": [], "2.0-2.5": [], "2.5-3.0": [], "3.0+": []}
            partial_size_groups = {"0.2-0.3": [], "0.3-0.4": [], "0.4-0.5": [], "0.5-0.6": [], "0.6+": []}
            
            for trade in condition_trades:
                # Group by ATR multiplier
                atr_mult = trade["parameters"]["atr_multiplier_sl"]
                if atr_mult < 2.0:
                    atr_mult_groups["1.5-2.0"].append(trade)
                elif atr_mult < 2.5:
                    atr_mult_groups["2.0-2.5"].append(trade)
                elif atr_mult < 3.0:
                    atr_mult_groups["2.5-3.0"].append(trade)
                else:
                    atr_mult_groups["3.0+"].append(trade)
                    
                # Group by TP multiplier
                tp_mult = trade["parameters"]["dynamic_tp_multiplier"]
                if tp_mult < 2.0:
                    tp_mult_groups["1.5-2.0"].append(trade)
                elif tp_mult < 2.5:
                    tp_mult_groups["2.0-2.5"].append(trade)
                elif tp_mult < 3.0:
                    tp_mult_groups["2.5-3.0"].append(trade)
                else:
                    tp_mult_groups["3.0+"].append(trade)
                
                # Group by partial size if available
                if "partial_profit_size" in trade["parameters"]:
                    partial_size = trade["parameters"]["partial_profit_size"]
                    if partial_size < 0.3:
                        partial_size_groups["0.2-0.3"].append(trade)
                    elif partial_size < 0.4:
                        partial_size_groups["0.3-0.4"].append(trade)
                    elif partial_size < 0.5:
                        partial_size_groups["0.4-0.5"].append(trade)
                    elif partial_size < 0.6:
                        partial_size_groups["0.5-0.6"].append(trade)
                    else:
                        partial_size_groups["0.6+"].append(trade)
            
            # Calculate performance metrics for each parameter group
            atr_performance = {}
            for group_name, trades in atr_mult_groups.items():
                if len(trades) < 3:
                    continue
                    
                wins = sum(1 for t in trades if t["profit_pips"] > 0)
                win_rate = wins / len(trades) if trades else 0
                avg_profit = sum(t["profit_pips"] for t in trades) / len(trades) if trades else 0
                
                atr_performance[group_name] = {
                    "win_rate": win_rate,
                    "avg_profit": avg_profit,
                    "trades": len(trades)
                }
                
            tp_performance = {}
            for group_name, trades in tp_mult_groups.items():
                if len(trades) < 3:
                    continue
                    
                wins = sum(1 for t in trades if t["profit_pips"] > 0)
                win_rate = wins / len(trades) if trades else 0
                avg_profit = sum(t["profit_pips"] for t in trades) / len(trades) if trades else 0
                
                tp_performance[group_name] = {
                    "win_rate": win_rate,
                    "avg_profit": avg_profit,
                    "trades": len(trades)
                }
            
            partial_performance = {}
            for group_name, trades in partial_size_groups.items():
                if len(trades) < 3:
                    continue
                    
                wins = sum(1 for t in trades if t["profit_pips"] > 0)
                win_rate = wins / len(trades) if trades else 0
                avg_profit = sum(t["profit_pips"] for t in trades) / len(trades) if trades else 0
                
                partial_performance[group_name] = {
                    "win_rate": win_rate,
                    "avg_profit": avg_profit,
                    "trades": len(trades)
                }
            
            # Store results for this market condition
            parameter_analysis[condition_name] = {
                "atr_multiplier_performance": atr_performance,
                "tp_multiplier_performance": tp_performance,
                "partial_size_performance": partial_performance,
                "total_trades": len(condition_trades)
            }
        
        # Find optimal parameter settings for each market condition
        optimal_settings = {}
        for condition, analysis in parameter_analysis.items():
            optimal_settings[condition] = {}
            
            # Find best ATR multiplier (prioritize profit)
            best_atr_mult = None
            best_atr_profit = -float('inf')
            
            for group, metrics in analysis["atr_multiplier_performance"].items():
                if metrics["avg_profit"] > best_atr_profit and metrics["trades"] >= 3:
                    best_atr_profit = metrics["avg_profit"]
                    best_atr_mult = group
            
            if best_atr_mult:
                optimal_settings[condition]["best_atr_multiplier"] = best_atr_mult
            
            # Find best TP multiplier (prioritize profit)
            best_tp_mult = None
            best_tp_profit = -float('inf')
            
            for group, metrics in analysis["tp_multiplier_performance"].items():
                if metrics["avg_profit"] > best_tp_profit and metrics["trades"] >= 3:
                    best_tp_profit = metrics["avg_profit"]
                    best_tp_mult = group
            
            if best_tp_mult:
                optimal_settings[condition]["best_tp_multiplier"] = best_tp_mult
            
            # Find best partial size (prioritize profit)
            best_partial_size = None
            best_partial_profit = -float('inf')
            
            for group, metrics in analysis["partial_size_performance"].items():
                if metrics["avg_profit"] > best_partial_profit and metrics["trades"] >= 3:
                    best_partial_profit = metrics["avg_profit"]
                    best_partial_size = group
            
            if best_partial_size:
                # Convert group name to actual value
                if best_partial_size == "0.2-0.3":
                    optimal_settings[condition]["best_partial_size"] = 0.25
                elif best_partial_size == "0.3-0.4":
                    optimal_settings[condition]["best_partial_size"] = 0.35
                elif best_partial_size == "0.4-0.5":
                    optimal_settings[condition]["best_partial_size"] = 0.45
                elif best_partial_size == "0.5-0.6":
                    optimal_settings[condition]["best_partial_size"] = 0.55
                else:  # 0.6+
                    optimal_settings[condition]["best_partial_size"] = 0.65
        
        return {
            "parameter_analysis": parameter_analysis,
            "optimal_settings": optimal_settings
        }

    def can_trade_based_on_time(self):
        """Check if we can trade based on time filters"""
        if not self.session_filter:
            return True
            
        chart = self.tfs_chart[self.tf]
        current_time = chart.iloc[-1]["Open time"]
        
        # Convert to datetime object if needed
        if not isinstance(current_time, datetime):
            current_time = pd.to_datetime(current_time)
            
        # Check trading sessions - London (7-16 UTC) or New York (12-21 UTC)
        hour = current_time.hour
        is_major_session = (7 <= hour < 16) or (12 <= hour < 21)
        
        # Check minimum interval between trades
        idx = len(chart) - 1
        enough_time_passed = (idx - self.last_trade_idx) >= self.min_trades_interval
        
        return is_major_session and enough_time_passed

    def calculate_dynamic_tpsl(self, side, entry_price):
        """Calculate dynamic TP/SL based on market conditions with better trend profit capture"""
        chart = self.tfs_chart[self.tf]
        point = self.trader.get_point()
        current_atr = self.atr.iloc[-1]
        current_condition = self.market_condition_series.iloc[-1]
        
        # Trend strength - important for long-term trend profit capture
        trend_strength = 1.0
        if current_condition == 2:  # Trending market
            # Calculate trend strength based on ADX and EMA alignment
            adx_value = self.adx.iloc[-1]
            trend_strength = min(3.0, max(1.0, adx_value / 25))  # Scale between 1.0-3.0 based on ADX
        
        # Base ATR multiplier on market condition
        if current_condition == 5:  # Disorderly
            sl_atr_multiplier = 3.0
            tp_rr_ratio = 1.5
        elif current_condition == 2:  # Trending
            # For trending markets, use wider targets for long-term profit capture
            sl_atr_multiplier = 2.0
            tp_rr_ratio = 2.5 * trend_strength  # Can go up to 7.5 for very strong trends
        elif current_condition == 3:  # Volatile
            sl_atr_multiplier = 2.5
            tp_rr_ratio = 2.0
        elif current_condition == 4:  # Ranging
            sl_atr_multiplier = 1.5
            tp_rr_ratio = 1.5
        else:  # Normal
            sl_atr_multiplier = 2.0
            tp_rr_ratio = 2.0
        
        # Use auto-optimized parameters if available
        sl_atr_multiplier = self.atr_multiplier_sl
        tp_rr_ratio = self.dynamic_tp_multiplier
        
        # Calculate SL distance in pips
        sl_pips = current_atr / point * sl_atr_multiplier
        
        # Calculate TP/SL prices
        if side == OrderSide.BUY:
            sl = entry_price - (point * sl_pips)
            tp = entry_price + (point * sl_pips * tp_rr_ratio)
        else:  # SELL
            sl = entry_price + (point * sl_pips)
            tp = entry_price - (point * sl_pips * tp_rr_ratio)
            
        return tp, sl

    def detect_optimal_exit_point(self, order):
        """
        Detect if current conditions suggest an optimal exit point for open positions
        
        Parameters:
        -----------
        order : Order
            The open order to check for optimal exit
            
        Returns:
        --------
        bool
            True if optimal exit conditions are met, False otherwise
        """
        if not self.exit_detection_enabled:
            return False
            
        chart = self.tfs_chart[self.tf]
        current_price = chart["Close"].iloc[-1]
        
        # Get current conditions
        current_condition = self.market_condition_series.iloc[-1]
        
        # Skip checking if in disorderly market - rely on stop loss instead
        if current_condition == 5:
            return False
            
        # Calculate profit in pips
        current_profit_pips = self._calculate_profit_pips(order, current_price)
        
        # Only consider exit if we have meaningful profit
        if current_profit_pips < 3.0:
            return False
        
        # Get trend duration
        trend_duration = len(chart) - 1 - self.trend_start_idx
        
        # Get condition-specific optimal exit indicators from historical data
        condition_name = {1: "normal", 2: "trending", 3: "volatile", 4: "ranging"}.get(current_condition, "normal")
        optimal_exits = self.market_stats["optimal_exits"][condition_name]
        
        # Skip if no historical optimal exits for this condition
        if not optimal_exits:
            return False
        
        # Optimal exit detection logic varies by market condition
        if current_condition == 2:  # Trending
            # In trending markets, look for reversal signs
            
            # 1. RSI divergence
            if (order.side == OrderSide.BUY and
                current_price > chart["Close"].iloc[-2] and  # Price making higher high
                self.rsi.iloc[-1] < self.rsi.iloc[-2]):      # RSI making lower high (bearish divergence)
                return True
                
            if (order.side == OrderSide.SELL and
                current_price < chart["Close"].iloc[-2] and  # Price making lower low
                self.rsi.iloc[-1] > self.rsi.iloc[-2]):      # RSI making higher low (bullish divergence)
                return True
            
            # 2. Extreme RSI values
            if (order.side == OrderSide.BUY and self.rsi.iloc[-1] > 85) or \
               (order.side == OrderSide.SELL and self.rsi.iloc[-1] < 15):
                return True
            
            # 3. Extended trend duration compared to historical optimal exits
            avg_optimal_duration = sum(exit["trend_duration"] for exit in optimal_exits) / len(optimal_exits)
            if trend_duration > avg_optimal_duration * 1.5 and current_profit_pips > 10.0:
                return True
            
            # 4. Price reached typical reversal zone
            typical_exit_rsi = sum(exit["rsi"] for exit in optimal_exits) / len(optimal_exits)
            if (order.side == OrderSide.BUY and self.rsi.iloc[-1] > typical_exit_rsi * 0.95) or \
               (order.side == OrderSide.SELL and self.rsi.iloc[-1] < typical_exit_rsi * 1.05):
                return True
            
        elif current_condition == 4:  # Ranging
            # In ranging markets, exit when price approaches range boundaries
            
            # 1. Price approaching typical range boundaries
            if (order.side == OrderSide.BUY and self.price_location.iloc[-1] > 0.85) or \
               (order.side == OrderSide.SELL and self.price_location.iloc[-1] < 0.15):
                return True
                
            # 2. RSI approaching extremes for ranging market
            if (order.side == OrderSide.BUY and self.rsi.iloc[-1] > 70) or \
               (order.side == OrderSide.SELL and self.rsi.iloc[-1] < 30):
                return True
        
        elif current_condition == 3:  # Volatile
            # In volatile markets, be quicker to take profits
            
            # 1. Significant profit compared to current volatility
            atr_in_pips = self.atr.iloc[-1] / self.trader.get_point()
            if current_profit_pips > atr_in_pips * 1.5:
                return True
                
            # 2. Momentum starting to fade
            if (order.side == OrderSide.BUY and self.macd.iloc[-1] < self.macd.iloc[-2]) or \
               (order.side == OrderSide.SELL and self.macd.iloc[-1] > self.macd.iloc[-2]):
                return True
        
        else:  # Normal market
            # In normal markets, use a blend of trend and range indicators
            
            # 1. Price moved significantly from entry
            if current_profit_pips > 8.0:
                # 2. Momentum starting to fade
                if (order.side == OrderSide.BUY and self.macd.iloc[-1] < self.macd.iloc[-2]) or \
                   (order.side == OrderSide.SELL and self.macd.iloc[-1] > self.macd.iloc[-2]):
                    return True
            
            # 3. Price reached extended target
            entry_to_sl_dist = abs(order.entry - order.sl)
            if (order.side == OrderSide.BUY and (current_price - order.entry) > entry_to_sl_dist * 2) or \
               (order.side == OrderSide.SELL and (order.entry - current_price) > entry_to_sl_dist * 2):
                # And momentum fading
                if (order.side == OrderSide.BUY and self.rsi_slope.iloc[-1] < 0) or \
                   (order.side == OrderSide.SELL and self.rsi_slope.iloc[-1] > 0):
                    return True
        
        # No optimal exit detected
        return False

    def check_required_params(self):
        # Minimal required parameters
        return True

    def is_params_valid(self):
        # Simple validation
        return self.check_required_params()

    def update(self, tf):
        # Update when new kline arrives
        super().update(tf)
        
        # Adapt parameters to current market conditions
        self.adapt_parameters_to_market()
        
        # Run auto-optimization if enabled and due
        if self.auto_optimization_enabled:
            self.trades_since_optimization += 1
            if self.trades_since_optimization >= self.optimization_interval:
                self.auto_optimize_parameters()
        
        # Check for partial profit opportunities 
        if self.partial_profit_enabled:
            self.check_partial_profit_opportunity()
        
        # Check for optimal exit points
        if self.exit_detection_enabled:
            self.check_optimal_exits()
        
        # Check order signals
        self.check_close_signal()
        self.check_signal()
        
        # Adjust stop losses dynamically for open positions
        self.adjust_dynamic_sl()

    def close_opening_orders(self):
        super().close_opening_orders(self.tfs_chart[self.tf].iloc[-1])

    def check_optimal_exits(self):
        """Check all open orders for optimal exit conditions"""
        chart = self.tfs_chart[self.tf]
        last_kline = chart.iloc[-1]
        
        for i in range(len(self.orders_opening) - 1, -1, -1):
            order = self.orders_opening[i]
            
            # Check if current conditions suggest an optimal exit
            if self.detect_optimal_exit_point(order):
                # Calculate profit
                current_profit = self._calculate_profit_pips(order, last_kline["Close"])
                
                # Close the position
                order.close(last_kline)
                self.trader.close_trade(order)
                
                if order.is_closed():
                    self.orders_closed.append(order)
                    # Record trade with exit reason
                    if current_profit > 0:
                        order["exit_reason"] = "optimal_exit"
                        self._record_trade(order, current_profit)
                        
                    bot_logger.info(f"Optimal exit triggered: {order.side}, {current_profit:.1f} pips")
                
                del self.orders_opening[i]

    def check_signal(self):
        """Check for new entry signals"""
        chart = self.tfs_chart[self.tf]
        last_kline = chart.iloc[-1]
        
        # Skip if market is in disorder and we want to avoid such conditions
        if self.market_disorder and not self.params.get("trade_in_disorder", False):
            bot_logger.info("Skipping signal check - market in disorder")
            return
            
        # Skip if time filters prevent trading
        # if not self.can_trade_based_on_time():
        #     return
        
        current_condition = self.market_condition_series.iloc[-1]
        
        # Reset trading flag
        self.trading_flag = 0
        
        # Different entry logic based on market condition
        if current_condition == 2:  # Trending market
            # Trend-following strategy
            if (self.ema_s.iloc[-1] > self.ema_m.iloc[-1] > self.ema_l.iloc[-1] and 
                self.rsi.iloc[-1] > 50 and self.rsi.iloc[-1] < 70 and
                last_kline["Close"] > self.middle_bb.iloc[-1]):
                # BUY signal in uptrend
                self.trading_flag = 1
                
            elif (self.ema_s.iloc[-1] < self.ema_m.iloc[-1] < self.ema_l.iloc[-1] and 
                 self.rsi.iloc[-1] < 50 and self.rsi.iloc[-1] > 30 and
                 last_kline["Close"] < self.middle_bb.iloc[-1]):
                # SELL signal in downtrend
                self.trading_flag = -1
                
        elif current_condition == 4:  # Ranging market
            # Mean-reversion strategy
            if (last_kline["Close"] < self.lower_bb.iloc[-1] and 
                self.rsi.iloc[-1] < 30 and
                self.rsi_slope.iloc[-1] > 0):  # RSI turning up
                # Buy at support
                self.trading_flag = 1
                
            elif (last_kline["Close"] > self.upper_bb.iloc[-1] and 
                 self.rsi.iloc[-1] > 70 and
                 self.rsi_slope.iloc[-1] < 0):  # RSI turning down
                # Sell at resistance
                self.trading_flag = -1
                
        elif current_condition in [1, 3]:  # Normal or Volatile market
            # Breakout strategy with confirmation
            if (last_kline["Close"] > self.upper_bb.iloc[-2] and  # Previous upper BB
                last_kline["Close"] > last_kline["Open"] and  # Bullish candle
                self.rsi_slope.iloc[-1] > 0):  # RSI increasing
                # Bullish breakout
                self.trading_flag = 1
                
            elif (last_kline["Close"] < self.lower_bb.iloc[-2] and  # Previous lower BB
                 last_kline["Close"] < last_kline["Open"] and  # Bearish candle
                 self.rsi_slope.iloc[-1] < 0):  # RSI decreasing
                # Bearish breakout
                self.trading_flag = -1
        
        # Execute BUY signal
        if self.trading_flag == 1:
            point = self.trader.get_point()
            if self.use_dynamic_tpsl:
                tp, sl = self.calculate_dynamic_tpsl(OrderSide.BUY, last_kline["Close"])
            else:
                tp = point * 5 + last_kline["Close"]
                sl = -point * 2 + last_kline["Close"]
                
            order = Order(
                OrderType.MARKET,
                OrderSide.BUY,
                last_kline["Close"],
                tp=tp,
                sl=sl,
                status=OrderStatus.FILLED,
            )
            order["FILL_TIME"] = last_kline["Open time"]
            order["strategy"] = self.name
            order["trading_price"] = last_kline["Close"]
            order["description"] = f"Forex algorithm - BUY ({current_condition})"
            
            # Add market context for later analysis
            order["market_context"] = {
                "atr": self.atr.iloc[-1],
                "bb_width": self.bb_width.iloc[-1],
                "adx": self.adx.iloc[-1],
                "is_trending": self.is_trending.iloc[-1],
                "rsi": self.rsi.iloc[-1],
                "market_condition": int(self.market_condition_series.iloc[-1]),
                "trend_direction": int(self.current_trend),
                "trend_duration": len(chart) - 1 - self.trend_start_idx
            }
            
            # Add parameter metadata for optimization analysis
            order["parameters"] = {
                "atr_multiplier_sl": self.atr_multiplier_sl,
                "dynamic_tp_multiplier": self.dynamic_tp_multiplier,
                "rsi_overbought": self.rsi_overbought,
                "rsi_oversold": self.rsi_oversold,
                "partial_profit_size": self.partial_profit_size,
                "partial_profit_min_pips": self.partial_profit_min_pips
            }
            
            # Apply sl fix mode if specified
            order = self.trader.fix_order(order, self.params.get("sl_fix_mode", "none"), self.max_sl_pct)
            if order:
                self.trader.create_trade(order, self.volume)
                self.orders_opening.append(order)
                self.last_trade_idx = len(chart) - 1
                
        # Execute SELL signal
        elif self.trading_flag == -1:
            point = self.trader.get_point()
            if self.use_dynamic_tpsl:
                tp, sl = self.calculate_dynamic_tpsl(OrderSide.SELL, last_kline["Close"])
            else:
                tp = -point * 5 + last_kline["Close"]
                sl = point * 2 + last_kline["Close"]
                
            order = Order(
                OrderType.MARKET,
                OrderSide.SELL,
                last_kline["Close"],
                tp=tp,
                sl=sl,
                status=OrderStatus.FILLED,
            )
            order["FILL_TIME"] = last_kline["Open time"]
            order["strategy"] = self.name
            order["trading_price"] = last_kline["Close"]
            order["description"] = f"Forex algorithm - SELL ({current_condition})"
            
            # Add market context for later analysis
            order["market_context"] = {
                "atr": self.atr.iloc[-1],
                "bb_width": self.bb_width.iloc[-1],
                "adx": self.adx.iloc[-1],
                "is_trending": self.is_trending.iloc[-1],
                "rsi": self.rsi.iloc[-1],
                "market_condition": int(self.market_condition_series.iloc[-1]),
                "trend_direction": int(self.current_trend),
                "trend_duration": len(chart) - 1 - self.trend_start_idx
            }
            
            # Add parameter metadata for optimization analysis
            order["parameters"] = {
                "atr_multiplier_sl": self.atr_multiplier_sl,
                "dynamic_tp_multiplier": self.dynamic_tp_multiplier,
                "rsi_overbought": self.rsi_overbought,
                "rsi_oversold": self.rsi_oversold,
                "partial_profit_size": self.partial_profit_size,
                "partial_profit_min_pips": self.partial_profit_min_pips
            }
            
            # Apply sl fix mode if specified
            order = self.trader.fix_order(order, self.params.get("sl_fix_mode", "none"), self.max_sl_pct)
            if order:
                self.trader.create_trade(order, self.volume)
                self.orders_opening.append(order)
                self.last_trade_idx = len(chart) - 1

    def check_close_signal(self):
        """Check for exit signals with improved trend-riding capability"""
        chart = self.tfs_chart[self.tf]
        last_kline = chart.iloc[-1]
        
        # Early exit during market disorder if configured
        if self.market_disorder and self.params.get("exit_on_disorder", True):
            bot_logger.info("Market in disorder - checking for early exit")
            for i in range(len(self.orders_opening) - 1, -1, -1):
                order = self.orders_opening[i]
                # Calculate P&L
                current_profit = self._calculate_profit_pips(order, last_kline["Close"])
                # Exit if in profit or small loss
                if current_profit > -1.0:  # Exit if profit or small loss
                    order.close(last_kline)
                    self.trader.close_trade(order)
                    if order.is_closed():
                        order["exit_reason"] = "disorder"
                        self.orders_closed.append(order)
                        # Record successful trade
                        if current_profit > 0:
                            self._record_trade(order, current_profit)
                    del self.orders_opening[i]
            return
            
        # Regular exit signals
        for i in range(len(self.orders_opening) - 1, -1, -1):
            order = self.orders_opening[i]
            
            # Get current market condition and trend strength
            current_condition = self.market_condition_series.iloc[-1]
            adx_value = self.adx.iloc[-1]
            trend_strength = adx_value / 25  # Normalize to a 0-1+ scale
            current_profit = self._calculate_profit_pips(order, last_kline["Close"])
            
            # Default to not exiting
            should_exit = False
            exit_reason = ""
            
            # For trending markets, be more conservative with exits
            if current_condition == 2 and trend_strength > 1.2:  # Strong trend
                # Only exit if we have clear trend reversal, not just pullbacks
                if order.side == OrderSide.BUY:
                    # Only exit long positions if clear downtrend emerges
                    should_exit = (
                        (self.ema_s.iloc[-1] < self.ema_m.iloc[-1] < self.ema_l.iloc[-1]) or  # Complete EMA flip
                        (self.rsi.iloc[-1] > 85 and self.rsi_slope.iloc[-1] < 0) or  # Extreme overbought + turning down
                        (self.macd.iloc[-1] < self.macdsignal.iloc[-1] and  # MACD bearish crossover
                         self.macdhist.iloc[-2] > 0 and self.macdhist.iloc[-1] < 0)  # Confirmed by histogram flip
                    )
                    if should_exit:
                        exit_reason = "trend_reversal"
                else:  # SELL position
                    # Only exit short positions if clear uptrend emerges
                    should_exit = (
                        (self.ema_s.iloc[-1] > self.ema_m.iloc[-1] > self.ema_l.iloc[-1]) or  # Complete EMA flip
                        (self.rsi.iloc[-1] < 15 and self.rsi_slope.iloc[-1] > 0) or  # Extreme oversold + turning up
                        (self.macd.iloc[-1] > self.macdsignal.iloc[-1] and  # MACD bullish crossover
                         self.macdhist.iloc[-2] < 0 and self.macdhist.iloc[-1] > 0)  # Confirmed by histogram flip
                    )
                    if should_exit:
                        exit_reason = "trend_reversal"
            else:
                # Use standard exit logic for non-trending or weakly trending markets
                if order.side == OrderSide.BUY:
                    # Exit signals based on market condition
                    if current_condition == 2:  # Trending
                        # Exit when trend weakens or reverses
                        if (self.ema_s.iloc[-1] < self.ema_m.iloc[-1] or
                            self.rsi.iloc[-1] > 80 or
                            (self.macd.iloc[-1] < self.macdsignal.iloc[-1] and
                             self.macdhist.iloc[-1] < 0)):
                            should_exit = True
                            exit_reason = "trend_weakening"
                    
                    elif current_condition == 4:  # Ranging
                        # Exit when approaching upper band or resistance
                        if (last_kline["Close"] > self.upper_bb.iloc[-1] * 0.98 or
                            self.rsi.iloc[-1] > 70):
                            should_exit = True
                            exit_reason = "range_resistance"
                    
                    else:  # Normal, Volatile
                        # More conservative exit
                        if (self.macd.iloc[-1] < self.macdsignal.iloc[-1] or
                            self.rsi.iloc[-1] > 75):
                            should_exit = True
                            exit_reason = "momentum_shift"
                    
                    # Take profit at significant resistance levels
                    if (last_kline["Close"] > self.upper_bb.iloc[-1] and
                        self.rsi.iloc[-1] > 70):
                        should_exit = True
                        exit_reason = "resistance_hit"
                
                elif order.side == OrderSide.SELL:
                    # Exit signals based on market condition
                    if current_condition == 2:  # Trending
                        # Exit when trend weakens or reverses
                        if (self.ema_s.iloc[-1] > self.ema_m.iloc[-1] or
                            self.rsi.iloc[-1] < 20 or
                            (self.macd.iloc[-1] > self.macdsignal.iloc[-1] and
                             self.macdhist.iloc[-1] > 0)):
                            should_exit = True
                            exit_reason = "trend_weakening"
                    
                    elif current_condition == 4:  # Ranging
                        # Exit when approaching lower band or support
                        if (last_kline["Close"] < self.lower_bb.iloc[-1] * 1.02 or
                            self.rsi.iloc[-1] < 30):
                            should_exit = True
                            exit_reason = "range_support"
                    
                    else:  # Normal, Volatile
                        # More conservative exit
                        if (self.macd.iloc[-1] > self.macdsignal.iloc[-1] or
                            self.rsi.iloc[-1] < 25):
                            should_exit = True
                            exit_reason = "momentum_shift"
                    
                    # Take profit at significant support levels
                    if (last_kline["Close"] < self.lower_bb.iloc[-1] and
                        self.rsi.iloc[-1] < 30):
                        should_exit = True
                        exit_reason = "support_hit"
            
            # Execute exit if conditions met
            if should_exit:
                order.close(last_kline)
                order["exit_reason"] = exit_reason
                self.trader.close_trade(order)
                if order.is_closed():
                    self.orders_closed.append(order)
                    # Record trade data
                    current_profit = self._calculate_profit_pips(order, last_kline["Close"])
                    if current_profit > 0:
                        self._record_trade(order, current_profit)
                    bot_logger.info(f"Exit signal: {exit_reason}, profit: {current_profit:.1f} pips")
                del self.orders_opening[i]

    def check_partial_profit_opportunity(self):
        """Check for opportunities to take partial profits during trends"""
        if len(self.orders_opening) == 0:
            return

        chart = self.tfs_chart[self.tf]
        last_kline = chart.iloc[-1]
        point = self.trader.get_point()
        
        # Get the current market condition
        current_condition = self.market_condition_series.iloc[-1]
        
        # Loop through open orders
        for i in range(len(self.orders_opening) - 1, -1, -1):
            order = self.orders_opening[i]
            
            # Calculate current profit in pips
            current_profit_pips = self._calculate_profit_pips(order, last_kline["Close"])
            
            # Only consider taking partial profits if we have sufficient profit
            if current_profit_pips <= self.partial_profit_min_pips:  # Minimum pips required
                continue

            # Check for short-term conditions within a trend
            take_partial = False
            partial_reason = ""
            
            # 1. Take partial profit if we detect short-term disorder within trend
            if self.market_disorder:
                take_partial = True
                partial_reason = "disorder_detected"
                
            # 2. Take partial profit if market suddenly becomes ranging within trend
            elif current_condition == 4 and current_profit_pips > 5.0:  # Ranging
                take_partial = True
                partial_reason = "ranging_market"
                
            # 3. Take partial profit on overbought/oversold conditions against trend
            elif order.side == OrderSide.BUY and self.rsi.iloc[-1] > 75:
                take_partial = True
                partial_reason = "overbought_condition"
                
            elif order.side == OrderSide.SELL and self.rsi.iloc[-1] < 25:
                take_partial = True
                partial_reason = "oversold_condition"
                
            # 4. Take partial profit when BB width narrows significantly (consolidation)
            elif self.bb_width.iloc[-1] < 0.01 and current_profit_pips > 8.0:
                take_partial = True
                partial_reason = "narrowing_bb"
                
            # 5. Take partial profit on waning momentum (MACD divergence)
            elif (order.side == OrderSide.BUY and 
                  self.macd.iloc[-1] < self.macd.iloc[-2] and 
                  last_kline["Close"] > chart["Close"].iloc[-2]):
                take_partial = True
                partial_reason = "bearish_macd_divergence"
                
            elif (order.side == OrderSide.SELL and 
                  self.macd.iloc[-1] > self.macd.iloc[-2] and 
                  last_kline["Close"] < chart["Close"].iloc[-2]):
                take_partial = True
                partial_reason = "bullish_macd_divergence"
                
            # 6. Take partial profit on extended trend without pullback
            if not take_partial and current_condition == 2:  # Trending
                # Calculate how many consecutive candles in this direction
                consecutive_count = 0
                direction = 1 if order.side == OrderSide.BUY else -1
                
                for j in range(len(chart) - 1, max(0, len(chart) - 10), -1):
                    if direction == 1 and chart["Close"].iloc[j] > chart["Open"].iloc[j]:
                        consecutive_count += 1
                    elif direction == -1 and chart["Close"].iloc[j] < chart["Open"].iloc[j]:
                        consecutive_count += 1
                    else:
                        break
                
                if consecutive_count >= 7:  # Seven consecutive candles in one direction
                    take_partial = True
                    partial_reason = "extended_move_no_pullback"
            
            # Execute partial profit if conditions met
            if take_partial:
                # Determine how much of the position to close
                partial_size = self.partial_profit_size
                
                try:
                    # Close the partial position
                    partial_order = order.copy() if hasattr(order, "copy") else Order(
                        order.type, order.side, order.entry, tp=order.tp, sl=order.sl
                    )
                    
                    # Set partial volume
                    original_volume = order.volume
                    partial_volume = original_volume * partial_size
                    partial_order.volume = partial_volume
                    
                    # Close the partial position
                    partial_order.close(last_kline)
                    partial_order["exit_reason"] = f"partial_{partial_reason}"
                    self.trader.close_trade(partial_order)
                    
                    # Update the original order's remaining volume
                    order.volume = original_volume * (1 - partial_size)
                    
                    # Log the partial profit
                    bot_logger.info(f"Taking {int(partial_size*100)}% partial profit: {order.side}, "
                                 f"{current_profit_pips:.1f} pips, reason: {partial_reason}")
                    
                    # Record the partial profit for analysis
                    if hasattr(order, "market_context"):
                        self._record_trade(partial_order, current_profit_pips, is_partial=True, 
                                          partial_reason=partial_reason)
                except Exception as e:
                    bot_logger.error(f"Error taking partial profit: {str(e)}")

    def _calculate_profit_pips(self, order, current_price):
        """Calculate current profit in pips"""
        point = self.trader.get_point()
        if order.side == OrderSide.BUY:
            return (current_price - order.entry) / point
        else:  # SELL
            return (order.entry - current_price) / point

    def _record_trade(self, order, profit_pips, is_partial=False, partial_reason=None):
        """Record successful trade pattern for analysis with parameter metadata"""
        if hasattr(order, "market_context"):
            # Record basic trade data
            trade_record = {
                "timestamp": order["FILL_TIME"],
                "side": order.side,
                "profit_pips": profit_pips,
                "market_context": order["market_context"],
                "is_partial": is_partial,
                "exit_reason": order.get("exit_reason", "tp_hit" if not is_partial else f"partial_{partial_reason}"),
                "hold_time": (order["CLOSE_TIME"] - order["FILL_TIME"]).total_seconds() / 60  # in minutes
            }
            
            # Add parameter metadata if available
            if hasattr(order, "parameters"):
                trade_record["parameters"] = order["parameters"]
            else:
                # Use current parameters
                trade_record["parameters"] = {
                    "atr_multiplier_sl": self.atr_multiplier_sl,
                    "dynamic_tp_multiplier": self.dynamic_tp_multiplier,
                    "rsi_overbought": self.rsi_overbought,
                    "rsi_oversold": self.rsi_oversold,
                    "partial_profit_size": self.partial_profit_size,
                    "partial_profit_min_pips": self.partial_profit_min_pips
                }
            
            self.trade_history.append(trade_record)
            # Keep history size manageable
            if len(self.trade_history) > self.max_history_size:
                self.trade_history.pop(0)
                
            # Log successful trade details
            log_msg = f"{'Partial ' if is_partial else ''}Trade closed: {order.side} profit={profit_pips:.1f} pips, "
            log_msg += f"market={trade_record['market_context']['market_condition']}, "
            log_msg += f"reason={trade_record['exit_reason']}"
            bot_logger.info(log_msg)

    def adjust_dynamic_sl(self):
        """Dynamically adjust stop losses to capture larger trend moves"""
        if len(self.orders_opening) == 0:
            return
            
        chart = self.tfs_chart[self.tf]
        last_kline = chart.iloc[-1]
        point = self.trader.get_point()
        
        for i in range(len(self.orders_opening)):
            order = self.orders_opening[i]
            entry_price = order.entry
            current_price = last_kline["Close"]
            
            # Calculate current profit in pips
            current_profit_pips = self._calculate_profit_pips(order, current_price)
            
            # Get current market condition
            current_condition = self.market_condition_series.iloc[-1]
            is_trending = current_condition == 2
            
            # Trailing stop logic with trend-awareness
            if order.side == OrderSide.BUY:
                # For BUY orders
                if current_profit_pips >= 3.0:  # Minimum profit to start trailing
                    # Base trailing stop - breakeven + small buffer
                    new_sl = max(entry_price + (point * 0.5), order.sl)
                    
                    # Enhanced trailing for trending markets
                    if is_trending:
                        # Looser trailing in strong trends to avoid early exit
                        if current_profit_pips >= 20.0:
                            # For big trends, use a percentage-based trailing stop (e.g., 30% of profit)
                            profit_distance = current_price - entry_price
                            trailing_distance = profit_distance * 0.3  # Only give back 30% max
                            trend_sl = current_price - trailing_distance
                            new_sl = max(new_sl, trend_sl)
                        elif current_profit_pips >= 10.0:
                            # Use EMA-based trailing for medium trends
                            trend_sl = self.ema_m.iloc[-1] - (point * 2.0)  # Trail below medium EMA
                            new_sl = max(new_sl, trend_sl)
                        else:
                            # Use swing-based trailing for smaller trends
                            past_5_low = chart["Low"].iloc[-5:].min()
                            trend_sl = past_5_low - (point * 1.0)
                            new_sl = max(new_sl, trend_sl)
                    else:
                        # Standard trailing for non-trending conditions
                        past_3_low = chart["Low"].iloc[-3:].min()
                        sl_level = past_3_low - (point * 1.0)
                        new_sl = max(new_sl, sl_level)
                    
                    # If we have a new, better SL, apply it
                    if new_sl > order.sl:
                        order.adjust_sl(new_sl)
                        self.trader.adjust_sl(order, new_sl)
                        bot_logger.info(f"Adjusted BUY SL to {new_sl} (profit: {current_profit_pips:.1f} pips)")
                        
            else:  # SELL orders
                if current_profit_pips >= 3.0:  # Minimum profit to start trailing
                    # Base trailing stop - breakeven - small buffer
                    new_sl = min(entry_price - (point * 0.5), order.sl)
                    
                    # Enhanced trailing for trending markets
                    if is_trending:
                        # Looser trailing in strong trends to avoid early exit
                        if current_profit_pips >= 20.0:
                            # For big trends, use a percentage-based trailing stop
                            profit_distance = entry_price - current_price
                            trailing_distance = profit_distance * 0.3  # Only give back 30% max
                            trend_sl = current_price + trailing_distance
                            new_sl = min(new_sl, trend_sl)
                        elif current_profit_pips >= 10.0:
                            # Use EMA-based trailing for medium trends
                            trend_sl = self.ema_m.iloc[-1] + (point * 2.0)  # Trail above medium EMA
                            new_sl = min(new_sl, trend_sl)
                        else:
                            # Use swing-based trailing for smaller trends
                            past_5_high = chart["High"].iloc[-5:].max()
                            trend_sl = past_5_high + (point * 1.0)
                            new_sl = min(new_sl, trend_sl)
                    else:
                        # Standard trailing for non-trending conditions
                        past_3_high = chart["High"].iloc[-3:].max()
                        sl_level = past_3_high + (point * 1.0)
                        new_sl = min(new_sl, sl_level)
                    
                    # If we have a new, better SL, apply it
                    if new_sl < order.sl:
                        order.adjust_sl(new_sl)
                        self.trader.adjust_sl(order, new_sl)
                        bot_logger.info(f"Adjusted SELL SL to {new_sl} (profit: {current_profit_pips:.1f} pips)")

    def plot_orders(self, fig=None, tf=None, row=1, col=1, dt2idx=None):
        """Plot the strategy with indicators"""
        if fig is None:
            # Create subplots with 5 rows for better visualization
            fig = make_subplots(5, 1, vertical_spacing=0.02, shared_xaxes=True, 
                              row_heights=[0.35, 0.15, 0.15, 0.15, 0.2])
            fig.update_layout(
                xaxis_rangeslider_visible=False,
                yaxis=dict(showgrid=False),
                plot_bgcolor="rgb(19, 23, 34)",
                paper_bgcolor="rgb(121,125,127)",
                font=dict(color="rgb(247,249,249)"),
            )
            
        if tf is None:
            tf = self.tf
            
        df = self.tfs_chart[tf]
        
        if dt2idx is None:
            dt2idx = dict(zip(df["Open time"], list(range(len(df)))))
            tmp_ot = df["Open time"]
            df["Open time"] = list(range(len(df)))
        
        # Plot price and orders (inherits from parent class)
        super().plot_orders(fig, tf, row, col, dt2idx=dt2idx)
        
        # Plot EMAs for trend identification
        fig.add_trace(go.Scatter(
            x=df["Open time"],
            y=self.ema_s,
            mode='lines',
            name=f'EMA {self.ema_short}',
            line=dict(color="blue", width=1)
        ), row=row, col=col)
        
        fig.add_trace(go.Scatter(
            x=df["Open time"],
            y=self.ema_m,
            mode='lines',
            name=f'EMA {self.ema_medium}',
            line=dict(color="green", width=1)
        ), row=row, col=col)
        
        fig.add_trace(go.Scatter(
            x=df["Open time"],
            y=self.ema_l,
            mode='lines',
            name=f'EMA {self.ema_long}',
            line=dict(color="purple", width=1)
        ), row=row, col=col)
        
        # Plot Bollinger Bands
        fig.add_trace(go.Scatter(
            x=df["Open time"],
            y=self.upper_bb,
            mode='lines',
            name='Upper BB',
            line=dict(color="rgba(173, 204, 255, 0.7)")
        ), row=row, col=col)
        
        fig.add_trace(go.Scatter(
            x=df["Open time"],
            y=self.middle_bb,
            mode='lines',
            name='Middle BB',
            line=dict(color="rgba(173, 204, 255, 0.7)")
        ), row=row, col=col)
        
        fig.add_trace(go.Scatter(
            x=df["Open time"],
            y=self.lower_bb,
            mode='lines',
            name='Lower BB',
            line=dict(color="rgba(173, 204, 255, 0.7)")
        ), row=row, col=col)
        
        # Plot RSI
        fig.add_trace(go.Scatter(
            x=df["Open time"],
            y=self.rsi,
            mode='lines',
            name='RSI',
            line=dict(color="orange")
        ), row=row+2, col=col)
        
        # Add overbought/oversold lines
        fig.add_trace(go.Scatter(
            x=[df["Open time"].iloc[0], df["Open time"].iloc[-1]],
            y=[self.rsi_overbought, self.rsi_overbought],
            mode='lines',
            name='Overbought',
            line=dict(color="red", dash="dash")
        ), row=row+2, col=col)
        
        fig.add_trace(go.Scatter(
            x=[df["Open time"].iloc[0], df["Open time"].iloc[-1]],
            y=[self.rsi_oversold, self.rsi_oversold],
            mode='lines',
            name='Oversold',
            line=dict(color="green", dash="dash")
        ), row=row+2, col=col)
        
        # Plot ADX for trend strength
        fig.add_trace(go.Scatter(
            x=df["Open time"],
            y=self.adx,
            mode='lines',
            name='ADX',
            line=dict(color="purple")
        ), row=row+3, col=col)
        
        # Add ADX threshold line
        fig.add_trace(go.Scatter(
            x=[df["Open time"].iloc[0], df["Open time"].iloc[-1]],
            y=[self.adx_threshold, self.adx_threshold],
            mode='lines',
            name='ADX Threshold',
            line=dict(color="purple", dash="dash")
        ), row=row+3, col=col)
        
        # Plot Market Condition
        fig.add_trace(go.Scatter(
            x=df["Open time"],
            y=self.market_condition_series,
            mode='lines',
            name='Market Condition',
            line=dict(color="teal")
        ), row=row+4, col=col)
        
        # Add market condition legend
        fig.add_annotation(
            x=df["Open time"].iloc[0],
            y=5,
            text="Market Conditions: 1=Normal, 2=Trending, 3=Volatile, 4=Ranging, 5=Disorderly",
            showarrow=False,
            xanchor="left",
            row=row+4, col=col
        )
        
        # Plot ATR for volatility
        fig.add_trace(go.Scatter(
            x=df["Open time"],
            y=self.atr,
            mode='lines',
            name='ATR',
            line=dict(color="brown")
        ), row=row+4, col=col)
        
        if dt2idx is not None and 'tmp_ot' in locals():
            df["Open time"] = tmp_ot
            
        # Update layout
        fig.update_layout(
            title={
                "text": f"Forex Algorithm Strategy ({self.trader.symbol_name}) (tf {self.tf})",
                "x": 0.5,
                "xanchor": "center",
            }
        )
        
        return fig
        
    def analyze_performance(self):
        """Analyze strategy performance to identify high-probability setups"""
        if len(self.trade_history) < 10:
            return "Not enough trades to analyze performance patterns"
            
        # Calculate win rate
        profitable_trades = [t for t in self.trade_history if t["profit_pips"] > 0]
        win_rate = len(profitable_trades) / len(self.trade_history) if self.trade_history else 0
        
        # Average profit
        avg_profit = sum(t["profit_pips"] for t in self.trade_history) / len(self.trade_history) if self.trade_history else 0
        
        # Analyze partial vs full trades
        partial_trades = [t for t in self.trade_history if t.get("is_partial", False)]
        full_trades = [t for t in self.trade_history if not t.get("is_partial", False)]
        
        partial_win_rate = len([t for t in partial_trades if t["profit_pips"] > 0]) / len(partial_trades) if partial_trades else 0
        full_win_rate = len([t for t in full_trades if t["profit_pips"] > 0]) / len(full_trades) if full_trades else 0
        
        partial_avg_profit = sum(t["profit_pips"] for t in partial_trades) / len(partial_trades) if partial_trades else 0
        full_avg_profit = sum(t["profit_pips"] for t in full_trades) / len(full_trades) if full_trades else 0
        
        # Analyze by market condition
        condition_results = {}
        for condition in range(1, 6):
            condition_trades = [t for t in self.trade_history 
                               if t["market_context"]["market_condition"] == condition]
            
            if condition_trades:
                condition_wins = [t for t in condition_trades if t["profit_pips"] > 0]
                condition_win_rate = len(condition_wins) / len(condition_trades)
                condition_avg_profit = sum(t["profit_pips"] for t in condition_trades) / len(condition_trades)
                
                condition_name = {
                    1: "Normal",
                    2: "Trending",
                    3: "Volatile",
                    4: "Ranging",
                    5: "Disorderly"
                }.get(condition, f"Unknown ({condition})")
                
                condition_results[condition_name] = {
                    "trades": len(condition_trades),
                    "win_rate": condition_win_rate,
                    "avg_profit": condition_avg_profit
                }
        
        # Analyze by exit reason
        exit_reason_results = {}
        exit_reasons = set(t.get("exit_reason", "unknown") for t in self.trade_history)
        
        for reason in exit_reasons:
            reason_trades = [t for t in self.trade_history if t.get("exit_reason") == reason]
            if reason_trades:
                reason_wins = [t for t in reason_trades if t["profit_pips"] > 0]
                reason_win_rate = len(reason_wins) / len(reason_trades)
                reason_avg_profit = sum(t["profit_pips"] for t in reason_trades) / len(reason_trades)
                
                exit_reason_results[reason] = {
                    "trades": len(reason_trades),
                    "win_rate": reason_win_rate,
                    "avg_profit": reason_avg_profit
                }
        
        # Analyze trending vs. ranging
        trending_trades = [t for t in self.trade_history if t["market_context"]["is_trending"]]
        ranging_trades = [t for t in self.trade_history if not t["market_context"]["is_trending"]]
        
        trending_win_rate = len([t for t in trending_trades if t["profit_pips"] > 0]) / len(trending_trades) if trending_trades else 0
        ranging_win_rate = len([t for t in ranging_trades if t["profit_pips"] > 0]) / len(ranging_trades) if ranging_trades else 0
        
        trending_avg_profit = sum(t["profit_pips"] for t in trending_trades) / len(trending_trades) if trending_trades else 0
        ranging_avg_profit = sum(t["profit_pips"] for t in ranging_trades) / len(ranging_trades) if ranging_trades else 0
        
        # Analyze optimal exits
        optimal_exit_trades = [t for t in self.trade_history if t.get("exit_reason") == "optimal_exit"]
        optimal_exit_win_rate = len([t for t in optimal_exit_trades if t["profit_pips"] > 0]) / len(optimal_exit_trades) if optimal_exit_trades else 0
        optimal_exit_avg_profit = sum(t["profit_pips"] for t in optimal_exit_trades) / len(optimal_exit_trades) if optimal_exit_trades else 0
        
        # Build report
        report = {
            "overall": {
                "win_rate": win_rate,
                "avg_profit_pips": avg_profit,
                "total_trades": len(self.trade_history)
            },
            "by_trade_type": {
                "partial": {
                    "win_rate": partial_win_rate,
                    "avg_profit": partial_avg_profit,
                    "count": len(partial_trades)
                },
                "full": {
                    "win_rate": full_win_rate,
                    "avg_profit": full_avg_profit,
                    "count": len(full_trades)
                }
            },
            "by_market_condition": condition_results,
            "by_exit_reason": exit_reason_results,
            "market_regime": {
                "trending": {
                    "win_rate": trending_win_rate,
                    "avg_profit": trending_avg_profit,
                    "count": len(trending_trades)
                },
                "ranging": {
                    "win_rate": ranging_win_rate,
                    "avg_profit": ranging_avg_profit,
                    "count": len(ranging_trades)
                }
            },
            "optimal_exits": {
                "win_rate": optimal_exit_win_rate,
                "avg_profit": optimal_exit_avg_profit,
                "count": len(optimal_exit_trades)
            }
        }
        
        # Identify highest probability setups
        high_prob_setups = []
        
        # Check which market conditions have best performance
        best_condition = None
        best_condition_win_rate = 0
        
        for condition, stats in condition_results.items():
            if stats["trades"] >= 5 and stats["win_rate"] > best_condition_win_rate:
                best_condition = condition
                best_condition_win_rate = stats["win_rate"]
        
        if best_condition and best_condition_win_rate > 0.6:
            high_prob_setups.append(f"Trading in {best_condition} market conditions")
        
        # Check which exit reasons perform best
        best_exit = None
        best_exit_profit = 0
        
        for reason, stats in exit_reason_results.items():
            if stats["trades"] >= 5 and stats["avg_profit"] > best_exit_profit:
                best_exit = reason
                best_exit_profit = stats["avg_profit"]
        
        if best_exit:
            high_prob_setups.append(f"Exit strategy '{best_exit}' yields best profits")
        
        # Check if trending or ranging performs better
        if trending_win_rate > ranging_win_rate + 0.1 and len(trending_trades) >= 5:
            high_prob_setups.append("Trading with trend (ADX > 25)")
        if ranging_win_rate > trending_win_rate + 0.1 and len(ranging_trades) >= 5:
            high_prob_setups.append("Trading in range (ADX < 25)")
        
        # Check if partial or full trades perform better
        if partial_win_rate > full_win_rate + 0.1 and len(partial_trades) >= 5:
            high_prob_setups.append("Taking partial profits is effective")
        
        # Check if optimal exits are effective
        if optimal_exit_win_rate > 0.7 and len(optimal_exit_trades) >= 5:
            high_prob_setups.append("Optimal exit detection is effective")
        
        report["high_probability_setups"] = high_prob_setups
        return report