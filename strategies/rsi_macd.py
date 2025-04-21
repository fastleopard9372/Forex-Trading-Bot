import logging
import talib as ta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .base_strategy import BaseStrategy
import indicators as mta
from order import Order, OrderType, OrderSide, OrderStatus, OrderCloseSide
from utils import get_line_coffs, find_uptrend_line, find_downtrend_line, get_y_on_line
from scipy.stats import linregress
import numpy as np
from typing import Dict

bot_logger = logging.getLogger("bot_logger")

# Enhanced RSI-MACD strategy implementation
class RsiMacd(BaseStrategy):
    def __init__(self, name: str, params: Dict, tfs: Dict[str, str]):
        super().__init__(name, params, tfs)
        self.tf = self.tfs.get("tf", list(self.tfs.keys())[0])
        
        # Initialize parameters from config
        self.rsi_len = self.params["rsi_inputs"]["rsi_len"]
        self.ob_rsi = self.params["rsi_inputs"]["ob_rsi"]  # Overbought level
        self.os_rsi = self.params["rsi_inputs"]["os_rsi"]  # Oversold level
        self.adx_threshold = self.params["adx_inputs"]["threshold"]
        self.atr_multiplier = 1.3 #self.params["atr_inputs"]["atr_multiplier"]
        self.min_volume_ratio = self.params["vol_ratio_ma"]
        self.min_zz_ratio = 0.01 * self.params["min_zz_pct"]
        # Initialize indicators and state
        self.indicators = {}
        self.trading_flag = None
        self.last_signal_time = None
        self.signal_cooldown = pd.Timedelta(minutes=self.params.get("signal_cooldown_minutes", 240))
        
        # Signal conditions trackers
        self.conditions = {
            "rsi": [],
            "macd": [],
            "ma": [],
            "adx": [],
            "atr": [],
            "bb": []
        }
        
        # Performance metrics
        self.metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_profit": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0
        }
        
        # Risk adjustment multipliers
        self.risk_multiplier = 1.0
        self.counter_trend_risk_multiplier = 0.8
        
        # Market structure tracking
        self.market_structure = {
            "uptrend": False,
            "downtrend": False,
            "bullish_reversal": False,
            "bearish_reversal": False,
            "long_term_bullish": False,
            "long_term_bearish": False,
            "swing_highs": [],
            "swing_lows": []
        }
        self.temp = []

    def attach(self, tfs_chart: Dict[str, pd.DataFrame]):
        """Attach price charts and initialize indicators"""
        self.tfs_chart = tfs_chart
        self.init_indicators()
        bot_logger.info(f"Attached {self.name} strategy to {len(tfs_chart)} timeframes")
   
    def is_params_valid(self):
        if not self.check_required_params():
            bot_logger.info("   [-] Missing required params")
            return False
        return True
    
    def init_main_zigzag(self):
        self.main_zz_idx = []
        if len(self.indicators["zz_points"]) > 0:
            self.main_zz_idx.append(0)
        else:
            return
        last_main_zz_idx = 0
        while last_main_zz_idx + 3 < len(self.indicators["zz_points"]):
            last_main_zz_type = self.indicators["zz_points"][last_main_zz_idx].ptype
            if last_main_zz_type == mta.POINT_TYPE.PEAK_POINT:
                if (
                    self.indicators["zz_points"][last_main_zz_idx + 2].pline.high < self.indicators["zz_points"][last_main_zz_idx].pline.high
                    and self.indicators["zz_points"][last_main_zz_idx + 3].pline.low < self.indicators["zz_points"][last_main_zz_idx + 1].pline.low
                ):
                    last_main_zz_idx += 2
                else:
                    last_main_zz_idx += 1
                    self.main_zz_idx.append(last_main_zz_idx)
            else:
                if (
                    self.indicators["zz_points"][last_main_zz_idx + 2].pline.low > self.indicators["zz_points"][last_main_zz_idx].pline.low
                    and self.indicators["zz_points"][last_main_zz_idx + 3].pline.high > self.indicators["zz_points"][last_main_zz_idx + 1].pline.low
                ):
                    last_main_zz_idx += 2
                else:
                    last_main_zz_idx += 1
                    self.main_zz_idx.append(last_main_zz_idx)

    def update_main_zigzag(self):
        last_main_zz_idx = self.main_zz_idx[-1]
        while last_main_zz_idx + 3 < len(self.indicators["zz_points"]):
            last_main_zz_type = self.indicators["zz_points"][last_main_zz_idx].ptype
            if last_main_zz_type == mta.POINT_TYPE.PEAK_POINT:
                if (
                    self.indicators["zz_points"][last_main_zz_idx + 2].pline.high < self.indicators["zz_points"][last_main_zz_idx].pline.high
                    and self.indicators["zz_points"][last_main_zz_idx + 3].pline.low < self.indicators["zz_points"][last_main_zz_idx + 1].pline.low
                ):
                    last_main_zz_idx += 2
                else:
                    last_main_zz_idx += 1
                    self.main_zz_idx.append(last_main_zz_idx)
            else:
                if (
                    self.indicators["zz_points"][last_main_zz_idx + 2].pline.low > self.indicators["zz_points"][last_main_zz_idx].pline.low
                    and self.indicators["zz_points"][last_main_zz_idx + 3].pline.high
                    > self.indicators["zz_points"][last_main_zz_idx + 1].pline.high
                ):
                    last_main_zz_idx += 2
                else:
                    last_main_zz_idx += 1
                    self.main_zz_idx.append(last_main_zz_idx)

        last_main_zz_type = self.indicators["zz_points"][last_main_zz_idx].ptype
        if last_main_zz_idx + 2 < len(self.indicators["zz_points"]):
            if last_main_zz_type == mta.POINT_TYPE.POKE_POINT:
                if self.indicators["zz_points"][last_main_zz_idx + 2].pline.low < self.indicators["zz_points"][last_main_zz_idx].pline.low:
                    last_main_zz_idx += 1
                    self.main_zz_idx.append(last_main_zz_idx)
            else:
                if self.indicators["zz_points"][last_main_zz_idx + 2].pline.high > self.indicators["zz_points"][last_main_zz_idx].pline.high:
                    last_main_zz_idx += 1
                    self.main_zz_idx.append(last_main_zz_idx)

        last_main_zz_type = self.indicators["zz_points"][last_main_zz_idx].ptype
        if last_main_zz_idx + 1 < len(self.indicators["zz_points"]):
            last_kline = self.tfs_chart[self.tf].iloc[-1]
            if last_main_zz_type == mta.POINT_TYPE.POKE_POINT:
                if last_kline["Low"] < self.indicators["zz_points"][last_main_zz_idx].pline.low:
                    last_main_zz_idx += 1
                    self.main_zz_idx.append(last_main_zz_idx)
            else:
                if last_kline["High"] > self.indicators["zz_points"][last_main_zz_idx].pline.high:
                    last_main_zz_idx += 1
                    self.main_zz_idx.append(last_main_zz_idx)
 
    def init_indicators(self):
        """Initialize all technical indicators"""
        if self.tf not in self.tfs_chart:
            bot_logger.error(f"Timeframe {self.tf} not found in chart data")
            return
            
        chart = self.tfs_chart[self.tf]
        
        try:
            # Volume indicators
            self.indicators["ma_vol"] = ta.SMA(chart["Tick volume"], timeperiod=self.params["ma_vol"])
            
            # MACD
            self.indicators["macd"], self.indicators["macdsignal"], self.indicators["macdhist"] = ta.MACD(
                chart["Close"],
                fastperiod=self.params["macd_inputs"]["fast_len"],
                slowperiod=self.params["macd_inputs"]["slow_len"],
                signalperiod=self.params["macd_inputs"]["signal"]
            )
            self.indicators["zz_points"] = mta.zigzag(chart, self.min_zz_ratio)
            self.init_main_zigzag()
            # RSI
            self.indicators["rsi"] = ta.RSI(chart["Close"], timeperiod=self.rsi_len)
            
            # Moving Averages
            self.indicators["ma_short"] = ta.EMA(chart["Close"], timeperiod=self.params["ma_inputs"]["short"])
            self.indicators["ma_medium"] = ta.EMA(chart["Close"], timeperiod=self.params["ma_inputs"]["medium"])
            self.indicators["ma_long"] = ta.EMA(chart["Close"], timeperiod=self.params["ma_inputs"]["long"])
            
            # Bollinger Bands
            self.indicators["bb_upper"], self.indicators["bb_middle"], self.indicators["bb_lower"] = ta.BBANDS(
                chart["Close"],
                timeperiod=self.params["bb_inputs"]["len"],
                nbdevup=self.params["bb_inputs"]["nbdevup"],
                nbdevdn=self.params["bb_inputs"]["nbdevdn"],
                matype=0  # Simple moving average
            )
            
            # BB Width
            self.indicators["bb_width"] = (self.indicators["bb_upper"] - self.indicators["bb_lower"]) / self.indicators["bb_middle"]
            
            # ADX
            self.indicators["adx"] = ta.ADX(chart["High"], chart["Low"], chart["Close"], timeperiod=self.params["adx_inputs"]["len"])
            self.indicators["plus_di"] = ta.PLUS_DI(chart["High"], chart["Low"], chart["Close"], timeperiod=self.params["adx_inputs"]["len"])
            self.indicators["minus_di"] = ta.MINUS_DI(chart["High"], chart["Low"], chart["Close"], timeperiod=self.params["adx_inputs"]["len"])
            
            # ATR
            self.indicators["atr"] = ta.ATR(chart["High"], chart["Low"], chart["Close"], timeperiod=self.params["atr_inputs"]["len"])
            self.indicators["atr_pct"] = self.indicators["atr"] / chart["Close"] * 100
            
            # Stochastic RSI - additional indicator for confirmation
            self.indicators["stoch_k"], self.indicators["stoch_d"] = ta.STOCH(
                self.indicators["rsi"], 
                self.indicators["rsi"], 
                self.indicators["rsi"], 
                fastk_period=14, 
                slowk_period=3, 
                slowd_period=3
            )
            
            # Initialize condition arrays
            for condition in self.conditions:
                self.conditions[condition] = [None] * len(chart)
                
            bot_logger.info(f"Successfully initialized all indicators for {self.name}")
            
        except Exception as e:
            bot_logger.error(f"Error initializing indicators: {str(e)}")
            raise

    def update_indicators(self, tf):
        """Update indicators when new candle arrives"""
        if tf != self.tf:
            return
            
        chart = self.tfs_chart[self.tf]
        if len(chart) == 0:
            return
        
        # Update ZigZag
        mta.zigzag_stream(chart, self.min_zz_ratio, self.indicators["zz_points"])
        last_main_idx = self.main_zz_idx[-1]
        self.update_main_zigzag()
        # if last_main_idx != self.main_zz_idx[-1]:
        #     self.temp.append((self.indicators["zz_points"][self.main_zz_idx[-1]].pidx, len(chart) - 1))
            # self.adjust_sl()
        # try:
        # Update Volume MA
        self.indicators["ma_vol"].loc[len(self.indicators["ma_vol"])] = ta.stream.SMA(chart["Tick volume"], timeperiod=self.params["ma_vol"])

        #Update Moving Averages
        self.indicators["ma_short"].loc[len(self.indicators["ma_short"])] = ta.stream.EMA(chart["Close"], timeperiod=self.params["ma_inputs"]["short"])
        self.indicators["ma_medium"].loc[len(self.indicators["ma_medium"])] = ta.stream.EMA(chart["Close"], timeperiod=self.params["ma_inputs"]["medium"])
        self.indicators["ma_long"].loc[len(self.indicators["ma_long"])] = ta.stream.EMA(chart["Close"], timeperiod=self.params["ma_inputs"]["long"])
        
        # Update MACD
        macd, macdsignal, macdhist = ta.stream.MACD(
            chart["Close"],
            fastperiod=self.params["macd_inputs"]["fast_len"],
            slowperiod=self.params["macd_inputs"]["slow_len"],
            signalperiod=self.params["macd_inputs"]["signal"]
        )
        self.indicators["macd"].loc[len(self.indicators["macd"])] = macd
        self.indicators["macdsignal"].loc[len(self.indicators["macdsignal"])] = macdsignal
        self.indicators["macdhist"].loc[len(self.indicators["macdhist"])] = macdhist
        
        # Update RSI
        self.indicators["rsi"].loc[len(self.indicators["rsi"])] = ta.stream.RSI(chart["Close"], timeperiod=self.rsi_len)
        
        # Update Moving Averages
        for ma_type in ["short", "medium", "long"]:
            self.indicators[f"ma_{ma_type}"].loc[len(self.indicators[f"ma_{ma_type}"])] = ta.stream.EMA(chart["Close"], timeperiod=self.params["ma_inputs"][ma_type])   
        
        # Update Bollinger Bands
        bb_upper, bb_middle, bb_lower = ta.stream.BBANDS(
            chart["Close"],
            timeperiod=self.params["bb_inputs"]["len"],
            nbdevup=self.params["bb_inputs"]["nbdevup"],
            nbdevdn=self.params["bb_inputs"]["nbdevdn"],
            matype=0
        )
        self.indicators["bb_upper"].loc[len(self.indicators["bb_upper"])] = bb_upper
        self.indicators["bb_middle"].loc[len(self.indicators["bb_middle"])] = bb_middle
        self.indicators["bb_lower"].loc[len(self.indicators["bb_lower"])] = bb_lower
        
        # Update BB Width
        self.indicators["bb_width"].loc[len(self.indicators["bb_width"])] = (bb_upper - bb_lower) / bb_middle
        
        # Update ADX
        self.indicators["adx"].loc[len(self.indicators["adx"])] = ta.stream.ADX(chart["High"], chart["Low"], chart["Close"], timeperiod=self.params["adx_inputs"]["len"])
        self.indicators["plus_di"].loc[len(self.indicators["plus_di"])] = ta.stream.PLUS_DI(chart["High"], chart["Low"], chart["Close"], timeperiod=self.params["adx_inputs"]["len"])
        self.indicators["minus_di"].loc[len(self.indicators["minus_di"])] = ta.stream.MINUS_DI(chart["High"], chart["Low"], chart["Close"], timeperiod=self.params["adx_inputs"]["len"])
        
        # Update ATR
        atr = ta.stream.ATR(chart["High"], chart["Low"], chart["Close"], timeperiod=self.params["atr_inputs"]["len"])
        self.indicators["atr"].loc[len(self.indicators["atr"])] = atr
        self.indicators["atr_pct"].loc[len(self.indicators["atr_pct"])] = atr / chart["Close"].iloc[-1] * 100
        
        # Update Stochastic RSI
        k, d = ta.stream.STOCH(
            self.indicators["rsi"],
            self.indicators["rsi"],
            self.indicators["rsi"],
            fastk_period=14,
            slowk_period=3,
            slowd_period=3
        )
        self.indicators["stoch_k"].loc[len(self.indicators["stoch_k"])] = k
        self.indicators["stoch_d"].loc[len(self.indicators["stoch_d"])] = d
        
        # Extend condition arrays
        for condition in self.conditions:
            self.conditions[condition].append(None) 
            
            
        # except Exception as e:
        #     bot_logger.error(f"Error updating indicators: {str(e)}")

    def check_required_params(self):
        """Validate required parameters are present"""
        required_params = [
            "ma_vol", "vol_ratio_ma", "kline_body_ratio", "sl_fix_mode",
            "macd_inputs", "rsi_inputs", "ma_inputs", "bb_inputs", 
            "adx_inputs", "atr_inputs"
        ]
        
        missing_params = [param for param in required_params if param not in self.params]
        if missing_params:
            bot_logger.error(f"Missing required parameters: {missing_params}")
            return False
        return True

    def update(self, tf):
        """Main update method called when new candle arrives"""
        super().update(tf)
        if tf != self.tf:
            return
            
        # Update all indicators with new candle data
        self.update_indicators(tf)
        
        # Check close signals first (manage open positions)
        self.check_close_signal()
        
        # Then check for new entry signals
        self.check_signal()

    def _adjust_take_profit_with_targets(self, order, last_kline):
        """
        Implements a multi-stage take profit strategy with partial position closure
        
        This advanced take profit management:
        1. Sets multiple profit targets
        2. Closes portions of the position at each target
        3. Moves stop loss to breakeven after first target hit
        4. Adjusts final target based on market conditions
        """
        # Skip if we don't have partial profit tracking
        if "partial_exits" not in order:
            # Initialize partial exit tracking 
            order["partial_exits"] = 0
            order["partial_exit_levels"] = []
            
            # Set initial profit targets as R-multiples of initial risk
            if order.side == OrderSide.BUY:
                initial_risk = order.entry - order.sl if order.sl else order.entry * 0.01
                order["tp_level1"] = order.entry + initial_risk * 1.5  # 1.5R
                order["tp_level2"] = order.entry + initial_risk * 2.5  # 2.5R
                order["tp_level3"] = order.entry + initial_risk * 4.0  # 4.0R (final target)
            else:  # SELL
                initial_risk = order.sl - order.entry if order.sl else order.entry * 0.01
                order["tp_level1"] = order.entry - initial_risk * 1.5
                order["tp_level2"] = order.entry - initial_risk * 2.5
                order["tp_level3"] = order.entry - initial_risk * 4.0

            # Set the first target as TP
            order.adjust_tp(order["tp_level1"])
            self.trader.adjust_tp(order, order["tp_level1"])
            
            bot_logger.info(f"Set multi-level targets: TP1={order['tp_level1']:.4f}, " +
                          f"TP2={order['tp_level2']:.4f}, TP3={order['tp_level3']:.4f}")
            return
        
        # Check if price has hit any of our targets
        current_price = last_kline["Close"]
        hit_target = False
        
        if order.side == OrderSide.BUY:
            # For buy orders, check if price went high enough to hit target
            high_price = last_kline["High"]
            
            # Check which level we're on and if target was hit
            if order["partial_exits"] == 0 and high_price >= order["tp_level1"]:
                # First target hit - close 1/3 of position
                hit_target = True
                next_target = order["tp_level2"]
                order["partial_exits"] = 1
                order["partial_exit_levels"].append({"price": order["tp_level1"], "time": last_kline["Open time"]})
                
                # Move stop loss to breakeven
                if order.sl < order.entry:
                    new_sl = order.entry + (order.entry * 0.001)  # Small buffer above entry
                    order.adjust_sl(new_sl)
                    self.trader.adjust_sl(order, new_sl)
                    bot_logger.info(f"Moved stop loss to breakeven+ at {new_sl:.4f}")

            elif order["partial_exits"] == 1 and high_price >= order["tp_level2"]:
                # Second target hit - close another 1/3
                hit_target = True
                next_target = order["tp_level3"]
                order["partial_exits"] = 2
                order["partial_exit_levels"].append({"price": order["tp_level2"], "time": last_kline["Open time"]})
                
                # Lock in more profit by moving stop further up
                new_sl = order["tp_level1"]
                order.adjust_sl(new_sl)
                self.trader.adjust_sl(order, new_sl)
                bot_logger.info(f"Moved stop loss to TP1 level at {new_sl:.4f}")
                
            # Check if we should adjust the final take profit target
            if order["partial_exits"] == 2:
                # Analyze trend strength for final portion
                trend_strengthening = (self.indicators["adx"].iloc[-1] > self.indicators["adx"].iloc[-2] * 1.1 and 
                                      self.indicators["plus_di"].iloc[-1] > self.indicators["plus_di"].iloc[-2])
                                      
                if trend_strengthening and self.indicators["adx"].iloc[-1] > self.adx_threshold * 1.5:
                    # Strong trend developing - extend final target
                    current_atr = self.indicators["atr"].iloc[-1]
                    new_final_tp = order["tp_level3"] + (current_atr * 2)
                    
                    # Only update if it's a meaningful extension
                    if new_final_tp > order["tp_level3"] * 1.05:
                        order["tp_level3"] = new_final_tp
                        bot_logger.info(f"Extended final target to {new_final_tp:.4f} due to strengthening trend")
        
        elif order.side == OrderSide.SELL:
            # For sell orders, check if price went low enough to hit target
            low_price = last_kline["Low"]
            
            # Check which level we're on and if target was hit
            if order["partial_exits"] == 0 and low_price <= order["tp_level1"]:
                # First target hit - close 1/3 of position
                hit_target = True
                next_target = order["tp_level2"]
                order["partial_exits"] = 1
                order["partial_exit_levels"].append({"price": order["tp_level1"], "time": last_kline["Open time"]})
                
                # Move stop loss to breakeven
                if order.sl > order.entry:
                    new_sl = order.entry - (order.entry * 0.001)  # Small buffer below entry
                    order.adjust_sl(new_sl)
                    self.trader.adjust_sl(order, new_sl)
                    bot_logger.info(f"Moved stop loss to breakeven+ at {new_sl:.4f}")

            elif order["partial_exits"] == 1 and low_price <= order["tp_level2"]:
                # Second target hit - close another 1/3
                hit_target = True
                next_target = order["tp_level3"]
                order["partial_exits"] = 2
                order["partial_exit_levels"].append({"price": order["tp_level2"], "time": last_kline["Open time"]})
                
                # Lock in more profit by moving stop further down
                new_sl = order["tp_level1"]
                order.adjust_sl(new_sl)
                self.trader.adjust_sl(order, new_sl)
                bot_logger.info(f"Moved stop loss to TP1 level at {new_sl:.4f}")
                
            # Check if we should adjust the final take profit target
            if order["partial_exits"] == 2:
                # Analyze trend strength for final portion
                trend_strengthening = (self.indicators["adx"].iloc[-1] > self.indicators["adx"].iloc[-2] * 1.1 and 
                                      self.indicators["minus_di"].iloc[-1] > self.indicators["minus_di"].iloc[-2])
                                      
                if trend_strengthening and self.indicators["adx"].iloc[-1] > self.adx_threshold * 1.5:
                    # Strong trend developing - extend final target
                    current_atr = self.indicators["atr"].iloc[-1]
                    new_final_tp = order["tp_level3"] - (current_atr * 2)
                    
                    # Only update if it's a meaningful extension
                    if new_final_tp < order["tp_level3"] * 0.95:
                        order["tp_level3"] = new_final_tp
                        bot_logger.info(f"Extended final target to {new_final_tp:.4f} due to strengthening trend")
        
        # If we hit a target, update to the next target
        if hit_target:
            # Simulate a partial position closure
            # In a real implementation, you would:
            # 1. Close a portion of the position
            # 2. Keep track of remaining position size
            bot_logger.info(f"Hit take profit target: {order.tp:.4f}")
            
            # Set next target
            if order["partial_exits"] <= 2:
                next_level_name = f"tp_level{order['partial_exits'] + 1}"
                next_target = order[next_level_name]
                order.adjust_tp(next_target)
                self.trader.adjust_tp(order, next_target)
                bot_logger.info(f"Updated take profit to target {order['partial_exits'] + 1}: {next_target:.4f}")

    def check_signal(self):
        """Check for entry signals based on indicator conditions"""
        chart = self.tfs_chart[self.tf]
        if len(chart) < 2:
            return
            
        last_kline = chart.iloc[-1]
        
        # Volume filter - ignore low volume periods
        # if last_kline["Tick volume"] < 50:
        #     return
            
        # Check if volume is below MA threshold
        # if last_kline["Tick volume"] < self.min_volume_ratio * self.indicators["ma_vol"].iloc[-1]:
        #     return
        
        # Signal cooldown check
        # if self.last_signal_time is not None:
        #     if last_kline["Open time"] - self.last_signal_time < self.signal_cooldown:
        #         return
                
        # Market structure analysis for trend context
        self.analyze_market_structure(last_kline)
        
    def analyze_market_structure(self, last_kline):
        chart = self.tfs_chart[self.tf]
        # === Higher Timeframe Trend Analysis ===
        medium_ma_values = self.indicators["ma_medium"][-10:]
        long_ma_values = self.indicators["ma_long"].iloc[-200:]

        self.indicators["zz_points"]


        medium_slope, _, _, _, _ = linregress(range(len(medium_ma_values)), medium_ma_values)
        long_slope, _, _, _, _ = linregress(range(len(long_ma_values)), long_ma_values)

        # Stronger trend identification with statistical significance
        long_term_bullish = medium_slope > 0.0002 and long_slope > 0.0001
        long_term_bearish = medium_slope < -0.0002 and long_slope < -0.0001
        
        # idx = 1
        # while idx < len(self.indicators["zz_points"]):
        #     zz_point_1 = self.indicators["zz_points"][-idx]
        #     zz_point_2 = self.indicators["zz_points"][-idx - 1]
        #     if zz_point_2.ptype == mta.POINT_TYPE.POKE_POINT:
        #         change = (zz_point_1.pline.high - zz_point_2.pline.low) / zz_point_2.pline.low
        #     else:
        #         change = (zz_point_2.pline.high - zz_point_1.pline.low) / zz_point_2.pline.high
        #     if change > self.params["zz_dev"] * self.min_zz_ratio:
        #         break
        #     idx += 1

        # n_df = chart[self.indicators["zz_points"][-idx].pidx : -1]
        # if len(n_df) < self.params["min_num_cuml"]:
        #     return
        # n_last_poke_points = []
        # n_last_peak_points = []
        # for i, kline in n_df.iterrows():
        #     n_last_poke_points.append((i, kline["Low"]))
        #     n_last_peak_points.append((i, kline["High"]))

        
        # self.up_trend_line = find_uptrend_line(n_last_poke_points)
        # self.down_trend_line = find_downtrend_line(n_last_peak_points)

        # self.up_pct = (self.up_trend_line[1][1] - self.up_trend_line[0][1]) / self.up_trend_line[0][1]
        # self.down_pct = (self.down_trend_line[1][1] - self.down_trend_line[0][1]) / self.down_trend_line[0][1]
        # delta_end = abs(self.down_trend_line[1][1] - self.up_trend_line[1][1]) / self.up_trend_line[1][1]
        # if delta_end > self.params["zz_dev"] * self.min_zz_ratio:
        #     return

        higher_highs = False
        higher_lows = False
        lower_highs = False
        lower_lows = False
        p_1 = None
        main_last_zigzag = self.indicators["zz_points"][self.main_zz_idx[-1]]
        if(len(self.indicators["zz_points"]) > 4):
            p_1 = self.indicators["zz_points"][-1]
            p_2 = self.indicators["zz_points"][-2]
            p_3 = self.indicators["zz_points"][-3]
            p_4 = self.indicators["zz_points"][-4]
            #        /\        /\   3
            #   /\  /  \/      5 \  /\    1
            #  /  \/  2  1        \/  \  /\
            # / 4  3              4    \/  
            # 5   
        if p_1 != None and p_1.ptype == mta.POINT_TYPE.POKE_POINT:  # Current point is a low
            higher_lows = p_1.pline.low > p_3.pline.low
            higher_highs = p_2.pline.high > p_4.pline.high
            
            lower_lows = p_1.pline.low < p_3.pline.low
            lower_highs = p_2.pline.high < p_4.pline.high
            
        elif p_1 != None and p_1.ptype == mta.POINT_TYPE.PEAK_POINT:  # Current point is a high
            higher_highs = p_1.pline.high > p_3.pline.high and p_3.pline.high
            higher_lows = p_2.pline.low > p_4.pline.low
            
            lower_highs = p_1.pline.high < p_3.pline.high and p_3.pline.high
            lower_lows = p_2.pline.low < p_4.pline.low

        print(chart.iloc[p_1.pidx]["Open time"],chart.iloc[p_1.pidx]["Close"])
        print(chart.iloc[p_2.pidx]["Open time"],chart.iloc[p_2.pidx]["Close"])
        print(chart.iloc[p_3.pidx]["Open time"],chart.iloc[p_3.pidx]["Close"])
        print(chart.iloc[p_4.pidx]["Open time"],chart.iloc[p_4.pidx]["Close"])
        # === Determine Current Market Structure ===
        uptrend = higher_highs and higher_lows
        downtrend = lower_highs and lower_lows
        
        # Handle mixed signals by checking longer-term trend
        
        if higher_highs and lower_lows:
            uptrend = long_term_bullish
            downtrend = long_term_bearish
            
        if lower_highs and higher_lows:
            uptrend = long_term_bullish
            downtrend = long_term_bearish
        
        # === Check for Potential Reversals ===
        # Bullish reversal: previous downtrend + higher low + break of previous high
        bullish_reversal = (downtrend and higher_lows and
                            last_kline["Close"] > main_last_zigzag.pline.high)

        # Bearish reversal: previous uptrend + lower high + break of previous low
        bearish_reversal = (uptrend and lower_highs and
                            last_kline["Close"] < main_last_zigzag.pline.low)
        
        
        # === Moving Average Condition ===
        # Buy: Short MA > Medium MA > Long MA and Short MA is rising
        long_up_trend = False
        long_down_trend = False
        
        if (self.indicators["ma_short"].iloc[-1] > self.indicators["ma_medium"].iloc[-1] and 
            self.indicators["ma_medium"].iloc[-1] > self.indicators["ma_long"].iloc[-1] and
            self.indicators["ma_short"].iloc[-5] < self.indicators["ma_short"].iloc[-1] and
            last_kline["Low"] > self.indicators["ma_short"].iloc[-1]):
            long_up_trend = True
        # Sell: Short MA < Medium MA < Long MA and Short MA is falling
        elif (self.indicators["ma_short"].iloc[-1] < self.indicators["ma_medium"].iloc[-1] and 
            self.indicators["ma_medium"].iloc[-1] < self.indicators["ma_long"].iloc[-1] and
            self.indicators["ma_short"].iloc[-5] > self.indicators["ma_short"].iloc[-1] and
            last_kline["High"] < self.indicators["ma_short"].iloc[-1]):
            long_down_trend = True
            
        # Store market structure state for strategy use
        self.market_structure = {
            "uptrend": uptrend,
            "downtrend": downtrend,
            "long_up_trend": long_up_trend,
            "long_down_trend": long_down_trend,
            "bullish_reversal": bullish_reversal,
            "bearish_reversal": bearish_reversal,
            "long_term_bullish": long_term_bullish,
            "long_term_bearish": long_term_bearish,
        }
        
        bot_logger.debug(f"Market structure: Uptrend={uptrend}, Downtrend={downtrend}, Long_Up_Trend={long_up_trend}, Long_Down_Trend={long_down_trend}, Bullish_Reversal={bullish_reversal}, Bearish_Reversal={bearish_reversal}")
        
        # Use market structure for risk adjustment
        if uptrend or long_up_trend:
            # In established uptrend, we can use looser stops for BUY trades
            self.risk_multiplier = self.params.get("uptrend_risk_multiplier", 0.8)
            # And tighter stops for SELL trades (counter-trend)
            self.counter_trend_risk_multiplier = self.params.get("counter_trend_multiplier", 0.5)
        elif downtrend or long_down_trend:
            # In established downtrend, we can use looser stops for SELL trades
            self.risk_multiplier = self.params.get("downtrend_risk_multiplier", 0.8)
            # And tighter stops for BUY trades (counter-trend)
            self.counter_trend_risk_multiplier = self.params.get("counter_trend_multiplier", 0.5)
        else:
            # In range or unclear trend, use standard risk settings
            self.risk_multiplier = 0.7
            self.counter_trend_risk_multiplier = 0.6
        
        # Get index for latest candle
        idx = len(self.conditions["rsi"]) - 1
        
        # Log current indicator values for debugging
        bot_logger.debug(f"MA: {self.indicators['ma_short'].iloc[-1]:.4f}, {self.indicators['ma_medium'].iloc[-1]:.4f}, {self.indicators['ma_long'].iloc[-1]:.4f}")
        bot_logger.debug(f"BB Width: {self.indicators['bb_width'].iloc[-1]:.4f}, RSI: {self.indicators['rsi'].iloc[-1]:.4f}")
        bot_logger.debug(f"ATR %: {float(self.indicators['atr_pct'].iloc[-1]):.4f}, ADX: {float(self.indicators['adx'].iloc[-1]):.4f}")
        
        # === RSI Condition ===
        print("RSI",round(self.indicators["rsi"].iloc[-3],4),round(self.indicators["rsi"].iloc[-2],4), round(self.indicators["rsi"].iloc[-1],4))
        if self.indicators["rsi"].iloc[-3] < self.os_rsi and self.indicators["rsi"].iloc[-3] < self.indicators["rsi"].iloc[-1]:
            self.conditions["rsi"][idx] = OrderSide.BUY
        elif self.indicators["rsi"].iloc[-3] > self.ob_rsi and self.indicators["rsi"].iloc[-3] > self.indicators["rsi"].iloc[-1]:
            self.conditions["rsi"][idx] = OrderSide.SELL
        else:
            self.conditions["rsi"][idx] = None
        # === MACD Condition ===
        print("MACD",round(self.indicators["macdhist"].iloc[-3],4),round(self.indicators["macd"].iloc[-3],4), round(self.indicators["macd"].iloc[-1],4))
        if self.indicators["macdhist"].iloc[-3] > 0 and self.indicators["macd"].iloc[-3] < self.indicators["macd"].iloc[-1] and self.indicators["macd"].iloc[-1] < 0:
            self.conditions["macd"][idx] = OrderSide.BUY
        elif self.indicators["macdhist"].iloc[-3] < 0 and self.indicators["macd"].iloc[-3] > self.indicators["macd"].iloc[-1] and self.indicators["macd"].iloc[-1] > 0:
            self.conditions["macd"][idx] = OrderSide.SELL
        else:
            self.conditions["macd"][idx] = None
        
        # === ADX Condition (Trend Strength) ===
        # Check if ADX indicates a strong trend (but not too strong) and which direction
        if self.indicators["adx"].iloc[-1] > self.adx_threshold and self.indicators["adx"].iloc[-1] < self.adx_threshold * 2.5:
            if self.indicators["plus_di"].iloc[-1] > self.indicators["minus_di"].iloc[-1]*1.5:
                self.conditions["adx"][idx] = OrderSide.BUY
            elif self.indicators["minus_di"].iloc[-1] > self.indicators["plus_di"].iloc[-1]*1.5:
                self.conditions["adx"][idx] = OrderSide.SELL
            else:
                self.conditions["adx"][idx] = None
        else:
            self.conditions["adx"][idx] = None
        
        # === ATR Condition (Volatility) ===
        # Check if there's enough volatility to trade
        min_atr_threshold = self.params["atr_inputs"].get("min_threshold", 0.02)
        if self.indicators["atr_pct"].iloc[-1] > min_atr_threshold:
            self.conditions["atr"][idx] = True
        else:
            self.conditions["atr"][idx] = False
        
        # === Bollinger Band Condition ===
        # Buy: Price near lower band and moving up
        if (self.tfs_chart[self.tf]["Close"].iloc[-1] < self.indicators["bb_lower"].iloc[-1] * 1.001 and 
            self.tfs_chart[self.tf]["Close"].iloc[-2] < self.tfs_chart[self.tf]["Close"].iloc[-1]):
            self.conditions["bb"][idx] = OrderSide.BUY
        # Sell: Price near upper band and moving down
        elif (self.tfs_chart[self.tf]["Close"].iloc[-1] > self.indicators["bb_upper"].iloc[-1] * 0.999 and 
              self.tfs_chart[self.tf]["Close"].iloc[-2] > self.tfs_chart[self.tf]["Close"].iloc[-1]):
            self.conditions["bb"][idx] = OrderSide.SELL
        else:
            self.conditions["bb"][idx] = None
        
        # Log all current conditions
        bot_logger.debug(f"RSI: {self.conditions['rsi'][idx]}, MACD: {self.conditions['macd'][idx]}")
        bot_logger.debug(f"MA: {self.conditions['ma'][idx]}, ADX: {self.conditions['adx'][idx]}")
        bot_logger.debug(f"ATR: {self.conditions['atr'][idx]}, BB: {self.conditions['bb'][idx]}")
        
        # Reset trading flag
        self.trading_flag = None
        
        # === BUY Signal ===
        # Require majority of conditions to align for stronger signals
        required_conditions = ['rsi', 'macd', 'bb', 'adx']
        required_cnd_market = [1.3, 1.5, 1.2, 1.2]
        optional_conditions = None  # BB condition is optional
        # optional_conditions = ['bb']  # BB condition is optional
        
        # Count buy signals from required conditions
        # buy_count = sum(1 for cond in required_conditions
        #                 if self.conditions[cond][idx] == OrderSide.BUY)
        buy_count = 0
        for i, cond in enumerate(required_conditions):
            if self.conditions[cond][idx] == OrderSide.BUY:
                buy_count += required_cnd_market[i]
        
        # Include optional conditions if they match
        # buy_optional_count = sum(1 for cond in optional_conditions 
        #                         if self.conditions[cond][idx] == OrderSide.BUY)
        buy_optional_count = 1

        sell_count = 0
        for i, cond in enumerate(required_conditions):
            if self.conditions[cond][idx] == OrderSide.SELL:
                sell_count += required_cnd_market[i]
        
        
        # Adjust signal threshold based on market structure
        signal_threshold = 0.6  # Default threshold: 60% of conditions
        sell_signal_threshold = 0.8
        buy_signal_threshold = 0.8
        if(self.tfs_chart[self.tf]["Close"].iloc[-1] > self.tfs_chart[self.tf]["Close"].iloc[-3]):
            if ((self.market_structure["uptrend"] or self.market_structure["long_up_trend"]) and self.conditions["atr"][idx]) or \
                ((buy_count >= len(required_conditions) * signal_threshold) and \
                self.conditions["adx"][idx] == OrderSide.BUY and \
                (buy_optional_count > 0 or self.market_structure["bearish_reversal"])):
                    if(self.indicators["rsi"].iloc[-1] < 80):
                        self.trading_flag = OrderSide.BUY
            elif (self.market_structure["downtrend"] or self.market_structure["long_down_trend"]) and \
                (buy_count >= len(required_conditions) * buy_signal_threshold):
                    self.trading_flag = OrderSide.BUY
            
        # === SELL Signal ===
        sell_count = 0
        for i, cond in enumerate(required_conditions):
            if self.conditions[cond][idx] == OrderSide.SELL:
                sell_count += required_cnd_market[i]
        
        sell_optional_count = 1
        if(self.tfs_chart[self.tf]["Close"].iloc[-1] < self.tfs_chart[self.tf]["Close"].iloc[-3]):
            if ((self.market_structure["downtrend"] or self.market_structure["long_down_trend"]) and self.conditions["atr"][idx]) or \
                ((sell_count >= len(required_conditions) * signal_threshold) and \
                self.conditions["adx"][idx] == OrderSide.SELL and \
                (sell_optional_count > 0 or self.market_structure["bullish_reversal"])):
                    if(self.indicators["rsi"].iloc[-1] > 20):
                        self.trading_flag = OrderSide.SELL
            elif (self.market_structure["uptrend"] or self.market_structure["long_up_trend"]) and \
                (sell_count >= len(required_conditions) * sell_signal_threshold):
                    self.trading_flag = OrderSide.SELL
            
        # Don't proceed if no trading signal
        if self.trading_flag is None:
            return
            
        # Calculate position sizing based on ATR
        atr_value = self.indicators["atr"].iloc[-1] * 10
        
        # Apply risk multiplier based on market structure
        if self.trading_flag == OrderSide.BUY:
            if self.market_structure["downtrend"]:
                # Counter-trend trade
                risk_factor = self.counter_trend_risk_multiplier
            else:
                # With-trend trade
                risk_factor = self.risk_multiplier
        else:  # SELL
            if self.market_structure["uptrend"]:
                # Counter-trend trade
                risk_factor = self.counter_trend_risk_multiplier
            else:
                # With-trend trade
                risk_factor = self.risk_multiplier
        
        # Set take profit and stop loss based on ATR volatility and market structure
        if self.trading_flag == OrderSide.BUY:
            # For buy orders, calculate take profit and stop loss with risk adjustment
            tp = last_kline["Close"] + (atr_value * self.atr_multiplier * 2)
            sl = last_kline["Close"] - (atr_value * self.atr_multiplier * risk_factor)
            
            # In strong uptrend, potentially use wider targets
            if (self.market_structure["uptrend"] or self.market_structure["long_up_trend"]) and self.indicators["adx"].iloc[-1] > self.adx_threshold * 1.05:
                tp = last_kline["Close"] + (atr_value * self.atr_multiplier * 3)
            
            # Create a new buy order
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
            order["description"] = f"{self.description} - BUY"
            
            # Store context for position management
            order["market_structure"] = self.market_structure.copy()
            order["entry_atr"] = atr_value
            
            # Validate the order
            order = self.trader.fix_order(order, self.params["sl_fix_mode"], self.max_sl_pct)
            if order:
                # Create the trade and track the order
                self.trader.create_trade(order, self.volume)
                self.orders_opening.append(order)
                self.last_signal_time = last_kline["Open time"]
                bot_logger.info(f"BUY signal triggered at {last_kline['Close']}, TP: {tp}, SL: {sl}")
                
        elif self.trading_flag == OrderSide.SELL:
            # For sell orders, calculate take profit and stop loss with risk adjustment
            tp = last_kline["Close"] - (atr_value * self.atr_multiplier * 2)
            sl = last_kline["Close"] + (atr_value * self.atr_multiplier * risk_factor)
            
            # In strong downtrend, potentially use wider targets
            if (self.market_structure["downtrend"] or self.market_structure["long_down_trend"]) and self.indicators["adx"].iloc[-1] > self.adx_threshold * 1.05:
                tp = last_kline["Close"] - (atr_value * self.atr_multiplier * 3)
            
            # Create a new sell order
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
            order["description"] = f"{self.description} - SELL"
            
            # Store context for position management
            order["market_structure"] = self.market_structure.copy()
            order["entry_atr"] = atr_value
            
            # Validate the order
            order = self.trader.fix_order(order, self.params["sl_fix_mode"], self.max_sl_pct)
            if order:
                # Create the trade and track the order
                self.trader.create_trade(order, self.volume)
                self.orders_opening.append(order)
                self.last_signal_time = last_kline["Open time"]
                bot_logger.info(f"SELL signal triggered at {last_kline['Close']}, TP: {tp}, SL: {sl}")
                
    def check_close_signal(self):
        """Check for close signals on open positions"""
        if not self.orders_opening:
            return
            
        chart = self.tfs_chart[self.tf]
        last_kline = chart.iloc[-1]
        
        for i in range(len(self.orders_opening) - 1, -1, -1):
            order = self.orders_opening[i]
            close_signal = False
            
            # Check for SL/TP hits
            if order.side == OrderSide.BUY:
                # Check stop loss
                if order.sl is not None and last_kline["Low"] <= order.sl:
                    bot_logger.info(f"Stop loss hit for BUY at {order.sl}")
                    close_signal = True
                # Check take profit
                elif order.tp is not None and last_kline["High"] >= order.tp:
                    bot_logger.info(f"Take profit hit for BUY at {order.tp}")
                    close_signal = True
                # Check for reversal signals
                elif (self.indicators["rsi"].iloc[-1] > self.ob_rsi and 
                      self.indicators["macdhist"].iloc[-1] < 0) :
                    bot_logger.info(f"Reversal signal detected for BUY position")
                    close_signal = True
                # Multiple adaptive trade management techniques
                else:
                    # Get current trend strength and volatility
                    trend_strengthening = (self.indicators["adx"].iloc[-1] > self.indicators["adx"].iloc[-3] and 
                                          self.indicators["plus_di"].iloc[-1] > self.indicators["plus_di"].iloc[-3])
                    current_atr = self.indicators["atr"].iloc[-1]
                    price_movement = (last_kline["Close"] - order.entry) / order.entry
                    
                    # === Adjust Stop Loss: Dynamic Trailing Stop ===
                    # Base condition to start trailing
                    if self.params.get("use_trailing_stop", True) and last_kline["Close"] > order.entry:
                        trailing_activated = price_movement >= self.params.get("trailing_start_pct", 0.003)
                        # Different trailing logic based on trend strength
                        if trailing_activated:
                            # Tighter trailing in choppy markets, looser in trends
                            if trend_strengthening:
                                # Wider trailing stop in strong trends
                                trailing_distance = current_atr * 1.5
                            else:
                                # Tighter trailing stop when trend weakens
                                trailing_distance = current_atr * 0.75
                                
                            # Calculate new stop loss
                            new_sl = max(order.sl, last_kline["Close"] - trailing_distance)
                            if new_sl > order.sl:
                                order.adjust_sl(new_sl)
                                self.trader.adjust_sl(order, new_sl)
                                bot_logger.info(f"Updated trailing stop for BUY to {new_sl:.4f} (ATR: {current_atr:.4f})")
                    
                    # === Adjust Take Profit: Extend in Strong Trends ===
                    # Only adjust TP if we're in profit and have a strong trend
                    tp_adjustment_threshold = self.params.get("tp_adjustment_threshold", 0.003)
                    if (order.tp is not None and 
                        price_movement >= tp_adjustment_threshold and 
                        trend_strengthening and 
                        self.indicators["adx"].iloc[-1] > self.adx_threshold * 1.5):
                        
                        # Calculate R-multiple (how many times current profit exceeds initial risk)
                        initial_risk = order.entry - order.sl
                        current_profit = last_kline["Close"] - order.entry
                        r_multiple = current_profit / initial_risk if initial_risk > 0 else 1
                        
                        # Scale TP extension based on R-multiple to be more aggressive with winners
                        tp_extension = current_atr * (1 + r_multiple * 0.5)
                        new_tp = order.tp + tp_extension
                        
                        # Use time-based dampening - reduce aggressiveness as trade duration increases
                        if "FILL_TIME" in order:
                            trade_duration = last_kline["Open time"] - order["FILL_TIME"]
                            # Dampen extension for trades open longer than 48 hours
                            hours_open = trade_duration.total_seconds() / (2 * 3600)
                            if hours_open > 2:
                                dampening_factor = max(0.5, 1 - (hours_open - 2) * 0.1)
                                new_tp = order.tp + (tp_extension * dampening_factor)
                        
                        if new_tp > order.tp:
                            order.adjust_tp(new_tp)
                            self.trader.adjust_tp(order, new_tp)
                            bot_logger.info(f"Extended take profit for BUY to {new_tp:.4f} (R-multiple: {r_multiple:.2f})")
                
            elif order.side == OrderSide.SELL:
                # Check stop loss
                if order.sl is not None and last_kline["High"] >= order.sl:
                    bot_logger.info(f"Stop loss hit for SELL at {order.sl}")
                    close_signal = True
                # Check take profit
                elif order.tp is not None and last_kline["Low"] <= order.tp:
                    bot_logger.info(f"Take profit hit for SELL at {order.tp}")
                    close_signal = True
                # Check for reversal signals
                elif (self.indicators["rsi"].iloc[-1] < self.os_rsi and 
                      self.indicators["macdhist"].iloc[-1] > 0) :
                    bot_logger.info(f"Reversal signal detected for SELL position")
                    close_signal = True
                # Multiple adaptive trade management techniques
                else:
                    # Get current trend strength and volatility
                    trend_strengthening = (self.indicators["adx"].iloc[-1] > self.indicators["adx"].iloc[-3] and 
                                          self.indicators["minus_di"].iloc[-1] > self.indicators["minus_di"].iloc[-3])
                    current_atr = self.indicators["atr"].iloc[-1]
                    price_movement = (order.entry - last_kline["Close"]) / order.entry
                    
                    # === Adjust Stop Loss: Dynamic Trailing Stop ===
                    # Base condition to start trailing
                    if self.params.get("use_trailing_stop", True) and last_kline["Close"] < order.entry:
                        trailing_activated = price_movement >= self.params.get("trailing_start_pct", 0.003)
                        # Different trailing logic based on trend strength
                        if trailing_activated:
                            # Tighter trailing in choppy markets, looser in trends
                            if trend_strengthening:
                                # Wider trailing stop in strong trends
                                trailing_distance = current_atr * 1.5
                            else:
                                # Tighter trailing stop when trend weakens
                                trailing_distance = current_atr * 0.75
                                
                            # Calculate new stop loss
                            new_sl = min(order.sl, last_kline["Close"] + trailing_distance)
                            if new_sl < order.sl:
                                order.adjust_sl(new_sl)
                                self.trader.adjust_sl(order, new_sl)
                                bot_logger.info(f"Updated trailing stop for SELL to {new_sl:.4f} (ATR: {current_atr:.4f})")
                    
                    # === Adjust Take Profit: Extend in Strong Trends ===
                    # Only adjust TP if we're in profit and have a strong trend
                    tp_adjustment_threshold = self.params.get("tp_adjustment_threshold", 0.003)
                    if (order.tp is not None and 
                        price_movement >= tp_adjustment_threshold and 
                        trend_strengthening and 
                        self.indicators["adx"].iloc[-1] > self.adx_threshold * 1.5):
                        
                        # Calculate R-multiple (how many times current profit exceeds initial risk)
                        initial_risk = order.sl - order.entry
                        current_profit = order.entry - last_kline["Close"]
                        r_multiple = current_profit / initial_risk if initial_risk > 0 else 1
                        
                        # Scale TP extension based on R-multiple to be more aggressive with winners
                        tp_extension = current_atr * (1 + r_multiple * 0.5)
                        new_tp = order.tp - tp_extension
                        
                        # Use time-based dampening - reduce aggressiveness as trade duration increases
                        if "FILL_TIME" in order:
                            trade_duration = last_kline["Open time"] - order["FILL_TIME"]
                            # Dampen extension for trades open longer than 48 hours
                            hours_open = trade_duration.total_seconds() / (2 * 3600)
                            if hours_open > 2:
                                dampening_factor = max(0.5, 1 - (hours_open - 2) * 0.1)
                                new_tp = order.tp - (tp_extension * dampening_factor)
                        
                        if new_tp < order.tp:
                            order.adjust_tp(new_tp)
                            self.trader.adjust_tp(order, new_tp)
                            bot_logger.info(f"Extended take profit for SELL to {new_tp:.4f} (R-multiple: {r_multiple:.2f})")
            
            # Apply multi-stage take profit strategy to open positions
            if not close_signal and self.params.get("use_multi_stage_tp", True):
                self._adjust_take_profit_with_targets(order, last_kline)
            
            # Close the position if signal detected
            if close_signal:
                order.close(last_kline)
                self.trader.close_trade(order)
                
                # Update metrics
                self.metrics["total_trades"] += 1
                if order.profit > 0:
                    self.metrics["winning_trades"] += 1
                else:
                    self.metrics["losing_trades"] += 1
                self.metrics["total_profit"] += order.profit
                
                # Move to closed orders list
                if order.is_closed():
                    self.orders_closed.append(order)
                    # Calculate win rate
                    self.metrics["win_rate"] = (self.metrics["winning_trades"] / self.metrics["total_trades"]) * 100
                    
                # Remove from opening orders
                del self.orders_opening[i]
    
    def close_opening_orders(self):
        super().close_opening_orders(self.tfs_chart[self.tf].iloc[-1])

    def plot_chart(self) :
        
        """Create an interactive chart with indicators and trades"""
        chart = self.tfs_chart[self.tf]
        
        # Create subplots with 4 rows
        fig = make_subplots(
            rows=4, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.4, 0.2, 0.2, 0.2],
            subplot_titles=("Price", "MACD", "RSI", "ADX")
        )
        
        # === Price Chart (1st row) ===
        # Candlestick chart
        candlestick = go.Candlestick(
            x=chart["Open time"],
            open=chart["Open"],
            high=chart["High"],
            low=chart["Low"],
            close=chart["Close"],
            name="Price"
        )
        fig.add_trace(candlestick, row=1, col=1)
        
        # Add Moving Averages
        for ma_type, color in [("short", "red"), ("medium", "blue"), ("long", "green")]:
            fig.add_trace(go.Scatter(
                x=chart["Open time"],
                y=self.indicators[f"ma_{ma_type}"],
                mode="lines",
                name=f"{self.params['ma_inputs'][ma_type]} EMA",
                line=dict(color=color)
            ), row=1, col=1)
        
        # Add Bollinger Bands
        for bb_type, name, color in [
            ("bb_upper", "Upper BB", "rgba(250, 120, 120, 0.7)"),
            ("bb_middle", "Middle BB", "rgba(120, 120, 250, 0.7)"),
            ("bb_lower", "Lower BB", "rgba(120, 250, 120, 0.7)")
        ]:
            fig.add_trace(go.Scatter(
                x=chart["Open time"],
                y=self.indicators[bb_type],
                mode="lines",
                name=name,
                line=dict(color=color, width=1)
            ), row=1, col=1)
            
        # Plot buy and sell signals
        buy_times, buy_prices = [], []
        sell_times, sell_prices = [], []
        
        for i, cond in enumerate(self.conditions["rsi"]):
            if i >= len(chart):
                break
                
            if all(self.conditions[c][i] == OrderSide.BUY for c in ["rsi", "macd", "ma", "adx"]) and self.conditions["atr"][i]:
                buy_times.append(chart["Open time"].iloc[i])
                buy_prices.append(chart["Close"].iloc[i])
            elif all(self.conditions[c][i] == OrderSide.SELL for c in ["rsi", "macd", "ma", "adx"]) and self.conditions["atr"][i]:
                sell_times.append(chart["Open time"].iloc[i])
                sell_prices.append(chart["Close"].iloc[i])
                
        fig.add_trace(go.Scatter(
            x=buy_times,
            y=buy_prices,
            mode="markers",
            name="Buy Signal",
            marker=dict(symbol="triangle-up", size=12, color="green")
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=sell_times,
            y=sell_prices,
            mode="markers",
            name="Sell Signal",
            marker=dict(symbol="triangle-down", size=12, color="red")
        ), row=1, col=1)
        
        # === MACD Chart (2nd row) ===
        # MACD Line
        fig.add_trace(go.Scatter(
            x=chart["Open time"],
            y=self.indicators["macd"],
            mode="lines",
            name="MACD",
            line=dict(color="blue")
        ), row=2, col=1)
        
        # Signal Line
        fig.add_trace(go.Scatter(
            x=chart["Open time"],
            y=self.indicators["macdsignal"],
            mode="lines",
            name="Signal",
            line=dict(color="red")
        ), row=2, col=1)
        
        # Histogram
        colors = ["green" if val >= 0 else "red" for val in self.indicators["macdhist"]]
        fig.add_trace(go.Bar(
            x=chart["Open time"],
            y=self.indicators["macdhist"],
            name="Histogram",
            marker_color=colors
        ), row=2, col=1)
        
        # Add zero line
        fig.add_trace(go.Scatter(
            x=[chart["Open time"].iloc[0], chart["Open time"].iloc[-1]],
            y=[0, 0],
            mode="lines",
            line=dict(color="gray", width=1, dash="dot"),
            showlegend=False
        ), row=2, col=1)
        
        # === RSI Chart (3rd row) ===
        fig.add_trace(go.Scatter(
            x=chart["Open time"],
            y=self.indicators["rsi"],
            mode="lines",
            name="RSI",
            line=dict(color="purple")
        ), row=3, col=1)
        
        # Add overbought/oversold lines
        for level, color in [(self.ob_rsi, "red"), (self.os_rsi, "green")]:
            fig.add_trace(go.Scatter(
                x=[chart["Open time"].iloc[0], chart["Open time"].iloc[-1]],
                y=[level, level],
                mode="lines",
                line=dict(color=color, width=1, dash="dash"),
                name=f"RSI {level}",
                showlegend=False
            ), row=3, col=1)
        
        # === ADX Chart (4th row) ===
        fig.add_trace(go.Scatter(
            x=chart["Open time"],
            y=self.indicators["adx"],
            mode="lines",
            name="ADX",
            line=dict(color="black")
        ), row=4, col=1)
        
        # DI+ and DI-
        fig.add_trace(go.Scatter(
            x=chart["Open time"],
            y=self.indicators["plus_di"],
            mode="lines",
            name="DI+",
            line=dict(color="green")
        ), row=4, col=1)
        
        fig.add_trace(go.Scatter(
            x=chart["Open time"],
            y=self.indicators["minus_di"],
            mode="lines",
            name="DI-",
            line=dict(color="red")
        ), row=4, col=1)
        
        # Add threshold line
        fig.add_trace(go.Scatter(
            x=[chart["Open time"].iloc[0], chart["Open time"].iloc[-1]],
            y=[self.adx_threshold, self.adx_threshold],
            mode="lines",
            line=dict(color="gray", width=1, dash="dot"),
            name="ADX Threshold",
            showlegend=False
        ), row=4, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"Enhanced RSI-MACD Strategy - {self.trader.symbol}",
            xaxis_rangeslider_visible=False,
            height=800,
            width=1200,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor="white",
            hovermode="x unified"
        )
        
        # Add performance metrics
        metrics_text = (
            f"Win Rate: {self.metrics['win_rate']:.1f}% | "
            f"Total Trades: {self.metrics['total_trades']} | "
            f"Profit: {self.metrics['total_profit']:.2f}"
        )
        
        fig.add_annotation(
            text=metrics_text,
            xref="paper", yref="paper",
            x=0.5, y=1.05,
            showarrow=False,
            font=dict(size=14),
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
            opacity=0.8
        )
        
        return fig
        
    def get_statistics(self):
        """Return strategy performance statistics"""
        if not self.orders_closed:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "max_drawdown": 0,
                "avg_profit": 0,
                "avg_loss": 0,
                "largest_win": 0,
                "largest_loss": 0
            }
            
        # Calculate basic metrics
        total_trades = len(self.orders_closed)
        winning_trades = sum(1 for order in self.orders_closed if order.profit > 0)
        losing_trades = total_trades - winning_trades
        
        win_rate = (winning_trades / total_trades) * 100
        
        # Calculate profit metrics
        total_profit = sum(order.profit for order in self.orders_closed if order.profit > 0)
        total_loss = abs(sum(order.profit for order in self.orders_closed if order.profit < 0))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate average and extreme values
        avg_profit = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
        
        largest_win = max((order.profit for order in self.orders_closed if order.profit > 0), default=0)
        largest_loss = min((order.profit for order in self.orders_closed if order.profit < 0), default=0)
        
        # Calculate drawdown
        equity_curve = []
        current_equity = 0
        
        for order in sorted(self.orders_closed, key=lambda x: x.close_time):
            current_equity += order.profit
            equity_curve.append(current_equity)
        
        max_equity = 0
        max_drawdown = 0
        
        for equity in equity_curve:
            max_equity = max(max_equity, equity)
            drawdown = max_equity - equity
            max_drawdown = max(max_drawdown, drawdown)
            
        # Return statistics
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "total_profit": sum(order.profit for order in self.orders_closed)
        }
        
    def get_params_description(self) -> str:
        """Return string description of strategy parameters"""
        return f"""
        Strategy: Enhanced RSI-MACD
        
        RSI Parameters:
        - Length: {self.rsi_len}
        - Overbought Level: {self.ob_rsi}
        - Oversold Level: {self.os_rsi}
        
        MACD Parameters:
        - Fast Length: {self.params["macd_inputs"]["fast_len"]}
        - Slow Length: {self.params["macd_inputs"]["slow_len"]}
        - Signal Length: {self.params["macd_inputs"]["signal"]}
        
        Moving Average Parameters:
        - Short: {self.params["ma_inputs"]["short"]}
        - Medium: {self.params["ma_inputs"]["medium"]}
        - Long: {self.params["ma_inputs"]["long"]}
        
        ADX Parameters:
        - Length: {self.params["adx_inputs"]["len"]}
        - Threshold: {self.adx_threshold}
        
        Bollinger Band Parameters:
        - Length: {self.params["bb_inputs"]["len"]}
        - Upper Deviation: {self.params["bb_inputs"]["nbdevup"]}
        - Lower Deviation: {self.params["bb_inputs"]["nbdevdn"]}
        
        ATR Parameters:
        - Length: {self.params["atr_inputs"]["len"]}
        - ATR Multiplier: {self.atr_multiplier}
        
        Volume Parameters:
        - MA Period: {self.params["ma_vol"]}
        - Minimum Volume Ratio: {self.min_volume_ratio}
        """
    
