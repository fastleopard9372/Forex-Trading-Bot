import logging
import pandas as pd
import numpy as np
from datetime import datetime
import talib as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .base_strategy import BaseStrategy
import indicators as mta
from order import Order, OrderType, OrderSide, OrderStatus
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

bot_logger = logging.getLogger("bot_logger")
IN_BUYING = 1
IN_SELLING = 2

class MarketRegime:
    """Market regime classification constants"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"

class MACross(BaseStrategy):

    def __init__(self, name, params, tfs):
        super().__init__(name, params, tfs)
        self.tf = self.tfs["tf"]
        self.name = "Enhanced Multi Strategy"
        self.description = "Multi Strategy"
        self.state = None
        self.trend = None
        self.regime = MarketRegime.UNKNOWN
        self.last_regime = MarketRegime.UNKNOWN
        
        # Set up signal weights
        self.signal_weights = self.params.get("signal_weights", {
            "price": 1.0,
            "ma": 1.5,
            "rsi": 1.0, 
            "adx": 1.0,
            "macd": 1.5,
            "volume": 0.5,
            "bbands": 0.7
        })
        
        # Set up MA type based on parameters
        self.ma_func = ta.SMA if self.params["ma_inputs"]["type"] == "SMA" else ta.EMA
        self.ma_stream_func = ta.stream.SMA if self.params["ma_inputs"]["type"] == "SMA" else ta.stream.EMA
            
        # Position sizing settings
        self.max_position_size = self.params.get("max_position_size", 1.0)
        self.min_position_size = self.params.get("min_position_size", 0.1)
        
        # Multiple timeframe settings
        self.use_mtf = self.params.get("use_mtf", True)
        self.mtf_weights = self.params.get("mtf_weights", {
            "1m": 0.2,
            "5m": 0.3,
            "15m": 0.5,
            "1h": 1.0,
            "4h": 1.2
        })
        
        # Initialize signal history
        self.signal_history = []
        self.mtf_indicators = {}

        self.use_ml = self.params.get("use_ml", False)
        if self.use_ml:
            self.feature_history = []
            self.performance_history = []
            self.ml_predictions = []
            self.ml_signal_probability = 0.5  # Default neutral
            self.ml_signal_strength = 0.0     # Default neutral
            self.ml_model = None
            self.ml_scaler = None
            self.last_train_trade_count = 0
            
            # Initialize dependencies needed for ML
            try:
                from sklearn.preprocessing import StandardScaler
                from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                bot_logger.info("ML dependencies loaded successfully")
            except ImportError:
                bot_logger.error("Missing scikit-learn dependency for ML features. Install with: pip install scikit-learn")
                self.use_ml = False

    def attach(self, tfs_chart):
        self.tfs_chart = tfs_chart
        self.init_indicators()
        if self.use_mtf:
            self.init_mtf_indicators()

    def init_indicators(self):
        """Initialize all technical indicators for the main timeframe"""
        chart = self.tfs_chart[self.tf]
        
        # Moving Averages
        self.fast_ma = self.ma_func(chart["Close"], self.params["ma_inputs"]["fast_ma"])
        self.slow_ma = self.ma_func(chart["Close"], self.params["ma_inputs"]["slow_ma"])
        self.trend_ma = self.ma_func(chart["Close"], self.params.get("trend_ma_period", 100))
        
        # Volatility indicators
        self.atr = ta.ATR(
            chart["High"], 
            chart["Low"], 
            chart["Close"], 
            timeperiod=self.params["atr_inputs"]["atr_period"]
        )
        self.atr_pct = self.atr / chart["Close"] * 100
        
        # Bollinger Bands
        self.upper_band, self.middle_band, self.lower_band = ta.BBANDS(
            chart["Close"], 
            timeperiod=self.params.get("bbands_inputs", {}).get("bbands_period", 20),
            nbdevup=self.params.get("bbands_inputs", {}).get("bbands_dev_up", 2),
            nbdevdn=self.params.get("bbands_inputs", {}).get("bbands_dev_down", 2),
            matype=0
        )
        
        # Trend indicators
        self.adx = ta.ADX(
            chart["High"], 
            chart["Low"], 
            chart["Close"], 
            timeperiod=self.params["adx_inputs"]["adx_period"]
        )
        self.plus_di = ta.PLUS_DI(
            chart["High"], 
            chart["Low"], 
            chart["Close"], 
            timeperiod=self.params["adx_inputs"]["adx_period"]
        )
        self.minus_di = ta.MINUS_DI(
            chart["High"], 
            chart["Low"], 
            chart["Close"], 
            timeperiod=self.params["adx_inputs"]["adx_period"]
        )
        
        # Momentum indicators
        self.rsi = ta.RSI(
            chart["Close"], 
            timeperiod=self.params["rsi_inputs"]["rsi_period"]
        )
        self.macd, self.macd_signal, self.macd_hist = ta.MACD(
            chart["Close"], 
            fastperiod=self.params["macd_inputs"]["fast"], 
            slowperiod=self.params["macd_inputs"]["slow"], 
            signalperiod=self.params["macd_inputs"]["signal"]
        )
        
        # Volume indicators
        self.obv = ta.OBV(chart["Close"], chart["Tick volume"])
        self.volume_ma = ta.SMA(chart["Tick volume"], timeperiod=20)
        
        # Initialize market regime detection
        self.detect_market_regime()
        
        self.start_trading_time = chart.iloc[-1]["Open time"]

    def init_mtf_indicators(self):
        """Initialize indicators for multiple timeframes"""
        self.mtf_indicators = {}
        for tf in self.tfs:
            if tf == "tf":  # Skip the main timeframe key
                continue
                
            chart = self.tfs_chart[tf]
            self.mtf_indicators[tf] = {
                "fast_ma": self.ma_func(chart["Close"], self.params["ma_inputs"]["fast_ma"]),
                "slow_ma": self.ma_func(chart["Close"], self.params["ma_inputs"]["slow_ma"]),
                "rsi": ta.RSI(chart["Close"], timeperiod=self.params["rsi_inputs"]["rsi_period"]),
                "adx": ta.ADX(
                    chart["High"], 
                    chart["Low"], 
                    chart["Close"], 
                    timeperiod=self.params["adx_inputs"]["adx_period"]
                )
            }

    def update_indicators(self, tf):
        if tf != self.tf:
            if self.use_mtf and tf in self.mtf_indicators:
                self.update_mtf_indicators(tf)
            return
            
        last_kline = self.tfs_chart[self.tf].iloc[-1]
        chart = self.tfs_chart[self.tf]
        
        # Update Moving Averages
        self.fast_ma.loc[len(self.fast_ma)] = self.ma_stream_func(
            chart["Close"], self.params["ma_inputs"]["fast_ma"]
        )
        self.slow_ma.loc[len(self.slow_ma)] = self.ma_stream_func(
            chart["Close"], self.params["ma_inputs"]["slow_ma"]
        )
        
        # Update trend MA
        self.trend_ma.loc[len(self.trend_ma)] = self.ma_stream_func(
            chart["Close"], self.params.get("trend_ma_period", 100)
        )
        
        # Update volatility indicators
        self.atr.loc[len(self.atr)] = ta.stream.ATR(
            chart["High"], chart["Low"], chart["Close"], 
            timeperiod=self.params["atr_inputs"]["atr_period"]
        )
        self.atr_pct.loc[len(self.atr_pct)] = self.atr.iloc[-1] / last_kline["Close"] * 100
        
        # Update Bollinger Bands
        self.upper_band.loc[len(self.upper_band)], self.middle_band.loc[len(self.middle_band)], self.lower_band.loc[len(self.lower_band)] = ta.stream.BBANDS(
            chart["Close"], 
            timeperiod=self.params.get("bbands_inputs", {}).get("bbands_period", 20),
            nbdevup=self.params.get("bbands_inputs", {}).get("bbands_dev_up", 2),
            nbdevdn=self.params.get("bbands_inputs", {}).get("bbands_dev_down", 2),
            matype=0
        )
        
        # Update trend indicators
        self.adx.loc[len(self.adx)] = ta.stream.ADX(
            chart["High"], chart["Low"], chart["Close"], 
            timeperiod=self.params["adx_inputs"]["adx_period"]
        )
        self.plus_di.loc[len(self.plus_di)] = ta.stream.PLUS_DI(
            chart["High"], chart["Low"], chart["Close"], 
            timeperiod=self.params["adx_inputs"]["adx_period"]
        )
        self.minus_di.loc[len(self.minus_di)] = ta.stream.MINUS_DI(
            chart["High"], chart["Low"], chart["Close"], 
            timeperiod=self.params["adx_inputs"]["adx_period"]
        )
        
        # Update momentum indicators
        self.rsi.loc[len(self.rsi)] = ta.stream.RSI(
            chart["Close"], 
            timeperiod=self.params["rsi_inputs"]["rsi_period"]
        )
        self.macd.loc[len(self.macd)], self.macd_signal.loc[len(self.macd_signal)], self.macd_hist.loc[len(self.macd_hist)] = ta.stream.MACD(
            chart["Close"], 
            fastperiod=self.params["macd_inputs"]["fast"], 
            slowperiod=self.params["macd_inputs"]["slow"], 
            signalperiod=self.params["macd_inputs"]["signal"]
        )
        
        # Update volume indicators
        self.obv.loc[len(self.obv)] = ta.stream.OBV(chart["Close"], chart["Tick volume"])
        self.volume_ma.loc[len(self.volume_ma)] = ta.stream.SMA(chart["Tick volume"], timeperiod=20)
        
        # Update market regime
        self.detect_market_regime()

    def update_mtf_indicators(self, tf):
        """
        Update indicators for multiple timeframes.
        
        Args:
            tf (str): Timeframe to update
        """
        if tf not in self.mtf_indicators:
            return
            
        chart = self.tfs_chart[tf]
        
        self.mtf_indicators[tf]["fast_ma"].loc[len(self.mtf_indicators[tf]["fast_ma"])] = self.ma_stream_func(
            chart["Close"], self.params["ma_inputs"]["fast_ma"]
        )
        self.mtf_indicators[tf]["slow_ma"].loc[len(self.mtf_indicators[tf]["slow_ma"])] = self.ma_stream_func(
            chart["Close"], self.params["ma_inputs"]["slow_ma"]
        )
        self.mtf_indicators[tf]["rsi"].loc[len(self.mtf_indicators[tf]["rsi"])] = ta.stream.RSI(
            chart["Close"], 
            timeperiod=self.params["rsi_inputs"]["rsi_period"]
        )
        self.mtf_indicators[tf]["adx"].loc[len(self.mtf_indicators[tf]["adx"])] = ta.stream.ADX(
            chart["High"], chart["Low"], chart["Close"], 
            timeperiod=self.params["adx_inputs"]["adx_period"]
        )

    def check_required_params(self):
        """Check if all required parameters are present"""
        return all([key in self.params.keys() for key in [
            "ma_inputs", "atr_inputs", "adx_inputs", "rsi_inputs", "macd_inputs"
        ]])

    def is_params_valid(self):
        """Check if parameters are valid"""
        if not self.check_required_params():
            bot_logger.info("   [-] Missing required params")
            return False
        
        return (self.params["ma_inputs"]["fast_ma"] < self.params["ma_inputs"]["slow_ma"] and 
                self.params["ma_inputs"]["type"] in ["SMA", "EMA"])

    def detect_market_regime(self):
        """Detect current market regime (trending, ranging, volatile)"""
        if len(self.atr_pct) < 20 or len(self.adx) < 20:
            self.regime = MarketRegime.UNKNOWN
            return
            
        # Get the latest values
        last_close = self.tfs_chart[self.tf]["Close"].iloc[-1]
        last_atr_pct = self.atr_pct.iloc[-1]
        last_adx = self.adx.iloc[-1]
        
        # Check if we're in a strong trend
        adx_threshold = self.params["regime_detection"]["adx_trend_threshold"]
        volatility_threshold = self.params["regime_detection"]["volatility_threshold"]
        
        if last_adx > adx_threshold:
            # We're in a trend, determine direction
            if self.plus_di.iloc[-1] > self.minus_di.iloc[-1]:
                self.regime = MarketRegime.TRENDING_UP
            else:
                self.regime = MarketRegime.TRENDING_DOWN
        elif last_atr_pct > volatility_threshold:
            self.regime = MarketRegime.VOLATILE
        else:
            self.regime = MarketRegime.RANGING
            
        # Log regime change
        if hasattr(self, 'last_regime') and self.last_regime != self.regime:
            bot_logger.info(f"Market regime changed from {self.last_regime} to {self.regime}")
            
        self.last_regime = self.regime

    def adapt_parameters(self):
        """Adapt strategy parameters based on current market regime"""
        if self.regime == MarketRegime.UNKNOWN:
            return
            
        # Adjust RSI thresholds based on trend
        if self.regime == MarketRegime.TRENDING_UP:
            # In uptrends, RSI can stay higher for longer
            self.params["rsi_inputs"]["ob_rsi"] = 75
            self.params["rsi_inputs"]["os_rsi"] = 35
            # Increase the weight of trend following indicators
            self.signal_weights["ma"] = 2.0
            self.signal_weights["adx"] = 1.5
        elif self.regime == MarketRegime.TRENDING_DOWN:
            # In downtrends, RSI can stay lower for longer
            self.params["rsi_inputs"]["ob_rsi"] = 65
            self.params["rsi_inputs"]["os_rsi"] = 25
            # Increase the weight of trend following indicators
            self.signal_weights["ma"] = 2.0
            self.signal_weights["adx"] = 1.5
        elif self.regime == MarketRegime.RANGING:
            # In ranging markets, use normal RSI thresholds
            self.params["rsi_inputs"]["ob_rsi"] = 65
            self.params["rsi_inputs"]["os_rsi"] = 35
            # Increase the weight of oscillators
            self.signal_weights["rsi"] = 2
            self.signal_weights["bbands"] = 1.0
            self.signal_weights["ma"] = 0.8
        elif self.regime == MarketRegime.VOLATILE:
            # In volatile markets, widen the thresholds
            self.params["rsi_inputs"]["ob_rsi"] = 75
            self.params["rsi_inputs"]["os_rsi"] = 25
            # Adjust ATR threshold to filter out noise

    def update(self, tf):
        
        super().update(tf)
        # Adapt parameters to current market conditions
        self.adapt_parameters()
        
        # Check for exit signals first
        self.check_close_signal()
        
        # Then check for entry signals
        self.check_signal()
        
        # Update machine learning model if enabled
        if self.use_ml and tf == self.tf:
            self.update_ml_model()

    def close_opening_orders(self):
        """Close all opening orders"""
        super().close_opening_orders(self.tfs_chart[self.tf].iloc[-1])

    def calculate_signal_score(self, indicators):
        score = 0
        p_score = 0
        m_score = 0
        max_score = 0
        
        print(self.signal_weights)

        # Price action signal
        if indicators["price_cnd"] == "Buy":
            p_score += self.signal_weights["price"]
        elif indicators["price_cnd"] == "Sell":
            m_score -= self.signal_weights["price"]
        max_score += abs(self.signal_weights["price"])
        
        # Moving average signal
        if indicators["ma_cnd"] == "Buy":
            p_score += self.signal_weights["ma"]
        elif indicators["ma_cnd"] == "Sell":
            m_score -= self.signal_weights["ma"]
        max_score += abs(self.signal_weights["ma"])
        
        # RSI signal
        if indicators["rsi_cnd"] == "Buy":
            p_score += self.signal_weights["rsi"]
        elif indicators["rsi_cnd"] == "Sell":
            m_score -= self.signal_weights["rsi"]
        max_score += abs(self.signal_weights["rsi"])
        
        # ADX signal
        if indicators["adx_cnd"] == "Buy":
            p_score += self.signal_weights["adx"]
        elif indicators["adx_cnd"] == "Sell":
            m_score -= self.signal_weights["adx"]
        max_score += abs(self.signal_weights["adx"])
        
        # MACD signal
        if indicators["macd_cnd"] == "Buy":
            p_score += self.signal_weights["macd"]
        elif indicators["macd_cnd"] == "Sell":
            m_score -= self.signal_weights["macd"]
        max_score += abs(self.signal_weights["macd"])
        
        # Volume signal
        if indicators["volume_cnd"] == "Buy":
            p_score += self.signal_weights["volume"]
        elif indicators["volume_cnd"] == "Sell":
            m_score -= self.signal_weights["volume"]
        max_score += abs(self.signal_weights["volume"])
        
        # Bollinger Bands signal
        if indicators["bbands_cnd"] == "Buy":
            p_score += self.signal_weights["bbands"]
        elif indicators["bbands_cnd"] == "Sell":
            m_score -= self.signal_weights["bbands"]
        max_score += abs(self.signal_weights["bbands"])
        

        # if self.use_ml and "ml" in self.signal_weights:
            # if indicators["ml_cnd"] == "Buy":
            #     p_score += self.signal_weights["ml"]
            # elif indicators["ml_cnd"] == "Sell":
            #     m_score -= self.signal_weights["ml"]
            # max_score += abs(self.signal_weights["ml"])


        # ATR condition (volatility filter)
        if not indicators["atr_cnd"]:
            return 0  # No signal if volatility is too low
            
        # Multiple timeframe analysis
        if self.use_mtf:
            mtf_score = self.calculate_mtf_score()
            # score += mtf_score
        
        if(p_score > -m_score):
            score = p_score
        else:
            score = m_score
        # Normalize score to be between -1 and 1
        print(f"score: {score}, max_score: {max_score}")
        normalized_score = score / max_score if max_score > 0 else 0
        
        return normalized_score

    def calculate_mtf_score(self):
        """
        Calculate score from multiple timeframe analysis.
        
        Returns:
            float: MTF score contribution
        """
        mtf_score = 0
        
        for tf, indicators in self.mtf_indicators.items():
            weight = self.mtf_weights.get(tf, 1.0)
            
            # Check if fast MA is above or below slow MA
            if indicators["fast_ma"].iloc[-1] > indicators["slow_ma"].iloc[-1]:
                mtf_score += weight
            else:
                mtf_score -= weight
                
            # Check RSI trend
            if indicators["rsi"].iloc[-1] > 50:
                mtf_score += 0.5 * weight
            else:
                mtf_score -= 0.5 * weight
                
            # Check ADX strength
            if indicators["adx"].iloc[-1] > 25:
                mtf_score += 0.3 * weight
                
        return mtf_score

    def calculate_position_size(self, signal_strength):
        # Signal strength should be between -1 and 1
        abs_strength = abs(signal_strength)
        
        # Scale position size between min and max
        position_size = self.min_position_size + (
            abs_strength * (self.max_position_size - self.min_position_size)
        )
        
        # Ensure it's within bounds
        position_size = max(min(position_size, self.max_position_size), self.min_position_size)
        
        return position_size

    def update_ml_model(self):
        if not self.use_ml:
            return
            
        # Extract features from current market state
        features = self._extract_features()
        
        # Add to historical feature data
        self.feature_history.append(features)
        
        # Keep only the last N records to manage memory
        max_history = self.params.get("ml_inputs", {}).get("max_history", 1000)
        if len(self.feature_history) > max_history:
            self.feature_history = self.feature_history[-max_history:]
        
        # Train model periodically or after new trades close
        should_train = False
        
        # Get number of trades closed since last training
        trades_since_train = 0
        if hasattr(self, 'last_train_trade_count'):
            trades_since_train = len(self.orders_closed) - self.last_train_trade_count
        else:
            self.last_train_trade_count = 0
        
        # Training schedule options
        min_trades_to_train = self.params.get("ml_inputs", {}).get("min_trades_to_train", 20)
        train_frequency = self.params.get("ml_inputs", {}).get("train_frequency", 10)
        
        # Decide if we should train
        if len(self.orders_closed) >= min_trades_to_train and trades_since_train >= train_frequency:
            should_train = True
            self.last_train_trade_count = len(self.orders_closed)
        
        # Perform training if conditions are met
        if should_train:
            self._train_model()
        
        # Generate prediction from current features
        if hasattr(self, 'ml_model') and self.ml_model is not None:
            self._predict_signal(features)

    def _extract_features(self):
        chart = self.tfs_chart[self.tf]
        last_kline = chart.iloc[-1]
        
        # Price-based features
        price_features = {
            "close": last_kline["Close"],
            "open": last_kline["Open"],
            "high": last_kline["High"],
            "low": last_kline["Low"],
            "volume": last_kline["Tick volume"],
            
            # Price changes (returns)
            "close_pct_change": (last_kline["Close"] - chart["Close"].iloc[-2]) / chart["Close"].iloc[-2] if len(chart) > 1 else 0,
            "close_open_ratio": last_kline["Close"] / last_kline["Open"] if last_kline["Open"] > 0 else 1.0,
            "high_low_ratio": last_kline["High"] / last_kline["Low"] if last_kline["Low"] > 0 else 1.0,
            
            # Calculate range metrics
            "candle_range": (last_kline["High"] - last_kline["Low"]) / last_kline["Close"] * 100,
            "candle_body": abs(last_kline["Close"] - last_kline["Open"]) / last_kline["Close"] * 100,
            "upper_wick": (last_kline["High"] - max(last_kline["Open"], last_kline["Close"])) / last_kline["Close"] * 100,
            "lower_wick": (min(last_kline["Open"], last_kline["Close"]) - last_kline["Low"]) / last_kline["Close"] * 100,
        }
        
        # Indicator values
        indicator_features = {
            # Moving averages
            "fast_ma": self.fast_ma.iloc[-1],
            "slow_ma": self.slow_ma.iloc[-1],
            "fast_ma_slope": self.fast_ma.iloc[-1] - self.fast_ma.iloc[-2] if len(self.fast_ma) > 1 else 0,
            "slow_ma_slope": self.slow_ma.iloc[-1] - self.slow_ma.iloc[-2] if len(self.slow_ma) > 1 else 0,
            "ma_cross_distance": (self.fast_ma.iloc[-1] - self.slow_ma.iloc[-1]) / last_kline["Close"] * 100,
            
            # Oscillators
            "rsi": self.rsi.iloc[-1],
            "rsi_slope": self.rsi.iloc[-1] - self.rsi.iloc[-2] if len(self.rsi) > 1 else 0,
            "rsi_ob_distance": self.rsi.iloc[-1] - self.params["rsi_inputs"]["ob_rsi"],
            "rsi_os_distance": self.params["rsi_inputs"]["os_rsi"] - self.rsi.iloc[-1],
            
            # MACD
            "macd": self.macd.iloc[-1],
            "macd_signal": self.macd_signal.iloc[-1],
            "macd_hist": self.macd_hist.iloc[-1],
            "macd_hist_slope": self.macd_hist.iloc[-1] - self.macd_hist.iloc[-2] if len(self.macd_hist) > 1 else 0,
            
            # Trend indicators
            "adx": self.adx.iloc[-1],
            "plus_di": self.plus_di.iloc[-1],
            "minus_di": self.minus_di.iloc[-1],
            "di_diff": self.plus_di.iloc[-1] - self.minus_di.iloc[-1],
            
            # Volatility
            "atr": self.atr.iloc[-1],
            "atr_pct": self.atr_pct.iloc[-1],
            "bb_width": (self.upper_band.iloc[-1] - self.lower_band.iloc[-1]) / self.middle_band.iloc[-1] * 100,
            "bb_position": (last_kline["Close"] - self.lower_band.iloc[-1]) / (self.upper_band.iloc[-1] - self.lower_band.iloc[-1]) if (self.upper_band.iloc[-1] - self.lower_band.iloc[-1]) > 0 else 0.5,
            
            # Volume
            "volume_ma_ratio": last_kline["Tick volume"] / self.volume_ma.iloc[-1] if len(self.volume_ma) > 0 and self.volume_ma.iloc[-1] > 0 else 1.0,
            "obv_slope": self.obv.iloc[-1] - self.obv.iloc[-2] if len(self.obv) > 1 else 0,
        }
        
        # Market regime features
        regime_features = {
            "is_trending_up": 1 if self.regime == MarketRegime.TRENDING_UP else 0,
            "is_trending_down": 1 if self.regime == MarketRegime.TRENDING_DOWN else 0,
            "is_ranging": 1 if self.regime == MarketRegime.RANGING else 0,
            "is_volatile": 1 if self.regime == MarketRegime.VOLATILE else 0,
        }
        
        # Multiple timeframe features
        mtf_features = {}
        if self.use_mtf:
            for tf, indicators in self.mtf_indicators.items():
                mtf_features.update({
                    f"{tf}_fast_ma": indicators["fast_ma"].iloc[-1],
                    f"{tf}_slow_ma": indicators["slow_ma"].iloc[-1],
                    f"{tf}_ma_cross": 1 if indicators["fast_ma"].iloc[-1] > indicators["slow_ma"].iloc[-1] else -1,
                    f"{tf}_rsi": indicators["rsi"].iloc[-1],
                    f"{tf}_adx": indicators["adx"].iloc[-1],
                })
        
        # Time-based features 
        dt = last_kline["Open time"]
        time_features = {
            "hour": dt.hour,
            "day_of_week": dt.weekday(),
            "day_of_month": dt.day,
            "week_of_year": dt.isocalendar()[1],
            "month": dt.month,
            "is_weekend": 1 if dt.weekday() >= 5 else 0,
        }
        
        # Merge all feature categories
        features = {**price_features, **indicator_features, **regime_features, **time_features}
        if mtf_features:
            features.update(mtf_features)
        
        return features

    def _train_model(self):
        bot_logger.info("Training machine learning model...")
        
        # First, gather training data from closed trades
        X = []  # Features at trade entry
        y = []  # Labels (profitable or not)
        
        # Find closest feature record for each trade entry
        for order in self.orders_closed:
            entry_time = order["FILL_TIME"]
            profit = order["profit"]
            
            # Find closest feature record based on time
            closest_features = None
            min_time_diff = float('inf')
            
            for feature_record in self.feature_history:
                if hasattr(feature_record, 'time'):
                    time_diff = abs((entry_time - feature_record['time']).total_seconds())
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_features = feature_record
            
            # Only use if we found matching features within a reasonable time window (e.g., 1 hour)
            if closest_features is not None and min_time_diff < 3600:  # 1 hour in seconds
                # Clean features (remove non-numeric values)
                numeric_features = {k: v for k, v in closest_features.items() if isinstance(v, (int, float)) and not k.startswith('time')}
                feature_values = list(numeric_features.values())
                
                X.append(feature_values)
                # 1 for profitable trades, 0 for losing trades
                y.append(1 if profit > 0 else 0)
        
        # Need enough samples to train
        if len(X) < 20:  # Minimum sample threshold
            bot_logger.info(f"Not enough trade samples for ML training ({len(X)} available, need at least 20)")
            return
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model_type = self.params.get("ml_inputs", {}).get("model_type", "random_forest")
        
        if model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=10,
                random_state=42
            )
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
        else:
            bot_logger.error(f"Unknown model type: {model_type}")
            return
        
        # Fit the model
        model.fit(X_scaled, y)
        
        # Save model and scaler for future predictions
        self.ml_model = model
        self.ml_scaler = scaler
        
        # Report feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            top_features = sorted_idx[:10]  # Top 10 features
            
            bot_logger.info("Top 10 important features:")
            for idx in top_features:
                bot_logger.info(f"  {list(numeric_features.keys())[idx]}: {importances[idx]:.4f}")
        
        # Report model accuracy
        train_accuracy = model.score(X_scaled, y)
        bot_logger.info(f"ML model trained with {len(X)} samples, accuracy: {train_accuracy:.2f}")
        
        # Save training timestamp
        self.last_train_time = self.tfs_chart[self.tf].iloc[-1]["Open time"]

    def _predict_signal(self, features):
        if not hasattr(self, 'ml_model') or self.ml_model is None:
            return
        # Extract numeric features in the same order as training
        numeric_features = {k: v for k, v in features.items() if isinstance(v, (int, float)) and not k.startswith('time')}
        feature_values = list(numeric_features.values())
        X = np.array([feature_values])
        
        # Scale features
        X_scaled = self.ml_scaler.transform(X)
        
        # Get prediction probability
        proba = self.ml_model.predict_proba(X_scaled)[0]
        
        # proba[1] is probability of profitable trade
        self.ml_signal_probability = proba[1]
        
        # Record prediction
        self.ml_predictions.append({
            'time': self.tfs_chart[self.tf].iloc[-1]["Open time"],
            'probability': self.ml_signal_probability,
            'features': features.copy()
        })
        
        # Keep only the last N predictions
        max_predictions = self.params.get("ml_inputs", {}).get("max_predictions", 100)
        if len(self.ml_predictions) > max_predictions:
            self.ml_predictions = self.ml_predictions[-max_predictions:]
        
        # Log prediction
        threshold = self.params.get("ml_inputs", {}).get("signal_threshold", 0.7)
        if self.ml_signal_probability > threshold:
            bot_logger.info(f"ML model predicts profitable trade with {self.ml_signal_probability:.2f} probability")
        
        # Store ML signal strength for use in trading decisions
        self.ml_signal_strength = (self.ml_signal_probability - 0.5) * 2  # Scale to [-1, 1]

    def check_signal(self):
        """Check for trading signals"""
        chart = self.tfs_chart[self.tf]
        last_kline = chart.iloc[-1]
        
        # Minimum volume filter
        if last_kline["Tick volume"] < 50:
            return
        
        # Initialize signal indicators
        indicators = {
            "price_cnd": "None",
            "ma_cnd": "None",
            "rsi_cnd": "None",
            "adx_cnd": "None",
            "macd_cnd": "None",
            "volume_cnd": "None",
            "bbands_cnd": "None",
            "ml_cnd": "None",
            "atr_cnd": False
        }
        
        # Price signal
        if last_kline["Close"] > last_kline["Open"] and chart["Close"].iloc[-1] > chart["Close"].iloc[-2]:
            indicators["price_cnd"] = "Buy"
        elif last_kline["Close"] < last_kline["Open"] and chart["Close"].iloc[-1] < chart["Close"].iloc[-2]:
            indicators["price_cnd"] = "Sell"
        
        # Moving Average signal
        if self.fast_ma.iloc[-1] > self.slow_ma.iloc[-1] and last_kline["Close"] >= self.slow_ma.iloc[-1]:
            if self.fast_ma.iloc[-1] > self.fast_ma.iloc[-2]:
                indicators["ma_cnd"] = "Buy"
        elif self.fast_ma.iloc[-1] < self.slow_ma.iloc[-1] and last_kline["Close"] <= self.slow_ma.iloc[-1]:
            if self.fast_ma.iloc[-1] < self.fast_ma.iloc[-2]:
                indicators["ma_cnd"] = "Sell"
        
        # RSI signal - adaptive thresholds based on market regime
        if self.rsi.iloc[-1] > self.params["rsi_inputs"]["os_rsi"] and self.rsi.iloc[-1] > self.rsi.iloc[-2]:
            indicators["rsi_cnd"] = "Buy"
        elif self.rsi.iloc[-1] < self.params["rsi_inputs"]["ob_rsi"] and self.rsi.iloc[-1] < self.rsi.iloc[-2]:
            indicators["rsi_cnd"] = "Sell"
        
        # ADX signal
        if self.adx.iloc[-1] > self.params["adx_inputs"]["adx_threshold"] and self.adx.iloc[-1] > self.adx.iloc[-2]:
            if self.plus_di.iloc[-1] > self.minus_di.iloc[-1] * 1.5:
                indicators["adx_cnd"] = "Buy"
            elif self.plus_di.iloc[-1] < self.minus_di.iloc[-1] * 1.5:
                indicators["adx_cnd"] = "Sell"
        
        # MACD signal
        if self.macd.iloc[-1] > self.macd_signal.iloc[-1] and self.macd.iloc[-1] > self.macd.iloc[-2]:
            indicators["macd_cnd"] = "Buy"
        elif self.macd.iloc[-1] < self.macd_signal.iloc[-1] and self.macd.iloc[-1] < self.macd.iloc[-2]:
            indicators["macd_cnd"] = "Sell"
        
        # Volume signal
        if last_kline["Tick volume"] > self.volume_ma.iloc[-1] * 1.3:
            if last_kline["Close"] > last_kline["Open"]:
                indicators["volume_cnd"] = "Buy"
            elif last_kline["Close"] < last_kline["Open"]:
                indicators["volume_cnd"] = "Sell"
        
        # Bollinger Bands signal
        if last_kline["Close"] < self.lower_band.iloc[-1] and self.rsi.iloc[-1] < self.params["rsi_inputs"]["os_rsi"]:
            indicators["bbands_cnd"] = "Buy"
        elif last_kline["Close"] > self.upper_band.iloc[-1] and self.rsi.iloc[-1] > self.params["rsi_inputs"]["ob_rsi"]:
            indicators["bbands_cnd"] = "Sell"
        
        # Volatility filter
        print(self.atr_pct.iloc[-1] , self.params["atr_inputs"]["atr_threshold"])
        if self.atr_pct.iloc[-1] > self.params["atr_inputs"]["atr_threshold"]:
            indicators["atr_cnd"] = True
        
        # Log indicators
        print(f"price_cnd: {indicators['price_cnd']}, ma_cnd: {indicators['ma_cnd']}, " + 
              f"rsi_cnd: {indicators['rsi_cnd']}, adx_cnd: {indicators['adx_cnd']}, " + 
              f"macd_cnd: {indicators['macd_cnd']}, volume_cnd: {indicators['volume_cnd']}, " +
              f"bbands_cnd: {indicators['bbands_cnd']}, atr_cnd: {indicators['atr_cnd']}")
        
        # Add ML prediction if enabled
        if self.use_ml and hasattr(self, 'ml_signal_strength'):
            ml_threshold = self.params.get("ml_inputs", {}).get("signal_threshold", 0.7)
            if self.ml_signal_probability > ml_threshold:
                indicators["ml_cnd"] = "Buy"
            elif self.ml_signal_probability < (1 - ml_threshold):
                indicators["ml_cnd"] = "Sell"
        
        # Update signal weights with ML
        if self.use_ml:
            self.signal_weights["ml"] = self.params.get("ml_inputs", {}).get("ml_weight", 1.5)
        
        # Calculate signal score
        signal_score = self.calculate_signal_score(indicators)
        # Log signal score
        print(f"Signal score: {signal_score:.4f}, Market regime: {self.regime}")
        
        # Store signal history for analysis
        self.signal_history.append({
            "time": last_kline["Open time"],
            "price": last_kline["Close"],
            "indicators": indicators.copy(),
            "score": signal_score,
            "regime": self.regime
        })
        
        # Keep only the last 300 signals
        if len(self.signal_history) > 300:
            self.signal_history.pop(0)
        
        # Execute trades based on signal score
        buy_threshold = self.params.get("signal_thresholds", {}).get("buy", 0.6)
        sell_threshold = self.params.get("signal_thresholds", {}).get("sell", -0.6)
        
        if signal_score > buy_threshold:
            self._execute_buy_signal(last_kline, signal_score)
                
        elif signal_score < sell_threshold:
            self._execute_sell_signal(last_kline, signal_score)

    def _execute_buy_signal(self, last_kline, signal_score):
        # Calculate position size based on signal strength
        position_size = self.calculate_position_size(signal_score)
        
        # Calculate adaptive stop loss based on ATR
        atr_multiplier = self.params.get("sl_atr_multiplier", 2.0)
        sl = last_kline["Close"] - (self.atr.iloc[-1] * atr_multiplier)
        
        # Calculate take profit (using risk:reward ratio)
        risk_reward_ratio = self.params.get("risk_reward_ratio", 2)
        risk = last_kline["Close"] - sl
        tp = last_kline["Close"] + (risk * risk_reward_ratio)
        
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
        order["description"] = f"{self.description}-BUY"
        order["regime"] = self.regime
        order["sl"] = sl
        order["tp"] = tp
        order["Close"] = last_kline["Close"]
        order["signal_score"] = signal_score
        
        order = self.trader.fix_order(order, self.params["sl_fix_mode"], self.max_sl_pct)
        if order:
            # Use position sizing based on signal strength
            self.trader.create_trade(order, self.volume)
            self.orders_opening.append(order)
            bot_logger.info(f"BUY order created: price={last_kline['Close']}, " + 
                         f"position_size={position_size:.2f}, signal_score={signal_score:.2f}")

    def _execute_sell_signal(self, last_kline, signal_score):
        
        # Calculate position size based on signal strength
        position_size = self.calculate_position_size(abs(signal_score))
        
        # Calculate adaptive stop loss based on ATR
        atr_multiplier = self.params.get("sl_atr_multiplier", 2.0)
        sl = last_kline["Close"] + (self.atr.iloc[-1] * atr_multiplier)
        
        # Calculate take profit (using risk:reward ratio)
        risk_reward_ratio = self.params.get("risk_reward_ratio", 2)
        risk = sl - last_kline["Close"]
        tp = last_kline["Close"] - (risk * risk_reward_ratio)
        
        order = Order(
            OrderType.MARKET,
            OrderSide.SELL,
            last_kline["Close"],    
            tp=tp,  
            sl=sl,
            status=OrderStatus.FILLED,
        )
        order["FILL_TIME"] = last_kline["Open time"]
        order["Close"] = last_kline["Close"]
        order["sl"] = sl
        order["tp"] = tp
        order["strategy"] = self.name
        order["description"] = f"{self.description}-SELL"
        order["regime"] = self.regime
        order["signal_score"] = signal_score

        order = self.trader.fix_order(order, self.params["sl_fix_mode"], self.max_sl_pct)
        if order:
            # Use position sizing based on signal strength
            self.trader.create_trade(order, self.volume)
            self.orders_opening.append(order)
            bot_logger.info(f"SELL order created: price={last_kline['Close']}, " + 
                         f"position_size={position_size:.2f}, signal_score={signal_score:.2f}")
                         
    def check_close_signal(self):
        """Check for trade exit signals with trailing stop implementation"""
        chart = self.tfs_chart[self.tf]
        last_kline = chart.iloc[-1]
        
        # Nothing to do if no open orders
        if not self.orders_opening:
            return
            
        # Process each open order
        for i in range(len(self.orders_opening) - 1, -1, -1):
            order = self.orders_opening[i]
            
            # Calculate current profit percentage
            if order.side == OrderSide.BUY:
                profit_pct = (last_kline["Close"] - order["Close"]) / order["Close"] * 100
            else:  # SELL order
                profit_pct = (order["Close"] - last_kline["Close"]) / order["Close"] * 100
                
            # Log current profit
            print(f"Order {i} profit: {profit_pct:.2f}%, fast_ma: {self.fast_ma.iloc[-1]:.4f}, " +
                 f"slow_ma: {self.slow_ma.iloc[-1]:.4f}, rsi: {self.rsi.iloc[-1]:.2f}")
            
            # Initialize trailing stop if not already set
            if "trailing_stop" not in order:
                order["trailing_stop"] = order["sl"]
                
            # Update trailing stop for profitable positions
            if profit_pct > self.params.get("trailing_stop", {}).get("activation_pct", 2.0):
                # For BUY orders, move stop loss up
                if order.side == OrderSide.BUY:
                    trail_value = last_kline["Close"] - (self.atr.iloc[-1] * self.params.get("trailing_stop", {}).get("atr_multiplier", 2.0))
                    if trail_value > order["trailing_stop"]:
                        order["trailing_stop"] = trail_value
                        bot_logger.info(f"Trailing stop updated for BUY order: {order['trailing_stop']:.4f}")
                # For SELL orders, move stop loss down
                else:
                    trail_value = last_kline["Close"] + (self.atr.iloc[-1] * self.params.get("trailing_stop", {}).get("atr_multiplier", 2.0))
                    if trail_value < order["trailing_stop"]:
                        order["trailing_stop"] = trail_value
                        bot_logger.info(f"Trailing stop updated for SELL order: {order['trailing_stop']:.4f}")
            
            # Check for exit conditions
            exit_signal = False
            
            # 1. Check trailing stop
            if order.side == OrderSide.BUY and last_kline["Close"] < order["trailing_stop"]:
                bot_logger.info(f"BUY order hit trailing stop: {order['trailing_stop']:.4f}")
                exit_signal = True
            elif order.side == OrderSide.SELL and last_kline["Close"] > order["trailing_stop"]:
                bot_logger.info(f"SELL order hit trailing stop: {order['trailing_stop']:.4f}")
                exit_signal = True
                
            # 2. Check indicator reversals for BUY orders
            if order.side == OrderSide.BUY:
                # Exit if indicators show reversal
                if (self.fast_ma.iloc[-1] < self.slow_ma.iloc[-1] and  # MA crossover
                    self.rsi.iloc[-1] < 40 and  # RSI trending down
                    profit_pct > 0.5):  # Only if in profit
                    bot_logger.info("BUY order exit due to indicator reversal")
                    exit_signal = True
                    
                # Exit if take profit hit
                if last_kline["Close"] >= order["tp"]:
                    bot_logger.info("BUY order hit take profit")
                    exit_signal = True
                    
            # 3. Check indicator reversals for SELL orders
            elif order.side == OrderSide.SELL:
                # Exit if indicators show reversal
                if (self.fast_ma.iloc[-1] > self.slow_ma.iloc[-1] and  # MA crossover
                    self.rsi.iloc[-1] > 60 and  # RSI trending up
                    profit_pct > 0.5):  # Only if in profit
                    bot_logger.info("SELL order exit due to indicator reversal")
                    exit_signal = True
                    
                # Exit if take profit hit
                if last_kline["Close"] <= order["tp"]:
                    bot_logger.info("SELL order hit take profit")
                    exit_signal = True
            
            # 4. Check for regime change exit
            if "regime" in order and order["regime"] != self.regime:
                # Don't exit immediately on regime change, but increase likelihood
                if profit_pct > 0.3:  # Small profit threshold
                    bot_logger.info(f"Order exit due to regime change from {order['regime']} to {self.regime}")
                    exit_signal = True
            
            # Execute exit if signal triggered
            if exit_signal:
                self._close_order(order, last_kline, profit_pct, i)
    
    def _close_order(self, order, last_kline, profit_pct, index):
        order.close(last_kline)
        self.trader.close_trade(order)
        
        # Record trade performance for ML training
        if self.use_ml:
            trade_performance = {
                "entry_price": order["Close"],
                "exit_price": last_kline["Close"],
                "profit_pct": profit_pct,
                "side": order.side,
                "duration": last_kline["Open time"] - order["FILL_TIME"],
                "regime": order.get("regime", "unknown"),
                "signal_score": order.get("signal_score", 0)
            }
            self.performance_history.append(trade_performance)
        
        if order.is_closed():
            self.orders_closed.append(order)
        del self.orders_opening[index]
    
    def get_performance_stats(self):
        if not self.orders_closed:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "avg_profit_pct": 0,
                "max_drawdown": 0
            }
            
        total_trades = len(self.orders_closed)
        winning_trades = sum(1 for order in self.orders_closed if order["profit"] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit = sum(order["profit"] for order in self.orders_closed if order["profit"] > 0)
        total_loss = abs(sum(order["profit"] for order in self.orders_closed if order["profit"] < 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")
        
        avg_profit_pct = sum(order["profit_pct"] for order in self.orders_closed) / total_trades if total_trades > 0 else 0
        
        # Calculate max drawdown
        equity_curve = []
        balance = 100  # Starting balance
        for order in self.orders_closed:
            balance *= (1 + order["profit_pct"] / 100)
            equity_curve.append(balance)
            
        max_balance = equity_curve[0]
        max_drawdown = 0
        
        for balance in equity_curve:
            max_balance = max(max_balance, balance)
            drawdown = (max_balance - balance) / max_balance * 100
            max_drawdown = max(max_drawdown, drawdown)
            
        return {
            "total_trades": total_trades,
            "win_rate": win_rate * 100,
            "profit_factor": profit_factor,
            "avg_profit_pct": avg_profit_pct,
            "max_drawdown": max_drawdown
        }
            
    def get_trades_summary(self):
        if not self.orders_closed:
            return {"No trades executed": True}
            
        # Initialize summary stats
        summary = {
            "by_regime": {},
            "by_side": {
                "BUY": {"count": 0, "win_rate": 0, "avg_profit": 0},
                "SELL": {"count": 0, "win_rate": 0, "avg_profit": 0}
            },
            "by_signal_strength": {
                "strong": {"count": 0, "win_rate": 0, "avg_profit": 0},
                "medium": {"count": 0, "win_rate": 0, "avg_profit": 0},
                "weak": {"count": 0, "win_rate": 0, "avg_profit": 0}
            }
        }
        
        # Collect statistics by regime
        regime_trades = {}
        for order in self.orders_closed:
            regime = order.get("regime", "unknown")
            
            if regime not in regime_trades:
                regime_trades[regime] = []
            regime_trades[regime].append(order)
            
            # Collect by side
            side = order.side.name
            summary["by_side"][side]["count"] += 1
            summary["by_side"][side]["avg_profit"] += order["profit_pct"]
            
            # Categorize by signal strength
            signal_score = abs(order.get("signal_score", 0))
            if signal_score > 0.8:
                category = "strong"
            elif signal_score > 0.6:
                category = "medium"
            else:
                category = "weak"
                
            summary["by_signal_strength"][category]["count"] += 1
            summary["by_signal_strength"][category]["avg_profit"] += order["profit_pct"]
        
        # Calculate statistics by regime
        for regime, trades in regime_trades.items():
            count = len(trades)
            winning = sum(1 for trade in trades if trade["profit"] > 0)
            win_rate = winning / count if count > 0 else 0
            avg_profit = sum(trade["profit_pct"] for trade in trades) / count if count > 0 else 0
            
            summary["by_regime"][regime] = {
                "count": count,
                "win_rate": win_rate * 100,
                "avg_profit": avg_profit
            }
            
        # Finalize calculations for side and signal strength
        for side in ["BUY", "SELL"]:
            count = summary["by_side"][side]["count"]
            if count > 0:
                summary["by_side"][side]["avg_profit"] /= count
                winning = sum(1 for order in self.orders_closed 
                                if order.side.name == side and order["profit"] > 0)
                summary["by_side"][side]["win_rate"] = (winning / count) * 100
                
        for category in ["strong", "medium", "weak"]:
            count = summary["by_signal_strength"][category]["count"]
            if count > 0:
                summary["by_signal_strength"][category]["avg_profit"] /= count
                
                # Calculate win rate based on signal strength category
                signal_score_threshold = 0.8 if category == "strong" else (0.6 if category == "medium" else 0)
                winning = sum(1 for order in self.orders_closed 
                                if abs(order.get("signal_score", 0)) > signal_score_threshold and order["profit"] > 0)
                summary["by_signal_strength"][category]["win_rate"] = (winning / count) * 100
                
        return summary


# # Update layout
#         fig.update_xaxes(showspikes=True, spikesnap="data")
#         fig.update_yaxes(showspikes=True, spikesnap="data")
#         fig.update_layout(hovermode="x", spikedistance=-1)
#         fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=16))
#         fig.update_layout(
#             title={
#                 "text": f"Enhanced Multi-Strategy (tf {self.tf}, Regime: {self.regime})", 
#                 "x": 0.5, 
#                 "xanchor": "center"
#             }
#         )
        
#         # Restore original time
#         df["Open time"] = tmp_ot
#         return fig