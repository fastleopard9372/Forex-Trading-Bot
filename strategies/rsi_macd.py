import logging
import talib as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .base_strategy import BaseStrategy
import indicators as mta
from order import Order, OrderType, OrderSide, OrderStatus, OrderCloseSide
from utils import get_line_coffs, find_uptrend_line, find_downtrend_line, get_y_on_line
from scipy.stats import linregress
import numpy as np

bot_logger = logging.getLogger("bot_logger")


class RsiMacd(BaseStrategy):

    def __init__(self, name, params, tfs):
        super().__init__(name, params, tfs)
        self.tf = self.tfs["tf"]
        self.state = None
        self.trend = None
        self.min_zz_ratio = 0.01 * self.params["min_zz_pct"]
        self.temp = []
        self.rsi_len = self.params["rsi_inputs"]["rsi_len"]
        self.ob_rsi = self.params["rsi_inputs"]["ob_rsi"]
        self.os_rsi = self.params["rsi_inputs"]["os_rsi"]
        self.adx_threshold = self.params["adx_inputs"]["threshold"]
        self.atr_pct_threshold = self.params["atr_inputs"]["atr_pct"]
        
        self.rsi = 50
        self.trading_flag = 0
        self.rsi_cur_avg = 50

        self.rsi_cond = []
        self.macd_cond = []
        self.ma_cond = []
        self.adx_cond = []
        self.bb_cond = []
        self.atr_cond = []

    def attach(self, tfs_chart):
        self.tfs_chart = tfs_chart
        self.init_indicators()

    def init_indicators(self):
        # calculate HA candelstick
        chart = self.tfs_chart[self.tf]
        self.ma_vol = ta.SMA(chart["Tick volume"], self.params["ma_vol"])
        #MACD
        self.macd, self.macdsignal, self.macdhist = ta.MACD(
            chart["Close"],
            self.params["macd_inputs"]["fast_len"],
            self.params["macd_inputs"]["slow_len"],
            self.params["macd_inputs"]["signal"],
        )
        # Momentum Oscillators
        self.rsi = ta.RSI(chart["Close"], self.rsi_len)

        # Moving Averages
        self.ma_short = ta.EMA(chart['Close'], timeperiod = self.params["ma_inputs"]["short"])
        self.ma_medium = ta.EMA(chart['Close'], timeperiod = self.params["ma_inputs"]["medium"])
        self.ma_long = ta.EMA(chart['Close'], timeperiod = self.params["ma_inputs"]["long"])
        
        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_Lower = ta.BBANDS(
            chart['Close'], 
            timeperiod = self.params["bb_inputs"]["len"], 
            nbdevup = self.params["bb_inputs"]["nbdevup"], 
            nbdevdn = self.params["bb_inputs"]["nbdevdn"], 
            matype = 0
        )
        
        # Bollinger Band Width (for congestion)
        self.bb_width =  (self.bb_upper - self.bb_Lower) / self.bb_middle
        
        # ADX (for trend strength)
        self.adx =  ta.ADX(chart['High'], chart['Low'], chart['Close'], timeperiod = self.params["adx_inputs"]["len"])
        self.plus_di = ta.PLUS_DI(chart['High'], chart['Low'], chart['Close'], timeperiod = self.params["adx_inputs"]["len"])
        self.minus_di = ta.MINUS_DI(chart['High'], chart['Low'], chart['Close'], timeperiod = self.params["adx_inputs"]["len"])
        # ATR (for volatility and congestion)
        atr =  ta.ATR(chart['High'], chart['Low'], chart['Close'], timeperiod = self.params["atr_inputs"]["len"])
        self.atr_pct =  atr / chart['Close'] * 100  # ATR as percentage of price

        self.start_trading_time = chart.iloc[-1]["Open time"]

    def update_indicators(self, tf):
        if tf != self.tf:
            return
        chart = self.tfs_chart[self.tf]

        self.ma_vol.loc[len(self.ma_vol)] = ta.stream.SMA(chart["Tick volume"], self.params["ma_vol"])

        macd, macdsignal, macdhist = ta.stream.MACD(
            chart["Close"],
            self.params["macd_inputs"]["fast_len"],
            self.params["macd_inputs"]["slow_len"],
            self.params["macd_inputs"]["signal"],
        )
        self.macd.loc[len(self.macd)] = macd
        self.macdsignal.loc[len(self.macdsignal)] = macdsignal
        self.macdhist.loc[len(self.macdhist)] = macdhist

        self.rsi.loc[len(self.rsi)] = ta.stream.RSI(chart["Close"], self.rsi_len)

        self.ma_short.loc[len(self.ma_short)] = ta.stream.EMA(chart['Close'], timeperiod=self.params["ma_inputs"]["short"])
        self.ma_medium.loc[len(self.ma_medium)] = ta.stream.EMA(chart['Close'], timeperiod=self.params["ma_inputs"]["medium"])
        self.ma_long.loc[len(self.ma_long)] = ta.stream.EMA(chart['Close'], timeperiod=self.params["ma_inputs"]["long"])
        
        bb_upper, bb_middle, bb_Lower = ta.stream.BBANDS(
            chart['Close'], 
            timeperiod = self.params["bb_inputs"]["len"], 
            nbdevup = self.params["bb_inputs"]["nbdevup"], 
            nbdevdn = self.params["bb_inputs"]["nbdevdn"], 
            matype = 0
        )
        self.bb_upper.loc[len(self.bb_upper)] = bb_upper
        self.bb_middle.loc[len(self.bb_middle)] = bb_middle
        self.bb_Lower.loc[len(self.bb_Lower)] = bb_Lower

        self.bb_width =  (self.bb_upper - self.bb_Lower)
        
        self.adx[len(self.adx)] =  ta.stream.ADX(chart['High'], chart['Low'], chart['Close'], timeperiod=self.params["adx_inputs"]["len"])
        self.plus_di[len(self.plus_di)] = ta.stream.PLUS_DI(chart['High'], chart['Low'], chart['Close'], timeperiod = self.params["adx_inputs"]["len"])
        self.minus_di[len(self.minus_di)] = ta.stream.MINUS_DI(chart['High'], chart['Low'], chart['Close'], timeperiod = self.params["adx_inputs"]["len"])
        
        atr =  ta.stream.ATR(chart['High'], chart['Low'], chart['Close'], timeperiod=self.params["atr_inputs"]["len"])
        self.atr_pct.loc[len(self.atr_pct)] =  atr / chart.iloc[-1]['Close'] * 100

    def check_required_params(self):
        return all(
            [
                key in self.params.keys()
                for key in [
                    "ma_vol",
                    "vol_ratio_ma",
                    "kline_body_ratio",
                    "sl_fix_mode",
                    "macd_inputs",
                    "rsi_inputs",
                    "ma_inputs",
                    "bb_inputs",
                    "adx_inputs",
                    "atr_inputs"
                ]
            ]
        )

    def is_params_valid(self):
        if not self.check_required_params():
            bot_logger.info("   [-] Missing required params")
            return False
        return True

    def update(self, tf):
        # update when new kline arrive
        super().update(tf)
        # check order signal
        self.check_close_signal()
        self.check_signal()

    def close_opening_orders(self):
        super().close_opening_orders(self.tfs_chart[self.tf].iloc[-1])

    def check_signal(self):
        chart = self.tfs_chart[self.tf]
        last_kline = chart.iloc[-1]
        if(last_kline["Tick volume"] < 30): return
        if last_kline["Tick volume"] < self.params["vol_ratio_ma"] * self.ma_vol.iloc[-1]:
            return
        bb_width = (self.bb_upper.iloc[-1] - self.bb_Lower.iloc[-1])
        print("MA:", self.ma_short.iloc[-1], self.ma_medium.iloc[-1], self.ma_long.iloc[-1])
        print("RSI:",round(self.rsi.iloc[-1], 4),
               "Atr_pct:",round(self.atr_pct.iloc[-1], 4),
                 "Adx:",round(self.adx.iloc[-1], 4), "DI+:",round(self.plus_di.iloc[-1],4), "DI-:",round(self.minus_di.iloc[-1],4), 
                 "MACD_HIST:",round(self.macdhist.iloc[-1],4), "MACD:",round(self.macd.iloc[-1],4), "MACD_Signal:",round(self.macdsignal.iloc[-1],4))

        last = len(self.rsi_cond)
        #RSI condition
        self.rsi_cond.append("None")
        offset = 15
        if(self.rsi.iloc[-1] < self.os_rsi + offset): self.rsi_cond[last] = "ENTRY_BUY"
        if(self.rsi.iloc[-1] > self.ob_rsi - offset): self.rsi_cond[last] = "ENTRY_SELL"

        #MACD_condition
        self.macd_cond.append("None")
        if(self.macdhist.iloc[-1] > 0 ): self.macd_cond[last] = "ENTRY_BUY"
        if(self.macdhist.iloc[-1] < 0 ): self.macd_cond[last] = "ENTRY_SELL"

        #MA_condition
        self.ma_cond.append("None")
        if(self.ma_short.iloc[-1] > self.ma_medium.iloc[-1] and self.ma_medium.iloc[-1] > self.ma_long.iloc[-1]
           ): self.ma_cond[last] = "ENTRY_BUY"
        if(self.ma_short.iloc[-1] < self.ma_medium.iloc[-1] and self.ma_medium.iloc[-1] < self.ma_long.iloc[-1]
           ): self.ma_cond[last] = "ENTRY_SELL"
        
        
        #ADX_condition
        self.adx_cond.append("None")
        if(self.adx.iloc[-1] > self.adx_threshold and self.adx.iloc[-1] < self.adx_threshold * 2.5):
           if(self.plus_di.iloc[-1] > self.plus_di.iloc[-1]): self.adx_cond[last] = "ENTRY_BUY"
           if(self.minus_di.iloc[-1] > self.minus_di.iloc[-1]): self.adx_cond[last] = "ENTRY_SELL"
        
        #ATR_condition
        self.atr_cond.append(False)
        if(self.atr_pct.iloc[-1] > self.atr_pct_threshold): self.atr_cond[last] = True

        #BB_condition
        # self.bb_cond.loc[last] = "None"
        # if(self.bb_upper.iloc[-1] > last_kline["Close"] and self.bb_upper.iloc[-2] < self.bb_upper.iloc[-1]): self.bb_cond.loc[last] = "ENTRY_BUY"
        # if(self.bb_upper.iloc[-1] < last_kline["Close"] and self.bb_upper.iloc[-2] > self.bb_upper.iloc[-1]): self.bb_cond.loc[last] = "ENTRY_SELL"
        

        print("RSI:", self.rsi_cond[last], "MACD:", self.macd_cond[last], "MA:", self.ma_cond[last], "ADX:", self.adx_cond[last], "ATR:", self.atr_cond[last])
        #ENTRY BUY
        if(self.ma_cond[last] == "ENTRY_BUY" and 
           self.rsi_cond[last] == "ENTRY_BUY" and 
           self.macd_cond[last] == "ENTRY_BUY" and 
           self.adx_cond[last] == "ENTRY_BUY" and 
           self.atr_cond[last]):  self.trading_flag = "ENTRY_BUY"
        
        #ENTRY SELL
        if(self.ma_cond[last] == "ENTRY_SELL" and 
           self.rsi_cond[last] == "ENTRY_SELL" and 
           self.macd_cond[last] == "ENTRY_SELL" and 
           self.adx_cond[last] == "ENTRY_SELL" and 
           self.atr_cond[last]):  self.trading_flag = "ENTRY_SELL"
        
        #CLOSE BUY
        if(self.rsi.iloc[-1] > self.ob_rsi or 
           (self.macdhist.iloc[-1] < 0 and self.macdhist.iloc[-2] > self.macdhist.iloc[-1]) or 
           (self.adx.iloc[-1] < self.adx_threshold) or 
           (self.bb_upper.iloc[-1] < last_kline["Close"])):
            self.trading_flag = 0

        # print("RSI:", self.rsi_cond.iloc[-1], "MACD:", self.macd_cond.iloc[-1], "MA:", self.ma_cond.iloc[-1], "ADX:", self.adx_cond.iloc[-1], "BB:", self.bb_cond.iloc[-1])
        # if(self.rsi.iloc[-1] < self.os_rsi and self.rsi.iloc[-2] < self.rsi.iloc[-1]): self.trading_flag = "ENTRY_BUY"
        # if(self.rsi.iloc[-1] > self.ob_rsi and self.rsi.iloc[-2] > self.rsi.iloc[-1]): self.trading_flag = "ENTRY_SELL"
        # if(self.rsi.iloc[-1] > self.ob_rsi - 10 or (self.macd.iloc[-1] < self.macdsignal.iloc[-1]) or (self.adx.iloc[-1] < self.adx_threshold) or (self.bb_upper.iloc[-1] < last_kline["Close"])):
        #     self.trading_flag = 0
        # if(self.rsi.iloc[-1] < self.os_rsi + 10 or (self.macd.iloc[-1] > self.macdsignal.iloc[-1]) or (self.adx.iloc[-1] < self.adx_threshold) or (self.bb_Lower.iloc[-1] > last_kline["Close"])):
        #     self.trading_flag = 0
        # if(self.rsi.iloc[-1] < self.os_rsi and self.rsi.iloc[-2] < self.rsi.iloc[-1]): self.trading_flag = "ENTRY_BUY"
        # if(self.rsi.iloc[-1] > self.ob_rsi and self.rsi.iloc[-2] > self.rsi.iloc[-1]): self.trading_flag = "ENTRY_SELL"
        # if(self.rsi.iloc[-1] > self.ob_rsi - 10 or (self.macd.iloc[-1] < self.macdsignal.iloc[-1]) or (self.adx.iloc[-1] < self.adx_threshold) or (self.bb_upper.iloc[-1] < last_kline["Close"])):
        #     self.trading_flag = 0




        # #Buy condition[entry period]
        # if(self.rsi.iloc[-1] <= self.os_rsi or (self.rsi.iloc[-2] <= self.os_rsi and self.rsi.iloc[-2] <= self.rsi.iloc[-1] <= self.os_rsi + 10)) and \
        #     (self.macd.iloc[-1] >= self.macdsignal.iloc[-1]) and \
        #     (self.adx.iloc[-1] >= self.adx_threshold) and \
        #     (self.bb_Lower.iloc[-1] <= last_kline["Close"]):
        #     self.trading_flag = "ENTRY_BUY"

        # #Sell condition[entry period]
        # elif(self.rsi.iloc[-1] >= self.ob_rsi or (self.rsi.iloc[-2] >= self.ob_rsi and self.rsi.iloc[-2] >= self.rsi.iloc[-1] >= self.ob_rsi - 10)) and \
        #     (self.macd.iloc[-1] <= self.macdsignal.iloc[-1]) and \
        #     (self.adx.iloc[-1] >= self.adx_threshold) and \
        #     (self.bb_upper.iloc[-1] >= last_kline["Close"]):
        #     self.trading_flag = "ENTRY_SELL"

        # #Buy condition[entry period end]
        # if(self.trading_flag == "ENTRY_BUY" and (self.rsi.iloc[-1] > self.ob_rsi - 10 or \
        #     (self.macd.iloc[-1] < self.macdsignal.iloc[-1]) or \
        #     (self.adx.iloc[-1] < self.adx_threshold) or \
        #     (self.bb_upper.iloc[-1] < last_kline["Close"]))):
        #     return
       
        # #Sell condition[entry period close]
        # elif(self.trading_flag == "ENTRY_SELL" and (self.rsi.iloc[-1] < self.os_rsi + 10 or \
        #     (self.macd.iloc[-1] > self.macdsignal.iloc[-1]) or \
        #     (self.adx.iloc[-1] < self.adx_threshold) or \
        #     (self.bb_Lower.iloc[-1] > last_kline["Close"]))):
        #     return
        
        if(self.trading_flag == 0): return
        pip = bb_width
        # if(bb_width < 0.002 * last_kline["Close"]): pip = 0.002 * last_kline["Close"]
        # elif (bb_width > 0.004 * last_kline["Close"]): pip = 0.004 * last_kline["Close"]
        # if(bb_width < 0.0015 * last_kline["Close"]): return
        pip = last_kline["Close"] * self.atr_pct[-1] / 100 * 10
        if self.trading_flag == "ENTRY_BUY" :
            tp = None
            # if(self.ma_short.iloc[-1] < self.ma_medium.iloc[-1]):
            #     tp = pip * 1.5 + last_kline["Close"]
            # else:
            tp = pip * 2 + last_kline["Close"]
            sl = -pip + last_kline["Close"]
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
            order["description"] = self.description
            order = self.trader.fix_order(order, self.params["sl_fix_mode"], self.max_sl_pct)
            if order:
                self.trader.create_trade(order, self.volume)
                self.orders_opening.append(order)
        
        elif self.trading_flag == "ENTRY_SELL" : 
            # if (last_kline["Close"] - last_kline["Low"]) > 0.5 * (last_kline["High"] - last_kline["Low"]):
            #     return
            tp = None
            # if(self.ma_short.iloc[-1] > self.ma_medium.iloc[-1]):
            #     tp = -pip * 1.5 + last_kline["Close"]
            # else:
            tp = -pip * 2 + last_kline["Close"]
            sl = pip  + last_kline["Close"]
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
            order["description"] =  self.description
            order = self.trader.fix_order(order, self.params["sl_fix_mode"], self.max_sl_pct)
            if order:
                self.trader.create_trade(order, self.volume)
                self.orders_opening.append(order)
        return

    def check_close_signal(self):
        chart = self.tfs_chart[self.tf]
        last_kline = chart.iloc[-1]
        return
        for i in range(len(self.orders_opening) - 1, -1, -1):
            order = self.orders_opening[i]
            trading_flag = 0
            if (order.side == OrderSide.BUY) and (self.rsi.iloc[-1] >= self.ob_rsi or (self.rsi.iloc[-2] >= self.ob_rsi and self.rsi.iloc[-2] >= self.rsi.iloc[-1] >= self.ob_rsi - 10)) and \
                (self.adx.iloc[-1] >= self.adx_threshold) and \
                (self.macd.iloc[-1] <= self.macdsignal.iloc[-1]) and \
                (self.bb_upper.iloc[-1] >= last_kline["Close"]):
                    trading_flag = OrderCloseSide.BUY_CL

            elif (order.side == "ENTRY_SELL") and (self.rsi.iloc[-1] <= self.os_rsi or (self.rsi.iloc[-2] <= self.os_rsi and self.rsi.iloc[-2] <= self.rsi.iloc[-1] <= self.os_rsi + 10)) and \
                (self.adx.iloc[-1] >= self.adx_threshold) and \
                (self.macd.iloc[-1] >= self.macdsignal.iloc[-1]) and \
                (self.bb_Lower.iloc[-1] <= last_kline["Close"]):
                    trading_flag = OrderCloseSide.SELL_CL
                    
            if(trading_flag == 0): return
            
            if order.side == OrderSide.BUY and trading_flag == OrderCloseSide.BUY_CL: #Close buy
                order.close(last_kline)
                self.trader.close_trade(order)
                if order.is_closed():
                    self.orders_closed.append(order)
                del self.orders_opening[i]

            elif order.side == "ENTRY_SELL" and trading_flag == OrderCloseSide.SELL_CL: #Close sell
                order.close(last_kline)
                self.trader.close_trade(order)
                if order.is_closed():
                    self.orders_closed.append(order)
                del self.orders_opening[i]
            
    def adjust_sl(self):
        last_main_zz = self.zz_points[self.main_zz_idx[-1]]
        if last_main_zz.ptype == mta.POINT_TYPE.PEAK_POINT:
            for i in range(len(self.orders_opening) - 1, -1, -1):
                order = self.orders_opening[i]
                if order.side == OrderSide.BUY:
                    continue
                if not order.has_sl() or order.sl > last_main_zz.pline.High:
                    order.adjust_sl(last_main_zz.pline.High)
                    self.trader.adjust_sl(order, last_main_zz.pline.High)
        else:
            for i in range(len(self.orders_opening) - 1, -1, -1):
                order = self.orders_opening[i]
                if order.side == "ENTRY_SELL":
                    continue
                if not order.has_sl() or order.sl < last_main_zz.pline.Low:
                    order.adjust_sl(last_main_zz.pline.Low)
                    self.trader.adjust_sl(order, last_main_zz.pline.Low)

    def plot_orders(self):
        # Create subplots with 4 rows: price, volume, MACD, and RSI
        fig = make_subplots(4, 1, vertical_spacing=0.02, shared_xaxes=True, 
                          row_heights=[0.4, 0.2, 0.2, 0.2])
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            xaxis2_rangeslider_visible=False,
            yaxis=dict(showgrid=False),
            yaxis2=dict(showgrid=False),
            xaxis=dict(showgrid=False),
            xaxis2=dict(showgrid=False),
            plot_bgcolor="rgb(19, 23, 34)",
            paper_bgcolor="rgb(121,125,127)",
            font=dict(color="rgb(247,249,249)"),
        )
        df = self.tfs_chart[self.tf]
        dt2idx = dict(zip(df["Open time"], list(range(len(df)))))
        tmp_ot = df["Open time"]
        df["Open time"] = list(range(len(df)))
        
        # Plot price and orders
        # super().plot_orders(fig, self.tf, 1, 1, dt2idx=dt2idx)
        
        # Plot price line
        fig.add_trace(go.Scatter(
            x=df["Open time"],
            y=df['Close'],
            mode='lines',
            name='Price',
            line=dict(color="orange")
        ), row=1, col=1)
        
        # Update layout
        fig.update_xaxes(showspikes=True, spikesnap="data")
        fig.update_yaxes(showspikes=True, spikesnap="data")
        fig.update_layout(hovermode="x", spikedistance=-1)
        fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=16))
        fig.update_layout(
            title={
                "text": "RSI MACD Strategy({}) (tf {})".format(self.trader.symbol_name, self.tf),
                "x": 0.5,
                "xanchor": "center",
            }
        )
        
        # Restore original timestamps
        df["Open time"] = tmp_ot
        return fig
