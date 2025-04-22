import logging
import pandas as pd
from datetime import datetime
import talib as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .base_strategy import BaseStrategy
import indicators as mta
from order import Order, OrderType, OrderSide, OrderStatus

bot_logger = logging.getLogger("bot_logger")
IN_BUYING = 1
IN_SELLING = 2


class MACross(BaseStrategy):

    def __init__(self, name, params, tfs):
        super().__init__(name, params, tfs)
        self.tf = self.tfs["tf"]
        self.name = "Multi Strategy"
        self.description = "Multi Strategy"
        self.state = None
        self.trend = None
        self.ma_func = ta.SMA if self.params["ma_inputs"]["type"] == "SMA" else ta.EMA
        self.ma_stream_func = ta.stream.SMA if self.params["ma_inputs"]["type"] == "SMA" else ta.stream.EMA

    def attach(self, tfs_chart):
        self.tfs_chart = tfs_chart
        self.init_indicators()

    def init_indicators(self):
        chart = self.tfs_chart[self.tf]
        self.fast_ma = self.ma_func(chart["Close"], self.params["ma_inputs"]["fast_ma"])
        self.slow_ma = self.ma_func(chart["Close"], self.params["ma_inputs"]["slow_ma"])

        self.atr = ta.ATR(chart["High"], chart["Low"], chart["Close"], timeperiod=self.params["atr_inputs"]["atr_period"])
        self.atr_pct = self.atr / chart["Close"] * 100
        self.adx = ta.ADX(chart["High"], chart["Low"], chart["Close"], timeperiod=self.params["adx_inputs"]["adx_period"])
        self.plus_di = ta.PLUS_DI(chart["High"], chart["Low"], chart["Close"], timeperiod=self.params["adx_inputs"]["adx_period"])
        self.minus_di = ta.MINUS_DI(chart["High"], chart["Low"], chart["Close"], timeperiod=self.params["adx_inputs"]["adx_period"])
        self.rsi = ta.RSI(chart["Close"], timeperiod=self.params["rsi_inputs"]["rsi_period"])
        self.macd, self.macd_signal, self.macd_hist = ta.MACD(chart["Close"], fastperiod=self.params["macd_inputs"]["fast"], slowperiod=self.params["macd_inputs"]["slow"], signalperiod=self.params["macd_inputs"]["signal"])

        self.start_trading_time = chart.iloc[-1]["Open time"]

    def update_indicators(self, tf):
        if tf != self.tf:
            return
        last_kline = self.tfs_chart[self.tf].iloc[-1]
        chart = self.tfs_chart[self.tf]
        self.fast_ma.loc[len(self.fast_ma)] = self.ma_stream_func(
            chart["Close"], self.params["ma_inputs"]["fast_ma"]
        )
        self.slow_ma.loc[len(self.slow_ma)] = self.ma_stream_func(
            chart["Close"], self.params["ma_inputs"]["slow_ma"]
        )
        self.atr.loc[len(self.atr)] = ta.stream.ATR(chart["High"], chart["Low"], chart["Close"], timeperiod=self.params["atr_inputs"]["atr_period"])
        self.atr_pct.loc[len(self.atr_pct)] = self.atr.iloc[-1] / last_kline["Close"] * 100
        self.adx.loc[len(self.adx)] = ta.stream.ADX(chart["High"], chart["Low"], chart["Close"], timeperiod=self.params["adx_inputs"]["adx_period"])
        self.plus_di.loc[len(self.plus_di)] = ta.stream.PLUS_DI(chart["High"], chart["Low"], chart["Close"], timeperiod=self.params["adx_inputs"]["adx_period"])
        self.minus_di.loc[len(self.minus_di)] = ta.stream.MINUS_DI(chart["High"], chart["Low"], chart["Close"], timeperiod=self.params["adx_inputs"]["adx_period"])
        self.rsi.loc[len(self.rsi)] = ta.stream.RSI(chart["Close"], timeperiod=self.params["rsi_inputs"]["rsi_period"])
        self.macd.loc[len(self.macd)], self.macd_signal.loc[len(self.macd_signal)], self.macd_hist.loc[len(self.macd_hist)] = ta.stream.MACD(chart["Close"], fastperiod=self.params["macd_inputs"]["fast"], slowperiod=self.params["macd_inputs"]["slow"], signalperiod=self.params["macd_inputs"]["signal"])

    def check_required_params(self):
        return all([key in self.params.keys() for key in ["ma_inputs", "atr_inputs", "adx_inputs", "rsi_inputs", "macd_inputs"]])

    def is_params_valid(self):
        if not self.check_required_params():
            bot_logger.info("   [-] Missing required params")
            return False
        return self.params["ma_inputs"]["fast_ma"] < self.params["ma_inputs"]["slow_ma"] and self.params["ma_inputs"]["type"] in ["SMA", "EMA"]

    def update(self, tf):
        super().update(tf)
        self.check_close_signal()
        self.check_signal()

    def close_opening_orders(self):
        super().close_opening_orders(self.tfs_chart[self.tf].iloc[-1])

    def check_signal(self):
        chart = self.tfs_chart[self.tf]
        last_kline = chart.iloc[-1]
        if last_kline["Tick volume"] < 50:
            return
        
        ma_cnd = "None"
        rsi_cnd = "None"
        adx_cnd = "None"
        macd_cnd = "None"
        price_cnd = "None"
        atr_cnd = False

        #price signal
        if(last_kline["Close"] > last_kline["Open"] and  chart["Close"].iloc[-1] > chart["Close"].iloc[-2]):
            price_cnd = "Buy"
        elif(last_kline["Close"] < last_kline["Open"] and  chart["Close"].iloc[-1] < chart["Close"].iloc[-2]):
            price_cnd = "Sell"

        #ma signal
        print("ma",self.fast_ma.iloc[-1] , self.fast_ma.iloc[-2], "atr", self.atr_pct.iloc[-1])
        if(self.fast_ma.iloc[-1] > self.slow_ma.iloc[-1] and last_kline["Close"] >= self.slow_ma.iloc[-1]):
            if(self.fast_ma.iloc[-1] > self.fast_ma.iloc[-2]):
                ma_cnd = "Buy"
        elif(self.fast_ma.iloc[-1] < self.slow_ma.iloc[-1] and last_kline["Close"] <= self.slow_ma.iloc[-1]):
            if(self.fast_ma.iloc[-1] < self.fast_ma.iloc[-2]):
                ma_cnd = "Sell"

        #rsi signal
        if(self.rsi.iloc[-1] > self.params["rsi_inputs"]["os_rsi"] and self.rsi.iloc[-1] > self.rsi.iloc[-2]):
            rsi_cnd = "Buy"
        elif(self.rsi.iloc[-1] < self.params["rsi_inputs"]["ob_rsi"] and self.rsi.iloc[-1] < self.rsi.iloc[-2]):
            rsi_cnd = "Sell"

        #adx signal
        if(self.adx.iloc[-1] > self.params["adx_inputs"]["adx_threshold"] and self.adx.iloc[-1] > self.adx.iloc[-2]):
            if(self.plus_di.iloc[-1] > self.minus_di.iloc[-1] * 1.5):
                adx_cnd = "Buy"
            elif(self.plus_di.iloc[-1] < self.minus_di.iloc[-1] * 1.5):
                adx_cnd = "Sell"

        #macd signal
        if(self.macd.iloc[-1] > self.macd_signal.iloc[-1] and self.macd.iloc[-1] > self.macd.iloc[-2]):
            macd_cnd = "Buy"
        elif(self.macd.iloc[-1] < self.macd_signal.iloc[-1] and self.macd.iloc[-1] < self.macd.iloc[-2]):
            macd_cnd = "Sell"
        #atr signal
        if(self.atr_pct.iloc[-1] > self.params["atr_inputs"]["atr_threshold"]):
            atr_cnd = True

        #trading signal
        trading_cnd = "None"
        if(price_cnd == "Buy" and ma_cnd == "Buy" and rsi_cnd == "Buy" and adx_cnd == "Buy" and macd_cnd == "Buy" and atr_cnd):
            trading_cnd = "Buy"
        elif(price_cnd == "Sell" and ma_cnd == "Sell" and rsi_cnd == "Sell" and adx_cnd == "Sell" and macd_cnd == "Sell" and atr_cnd):
            trading_cnd = "Sell"

        print(f"price_cnd: {price_cnd}, ma_cnd: {ma_cnd}, rsi_cnd: {rsi_cnd}, adx_cnd: {adx_cnd}, macd_cnd: {macd_cnd}, atr_cnd: {atr_cnd}, trading_cnd: {trading_cnd}")

        if(trading_cnd == "Buy"):
            tp = None
            sl = None
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
            order = self.trader.fix_order(order, self.params["sl_fix_mode"], self.max_sl_pct)
            if order:
                self.trader.create_trade(order, self.volume)
                self.orders_opening.append(order)
        elif(trading_cnd == "Sell"):
            tp = None
            sl = None
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
            order["strategy"] = self.name
            order["description"] = f"{self.description}-SELL"

            order = self.trader.fix_order(order, self.params["sl_fix_mode"], self.max_sl_pct)
            if order:
                self.trader.create_trade(order, self.volume)
                self.orders_opening.append(order)

    def check_close_signal(self):
        chart = self.tfs_chart[self.tf]
        last_kline = chart.iloc[-1]
        if last_kline["Close"] < last_kline["Open"]:
            for i in range(len(self.orders_opening) - 1, -1, -1):
                order = self.orders_opening[i]
                if order.side == OrderSide.SELL:
                    continue
                inc_pct = (last_kline["Close"] - order["Close"]) / last_kline["Close"] * 100 
                print(f"inc_pct: {inc_pct}, fast_ma: {self.fast_ma.iloc[-1]}, slow_ma: {self.slow_ma.iloc[-1]}, rsi: {self.rsi.iloc[-1]}")
                if inc_pct > 0.5 and (self.fast_ma.iloc[-1] < self.slow_ma.iloc[-1] or self.rsi.iloc[-1] < 20):
                    order.close(last_kline)
                    self.trader.close_trade(order)
                    if order.is_closed():
                        self.orders_closed.append(order)
                    del self.orders_opening[i]
        elif last_kline["Close"] > last_kline["Open"]:
            for i in range(len(self.orders_opening) - 1, -1, -1):
                order = self.orders_opening[i]
                if order.side == OrderSide.BUY:
                    continue    
                inc_pct = (last_kline["Close"] - order["Close"]) / order["Close"] * 100 
                print(f"inc_pct: {inc_pct}, fast_ma: {self.fast_ma.iloc[-1]}, slow_ma: {self.slow_ma.iloc[-1]}, rsi: {self.rsi.iloc[-1]}")
                if inc_pct > 0.5 and (self.fast_ma.iloc[-1] > self.slow_ma.iloc[-1] or self.rsi.iloc[-1] > 80):
                    order.close(last_kline)
                    self.trader.close_trade(order)
                    if order.is_closed():
                        self.orders_closed.append(order)
                    del self.orders_opening[i]
    
    def plot_orders(self):
        fig = make_subplots(2, 1, vertical_spacing=0.02, shared_xaxes=True, row_heights=[0.8, 0.2])
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            xaxis2_rangeslider_visible=False,
            yaxis=dict(showgrid=False),
            xaxis=dict(showgrid=False),
            yaxis2=dict(showgrid=False),
            plot_bgcolor="#2E4053",
            paper_bgcolor="#797D7F",
            font=dict(color="#F7F9F9"),
        )
        df = self.tfs_chart[self.tf]
        dt2idx = dict(zip(df["Open time"], list(range(len(df)))))
        tmp_ot = df["Open time"]
        df["Open time"] = list(range(len(df)))
        super().plot_orders(fig, self.tf, 1, 1, dt2idx=dt2idx)
        df = self.tfs_chart[self.tf]
        fig.add_trace(
            go.Scatter(
                x=df["Open time"],
                y=self.fast_ma,
                mode="lines",
                line=dict(color="blue"),
                name="FastMA_{}".format(self.params["fast_ma"]),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["Open time"],
                y=self.slow_ma,
                mode="lines",
                line=dict(color="red"),
                name="SlowMA_{}".format(self.params["slow_ma"]),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["Open time"], y=ta.SMA(df["Close"], 100), mode="lines", line=dict(color="orange"), name="MA_100"
            ),
            row=1,
            col=1,
        )

        colors = ["red" if kline["Open"] > kline["Close"] else "green" for i, kline in df.iterrows()]
        fig.add_trace(go.Bar(x=df["Open time"], y=df["Volume"], marker_color=colors), row=2, col=1)

        fig.update_xaxes(showspikes=True, spikesnap="data")
        fig.update_yaxes(showspikes=True, spikesnap="data")
        fig.update_layout(hovermode="x", spikedistance=-1)
        fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=16))
        fig.update_layout(
            title={"text": "MA Cross Strategy (tf {})".format(self.tf), "x": 0.5, "xanchor": "center"}
        )
        df["Open time"] = tmp_ot
        return fig
