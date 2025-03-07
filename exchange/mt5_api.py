import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import MetaTrader5 as mt5

bot_logger = logging.getLogger("bot_logger")


class MT5API:
    def __init__(self, config):
        self.config = config

    def initialize(self):
        # connect to MetaTrader 5
        if not mt5.initialize():
            mt5.shutdown()
            return False
        return True

    def login(self):
        # connect to the trade account specifying a server
        authorized = mt5.login(self.config["account"], server=self.config["server"])
        if authorized:
            print("[+] Login success, account info: ")
            self.account_info = mt5.account_info()._asdict()
            if not mt5.account_info().trade_allowed:
                print("AutoTrading is disabled! Enable it in MT5.")
            print(self.account_info)
            return True
        else:
            print("[-] Login failed, check account infor")
            return False

    def round_price(self, symbol, price):
        symbol_info = mt5.symbol_info(symbol)
        return round(price, symbol_info.digits)

    def get_assets_balance(self, assets=["USD"]):
        assets = {}
        return assets

    def tick_ask_price(self, symbol):
        return mt5.symbol_info_tick(symbol).ask

    def tick_bid_price(self, symbol):
        return mt5.symbol_info_tick(symbol).bid

    def klines(self, symbol: str, interval: str, **kwargs): #Klines (also known as candlesticks) represent price movements over a fixed time period
        
        symbol_rates = mt5.copy_rates_from_pos(
            symbol, getattr(mt5, "TIMEFRAME_" + (interval[-1:] + interval[:-1]).upper()), 0, kwargs["limit"]
        )
        # print(mt5.last_error())
        df = pd.DataFrame(symbol_rates)
        df["time"] += -time.timezone
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.columns = ["Open time", "Open", "High", "Low", "Close", "Volume", "Spread", "Real_Volume"]
        df = df[["Open time", "Open", "High", "Low", "Close", "Volume"]]
        return df
    
    def klinesDate(self, symbol: str, interval:str, from_date: datetime, to_date: datetime): 
        from_date = from_date + timedelta(seconds=-time.timezone)
        to_date = to_date + timedelta(seconds=-time.timezone)

        symbol_rates = mt5.copy_rates_range(
            symbol, getattr(mt5, "TIMEFRAME_" + (interval[-1:] + interval[:-1]).upper()), from_date, to_date
        )
        df = pd.DataFrame(symbol_rates)
        if len(df)==0: return False
        df["time"] += -time.timezone
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.columns = ["Open time", "Open", "High", "Low", "Close", "Volume", "Spread", "Real_Volume"]
        df = df[["Open time", "Open", "High", "Low", "Close", "Volume"]]
        return df
    
    def klinesCount(self, symbol: str, interval:str, from_date: datetime, limit: int): 
        from_date = from_date + timedelta(seconds=-time.timezone)
        symbol_rates = mt5.copy_rates_from(
            symbol, getattr(mt5, "TIMEFRAME_" + (interval[-1:] + interval[:-1]).upper()), from_date, limit
        )
        df = pd.DataFrame(symbol_rates)
        if len(df)==0: return []
        df["time"] += -time.timezone
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.columns = ["Open time", "Open", "High", "Low", "Close", "Volume", "Spread", "Real_Volume"]
        df = df[["Open time", "Open", "High", "Low", "Close", "Volume"]]
        return df

    def place_order(self, params):
        return mt5.order_send(params)

    def history_deals_get(self, position_id):
        return mt5.history_deals_get(position=position_id)
    
    def history_deals_get(self, from_time, to_time, group="*"):
        return mt5.history_deals_get(from_time, to_time, group = group)
