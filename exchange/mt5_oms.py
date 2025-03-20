import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from trade import Trade
from order import Order, OrderType, OrderSide, OrderType
import MetaTrader5 as mt5

bot_logger = logging.getLogger("bot_logger")


class MT5OrderTemplate:
    def __init__(self, symbol, volume, entry, tp, sl, order_side: OrderSide, order_type: OrderType):
        self.symbol = symbol
        self.volume = volume
        self.price = entry
        self.tp = tp
        self.sl = sl
        self.order_type = order_type
        self.order_side = order_side

    def get_main_order(self):
        params = {
            "symbol": self.symbol,
            "volume": self.volume,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "type_time": mt5.ORDER_TIME_GTC,
        }
        if self.tp:
            params["tp"] = self.tp
        if self.sl:
            params["sl"] = self.sl
        if self.order_type == OrderType.LIMIT:
            params["action"] = mt5.TRADE_ACTION_PENDING
            params["type"] = mt5.ORDER_TYPE_BUY_LIMIT if self.order_side == OrderSide.BUY else mt5.ORDER_TYPE_SELL_LIMIT
            params["price"] = self.price
        else:
            params["action"] = mt5.TRADE_ACTION_DEAL
            params["type"] = mt5.ORDER_TYPE_BUY if self.order_side == OrderSide.BUY else mt5.ORDER_TYPE_SELL
        return params

    def get_close_order(self):
        params = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.volume,
            "position": None,
            "price": None,
            "type": mt5.ORDER_TYPE_BUY if self.order_side == OrderSide.SELL else mt5.ORDER_TYPE_SELL,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        return params


class MT5OMS:
    def __init__(self, exchange):
        self.mt5_api = exchange
        self.active_trades = {}
        self.closed_trades = {}
        self.start_time = datetime.now()

    def create_trade(self, order: Order, volume):
        # round tp/sl price
        if order.has_sl():
            order.sl = self.mt5_api.round_price(order["symbol"], order.sl)
        if order.has_tp():
            order.tp = self.mt5_api.round_price(order["symbol"], order.tp)
        order.entry = self.mt5_api.round_price(order["symbol"], order.entry)
        print(order.entry, order.sl, order.tp)
        trade = Trade(order, volume)
        print("   [*] create trade, trade_id: {}".format(trade.trade_id))
        order_tpl = MT5OrderTemplate(order["symbol"], volume, order.entry, order.tp, order.sl, order.side, order.type)
        trade.main_order_params = order_tpl.get_main_order()
        trade.close_order_params = order_tpl.get_close_order()
        # update ask/bid price for market order
        if order.type == OrderType.MARKET:
            if order.side == OrderSide.BUY:
                trade.main_order_params["price"] = self.mt5_api.tick_ask_price(order["symbol"])
            else:
                trade.main_order_params["price"] = self.mt5_api.tick_bid_price(order["symbol"])
        trade.main_order_params["comment"] = order["description"]
        result = self.mt5_api.place_order(trade.main_order_params)
        result_dict = result._asdict()
        result_dict["request"] = result_dict["request"]._asdict()
        trade.main_order = result_dict
        trade.close_order_params["position"] = result_dict["order"]
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print("       [+] create main order success")
            self.active_trades[trade.trade_id] = trade
            return trade.trade_id
        else:
            print("       [+] create main order failed: {}".format(result_dict))
            self.closed_trades[trade.trade_id] = trade
            return None

    def get_trade(self, trade_id):
        return self.active_trades.get(trade_id)

    def get_point(self, symbol):
        return self.mt5_api.get_point(symbol)

    def close_trade(self, trade_id):
        print("  [*] close trade: {}".format(trade_id))
        trade = self.get_trade(trade_id)
        if trade is None:
            print("  [*] trade: {} not exist or already closed".format(trade_id))
            return
        # check if order is type limit and pending
        result = mt5.orders_get(ticket=trade.main_order["order"])
        if len(result) > 0:
            pending_order = result[0]
            if pending_order.state == mt5.ORDER_STATE_PLACED:
                request = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "symbol": trade.order["symbol"],
                    "order": trade.main_order["order"],
                }
                result = self.mt5_api.place_order(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print("       [+] close limit trade success")
                else:
                    print("       [+] close limit trade failed: {}".format(result._asdict()))
        else:
            # update ask/bid price for market order
            if trade.order.side == OrderSide.BUY:
                trade.close_order_params["price"] = self.mt5_api.tick_bid_price(trade.order["symbol"])
            else:
                trade.close_order_params["price"] = self.mt5_api.tick_ask_price(trade.order["symbol"])
            result = self.mt5_api.place_order(trade.close_order_params)
            result_dict = result._asdict()
            result_dict["request"] = result_dict["request"]._asdict()
            trade.close_order = result_dict
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print("       [+] close trade success")
            else:
                print("       [+] close trade failed: {}".format(result_dict))
        self.closed_trades[trade_id] = self.active_trades[trade_id]
        del self.active_trades[trade_id]

    def adjust_sl(self, trade_id, sl):
        print("  [+] adjust sl, trade: {}, sl: {}".format(trade_id, sl))
        trade = self.get_trade(trade_id)
        if trade is None:
            print("  [*] trade: {} not exist or already closed".format(trade_id))
            return
        sl = self.mt5_api.round_price(trade.order["symbol"], sl)
        params = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": trade.order["symbol"],
            "position": trade.main_order["order"],
            "sl": sl,
        }
        if trade.order.tp:
            params["tp"] = trade.order.tp

        result = self.mt5_api.place_order(params)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print("       [+] adjust sl success")
            trade.order.sl = sl
        else:
            print("       [+] adjust sl failed")

    def adjust_tp(self, trade_id, tp):
        print("  [+] adjust tp, trade: {}, tp: {}".format(trade_id, tp))
        trade = self.get_trade(trade_id)
        if trade is None:
            print("  [*] trade: {} not exist or already closed".format(trade_id))
            return
        tp = self.mt5_api.round_price(trade.order["symbol"], tp)
        params = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": trade.order["symbol"],
            "position": trade.main_order["order"],
            "tp": tp,
        }
        if trade.order.sl:
            params["sl"] = trade.order.sl

        result = self.mt5_api.place_order(params)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print("       [+] adjust tp success")
            trade.order.tp = tp
        else:
            print("       [+] adjust tp failed")

    def monitor_trades(self):
        if len(self.active_trades) == 0:
            return

    def close_all_trade(self):
        print("  [*] close all trade")
        for trade_id in list(self.active_trades.keys()):
            self.close_trade(trade_id)

    def get_income_history(self):
        total_deals = []
        for _, trade in self.closed_trades.items():
            result = self.mt5_api.history_deals_get(trade.main_order["order"])
            if result:
                total_deals.extend([res._asdict() for res in result])
        return pd.DataFrame(total_deals)
    
    def get_income_history(self, from_time,to_time, group="*"):
        from_time = from_time + timedelta(seconds=-time.timezone)
        to_time = to_time + timedelta(seconds=-time.timezone)
        result = self.mt5_api.history_deals_get(from_time, to_time, group=group)
        df=[]
        if result:
            df=pd.DataFrame(list(result),columns=result[0]._asdict().keys())
            df["time"] += -time.timezone
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df[['symbol', 'type', 'price', 'volume', 'profit','time','time_msc']]
        return df
    
    def klinesDate(self, symbol: str, interval:str, from_date: datetime, to_date: datetime): 
        return self.mt5_api.klinesDate(symbol, interval, from_date, to_date)
    
    def klinesCount(self, symbol: str, interval:str, from_date: datetime, limit: int): 
        return self.mt5_api.klinesCount(symbol, interval, from_date, limit)

    def get_trades(self):
        return self.closed_trades, self.active_trades
