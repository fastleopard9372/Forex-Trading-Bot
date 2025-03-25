import os
from datetime import datetime, timedelta
import logging
import json
import shutil
from typing import List
import MetaTrader5 as mt5
import pandas as pd
from trader import Trader
import threading
import time
from utils import tf_cron, NUM_KLINE_INIT, CANDLE_COLUMNS
from utils import get_pretty_table


bot_logger = logging.getLogger("bot_logger")


class BackTestEngine:
    def __init__(self, exch, symbols_trading_cfg_file, data_dir, start_date=None, end_date=None):
        self.exch = exch
        self.data_dir = data_dir
        self.bot_traders: List[Trader] = []
        self.symbols_trading_cfg_file = symbols_trading_cfg_file
        self.debug_dir = os.environ["DEBUG_DIR"]
        self.thread = None,
        self.from_date = start_date
        self.to_date = end_date
        self.start_flag = False

    def get_klines_data(self, symbol, interval, start_date, end_date):
        tf = getattr(mt5, "TIMEFRAME_" + (interval[-1:] + interval[:-1]).upper())
        print(f"Fetching {symbol} {interval} data from {self.from_date} to {self.to_date}...")
        self.from_date = start_date
        self.to_date = end_date
        mt5.initialize()
        rates = mt5.copy_rates_range(symbol, tf, self.from_date, self.to_date)
        mt5.shutdown()
        # Check if data was retrieved
        if rates is None:
            print(f"Failed to retrieve data for {symbol}, error code:", mt5.last_error())
            mt5.shutdown()
            quit()

        # Convert data to pandas DataFrame
        df = pd.DataFrame(rates)
        new_header = ['Open time', 'Open', 'High', 'Low', 'Close', 'Tick Volume', 'Spread', 'Volume']
        df.columns = new_header
        df['Open time'] = pd.to_datetime(df['Open time'],unit='s')
        return df

    def backtest_bot_trader(self, symbol_cfg):
        bot_trader = Trader(symbol_cfg)
        bot_trader.init_strategies()
        tfs_chart = {}
        # Load kline data for all required timeframes
        for tf in bot_trader.get_required_tfs():
            chart_df = pd.concat(
                [
                    self.get_klines_data(symbol_cfg["symbol"], tf, self.from_date, self.to_date)
                ],
                ignore_index=True,
            )
            tfs_chart[tf] = chart_df
        max_time = max([tf_chart.iloc[NUM_KLINE_INIT - 1]["Open time"] for tf_chart in tfs_chart.values()])
        end_time = max([tf_chart.iloc[-1]["Open time"] for tf_chart in tfs_chart.values()])
        tfs_chart_init = {}
        for tf, tf_chart in tfs_chart.items():
            tfs_chart_init[tf] = tf_chart[tf_chart["Open time"] <= max_time][-NUM_KLINE_INIT:]
            tfs_chart[tf] = tf_chart[tf_chart["Open time"] > max_time]
            start_index = tfs_chart_init[tf].index[0]
            tfs_chart_init[tf].index -= start_index
            tfs_chart[tf].index -= start_index
        bot_trader.init_chart(tfs_chart_init)
        bot_trader.attach_oms(None)  # for backtesting don't need oms

        timer = max_time
        end_time = end_time
        print("   [+] Start timer from: {} to {}".format(timer, end_time))
        required_tfs = [tf for tf in tf_cron.keys() if tf in bot_trader.get_required_tfs()]
        # c = 0
        while timer <= end_time:
            timer += timedelta(seconds=60)
            hour, minute = timer.hour, timer.minute
            for tf in required_tfs:
                cron_time = tf_cron[tf]
                if ("hour" not in cron_time or hour in cron_time["hour"]) and (
                    "minute" not in cron_time or minute in cron_time["minute"]
                ):
                    last_kline = tfs_chart[tf][:1]
                    tfs_chart[tf] = tfs_chart[tf][1:]
                    print(timer)
                    bot_trader.on_kline(tf, last_kline)
            # c += 1
            # if c > 1000:
            #     break
                if(self.start_flag == False): return bot_trader
        return bot_trader
    
    def start(self):
        if self.start_flag == False:  
            self.start_flag = True  
            self.thread = threading.Thread(target=self.start_backtest)
            self.thread.start()
        return self.start_flag
    
    def stop(self):
        self.start_flag = False
        time.sleep(3)  # Give some time for the thread to finish
        return self.backtest_result()
            
    def start_backtest(self):
        with open(self.symbols_trading_cfg_file) as f:
            symbols_config = json.load(f)
        print("[*] Start backtesting ...")
        
        for symbol_cfg in symbols_config:
            print("[+] Backtest bot for symbol: {}".format(symbol_cfg["symbol"]))
            bot_trader = self.backtest_bot_trader(symbol_cfg)
            self.bot_traders.append(bot_trader)
        print("[*] Backtesting finished")
        self.start_flag = False

    def backtest_result(self):
        final_backtest_stats = []
        for bot_trader in self.bot_traders:
            backtest_stats = bot_trader.statistic_trade()
            print(get_pretty_table(backtest_stats, bot_trader.get_symbol_name(), transpose=True, tran_col="NAME"))
            backtest_stats.loc[len(backtest_stats) - 1, "NAME"] = bot_trader.get_symbol_name()
            final_backtest_stats.append(backtest_stats.loc[len(backtest_stats) - 1 :])

            bot_trader.log_orders()
            bot_trader.plot_strategy_orders()

        table_stats = pd.concat(final_backtest_stats, axis=0, ignore_index=True)
        s = table_stats.sum(axis=0)
        table_stats.loc[len(table_stats)] = s
        table_stats.loc[len(table_stats) - 1, "NAME"] = "TOTAL"
        print(get_pretty_table(table_stats, "SUMMARY", transpose=True, tran_col="NAME"))
        return table_stats
    
    def summary_trade_result(self):
        final_backtest_stats = []
        for bot_trader in self.bot_traders:
            bot_trader.close_opening_orders()
            backtest_stats = bot_trader.statistic_trade()
            print(get_pretty_table(backtest_stats, bot_trader.get_symbol_name(), transpose=True, tran_col="NAME"))
            backtest_stats.loc[len(backtest_stats) - 1, "NAME"] = bot_trader.get_symbol_name()
            final_backtest_stats.append(backtest_stats.loc[len(backtest_stats) - 1 :])

            bot_trader.log_orders()
            bot_trader.plot_strategy_orders()

        table_stats = pd.concat(final_backtest_stats, axis=0, ignore_index=True)
        s = table_stats.sum(axis=0)
        table_stats.loc[len(table_stats)] = s
        table_stats.loc[len(table_stats) - 1, "NAME"] = "TOTAL"
        print(get_pretty_table(table_stats, "SUMMARY", transpose=True, tran_col="NAME"))
        return table_stats

