from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
import os
import time
import argparse
import logging
import logging.config
from trade_engine import TradeEngine
from backtest_engine import BackTestEngine
from datetime import datetime, timedelta, timezone
from utils import datetime_to_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dmf;aowur2304u234;l23=-412;'
socketio = SocketIO(app, cors_allowed_origins="*")

def config_logging(exchange):
    if not os.path.isdir(os.path.join(os.environ["LOG_DIR"], exchange)):
        os.makedirs(os.path.join(os.environ["LOG_DIR"], exchange), exist_ok=True)
    curr_time = datetime.now()
    logging.config.fileConfig(
        "logging_config.ini",
        defaults={"logfilename": "logs/{}/bot_{}.log".format(exchange, datetime_to_filename(curr_time))},
    )
    logging.getLogger().setLevel(logging.WARNING)

parser = None
args = None
trade_engine = None
backtest_engine = None
trade_init_flag = False
trade_start_flag = False
backtest_engine_start_flag = False

def set_config(args):
    global trade_engine
    os.environ["DEBUG_DIR"] = "debug"
    os.environ["LOG_DIR"] = "logs"
    config_logging(args.exch)
    if not os.path.isdir(os.environ["DEBUG_DIR"]):
        os.mkdir(os.environ["DEBUG_DIR"])

def get_count(tf, limit):
    tf[-1:] + tf[:-1]
    tm = len(tf[:-1])
    count = limit
    if tf[-1:].upper()== "H":
        count = count * 60
    elif tf[-1:].upper()== "D":
        count = count * 60* 24
    elif tf[-1:].upper()== "W":
        count = count * 60* 24 * 7
    count = count * tm
    return count


@app.route('/api/candlestick', methods=['POST'])
def get_candlestick():
    data = request.json
    starttime = data.get('starttime')
    endtime = data.get('endtime')
    trades = data.get('trades')

    if not starttime or not endtime or not trades:
        return jsonify({"error": "Missing required parameters"}), 400

    try:
        starttime = datetime.fromisoformat(starttime)
        endtime = datetime.fromisoformat(endtime)
    except ValueError:
        return jsonify({"error": "Invalid datetime format"}), 400

    candlestick = create_candlestick(starttime, endtime, trades)
    if candlestick is None:
        return jsonify({"error": "No trades in the specified time range"}), 404

    return jsonify(candlestick)

@app.route('/api/start_trade', methods=['POST'])
def create_candlestick(starttime, endtime, trades):
    filtered_trades = [trade for trade in trades if starttime <= trade['timestamp'] <= endtime]
    if not filtered_trades:
        return None

    open_price = filtered_trades[0]['price']
    close_price = filtered_trades[-1]['price']
    high_price = max(trade['price'] for trade in filtered_trades)
    low_price = min(trade['price'] for trade in filtered_trades)
    volume = sum(trade['quantity'] for trade in filtered_trades)

    return {
        "starttime": starttime,
        "endtime": endtime,
        "open": open_price,
        "close": close_price,
        "high": high_price,
        "low": low_price,
        "volume": volume
    }

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('message', {'data': 'Connected to server'})

@socketio.on('start_backtest')
def handle_start_backtest(data):
    global backtest_engine
    global backtest_engine_start_flag
    from_date = data.get('from_date')
    from_date = datetime.strptime(from_date, "%Y-%m-%dT%H:%M:%S.%fZ")
    from_date = from_date.replace(tzinfo=timezone.utc)
    to_date = data.get('to_date')
    to_date = datetime.strptime(to_date, "%Y-%m-%dT%H:%M:%S.%fZ")
    to_date = to_date.replace(tzinfo=timezone.utc)
    if backtest_engine_start_flag:
        emit('backtest_status', {'error': 'Backtest engine is already running'})
        return
    # Initialize backtest engine)
    backtest_engine = BackTestEngine(args.exch, args.sym_cfg_file, args.data_dir, from_date, to_date)
    backtest_engine_start_flag = True
    backtest_engine.start()
    emit('backtest_status', {'status': 'Backtesting started'})

@socketio.on('stop_backtest')
def handle_stop_backtest():
    global backtest_engine
    global backtest_engine_start_flag
    if backtest_engine_start_flag:
        data = backtest_engine.stop()
        backtest_engine_start_flag = False
        emit('backtest_status', {'status': 'Backtesting stopped', 'data':data})
    else:
        emit('backtest_status', {'error': 'Backtest engine is not running'})

@socketio.on('start_trade')
def handle_start_trade():
    global trade_start_flag
    trade_engine = TradeEngine(args.exch, args.exch_cfg_file, args.sym_cfg_file)
    trade_init_flag = trade_engine.init()
    if trade_init_flag:
        trade_start_flag = True
        trade_engine.start()
        emit('trade_status', {'status': 'Trading engine started'})
    else:
        trade_start_flag = False
        emit('trade_status', {'error': 'Failed to initialize trading engine'})

@socketio.on('stop_trade')
def handle_stop_trade():
    global trade_start_flag
    if trade_start_flag:
        trade_engine.stop()
        trade_start_flag = False
        emit('trade_status', {'status': 'Trading engine stopped'})
    else:
        emit('trade_status', {'error': 'Trading engine is not running'})

@socketio.on('get_trading_history')
def handle_get_trading_history(data):
    from_date = data.get('from_date')
    from_date = datetime.strptime(from_date, "%Y-%m-%dT%H:%M:%S.%fZ")
    from_date = from_date.replace(tzinfo=timezone.utc)
    to_date = data.get('to_date')
    to_date = datetime.strptime(to_date, "%Y-%m-%dT%H:%M:%S.%fZ")
    to_date = to_date.replace(tzinfo=timezone.utc)
    symbol = data.get('symbol', "EURUSD")
    history = trade_engine.log_income_history(from_date, to_date,symbol)
    
    if len(history) > 0:
        history_data = history.to_json(orient="records")
    else:
        history_data = []
    emit('trading_history', history_data)

@socketio.on('get_profit_loss')
def handle_get_profit_loss(data):
    symbol = data.get('symbol', "EURUSD")
    now = datetime.now()
    now = now.replace(tzinfo=timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)
    week_start = today_start - timedelta(days=today_start.weekday())
    month_start = today_start.replace(day=1)
    hour_ago = now - timedelta(hours=1)
    # Initialize counters for yesterday, week, two hours ago, and month
    counters = {
        "today": {"long_trades": 0, "short_trades": 0, "profit": 0, "loss": 0},
        "yesterday": {"long_trades": 0, "short_trades": 0, "profit": 0, "loss": 0},
        "week": {"long_trades": 0, "short_trades": 0, "profit": 0, "loss": 0},
        "hour_ago": {"long_trades": 0, "short_trades": 0, "profit": 0, "loss": 0},
        "month": {"long_trades": 0, "short_trades": 0, "profit": 0, "loss": 0},
    }

    # Fetch trading history for different time ranges
    history_yesterday = trade_engine.log_income_history(symbol, yesterday_start, today_start)
    history_week = trade_engine.log_income_history(symbol, week_start, now)
    history_hour = trade_engine.log_income_history(symbol, hour_ago, now)
    history_month = trade_engine.log_income_history(symbol, month_start, now)

    # Helper function to calculate counters
    def calculate_counters(history, time_key):
        if len(history) > 0:
            counters[time_key]["long_trades"] = len(history.loc[history['type'] == 0])
            counters[time_key]["short_trades"] = len(history.loc[history['type'] == 1])
            counters[time_key]["profit"] = history.loc[history['profit'] > 0, 'profit'].sum()
            counters[time_key]["loss"] = history.loc[history['profit'] < 0, 'profit'].sum()

    # Calculate counters for each time range
    calculate_counters(history_yesterday, "yesterday")
    calculate_counters(history_yesterday, "today")
    calculate_counters(history_week, "week")
    calculate_counters(history_hour, "hour_ago")
    calculate_counters(history_month, "month")

    # Calculate percentages
    for time_key in counters:
        total_trades = counters[time_key]["long_trades"] + counters[time_key]["short_trades"]
        counters[time_key]["profit_percent"] = (
            (counters[time_key]["profit"] / (counters[time_key]["profit"] + abs(counters[time_key]["loss"]))) * 100
            if counters[time_key]["profit"] + abs(counters[time_key]["loss"]) > 0
            else 0
        )
        counters[time_key]["total_trades"] = total_trades
        counters[time_key]["loss_percent"] = 100 - counters[time_key]["profit_percent"]
        if(counters[time_key]["profit"] == 0 and abs(counters[time_key]["loss"]) == 0):
            counters[time_key]["loss_percent"] = 0
    emit('profit_loss_counters', counters)

@socketio.on('get_kline')
def handle_get_kline(data):
    symbol = data.get('symbol','EURUSD')
    interval = data.get('interval', '1M')
    length = data.get('length',0)
    from_date = data.get('from_date')
    from_date = datetime.strptime(from_date, "%Y-%m-%dT%H:%M:%S.%fZ")
    from_date = from_date.replace(tzinfo=timezone.utc)
    to_date = data.get('to_date')
    to_date = datetime.strptime(to_date, "%Y-%m-%dT%H:%M:%S.%fZ")
    to_date = to_date.replace(tzinfo=timezone.utc)
    if length == 0:
        kline = trade_engine.klinesDate(symbol, interval, from_date, to_date)
    else:
        from_date = to_date - timedelta(minutes=get_count(interval, length))
        kline = trade_engine.klinesCount(symbol, interval, from_date, length)
    
    if len(kline) > 0:
        kline_json = kline.to_json(orient="records")
    else:
        kline_json = []
    emit('kline_data', kline_json)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Socket server for trading engine")
    # parser.add_argument("--mode", type=str, default="live", help="Mode of operation (live/backtest)")
    parser.add_argument("--exch", type=str, default="mt5", help="Exchange name")
    parser.add_argument("--exch_cfg_file", type=str, default="configs/exchange_config.json", help="Exchange config file")
    parser.add_argument("--sym_cfg_file", type=str, default="configs/symbols_trading_config.json", help="Symbols trading config file")
    parser.add_argument("--data_dir", type=str, default="D:\\MT5_Data", help="Data directory")
    args = parser.parse_args()
    set_config(args)
    trade_engine = TradeEngine(args.exch, args.exch_cfg_file, args.sym_cfg_file)
    trade_init_flag = trade_engine.init()
    socketio.run(app, host='0.0.0.0', port=5000)    

    # py main.py --mode live --exch mt5 --exch_cfg_file configs/exchange_config.json --sym_cfg_file configs/symbols_trading_config.json
    # py main.py --mode test --exch mt5 --exch_cfg_file configs/exchange_config.json --sym_cfg_file configs/symbols_trading_config.json --data_dir D:\MT5_Data
