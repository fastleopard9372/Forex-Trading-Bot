from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from datetime import datetime
import os
import time
from datetime import datetime
import argparse
import logging
import logging.config
from trade_engine import TradeEngine
from backtest import BackTest
from utils import datetime_to_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dmf;aowur2304u234;l23=-412;'
socketio = SocketIO(app, cors_allowed_origins="*")

# Sample data for trading history and profit/loss
trading_history = [
    {"id": 1, "symbol": "AAPL", "action": "buy", "quantity": 10, "price": 150},
    {"id": 2, "symbol": "GOOGL", "action": "sell", "quantity": 5, "price": 2800}
]

profit_loss = {
    "total_profit": 500,
    "total_loss": 200
}
# Function to create a candlestick from starttime to endtime
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

# Example usage
@app.route('/api/trading-history', methods=['GET'])
def get_trading_history():
    return jsonify(trading_history)

@app.route('/api/profit-loss', methods=['GET'])
def get_profit_loss():
    return jsonify(profit_loss)

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('message', {'data': 'Connected to server'})

@socketio.on('get_trading_history')
def handle_get_trading_history():
    emit('trading_history', trading_history)

@socketio.on('get_profit_loss')
def handle_get_profit_loss():
    emit('profit_loss', profit_loss)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)