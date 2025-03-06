import os
from datetime import datetime, timedelta
import argparse
import logging
import logging.config
import time
from trade_engine import TradeEngine
from backtest import BackTest
from utils import datetime_to_filename
from flask_marshmallow import Marshmallow
import json

from flask import Flask, render_template, redirect, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_session import Session 


app = Flask(__name__)
CORS(app)
# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/trading'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = "sdlfjowehfwler2339423!*!l4"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app) 

db = SQLAlchemy(app)
ma = Marshmallow(app)
bcrypt = Bcrypt(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    account = db.Column(db.String(80), unique=True, nullable=False)
    server = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)

class Strategies(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(80), nullable=False)
    min_num_cuml = db.Column(db.Float, nullable=False)
    min_zz_pct = db.Column(db.Float, nullable=False)
    zz_dev = db.Column(db.Float, nullable=False)
    ma_vol = db.Column(db.Float, nullable=False)
    vol_ratio_ma = db.Column(db.Float, nullable=False)
    kline_body = db.Column(db.Float, nullable=False)
    sl_fix_mode = db.Column(db.String(255), nullable=False)
    tf = db.Column(db.String(255), nullable=False)
    max_sl_pct = db.Column(db.Float, nullable=False)
    volume = db.Column(db.Float, nullable=False)
    months = db.Column(db.String(128), nullable=False)
    year = db.Column(db.Integer, nullable=False)

class StrategySchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Strategies
class UserSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = User

parser = argparse.ArgumentParser(description="Monn auto trading bot")
# parser.add_argument("--mode", required=True, type=str, choices=["live", "test"])
# parser.add_argument("--exch", required=True, type=str)
# parser.add_argument("--data_dir", required=False, type=str)
# parser.add_argument("--exch_cfg_file", required=True, type=str)
# parser.add_argument("--sym_cfg_file", required=True, type=str)
# args = parser.parse_args()

os.environ["DEBUG_DIR"] = "debug"
os.environ["LOG_DIR"] = "logs"
# config_logging(args.exch)

if not os.path.isdir(os.environ["DEBUG_DIR"]):
    os.mkdir(os.environ["DEBUG_DIR"])
                        
# Create the database
with app.app_context():
    db.create_all()

# def get_trade_history():
#     """Fetch and display trade history."""
#     history = mt5.history_deals_get()
#     df = pd.DataFrame(list(history), columns=['symbol', 'type', 'price', 'volume', 'profit'])
#     return df.to_dict(orient='records')

# @app.route('/')
# def home():
#     history = get_trade_history()
#     print(history)
#     return render_template('index.html', trades=history)

trade_engine = None

def strategies():
    strategies = Strategies.query.all()
    strategy_list = []
    for strategy in strategies:
        strategy_list.append({
            "symbol": "USDJPY",
            "strategies": [
                {
                    "name": "break_strategy",
                    "params": {
                        "min_num_cuml": strategy.min_num_cuml,
                        "min_zz_pct": strategy.min_zz_pct,
                        "zz_dev": strategy.zz_dev,
                        "ma_vol": strategy.ma_vol,
                        "vol_ratio_ma": strategy.vol_ratio_ma,
                        "kline_body_ratio": strategy.kline_body,
                        "sl_fix_mode": strategy.sl_fix_mode
                    },
                    "tfs": {
                        "tf": strategy.tf
                    },
                    "max_sl_pct": strategy.max_sl_pct,
                    "volume": strategy.volume
                }
            ],
            "months": [
                strategy.months
            ],
            "year": strategy.year
        })
    return strategy_list

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

@app.route('/')
def home():
    print("home")
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    account = data.get('account')
    server = data.get('server')
    password = data.get('password')

    if not account or not password or not server:
        return jsonify({"message": "All fields are required!"}), 400
    existing_user = User.query.filter_by(account=account).first()
    if existing_user:
        return jsonify({"message": "Account already exists!"}), 400
    
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

    new_user = User(account=account, server=server, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "User  registered successfully!"}), 201


@app.route('/signin', methods=['POST'])
def signin():
    data = request.get_json()
    account = data.get('account')
    password = data.get('password')

    if not account or not password:
        return jsonify({"message": "Account and password are required!"}), 400

    user = User.query.filter_by(account=account).first()
    if not user:
        return jsonify({"message": "User  not found!"}), 404

    User_schema = UserSchema(many=False) 
    session["user"] = User_schema.dump(user)

    if not bcrypt.check_password_hash(user.password, password):
        return jsonify({"message": "Invalid password!"}), 401
    
    user_data = User.query.filter_by(account=user.account).first()
    exc_cfg = {
        "mt5": {
            "account": int(user_data.account),
            "server": user_data.server
        }
    }
    global trade_engine
    trade_engine = TradeEngine("mt5", exc_cfg, strategies())
    
    if trade_engine.init():
        return jsonify({"message":"success","data":User_schema.dump(data)}), 200
    else:
        return jsonify({"message":"error"}), 400
   
@app.route('/get_strategies', methods=['POST'])
def get_strategies():
    data = Strategies.query.all()
    strategy_schema = StrategySchema(many=True) 
    return jsonify({"message":"success","data":strategy_schema.dump(data)}), 200

@app.route('/get_strategy', methods=['POST'])
def get_strategy():
    no = request.get_json().get('no')
    data = Strategies.query.filter_by(id=no).first()
    strategy_schema = StrategySchema(many=False) 
    return jsonify({"message":"success","data":strategy_schema.dump(data)}), 200

@app.route('/update_strategy', methods=['POST'])
def update_strategy():
    no = request.get_json().get('id')
    req_data = request.get_json()
    strategy = Strategies.query.filter_by(id=no).first()
    if not strategy:
        return jsonify({"message": "Strategy not found"}), 404
    strategy.symbol = req_data['symbol']
    strategy.min_num_cuml = req_data['min_num_cuml']
    strategy.min_zz_pct = req_data['min_zz_pct']
    strategy.zz_dev = req_data['zz_dev']
    strategy.ma_vol = req_data['ma_vol']
    strategy.vol_ratio_ma = req_data['vol_ratio_ma']
    strategy.kline_body = req_data['kline_body']
    strategy.sl_fix_mode = req_data['sl_fix_mode']
    strategy.tf = req_data['tf']
    strategy.max_sl_pct = req_data['max_sl_pct']
    strategy.volume = req_data['volume']
    strategy.months = req_data['months']
    strategy.year = req_data['year']
    db.session.commit()
    data = Strategies.query.all()
    strategy_schema = StrategySchema(many=True) 
    return jsonify({"message":"success","data":strategy_schema.dump(data)}), 200

@app.route('/get_mt5_data')
def get_mt5_data():
    # user_data = session.get("user")
    # print(user_data)
    # if not session.get('user'):
    #     return jsonify({"message":"Authentication Error!"}), 401
    # user_data = User.query.filter_by(account="90610739").first()
    
    to_date = datetime.now()
    tf = "1m"
    count = 75
    
    from_date = datetime.now() - timedelta(minutes=get_count(tf, count))
    history = trade_engine.log_income_history("EURUSD", datetime(2022,1,1), to_date)
    
    # total_profit = total_history["profit"].sum()
    # today = (datetime.now()+timedelta(seconds=-time.timezone)).date()
    # today_profit = total_history[total_history["time"].dt.date == today]["profit"].sum()

    # first_day_of_month = datetime(today.year, today.month, 1)
    # month_profit = total_history[total_history["time"] >= first_day_of_month]["profit"].sum()


    kline = trade_engine.klinesCount("EURUSD", tf, to_date, count)
    history_data = [] 
    kline_data = []
    if len(history) > 0:
        history_data = history.to_json(orient="records")
    if len(kline) > 0:
        kline_data = kline.to_json(orient="records")

    data = {
        "kline" : kline_data,
        "history" : history_data,
        "current_time" : datetime.now(),
    }
    return jsonify({"message":"success","data":data}), 200
    # trade_engine.start()
    # try:
    #     while True:
    #         time.sleep(1)
    # except (KeyboardInterrupt, SystemExit):
    #     trade_engine.stop()
    #     trade_engine.summary_trade_result()
    #     trade_engine.log_all_trades()
    #     time.sleep(3)  # Wait for exchange return income
    #     trade_engine.log_income_history()

@app.route('/trade_start', methods=['POST'])
def trade_start():
    trade_engine.start()
    while True:
        time.sleep(1)
    return jsonify({"message":"success"}), 200

if __name__ == '__main__':
    trade_engine = TradeEngine("mt5", {}, {})
    app.run(debug=True)