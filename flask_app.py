import os
import time
from datetime import datetime
import argparse
import logging
import logging.config
from trade_engine import TradeEngine
from backtest import BackTest
from utils import datetime_to_filename
from flask_marshmallow import Marshmallow

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/trading'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

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

def user():
    user = User.query.all().first()
    return user

@app.route('/')
def home():
    return jsonify({"message": "success"}), 200

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

    if not bcrypt.check_password_hash(user.password, password):
        return jsonify({"message": "Invalid password!"}), 401

    return jsonify({"message": "Sign-in successful!","data":user}), 200

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

@app.route('/get_trade_history')
def get_trade_history():
    trade_engine = TradeEngine("live", user(), strategies())
    if trade_engine.init():
        from_date=datetime(2025,1,1)
        to_date=datetime.now()
        data = trade_engine.log_income_history(from_date, to_date)
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
    else:
        return jsonify({"message":"error"}), 400
@app.route('/')
def trade_engin_start():
    str
    
if __name__ == '__main__':
    app.run(debug=True)