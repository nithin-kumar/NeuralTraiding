from app.mod_trades import transaction_api, getmarketsummary, bitcoin_summary, trade_api
from app.mod_neural_training import coin_latest_data, coin_fetch_api
import requests
from datetime import timedelta
import datetime
from datetime import datetime as dtime
from app.mod_neural_training.models import Coin, Parameters, Training
from app.mod_transactions.models import Transaction
import json
import atexit
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from flask import Blueprint, request, jsonify, abort, render_template, session, flash, redirect, url_for
mod_trades = Blueprint('trades', __name__, url_prefix='/trades')
from app import db
from app import app_name, config
from celery import Celery
import numpy as np
import operator
from app import redis_store
from random import shuffle
import time
celery = Celery(app_name, broker=config['CELERY_BROKER_URL'])
celery.conf.update(config)

@mod_trades.route('/monitor', methods=['GET'])
def index():
	market = request.args.get('market')
	coin = Coin.query.filter_by(code=market).first()
	map_ = { 'name' : coin.code }
	return render_template('trades/index.html', coin=map_)


def fetch_transaction_data(market):
	print market, '#######################################################'
	market_transaction_api = transaction_api.replace('MARKET', market)
	resp = requests.get(market_transaction_api)
	response = resp.json()['result']
	buy_vol = buy_price = sell_vol = sell_price = highest_buy_vol = highest_buy_price = highest_sell_vol = highest_sell_price = max_buy_rate = max_sell_rate = max_buy_rate_price = 0
	lowest_buy_vol = lowest_buy_price = lowest_sell_vol = lowest_sell_price = min_buy_rate = min_sell_rate = 9999999999999
	low_vol_coin = 1
	low_vol_coin_min_buy_wall = low_vol_coin_min_sell_wall = 2
	high_vol_coin_min_buy_wall = high_vol_coin_min_sell_wall = 10
	buy_wall_appeared = sell_wall_appeared = 0
	if response is None:
		return -1
	date_time = datetime.datetime.now()
	day_of_month = date_time.day
	time_of_the_day = date_time.hour
	try:
		coin_summary = requests.get(getmarketsummary + market).json()['result'][0]
	except Exception as e:
		return -1
	high = coin_summary['High']
	low = coin_summary['Low']
	volume = coin_summary['Volume']
	last = coin_summary['Last'] # Out put
	base_volume = coin_summary['BaseVolume']
	buy_wall_data = {}
	if base_volume > 100:
		low_vol_coin = 0
	if response['buy'] is not None:
		for i in response['buy']:
			buy_vol += i['Quantity']
			buy_btc_volume = i['Quantity'] * i['Rate']
			if max_buy_rate < buy_btc_volume:
				max_buy_rate = buy_btc_volume
			if min_buy_rate > buy_btc_volume:
				min_buy_rate = buy_btc_volume
				min_buy_rate_price = i['Rate']
			buy_price += buy_btc_volume # BTC volume
			if highest_buy_vol < i['Quantity']:
				highest_buy_vol = i['Quantity']
			if highest_buy_price < i['Rate']:
				highest_buy_price = i['Rate']
			if lowest_buy_vol > i['Quantity']:
				lowest_buy_vol = i['Quantity']
			if lowest_buy_price > i['Rate']:
				lowest_buy_price = i['Rate']
			if low_vol_coin == 1 and buy_btc_volume > low_vol_coin_min_buy_wall:
				buy_wall_appeared = 1
				buy_wall_data[buy_btc_volume] = i['Rate']
			if low_vol_coin == 0 and buy_btc_volume > high_vol_coin_min_buy_wall:
				buy_wall_appeared = 1
				buy_wall_data[buy_btc_volume] = i['Rate']
	sell_wall_data = {}
	if response['sell'] is not None:
		for i in response['sell']:
			sell_vol += i['Quantity']
			sell_btc_volume = i['Quantity'] * i['Rate']
			if max_sell_rate < sell_btc_volume:
				max_sell_rate = sell_btc_volume
				max_sell_rate_price = i['Rate']
			if min_sell_rate > sell_btc_volume:
				min_sell_rate = sell_btc_volume
				min_sell_rate_price = i['Rate']
			sell_price += sell_btc_volume
			if highest_sell_vol < i['Quantity']:
				highest_sell_vol = i['Quantity']
			if highest_sell_price < i['Rate']:
				highest_sell_price = i['Rate']
			if lowest_sell_vol > i['Quantity']:
				lowest_sell_vol = i['Quantity']
			if lowest_sell_price > i['Rate']:
				lowest_sell_price = i['Rate']
			if low_vol_coin == 1 and sell_btc_volume > low_vol_coin_min_sell_wall:
				sell_wall_appeared = 1
				sell_wall_data[sell_btc_volume] = i['Rate']
			if low_vol_coin == 0 and sell_btc_volume > high_vol_coin_min_sell_wall:
				sell_wall_appeared = 1
				sell_wall_data[sell_btc_volume] = i['Rate']
	min_diff_buy_wall_last_rate = 1
	min_diff_buy_wall_last_vol = 1
	for key in buy_wall_data:
		if last - buy_wall_data[key] < min_diff_buy_wall_last_rate:
			min_diff_buy_wall_last_rate = last - buy_wall_data[key]
			min_diff_buy_wall_last_vol = key
	min_diff_sell_wall_last_rate = 1
	min_diff_sell_wall_last_vol = 1
	for key in sell_wall_data:
		if sell_wall_data[key] - last < min_diff_sell_wall_last_rate:
			min_diff_sell_wall_last_rate = sell_wall_data[key] - last
			min_diff_sell_wall_last_vol = key
	buy_kill = min_diff_buy_wall_last_vol / min_diff_sell_wall_last_vol

	sorted_buy_wall_data = sorted(buy_wall_data.items(), key=operator.itemgetter(1), reverse=True)[:3]
	sorted_sell_wall_data = sorted(sell_wall_data.items(), key=operator.itemgetter(1))[:3]
	bid = coin_summary['Bid']
	ask = coin_summary['Ask']
	open_buy_orders = coin_summary['OpenBuyOrders']
	open_sell_orders = coin_summary['OpenSellOrders']
	prev_day = coin_summary['PrevDay']
	resp = requests.get(bitcoin_summary)
	response = resp.json()
	try:
		current_bit_coin_price = response['bpi']['USD']['rate']
	except Exception as e:
		current_bit_coin_price = 8000
	return [[buy_vol, buy_price, sell_vol, sell_price, highest_buy_vol,
			highest_buy_price, highest_sell_vol, highest_sell_price, lowest_buy_vol, lowest_buy_price,
			lowest_sell_vol, lowest_sell_price, high, low, volume,
			base_volume, bid, ask, open_buy_orders, open_sell_orders,
			prev_day, buy_vol - sell_vol, buy_price - sell_price, max_buy_rate, min_buy_rate,
			max_sell_rate, min_sell_rate, low_vol_coin, buy_wall_appeared, sell_wall_appeared,
			current_bit_coin_price, day_of_month, time_of_the_day, min_diff_buy_wall_last_rate, min_diff_buy_wall_last_vol,
			buy_kill, last], sorted_buy_wall_data, sorted_sell_wall_data]

def collect_trade_data():
	resp = requests.get(trade_api)
	response = resp.json()
	response = response['result']
	#shuffle(markets)
	for k in markets:
		market = k['MarketName']
		if market.split('-')[0] == 'BTC':
			row_data = fetch_transaction_data(market)
			if row_data[0] != -1:	
				coin = Coin.query.filter_by(code=market).first()
				transaction = Transaction(coin.id)
				transaction.data = json.dumps(row_data[0])
				transaction.buy_wall_data = json.dumps(row_data[1])
				transaction.sell_wall_data = json.dumps(row_data[2])
				db.session.add(transaction)
				try:
					db.session.commit()
				except:
					db.session.rollback()
					raise
				finally:
					db.session.close()
				redis_store.set(market, json.dumps(row_data))



@celery.task(bind=True)
def data_collector_wrapper(self):
	i = 0
	while i <= 10000:
		collect_trade_data()
		i += 1

@mod_trades.route('/api/v1.0/collect_trade_data/', methods=['GET', 'POST'])
def collect_trade_data():
	task = data_collector_wrapper.apply_async()
	#data_collector_wrapper()
	#scheduler = BackgroundScheduler()
	#scheduler.start()
	#scheduler.add_job(
    #	func=collect_data,
    #	trigger=IntervalTrigger(seconds=300),
    #	id='data_collection_job',
    #	name='Collect data from bittrex',
    #	replace_existing=True)
	# Shut down the scheduler when exiting the app
	#atexit.register(lambda: scheduler.shutdown())
	return jsonify({'message': 'Data Collection started!'}), 200

def prepare_data():
	data_matrix = []
	out_matrix = []
	for transaction in transactions:
		row = json.loads(transaction.data)
		out_matrix.append(row.pop(-1))
@celery.task(bind=True)
def train_neural_network(self, market, training_mode):
    coin = Coin.query.filter_by(code=market).first()
    transactions = Transaction.query.filter_by(coin_id=coin.id)
    prepare_data(transactions)
    
@mod_trades.route('/api/v1.0/train/', methods=['POST'])
def train():
    market = request.json['market']
    training_mode = request.json['training_mode']
    task = train_neural_network.apply_async(args=[market, training_mode])
    return jsonify({'Location': '/transaction/api/v1.0/status/' + task.id }), 202

@mod_trades.route('/api/v1.0/update_transactions/', methods=['POST'])
def update_transactions():
	markets = request.json['markets']
	map_ = {'status' : 1}
	for market in markets:
		#data = fetch_transaction_data(market)
		#coin = Coin.query.filter_by(code=market).first()
		#data = Transaction.query.filter_by(coin_id=coin.id).order_by('-id').first()
		data = redis_store.get(market)
		if data is not None:
			tmp = json.loads(data)
			data = tmp[0]
			buy_wall_data = tmp[1]
			sell_wall_data = tmp[2]
			low_high_vol = 'LOW('+ str(round(data[15], 3)) +')' if data[27] == 1 else 'HIGH(' + str(round(data[15], 3)) + ')'
			buy_wall_appeared = 'YES' if data[28] == 1 else 'NO'
			sell_wall_appeared = 'YES' if data[29] == 1 else 'NO'
			if len(buy_wall_data) != 0 and len(sell_wall_data) != 0:
				buy_sell_percentage = str((buy_wall_data[0][0]/sell_wall_data[0][0]))
			else:
				buy_sell_percentage = 0
			if len(buy_wall_data) != 0:
				closest_buy_wall = buy_wall_data[0][0]
			else:
				closest_buy_wall = 'NA'
			if len(sell_wall_data) != 0:
				closest_sell_wall = sell_wall_data[0][0]
			else:
				closest_sell_wall = 'NA'
			map_[market] = [low_high_vol,  closest_buy_wall, closest_sell_wall, buy_wall_appeared, sell_wall_appeared, data[36], buy_wall_data, sell_wall_data,  buy_sell_percentage, data[12], data[20]]
	return jsonify(map_), 200








