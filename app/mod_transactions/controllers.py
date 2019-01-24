from app.mod_transactions import transaction_api, getmarketsummary, bitcoin_summary
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
mod_transactions = Blueprint('transaction', __name__, url_prefix='/transaction')
from app import db
from app import app_name, config

from celery import Celery
from celery.decorators import periodic_task
import numpy as np
import operator
from app import redis_store
from random import shuffle
import time
import matplotlib.pyplot as plt
from app.lib.utils import Utils
#import matplotlib.animation as animation
#from matplotlib import style
#style.use('fivethirtyeight')
#fig = plt.figure()
#ax1 = fig.add_subplot(2,1,1)
#ax2 = fig.add_subplot(2,1,1)
#plt.hold(True)
current_market = 'BTC-NXS'
celery = Celery(app_name, broker=config['CELERY_BROKER_URL'])
celery.conf.update(config)

@periodic_task(run_every=timedelta(seconds=10))
def every_2_seconds():
	fetch_transaction_data(current_market)

@periodic_task(run_every=timedelta(seconds=180))
def every_180_seconds():
	print "######  Prediction ######"
	coin = Coin.query.filter_by(code=current_market).first()
	parameter = Parameters.query.filter_by(coin_id=coin.id, training_mode='transactions').first()
	if parameter is None:
		print "###### No trained models.. ######"
		return
	syn0 = np.array(json.loads(parameter.syn0))
	syn1 = np.array(json.loads(parameter.syn1))
	transactions = Transaction.query.filter_by(coin_id=coin.id).order_by('-id').limit(300)
	x, y, x_orig, y_orig = prepare_data(transactions, coin.id)
	predict_value(current_market, syn0, syn1, x_orig, y_orig)

@periodic_task(run_every=timedelta(seconds=1200))
def every_1200_seconds():
	syn0, syn1, x_orig, y_orig = train_neural_network(current_market)
	#print "######  Prediction ######"
	#if syn0.shape[0] == 0:
	#	coin = Coin.query.filter_by(code=current_market).first()
	#	parameter = Parameters.query.filter_by(coin_id=coin.id, training_mode='transactions').first()
	#	syn0 = np.array(json.loads(parameter.syn0))
	#	syn1 = np.array(json.loads(parameter.syn1))
	#predict_value(current_market, syn0, syn1, x_orig, y_orig)

@mod_transactions.route('/', methods=['GET', 'POST'])
def index():
    coins = Coin.query.all()
    map_ = {}
    coin_arr =[]
    for coin in coins:
    	if coin.code.split('-')[0] != 'BTC':
    		continue
    	coin_arr.append(coin.code.encode("ascii","replace"))
        parameter = Parameters.query.filter_by(coin_id=coin.id, training_mode='transactions').first()
        map_[coin.code.encode("ascii","replace")] = {'code' : coin.code.encode("ascii","replace"), 'url' : coin.url.encode("ascii","replace"), 'error': 0}
        if parameter is not None:
        	map_[coin.code]['error'] = parameter.error
    return render_template('transactions/index.html', coins=map_, coin_arr=coin_arr)


def get_bittrex_data(market):
	market_transaction_api = transaction_api.replace('MARKET', market)
	resp = requests.get(market_transaction_api)
	response = resp.json()['result']
	coin_summary = requests.get(getmarketsummary + market).json()['result'][0]
	return response, coin_summary, requests.get(bitcoin_summary)

def preprocess_data(response, coin_summary, bitcoin_summary):
	high = coin_summary['High']
	low = coin_summary['Low']
	volume = coin_summary['Volume']
	last = coin_summary['Last'] # Out put
	base_volume = coin_summary['BaseVolume']
	threshold = 40
	lower_boundary = last - threshold
	upper_boundary = last + threshold

	buy_orders = sell_orders = buy_volume = sell_volume = max_buy_rate = max_sell_rate = 0
	buy_vol_rate = sell_vol_rate = 0
	buy_avg_distance = sell_avg_distance = 0
	if response['buy'] is not None:
		for i in response['buy']:
			if lower_boundary < i['Rate']:
				buy_orders += 1
				buy_volume += i['Quantity']
				buy_vol_rate += i['Quantity'] * i['Rate']
				if i['Quantity'] * i['Rate'] > max_buy_rate:
					max_buy_rate = i['Quantity'] * i['Rate']
				buy_avg_distance += abs(last - i['Rate'])
		buy_avg_distance /= float(buy_orders)
	if response['sell'] is not None:
		for i in response['sell']:
			if upper_boundary > i['Rate']:
				sell_orders += 1
				sell_volume += i['Quantity']
				sell_vol_rate += i['Quantity'] * i['Rate']
				if i['Quantity'] * i['Rate'] > max_sell_rate:
					max_sell_rate = i['Quantity'] * i['Rate']
				sell_avg_distance += abs(last - i['Rate'])
		sell_avg_distance /= float(sell_orders)
	resp = bitcoin_summary
	response = resp.json()
	current_bit_coin_price = response['bpi']['USD']['rate']
	current_bit_coin_price = float(current_bit_coin_price.replace(',', ''))
	# Computed all data
	# Prepare the data to store
	# high[0], low[1], volume[2], base_volume[3], buy_orders[4], sell_orders[5], buy_volume[6], sell_volume[7], max_buy_rate[8], max_sell_rate[9]
	# buy_vol_rate[10], sell_vol_rate[11], current_bit_coin_price[12], buy_avg_distance[13], sell_avg_distance[14], last[15]
	data = [high, low, volume, base_volume, buy_orders, sell_orders, buy_volume,
			sell_volume, max_buy_rate, max_sell_rate, buy_vol_rate, sell_vol_rate, current_bit_coin_price, buy_avg_distance, sell_avg_distance, last]
	return data
def fetch_transaction_data(market):
	response, coin_summary, bitcoin_summary = get_bittrex_data(market)
	data = preprocess_data(response, coin_summary, bitcoin_summary)
	coin = Coin.query.filter_by(code=market).first()
	transaction = Transaction(coin.id)
	transaction.data = json.dumps(data)
	db.session.add(transaction)
	try:
		db.session.commit()
	except:
		db.session.rollback()
		raise
	finally:
		db.session.close()

def prepare_data(transactions, coin_id):
	X =  []
	Y = []
	x_back = []
	y_back = []
	count = 0
	transactions_array = []
	for transaction in transactions:
		transactions_array.append(json.loads(transaction.data))
	transactions_array = transactions_array[::-1]
	training_history = Training.query.filter_by(coin_id=coin_id, granularity='trading').first()
	flag = False
	for i in range(800 - 300):
		transactions_array[i].pop()
		y_elem = transactions_array[i + 300][-1]
		x_back.append(transactions_array[i])
		y_back.append([y_elem])
		if training_history is None:
			flag = True
			X.append(transactions_array[i])
			Y.append([y_elem])
		else:
			if transactions_array[i]['date_created'] >= training_history.last_trained_timestamp:
				flag = True
				X.append(transactions_array[i])
				Y.append([transactions_array[i + 300].pop()])
	# Normalize the data
	if not flag:
		return np.array([]), np.array([]), np.array(x_back), np.array(y_back)
	x = np.array(X)
	y = np.array(Y)
	x_normed = x / x.max(axis=0)
	y_normed = y / y.max(axis=0)
	return x_normed, y_normed, x, y

def nonlin(x,deriv=False):
        if(deriv == True):
            return x*(1-x)
        return 1/(1+np.exp(-x))


def train_neural_network(market):
    coin = Coin.query.filter_by(code=market).first()
    transaction_count = Transaction.query.filter_by(coin_id=coin.id).order_by('-id').count()
    if transaction_count < 800:
    	print "###### Data is not sufficient for training ######"
    	return np.array([]), np.array([]), np.array([]), np.array([])
    transactions = Transaction.query.filter_by(coin_id=coin.id).order_by('-id').limit(800)
    x, y, x_orig, y_orig = prepare_data(transactions, coin.id)
    if x.shape[0] == 0:
    	print "###### No new data added to training set ######"
    	return np.array([]), np.array([]), x_orig, y_orig
    np.random.seed(1)
    coin = Coin.query.filter_by(code=market).first()
    parameter = Parameters.query.filter_by(coin_id=coin.id, training_mode='transactions').first()


    if parameter is None:
    	# randomly initialize our weights with mean 0
    	parameter = Parameters(coin.id, '5Min', 'transactions')
    	syn0 = 2*np.random.random((x.shape[1], x.shape[0])) - 1
    	syn1 = 2*np.random.random((x.shape[0],1)) - 1
    else:
        syn0 = np.array(json.loads(parameter.syn0))
        syn1 = np.array(json.loads(parameter.syn1))

    # Training Begind
    training_count = 3000000
    print "##################################### Training Started ######################################"
    for j in xrange(training_count):
        # Feed forward through layers 0, 1, and 2
        l0 = x
        l1 = nonlin(np.dot(l0,syn0))
        l2 = nonlin(np.dot(l1,syn1))
        # how much did we miss the target value?
        l2_error = y - l2
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        l2_delta = l2_error*nonlin(l2,deriv=True)
        # how much did each l1 value contribute to the l2 error (according to the weights)?
        l1_error = l2_delta.dot(syn1.T)
        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        l1_delta = l1_error * nonlin(l1,deriv=True)
        syn1 += l1.T.dot(l2_delta)
        syn0 += l0.T.dot(l1_delta)
    l2_error_list = l2_error.tolist() 
    abs_error_list = []
    for x in l2_error_list:
        abs_error_list.append(abs(x[0])) 
    mean = sum(abs_error_list) /  len(abs_error_list)
    print '###### ERROR ######  ' + str(mean)
    parameter.syn0 = json.dumps(syn0.tolist())
    parameter.syn1 = json.dumps(syn1.tolist())
    parameter.error = mean
    db.session.add(parameter)
    db.session.commit()
    training_history = Training.query.filter_by(coin_id=coin.id, granularity='trading').first()
    if training_history is None:
    	training_history = Training(coin.id, datetime.datetime.now(), 'trading')
    else:
    	training_history.last_trained_timestamp = datetime.datetime.now()
	db.session.add(training_history)
	db.session.commit()
    return syn0, syn1, x_orig, y_orig
def predict_value(market, syn0, syn1, x_orig, y_orig):
	response, coin_summary, bitcoin_summary = get_bittrex_data(market)
	data = preprocess_data(response, coin_summary, bitcoin_summary)
	data.pop()
	x_normed = data / x_orig.max(axis=0)
	l1 = nonlin(np.dot(x_normed, syn0))
	l2 = nonlin(np.dot(l1, syn1))
	prediction = l2[0] * y_orig.max(axis=0)
	current_time = datetime.datetime.now()
	source_time = datetime.datetime.strptime(coin_summary['TimeStamp'], "%Y-%m-%dT%H:%M:%S.%f")
	predicted_for = Utils.utc_to_local(source_time) + timedelta(minutes = 3)
	#ax1.plot([predicted_for], [prediction])
	#plt.show()
	print "###### Predicted for ######  " + str(predicted_for)
	print "###### Predicted price ######  " + str(prediction)
	print "###### Last Price ######  " + repr(coin_summary['Last'])
	print "######  Buy Price ######  " + str(coin_summary['Last'] + coin_summary['Last'] * 0.005)
	print "\n\n\n\n\n"





