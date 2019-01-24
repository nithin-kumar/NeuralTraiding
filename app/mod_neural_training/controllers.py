"""Authentication Module Controllers."""
from flask import Blueprint, request, jsonify, abort, render_template, session, flash, redirect, url_for
import requests
from app.mod_neural_training import coin_fetch_api, coin_data, coin_latest_data, bittrex_commission, coin_page
import csv
from celery import Celery
import json
import random
import time
import numpy as np
import calendar
from datetime import timedelta
import datetime
from datetime import datetime as dtime
from app import redis_store
# Import password / encryption helper tools

# Import the database object from the main app module
from app import db
from app import app_name, config
# Import module models (i.e. User)
from app.mod_neural_training.models import Coin, Parameters, Training
from app.lib.neural_training import NeuralTraining
from app.lib.utils import Utils
# Define the blueprint: 'auth', set its url prefix: app.url/auth
mod_neural_training = Blueprint('neural_training', __name__, url_prefix='/neural')
celery = Celery(app_name, broker=config['CELERY_BROKER_URL'])
celery.conf.update(config)

@mod_neural_training.route('/', methods=['GET', 'POST'])
def index():
    coins = Coin.query.all()
    map_ = {}
    for coin in coins:
        parameters = Parameters.query.filter_by(coin_id=coin.id)
        map_[coin.code] = {'code' : coin.code, 'url' : coin.url, 'error': []}
        for parameter in parameters:
            tmp = [parameter.granularity.encode("ascii","replace"), str(parameter.error)]
            map_[coin.code]['error'].append(tmp)
    #for parameter in parameters:
    #    map_[parameter.coin_id] = parameter.error
    return render_template('neural_training/index.html', coins=map_)
# Onboard Bus
@mod_neural_training.route('/api/v1.0/get_coins/', methods=['GET'])
def get_coins():
    resp = requests.get(coin_fetch_api)
    response = resp.json()
    markets = []
    for i in response['result']:
        if i['MarketName'].split('-')[0] != 'BTC':
            continue
        markets.append(i['MarketName'])
        coin = Coin.query.filter_by(code=i['MarketName']).first()
        if coin is None:
            coin_url = coin_page + i['MarketName']
            coin = Coin(name=i['MarketCurrencyLong'])
            coin.code = i['MarketName']
            coin.url = coin_url
            db.session.add(coin)
            db.session.commit()
    return jsonify({'message': 'Coin base updated successfully!'}), 201

def nonlin(x,deriv=False):
        if(deriv == True):
            return x*(1-x)
        return 1/(1+np.exp(-x))

@celery.task(bind=True)
def train_neural_network(self, market, granularity, training_mode):
    coin_data_api = coin_data.replace('MARKET', market)
    coin_data_api = coin_data_api.replace('INTERVAL', granularity)
    resp = requests.get(coin_data_api)
    response = resp.json()
    neural_obj = NeuralTraining(response, market, granularity, training_mode)
    data = neural_obj.data
    training_count = 400
    report_count = 10000
    if data == -1:
        return {'current': training_count, 'total': training_count, 'status': 'No New data!!'}
    
    X = np.array(data[0])
    y = np.array([data[1]]).T
    print X.shape, "@@@@@@@@@@@@@@@@", y.shape
    np.random.seed(1)
    coin = Coin.query.filter_by(code=market).first()
    parameter = Parameters.query.filter_by(coin_id=coin.id, granularity=granularity).first()
    if parameter is None:
        # randomly initialize our weights with mean 0
        parameter = Parameters(coin.id, granularity, 'Summary')
        try:
            syn0 = 2*np.random.random((len(data[0][0]), len(data[0]))) - 1
            syn1 = 2*np.random.random((len(data[0]),1)) - 1
        except Exception as e:
            print "Exception for " + market
            return -1
    else:
        print "Hereeeeee........................"
        syn0 = np.array(json.loads(parameter.syn0))
        syn1 = np.array(json.loads(parameter.syn1))
    for j in xrange(training_count):
        # Feed forward through layers 0, 1, and 2
        l0 = X
        l1 = nonlin(np.dot(l0,syn0))
        print l1.shape, "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
        l2 = nonlin(np.dot(l1,syn1))
        # how much did we miss the target value?
        l2_error = y - l2
        if (j % report_count) == 0:
            self.update_state(state='PROGRESS',
                          meta={'current': j, 'total': training_count,
                                'status': 'Training....'})
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
    parameter.syn0 = json.dumps(syn0.tolist())
    parameter.syn1 = json.dumps(syn1.tolist())
    parameter.min_max_parameters = json.dumps(data[2])
    parameter.error = mean
    db.session.add(parameter)
    db.session.commit()
    return {'current': training_count, 'total': training_count, 'status': 'Training completed with error '+str(mean),
            'network_error': mean}

@mod_neural_training.route('/api/v1.0/train/', methods=['POST'])
def train():
    market = request.json['market']
    granularity = request.json['granularity']
    training_mode = request.json['training_mode']
    task = train_neural_network.apply_async(args=[market, granularity, training_mode])
    return jsonify({'Location': '/neural/api/v1.0/status/' + task.id }), 202

@mod_neural_training.route('/api/v1.0/get_network_error', methods=['GET'])
def get_network_error():
    market = request.args.get('market')
    coin = Coin.query.filter_by(code=market).first()
    parameters = Parameters.query.filter_by(coin_id=coin.id)
    map_ = {}
    for parameter in parameters:
        map_[parameter.granularity] = str(parameter.error)
    return jsonify({'parameters': map_}), 200

@mod_neural_training.route('/api/v1.0/predict', methods=['GET'])
def predict():
    market = request.args.get('market')
    granularity = request.args.get('granularity')
    coin = Coin.query.filter_by(code=market).first()
    parameter = Parameters.query.filter_by(coin_id=coin.id, granularity=granularity).first()
    if parameter is None:
        return jsonify({'message': 'Market is not trained to predict the value'}), 200
    
    coin_data_api = coin_data.replace('MARKET', market)
    coin_data_api = coin_data_api.replace('INTERVAL', granularity)
    resp = requests.get(coin_data_api)
    last_value = resp.json()['result'][-1]
    source_time_str = last_value['T']
    source_time = datetime.datetime.strptime(source_time_str, "%Y-%m-%dT%H:%M:%S")
    ist_time = Utils.utc_to_local(source_time) + timedelta(minutes = 30)
    that_day = datetime.datetime.combine(source_time.date(), datetime.time(0))
    diff  = source_time - that_day
    diff_in_minutes = (diff.seconds / 3600*60)
    min_max_parameters = json.loads(parameter.min_max_parameters)
    # Prepare input to predict
    l0 = [(last_value['O'] - min_max_parameters['min_o']) if (last_value['O'] - min_max_parameters['min_o']) != 0 else min_max_parameters['min_o'] / (min_max_parameters['max_o'] - min_max_parameters['min_o']) if min_max_parameters['max_o'] - min_max_parameters['min_o'] != 0 else min_max_parameters['min_o'],
          (last_value['C'] - min_max_parameters['min_c']) if (last_value['C'] - min_max_parameters['min_c']) != 0 else min_max_parameters['min_c'] / (min_max_parameters['max_c'] - min_max_parameters['min_c']) if min_max_parameters['max_c'] - min_max_parameters['min_c'] != 0 else min_max_parameters['min_c'],
          (last_value['L'] - min_max_parameters['min_l']) if (last_value['L'] - min_max_parameters['min_l']) != 0 else min_max_parameters['min_l'] / (min_max_parameters['max_l'] - min_max_parameters['min_l']) if min_max_parameters['max_l'] - min_max_parameters['min_l'] != 0 else min_max_parameters['min_l'],
          (last_value['V'] - min_max_parameters['min_v']) if (last_value['V'] - min_max_parameters['min_v']) != 0 else min_max_parameters['min_v'] / (min_max_parameters['max_v'] - min_max_parameters['min_v']) if min_max_parameters['max_v'] - min_max_parameters['min_v'] != 0 else min_max_parameters['min_v'],
          (last_value['BV'] - min_max_parameters['min_bv']) if (last_value['BV'] - min_max_parameters['min_bv']) != 0 else min_max_parameters['min_bv'] / (min_max_parameters['max_bv'] - min_max_parameters['min_bv']) if min_max_parameters['max_bv'] - min_max_parameters['min_bv'] != 0 else min_max_parameters['min_bv'],
          diff_in_minutes]
    
    #### Predcit using trained network ########
    syn0 = np.array(json.loads(parameter.syn0))
    syn1 = np.array(json.loads(parameter.syn1))
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    # Denormalise the prediction value
    prediction = (l2[0] * (min_max_parameters['max_'] - min_max_parameters['min_']) if (min_max_parameters['max_'] - min_max_parameters['min_']) != 0 else min_max_parameters['min_']) + min_max_parameters['min_']
    
    #### Calculate the profit ################
    resp = requests.get(coin_latest_data + market )
    current_data = resp.json()
    current_price = current_data['result']['Ask']
    buy_price = current_price + current_price * (bittrex_commission/2)
    profit = ((prediction - buy_price)/ buy_price) * 100
    if profit < 0:
        result = 'LOSS'
    else:
        result = 'PROFIT'
    map_ = {'current_price' : current_price, 'buy_price' : buy_price, 'predicted_price' : prediction, 'profit' : profit, 'status' : result, 'time' : ist_time }
    return jsonify({'parameters': map_}), 200


@mod_neural_training.route('/api/v1.0/status/<task_id>')
def taskstatus(task_id):
    task = train_neural_network.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)


