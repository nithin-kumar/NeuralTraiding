
import numpy as np
import os
import random
import time
import datetime
from flask import Flask, request, render_template, session, flash, redirect, \
    url_for, jsonify
from celery import Celery
from app.mod_neural_training.models import Coin, Training
from app import db
class NeuralTraining(object):
	"""docstring for NeuralTraining"""
	def __init__(self, data, market, granularity, training_mode):
		super(NeuralTraining, self).__init__()
		self.market = market
		self.granularity = granularity
		self.training_mode = training_mode
		self.data = self.prepare_data(data)
		#[X, Y, min_max]


	def prepare_data(self, data):
		X = []
		Y = []
		############################# Data Normalization ############################
		max_  = max_o = max_h = max_l = max_bv = max_v = max_c = 0
		min_ = min_o = min_h = min_l = min_bv = min_v = min_c = 999999999
		data = data['result'][-21:]
		if len(data) >= 20:
			iterations = 20
		else:
			iterations = len(data)
		coin = Coin.query.filter_by(code=self.market).first()
		training_history = Training.query.filter_by(coin_id=coin.id, granularity=self.granularity).first()
		flag = False
		for i in range (iterations):
			result = data[i]
			source_time_str = result['T']
			source_time = datetime.datetime.strptime(source_time_str, "%Y-%m-%dT%H:%M:%S")
			if self.training_mode == 'online_training' and training_history is not None and source_time <= training_history.last_trained_timestamp:
				continue
			flag = True
			if result['O'] > max_o:
				max_o = result['O']
			if result['O'] < min_o:
				min_o = result['O']

			if result['C'] > max_c:
				max_c = result['C']
			if result['C'] < min_c:
				min_c = result['C']

			if result['L'] > max_l:
				max_l = result['L']
			if result['L'] < min_l:
				min_l = result['L']

			if result['BV'] > max_bv:
				max_bv = result['BV']
			if result['BV'] < min_bv:
				min_bv = result['BV']

			if result['V'] > max_v:
				max_v = result['V']
			if result['V'] < min_v:
				min_v = result['V']

			if result['H'] > max_:
				max_ = result['H']
			if result['H'] < min_:
				min_ = result['H']
		min_max = {'min_o' : min_o, 'max_o': max_o, 'min_c' : min_c, 'max_c': max_c,
				   'min_l' : min_l, 'max_l': max_l, 'min_v' : min_v, 'max_v': max_v,
				   'min_bv': min_bv, 'max_bv' : max_bv, 'min_' : min_, 'max_' :  max_ }
		if not flag:
			return -1
		#print min_max, "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
		for i in range (iterations):
			result = data[i]
			source_time_str = result['T']
			source_time = datetime.datetime.strptime(source_time_str, "%Y-%m-%dT%H:%M:%S")
			if self.training_mode == 'online_training' and training_history is not None and source_time < training_history.last_trained_timestamp:
				continue
			that_day = datetime.datetime.combine(source_time.date(), datetime.time(0))
			diff  = source_time - that_day
			diff_in_minutes = diff.seconds / 3600
			#print result, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
			tmp = [(result['O'] - min_o) if (result['O'] - min_o) != 0 else min_o / (max_o - min_o) if (max_o - min_o) != 0 else min_o,
				   (result['C'] - min_c) if (result['C'] - min_c) != 0 else min_c / (max_c - min_c) if (max_c - min_c) != 0 else min_c,
				   (result['L'] - min_l) if (result['L'] - min_l) != 0 else min_l / (max_l - min_l) if (max_l - min_l) != 0 else min_l,
				   (result['V'] - min_v) if (result['V'] - min_v) != 0 else min_v / (max_v - min_v) if (max_v - min_v) != 0 else min_v,
				   (result['BV'] - min_bv) if (result['BV'] - min_bv) != 0 else min_bv / (max_bv - min_bv) if (max_bv - min_bv) != 0 else min_bv,
				   diff_in_minutes / 60]
			X.append(tmp)
			Y.append((data[i + 1]['H'] - min_) if (data[i + 1]['H'] - min_) != 0 else min_ / (max_ - min_) if (max_ - min_) != 0 else min_)
		if training_history is None:
			training_history = Training(coin.id, source_time, self.granularity)
		else:
			training_history.last_trained_timestamp = source_time
		db.session.add(training_history)
		db.session.commit()
		return [X, Y, min_max]