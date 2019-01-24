"""Import flask and template operators."""
from flask import Flask

# Import SQLAlchemy
from flask_sqlalchemy import SQLAlchemy

from flask_migrate import Migrate
import redis
#from flask.ext.redis import FlaskRedis
# Import Flask-MySQLdb

#import atexit
#from apscheduler.scheduler import Scheduler

# Define the WSGI application object
app = Flask(__name__)
# Configurations
app.config.from_object('config')
redis_store = redis.StrictRedis(host='localhost', port=6379, db=0)
app_name = app.name
config = app.config
# Define the database object which is imported
# by modules and controllers
db = SQLAlchemy(app)
#cron = Scheduler(daemon=True)

#@cron.interval_schedule(hours=1)
#def fetch_buy_sell_info():
	

# Import a module / component using its blueprint handler variable (mod_auth)
from app.mod_neural_training.controllers import mod_neural_training as neural_training_module
from app.mod_transactions.controllers import mod_transactions as transaction_module
from app.mod_trades.controllers import mod_trades as trade_module

# Register blueprint(s)
app.register_blueprint(neural_training_module)
app.register_blueprint(transaction_module)
app.register_blueprint(trade_module)

db.create_all()
