"""Model files for authentication module."""
from app import db


# Define a base model for other database tables to inherit
class Base(db.Model):
    """Base Model."""

    __abstract__ = True
    id = db.Column(db.Integer, primary_key=True)
    date_created = db.Column(db.DateTime, default=db.func.current_timestamp())
    date_modified = db.Column(db.DateTime, default=db.func.current_timestamp(),
                              onupdate=db.func.current_timestamp())


class Coin(Base):
    __tablename__ = 'coin'

    name = db.Column(db.String(150), nullable=False)
    code = db.Column(db.String(100), nullable=False)
    url =  db.Column(db.Text, nullable=False)
    def __init__(self, name):
        self.name = name
    #catogory_id = db.Column(db.Integer, db.ForeignKey('catogory.id'))
    #is_active = db.Column(db.SmallInteger, nullable=False)
    #choices = db.relationship('Choice', backref='question', lazy='dynamic')


class Training(Base):
    """docstring for Training"""
    __tablename__ = 'training'
    coin_id = db.Column(db.Integer, db.ForeignKey('coin.id'))
    last_trained_timestamp = db.Column(db.DateTime)
    granularity = db.Column(db.String(150), nullable=False)
    def __init__(self, id, time, granularity):
        self.coin_id = id
        self.last_trained_timestamp = time
        self.granularity = granularity

class Parameters(Base):
    """docstring for Choice."""

    __tablename__ = 'parameters'
    coin_id = db.Column(db.Integer, db.ForeignKey('coin.id'))
    granularity = db.Column(db.String(150), nullable=False)
    # Granularity can be day, oneMin, fiveMin, thirtyMin, hour
    training_mode = db.Column(db.String(150), nullable=False)
    # training_mode can be Transaction, Summary
    #error = db.Column(db.Numeric, precision=10, scale=8, nullable=False)
    error = db.Column(db.Numeric(precision=10, scale=8), nullable=False)
    syn0 = db.Column(db.Text, nullable=False)
    syn1 = db.Column(db.Text, nullable=False)
    #min_max_parameters = db.Column(db.Text, nullable= False)
    def __init__(self, coin_id, granularity, training_mode):
        self.coin_id = coin_id
        self.granularity = granularity
        self.training_mode = training_mode
