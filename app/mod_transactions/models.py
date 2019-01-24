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


class Transaction(Base):
    __tablename__ = 'transactions'

    coin_id = db.Column(db.Integer, db.ForeignKey('coin.id'))
    data = db.Column(db.Text, nullable=False)
    #buy_wall_data = db.Column(db.Text, nullable=False)
    #sell_wall_data = db.Column(db.Text, nullable=False)
    def __init__(self, coin_id):
        self.coin_id = coin_id
