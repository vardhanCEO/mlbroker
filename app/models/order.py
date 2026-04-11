from datetime import datetime
from ..extensions import db


class Order(db.Model):
    __tablename__ = 'orders'

    id               = db.Column(db.Integer, primary_key=True)
    user_id          = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    alpaca_id        = db.Column(db.String(50), nullable=True)
    symbol           = db.Column(db.String(20), nullable=False)
    side             = db.Column(db.String(10), nullable=False)       # buy | sell
    order_type       = db.Column(db.String(20), default='market')
    qty              = db.Column(db.Float,   nullable=False)
    limit_price      = db.Column(db.Float,   nullable=True)
    time_in_force    = db.Column(db.String(10), default='gtc')
    status           = db.Column(db.String(30), default='submitted')
    filled_avg_price = db.Column(db.Float, nullable=True)
    filled_qty       = db.Column(db.Float, nullable=True)
    signal_id        = db.Column(db.Integer, db.ForeignKey('signals.id'), nullable=True)
    raw_response     = db.Column(db.JSON, nullable=True)
    created_at       = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at       = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id':               self.id,
            'alpaca_id':        self.alpaca_id,
            'symbol':           self.symbol,
            'side':             self.side,
            'order_type':       self.order_type,
            'qty':              self.qty,
            'limit_price':      self.limit_price,
            'time_in_force':    self.time_in_force,
            'status':           self.status,
            'filled_avg_price': self.filled_avg_price,
            'filled_qty':       self.filled_qty,
            'signal_id':        self.signal_id,
            'created_at':       self.created_at.isoformat(),
        }
