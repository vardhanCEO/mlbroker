from datetime import datetime
from ..extensions import db

class BotTrade(db.Model):
    __tablename__ = 'bot_trades'
    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    symbol        = db.Column(db.String(20), nullable=False)
    qty           = db.Column(db.Float, nullable=False)
    amount_usd    = db.Column(db.Float, nullable=True)
    buy_order_id  = db.Column(db.String(50), nullable=True)
    sell_order_id = db.Column(db.String(50), nullable=True)
    buy_price     = db.Column(db.Float, nullable=True)
    sell_price    = db.Column(db.Float, nullable=True)
    filled_qty    = db.Column(db.Float, nullable=True)
    sell_at       = db.Column(db.DateTime, nullable=False)
    sold_at       = db.Column(db.DateTime, nullable=True)
    direction     = db.Column(db.String(10), default='long')
    # direction: long (buy→sell) | short (sell→buy)
    status        = db.Column(db.String(20), default='buying')
    # statuses: buying | active | selling | completed | error | cancelled
    profit        = db.Column(db.Float, nullable=True)
    profit_pct    = db.Column(db.Float, nullable=True)
    logs          = db.Column(db.JSON, default=list)
    error_msg     = db.Column(db.Text, nullable=True)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id':           self.id,
            'symbol':       self.symbol,
            'qty':          self.qty,
            'amount_usd':   self.amount_usd,
            'buy_order_id': self.buy_order_id,
            'sell_order_id':self.sell_order_id,
            'buy_price':    self.buy_price,
            'sell_price':   self.sell_price,
            'filled_qty':   self.filled_qty,
            'sell_at':      self.sell_at.isoformat() + 'Z',
            'sold_at':      self.sold_at.isoformat() + 'Z' if self.sold_at else None,
            'direction':    self.direction or 'long',
            'status':       self.status,
            'profit':       self.profit,
            'profit_pct':   self.profit_pct,
            'logs':         self.logs or [],
            'error_msg':    self.error_msg,
            'created_at':   self.created_at.isoformat() + 'Z',
        }
