from datetime import datetime
from ..extensions import db


class BinaryTrade(db.Model):
    __tablename__ = 'binary_trades'

    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    symbol      = db.Column(db.String(20), nullable=False)
    direction   = db.Column(db.String(4),  nullable=False)   # 'up' | 'down'
    amount      = db.Column(db.Float,      nullable=False)   # USD wagered
    payout_pct  = db.Column(db.Float,      default=80.0)     # win payout %
    entry_price = db.Column(db.Float,      nullable=False)
    exit_price  = db.Column(db.Float,      nullable=True)
    expire_at   = db.Column(db.DateTime,   nullable=False)
    status      = db.Column(db.String(10), default='open')   # open | won | lost | tie
    profit      = db.Column(db.Float,      nullable=True)
    created_at  = db.Column(db.DateTime,   default=datetime.utcnow)

    def to_dict(self):
        return {
            'id':          self.id,
            'symbol':      self.symbol,
            'direction':   self.direction,
            'amount':      self.amount,
            'payout_pct':  self.payout_pct,
            'entry_price': self.entry_price,
            'exit_price':  self.exit_price,
            'expire_at':   self.expire_at.isoformat() + 'Z',
            'status':      self.status,
            'profit':      self.profit,
            'created_at':  self.created_at.isoformat() + 'Z',
        }
