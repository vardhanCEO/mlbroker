from datetime import datetime
from ..extensions import db


class Strategy(db.Model):
    __tablename__ = 'strategies'

    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    name       = db.Column(db.String(100), nullable=False)
    type       = db.Column(db.String(50),  nullable=False)   # sma_crossover | rsi | bollinger | macd
    symbol     = db.Column(db.String(20),  nullable=False)
    timeframe  = db.Column(db.String(20),  default='1Day')
    parameters = db.Column(db.JSON, default=dict)
    is_active  = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    signals = db.relationship('Signal', backref='strategy', lazy='dynamic',
                              cascade='all, delete-orphan')

    @property
    def latest_signal(self):
        return self.signals.order_by(Signal.signal_time.desc()).first()

    @property
    def signal_count(self):
        return self.signals.count()

    def to_dict(self):
        latest = self.latest_signal
        return {
            'id':            self.id,
            'name':          self.name,
            'type':          self.type,
            'symbol':        self.symbol,
            'timeframe':     self.timeframe,
            'parameters':    self.parameters,
            'is_active':     self.is_active,
            'signal_count':  self.signal_count,
            'latest_signal': latest.to_dict() if latest else None,
            'created_at':    self.created_at.isoformat(),
        }


class Signal(db.Model):
    __tablename__ = 'signals'

    id          = db.Column(db.Integer, primary_key=True)
    strategy_id = db.Column(db.Integer, db.ForeignKey('strategies.id'), nullable=False)
    symbol      = db.Column(db.String(20), nullable=False)
    action      = db.Column(db.String(10), nullable=False)   # buy | sell
    price       = db.Column(db.Float, nullable=False)
    signal_time = db.Column(db.DateTime, nullable=False)
    executed    = db.Column(db.Boolean, default=False)
    order_id    = db.Column(db.String(50))
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id':            self.id,
            'strategy_id':   self.strategy_id,
            'strategy_name': self.strategy.name if self.strategy else None,
            'symbol':        self.symbol,
            'action':        self.action,
            'price':         self.price,
            'signal_time':   self.signal_time.isoformat(),
            'executed':      self.executed,
            'order_id':      self.order_id,
            'created_at':    self.created_at.isoformat(),
        }
