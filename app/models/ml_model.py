from datetime import datetime
from ..extensions import db


class MLModel(db.Model):
    __tablename__ = 'ml_models'

    id          = db.Column(db.Integer, primary_key=True)
    symbol      = db.Column(db.String(20), nullable=False)
    timeframe   = db.Column(db.String(10), nullable=False, default='15Min')
    model_type  = db.Column(db.String(50), default='xgb_lgb_ensemble')
    lookforward = db.Column(db.Integer, default=1)
    accuracy    = db.Column(db.Float)       # validation accuracy (%)
    n_samples   = db.Column(db.Integer)
    is_active   = db.Column(db.Boolean, default=True)
    trained_at  = db.Column(db.DateTime, default=datetime.utcnow)

    signals = db.relationship('MLSignal', backref='model', lazy='dynamic')

    def to_dict(self):
        return {
            'id':         self.id,
            'symbol':     self.symbol,
            'timeframe':  self.timeframe,
            'model_type': self.model_type,
            'accuracy':   self.accuracy,
            'n_samples':  self.n_samples,
            'trained_at': self.trained_at.isoformat() if self.trained_at else None,
        }


class MLSignal(db.Model):
    __tablename__ = 'ml_signals'

    id            = db.Column(db.Integer, primary_key=True)
    model_id      = db.Column(db.Integer, db.ForeignKey('ml_models.id'))
    symbol        = db.Column(db.String(20), nullable=False)
    direction     = db.Column(db.String(10))    # 'buy' | 'sell'
    confidence    = db.Column(db.Float)
    regime        = db.Column(db.String(20))
    kelly_frac    = db.Column(db.Float)
    suggested_usd = db.Column(db.Float)
    acted_on      = db.Column(db.Boolean, default=False)
    bot_trade_id  = db.Column(db.Integer, db.ForeignKey('bot_trades.id'), nullable=True)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id':            self.id,
            'symbol':        self.symbol,
            'direction':     self.direction,
            'confidence':    self.confidence,
            'regime':        self.regime,
            'kelly_frac':    self.kelly_frac,
            'suggested_usd': self.suggested_usd,
            'acted_on':      self.acted_on,
            'created_at':    self.created_at.isoformat() if self.created_at else None,
        }
