from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from ..extensions import db


class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id         = db.Column(db.Integer, primary_key=True)
    username   = db.Column(db.String(64),  unique=True, nullable=False)
    email      = db.Column(db.String(120), unique=True, nullable=False)
    # PBKDF2-SHA256 with 600 000 iterations — Werkzeug default, OWASP-compliant
    _pw_hash   = db.Column('password_hash', db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    strategies = db.relationship('Strategy', backref='owner', lazy='dynamic',
                                 cascade='all, delete-orphan')
    orders     = db.relationship('Order', backref='owner', lazy='dynamic',
                                 cascade='all, delete-orphan')

    def set_password(self, password: str) -> None:
        self._pw_hash = generate_password_hash(password, method='pbkdf2:sha256:600000')

    def check_password(self, password: str) -> bool:
        return check_password_hash(self._pw_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'
