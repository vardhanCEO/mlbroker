import os
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mlbroker-dev-secret')

    _db_url = os.environ.get('DATABASE_URL')
    if _db_url:
        SQLALCHEMY_DATABASE_URI = _db_url
    else:
        _pw = quote_plus('test@123')
        # Use psycopg (v3) driver — works on Python 3.13
        SQLALCHEMY_DATABASE_URI = f'postgresql+psycopg://postgres:{_pw}@localhost/mlbroker'

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    ALPACA_API_KEY    = os.environ.get('ALPACA_API_KEY',    'PKK24AZ6VGINBHULR27CBQD3G5')
    ALPACA_SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY', 'HkwaJaQpfB6Ej1wX8z3C5mBcQ2GXmBurjdEU2pYvN3n6')
    ALPACA_BASE_URL   = os.environ.get('ALPACA_BASE_URL',   'https://paper-api.alpaca.markets/v2')
    ALPACA_DATA_URL   = os.environ.get('ALPACA_DATA_URL',   'https://data.alpaca.markets/v2')

    BINANCE_API_KEY    = os.environ.get('BINANCE_API_KEY',    '')
    BINANCE_SECRET_KEY = os.environ.get('BINANCE_SECRET_KEY', '')

    # Flask-Login / session security
    REMEMBER_COOKIE_HTTPONLY  = True
    REMEMBER_COOKIE_SAMESITE  = 'Lax'
    SESSION_COOKIE_HTTPONLY   = True
    SESSION_COOKIE_SAMESITE   = 'Lax'
