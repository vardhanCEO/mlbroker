"""
Alpaca API wrapper — crypto only, using the official alpaca-py SDK.
Max bars: 10,000 per request (Alpaca crypto API limit).
"""
import json
import requests as _requests
from flask import current_app

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest,
    GetOrdersRequest, ClosePositionRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import (
    CryptoBarsRequest, CryptoLatestBarRequest, CryptoSnapshotRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

MAX_BARS = 10_000   # Alpaca crypto API hard limit per request

TICKER_PAIRS = [
    'BTC/USD', 'ETH/USD', 'SOL/USD', 'BNB/USD',
    'AVAX/USD', 'DOGE/USD', 'XRP/USD', 'LINK/USD',
]

_DAILY_TFS = ('1Day', '1Week', '1Month')

_TF_MAP = {
    '1Min':   TimeFrame.Minute,
    '5Min':   TimeFrame(5,  TimeFrameUnit.Minute),
    '15Min':  TimeFrame(15, TimeFrameUnit.Minute),
    '30Min':  TimeFrame(30, TimeFrameUnit.Minute),
    '1Hour':  TimeFrame.Hour,
    '4Hour':  TimeFrame(4,  TimeFrameUnit.Hour),
    '1Day':   TimeFrame.Day,
    '1Week':  TimeFrame.Week,
    '1Month': TimeFrame.Month,
}


def _serialize(obj):
    if obj is None:
        return None
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    if hasattr(obj, 'model_dump_json'):
        return json.loads(obj.model_dump_json())
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


def _bars_to_list(bar_set, symbol, timeframe='1Day'):
    """Convert Alpaca bar objects to Lightweight Charts-compatible dicts.
    Daily+ → 'YYYY-MM-DD' string; intraday → Unix int (seconds).
    """
    daily = timeframe in _DAILY_TFS
    return [
        {
            't': b.timestamp.strftime('%Y-%m-%d') if daily
                 else int(b.timestamp.timestamp()),
            'o': float(b.open),
            'h': float(b.high),
            'l': float(b.low),
            'c': float(b.close),
            'v': float(b.volume),
        }
        for b in bar_set[symbol]
    ]


class AlpacaClient:
    def __init__(self):
        self.api_key    = current_app.config['ALPACA_API_KEY']
        self.secret_key = current_app.config['ALPACA_SECRET_KEY']

        self.trade = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=True,
        )
        # Crypto historical data requires NO authentication
        self.crypto_data = CryptoHistoricalDataClient()

    # ── Account / Portfolio ─────────────────────────────────────────────────

    def get_account(self):
        return _serialize(self.trade.get_account())

    def get_portfolio_history(self, period='1M', timeframe='1D'):
        resp = _requests.get(
            'https://paper-api.alpaca.markets/v2/account/portfolio/history',
            headers={
                'APCA-API-KEY-ID':     self.api_key,
                'APCA-API-SECRET-KEY': self.secret_key,
            },
            params={'period': period, 'timeframe': timeframe},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    # ── Positions ───────────────────────────────────────────────────────────

    def get_positions(self):
        return _serialize(self.trade.get_all_positions())

    def close_position(self, symbol, qty=None):
        opts = ClosePositionRequest(qty=str(qty)) if qty else None
        return _serialize(self.trade.close_position(symbol, close_options=opts))

    # ── Orders ──────────────────────────────────────────────────────────────

    def get_orders(self, status='open', limit=50):
        _map = {
            'open':   QueryOrderStatus.OPEN,
            'closed': QueryOrderStatus.CLOSED,
            'all':    QueryOrderStatus.ALL,
        }
        req = GetOrdersRequest(status=_map.get(status, QueryOrderStatus.OPEN), limit=limit)
        return _serialize(self.trade.get_orders(req))

    def place_order(self, symbol, qty=None, side='buy', order_type='market',
                    time_in_force='gtc', limit_price=None, notional=None):
        """
        Place an order.  Pass either qty (units) OR notional (USD amount).
        Notional market orders let Alpaca calculate the exact qty from the
        live price — no external price lookup needed.
        """
        side_e = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
        # Crypto only supports GTC and IOC
        tif = TimeInForce.IOC if time_in_force == 'ioc' else TimeInForce.GTC

        if order_type == 'limit' and limit_price:
            req = LimitOrderRequest(
                symbol=symbol, qty=float(qty), side=side_e,
                time_in_force=tif, limit_price=float(limit_price),
            )
        elif notional is not None:
            # Dollar-amount order — Alpaca resolves qty at execution time
            req = MarketOrderRequest(
                symbol=symbol, notional=float(notional),
                side=side_e, time_in_force=tif,
            )
        else:
            req = MarketOrderRequest(
                symbol=symbol, qty=float(qty), side=side_e, time_in_force=tif,
            )
        return _serialize(self.trade.submit_order(req))

    def cancel_order(self, order_id):
        self.trade.cancel_order_by_id(order_id)
        return {}

    def cancel_all_orders(self):
        self.trade.cancel_orders()
        return {}

    def get_order(self, order_id: str) -> dict:
        """Fetch a single order by its Alpaca UUID."""
        return _serialize(self.trade.get_order_by_id(order_id))

    # ── Clock ────────────────────────────────────────────────────────────────

    def get_clock(self):
        return _serialize(self.trade.get_clock())

    # ── Crypto bars ──────────────────────────────────────────────────────────

    def get_bars(self, symbol, timeframe='1Day', limit=500):
        """
        Fetch up to MAX_BARS (10,000) crypto bars for symbol.
        symbol must be in Alpaca crypto format, e.g. 'BTC/USD'.
        """
        limit = min(int(limit), MAX_BARS)
        tf = _TF_MAP.get(timeframe, TimeFrame.Day)
        req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=tf, limit=limit)
        bar_set = self.crypto_data.get_crypto_bars(req)
        return {'bars': _bars_to_list(bar_set, symbol, timeframe), 'symbol': symbol}

    def get_latest_bar(self, symbol):
        req = CryptoLatestBarRequest(symbol_or_symbols=symbol)
        bar_set = self.crypto_data.get_crypto_latest_bar(req)
        b = bar_set[symbol]
        return {
            'bar': {
                't': b.timestamp.isoformat(),
                'o': float(b.open),
                'h': float(b.high),
                'l': float(b.low),
                'c': float(b.close),
                'v': float(b.volume),
            },
            'symbol': symbol,
        }

    def get_snapshot(self, symbol: str) -> dict:
        """Latest price + 24-h stats for one symbol via Alpaca snapshot API."""
        req  = CryptoSnapshotRequest(symbol_or_symbols=symbol)
        snap = self.crypto_data.get_crypto_snapshot(req)
        s    = snap[symbol]
        price = float(s.latest_trade.price) if s.latest_trade else float(s.daily_bar.close)
        db    = s.daily_bar
        open_ = float(db.open)   if db else price
        high_ = float(db.high)   if db else price
        low_  = float(db.low)    if db else price
        vol_  = float(db.volume) if db else 0.0
        chg   = price - open_
        pct   = (chg / open_ * 100) if open_ else 0.0
        return {
            'symbol':       symbol,
            'price':        price,
            'open':         open_,
            'high':         high_,
            'low':          low_,
            'volume':       vol_,
            'quote_volume': price * vol_,
            'change':       round(chg, 6),
            'change_pct':   round(pct, 4),
        }

    def get_snapshots(self, symbols: list = None) -> list:
        """Latest price + 24-h stats for multiple symbols (one batch call)."""
        syms = symbols or TICKER_PAIRS
        try:
            req   = CryptoSnapshotRequest(symbol_or_symbols=syms)
            snaps = self.crypto_data.get_crypto_snapshot(req)
            result = []
            for sym in syms:
                try:
                    s     = snaps[sym]
                    price = float(s.latest_trade.price) if s.latest_trade \
                            else float(s.daily_bar.close)
                    db    = s.daily_bar
                    open_ = float(db.open)   if db else price
                    high_ = float(db.high)   if db else price
                    low_  = float(db.low)    if db else price
                    vol_  = float(db.volume) if db else 0.0
                    chg   = price - open_
                    pct   = (chg / open_ * 100) if open_ else 0.0
                    result.append({
                        'symbol':       sym,
                        'price':        price,
                        'open':         open_,
                        'high':         high_,
                        'low':          low_,
                        'volume':       vol_,
                        'quote_volume': price * vol_,
                        'change':       round(chg, 6),
                        'change_pct':   round(pct, 4),
                    })
                except Exception:
                    pass
            return result
        except Exception:
            # Fallback: use latest bar for each symbol (no 24-h stats)
            result = []
            for sym in syms:
                try:
                    bar   = self.get_latest_bar(sym)['bar']
                    price = float(bar['c'])
                    result.append({
                        'symbol': sym, 'price': price,
                        'open': price, 'high': price, 'low': price,
                        'volume': float(bar['v']), 'quote_volume': 0.0,
                        'change': 0.0, 'change_pct': 0.0,
                    })
                except Exception:
                    pass
            return result
