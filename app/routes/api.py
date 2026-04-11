from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user
from ..services.alpaca import AlpacaClient
from ..services.strategy_engine import run_strategy
from ..models.strategy import Strategy, Signal
from ..models.order import Order
from ..models.binary_trade import BinaryTrade
from ..extensions import db
from datetime import datetime, timedelta

api_bp = Blueprint('api', __name__)


def alpaca():
    return AlpacaClient()


# ── Account & Portfolio ──────────────────────────────────────────────────────

@api_bp.route('/account')
@login_required
def account():
    try:
        return jsonify(alpaca().get_account())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/portfolio/history')
@login_required
def portfolio_history():
    period    = request.args.get('period', '1M')
    timeframe = request.args.get('timeframe', '1D')
    try:
        return jsonify(alpaca().get_portfolio_history(period, timeframe))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Positions ────────────────────────────────────────────────────────────────

@api_bp.route('/positions')
@login_required
def positions():
    try:
        return jsonify(alpaca().get_positions())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Orders (Alpaca live state) ───────────────────────────────────────────────

@api_bp.route('/orders')
@login_required
def orders():
    status = request.args.get('status', 'open')
    limit  = request.args.get('limit', 50, type=int)
    try:
        return jsonify(alpaca().get_orders(status=status, limit=limit))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/orders/place', methods=['POST'])
@login_required
def place_order():
    d = request.json or {}
    symbol      = d.get('symbol', '').upper()
    qty         = d.get('qty')
    side        = d.get('side', 'buy')
    order_type  = d.get('type', 'market')
    tif         = d.get('time_in_force', 'gtc')
    limit_price = d.get('limit_price')
    signal_id   = d.get('signal_id')

    if not symbol or not qty:
        return jsonify({'error': 'symbol and qty are required'}), 400

    db_order = Order(
        user_id=current_user.id,
        symbol=symbol, side=side, order_type=order_type,
        qty=float(qty), limit_price=float(limit_price) if limit_price else None,
        time_in_force=tif, status='pending', signal_id=signal_id,
    )
    db.session.add(db_order)
    db.session.commit()

    try:
        resp = alpaca().place_order(symbol, qty, side, order_type, tif, limit_price)
        db_order.alpaca_id    = resp.get('id')
        db_order.status       = resp.get('status', 'submitted')
        db_order.raw_response = resp
        db.session.commit()
        return jsonify({**resp, '_db_id': db_order.id})
    except Exception as e:
        db_order.status = 'rejected'
        db.session.commit()
        return jsonify({'error': str(e), '_db_id': db_order.id}), 500


@api_bp.route('/orders/<order_id>/cancel', methods=['DELETE'])
@login_required
def cancel_order(order_id):
    try:
        alpaca().cancel_order(order_id)
        row = Order.query.filter_by(alpaca_id=order_id, user_id=current_user.id).first()
        if row:
            row.status = 'canceled'
            db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/orders/cancel-all', methods=['DELETE'])
@login_required
def cancel_all():
    try:
        alpaca().cancel_all_orders()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Local order log (scoped to current user) ─────────────────────────────────

@api_bp.route('/orders/local')
@login_required
def local_orders():
    limit = request.args.get('limit', 50, type=int)
    rows  = (Order.query
             .filter_by(user_id=current_user.id)
             .order_by(Order.created_at.desc())
             .limit(limit).all())
    return jsonify([r.to_dict() for r in rows])


# ── Market data  (Binance) ───────────────────────────────────────────────────

@api_bp.route('/bars/<path:symbol>')
@login_required
def bars(symbol):
    timeframe = request.args.get('timeframe', '1Day')
    limit     = request.args.get('limit', 500, type=int)
    try:
        data = alpaca().get_bars(symbol.upper(), timeframe=timeframe, limit=limit)
        data['source'] = 'alpaca'
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/tickers')
@login_required
def tickers():
    try:
        return jsonify(alpaca().get_snapshots())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Alpaca latest bar ────────────────────────────────────────────────────────

@api_bp.route('/latest-bar/<path:symbol>')
@login_required
def latest_bar(symbol):
    try:
        return jsonify(alpaca().get_latest_bar(symbol.upper()))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Market clock (Alpaca) ────────────────────────────────────────────────────

@api_bp.route('/clock')
@login_required
def clock():
    try:
        return jsonify(alpaca().get_clock())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Strategy engine ───────────────────────────────────────────────────────────

@api_bp.route('/strategies/run/<int:strategy_id>')
@login_required
def run_strategy_route(strategy_id):
    # Ensure the strategy belongs to the current user
    strategy = Strategy.query.filter_by(id=strategy_id, user_id=current_user.id).first_or_404()
    limit    = request.args.get('limit', 500, type=int)
    try:
        bars_resp = alpaca().get_bars(strategy.symbol, timeframe=strategy.timeframe, limit=limit)
        data = bars_resp.get('bars', [])
        if not data:
            return jsonify({'error': 'No bar data returned from Alpaca'}), 400

        result = run_strategy(strategy.type, data, strategy.parameters or {},
                              timeframe=strategy.timeframe)

        existing  = {s.signal_time for s in Signal.query.filter_by(strategy_id=strategy_id).all()}
        new_count = 0
        for sig in result['signals']:
            dt = _parse_iso(sig['time'])
            if dt not in existing:
                db.session.add(Signal(
                    strategy_id=strategy_id, symbol=strategy.symbol,
                    action=sig['action'], price=sig['price'], signal_time=dt,
                ))
                new_count += 1
        db.session.commit()
        result['new_signals_count'] = new_count
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Signals ───────────────────────────────────────────────────────────────────

@api_bp.route('/signals')
@login_required
def signals():
    limit = request.args.get('limit', 20, type=int)
    # Only signals for this user's strategies
    rows = (Signal.query
            .join(Strategy, Signal.strategy_id == Strategy.id)
            .filter(Strategy.user_id == current_user.id)
            .order_by(Signal.created_at.desc())
            .limit(limit).all())
    return jsonify([s.to_dict() for s in rows])


@api_bp.route('/signals/<int:signal_id>/execute', methods=['POST'])
@login_required
def execute_signal(signal_id):
    signal   = Signal.query.get_or_404(signal_id)
    strategy = Strategy.query.filter_by(id=signal.strategy_id, user_id=current_user.id).first_or_404()
    qty      = (request.json or {}).get('qty', 1)
    try:
        resp = alpaca().place_order(signal.symbol, qty, signal.action)
        db_order = Order(
            user_id=current_user.id,
            alpaca_id=resp.get('id'), symbol=signal.symbol, side=signal.action,
            qty=float(qty), status=resp.get('status', 'submitted'),
            signal_id=signal_id, raw_response=resp,
        )
        db.session.add(db_order)
        signal.executed = True
        signal.order_id = resp.get('id')
        db.session.commit()
        return jsonify({'success': True, 'order': resp})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Binary options ────────────────────────────────────────────────────────────

@api_bp.route('/binary/place', methods=['POST'])
@login_required
def binary_place():
    d         = request.json or {}
    symbol    = d.get('symbol', '').upper()
    direction = d.get('direction', '').lower()
    amount    = d.get('amount')
    duration  = int(d.get('duration', 60))

    if not symbol or direction not in ('up', 'down') or not amount:
        return jsonify({'error': 'symbol, direction (up/down), and amount required'}), 400

    amount = float(amount)
    if amount <= 0:
        return jsonify({'error': 'Amount must be positive'}), 400

    try:
        price = alpaca().get_latest_bar(symbol)['bar']['c']
    except Exception as e:
        return jsonify({'error': f'Cannot get current price: {e}'}), 500

    expire_at = datetime.utcnow() + timedelta(seconds=duration)
    trade = BinaryTrade(
        user_id=current_user.id,
        symbol=symbol, direction=direction,
        amount=amount, payout_pct=80.0,
        entry_price=price, expire_at=expire_at,
    )
    db.session.add(trade)
    db.session.commit()
    return jsonify(trade.to_dict())


@api_bp.route('/binary/open')
@login_required
def binary_open():
    rows = BinaryTrade.query.filter_by(user_id=current_user.id, status='open').all()
    return jsonify([r.to_dict() for r in rows])


@api_bp.route('/binary/history')
@login_required
def binary_history():
    limit = request.args.get('limit', 20, type=int)
    rows  = (BinaryTrade.query
             .filter_by(user_id=current_user.id)
             .filter(BinaryTrade.status != 'open')
             .order_by(BinaryTrade.created_at.desc())
             .limit(limit).all())
    return jsonify([r.to_dict() for r in rows])


@api_bp.route('/binary/<int:trade_id>/resolve', methods=['POST'])
@login_required
def binary_resolve(trade_id):
    trade = BinaryTrade.query.filter_by(
        id=trade_id, user_id=current_user.id).first_or_404()

    if trade.status != 'open':
        return jsonify(trade.to_dict())

    try:
        exit_price = alpaca().get_latest_bar(trade.symbol)['bar']['c']
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    trade.exit_price = exit_price
    price_rose = exit_price > trade.entry_price

    if exit_price == trade.entry_price:
        trade.status = 'tie'
        trade.profit = 0.0
    elif (trade.direction == 'up' and price_rose) or \
         (trade.direction == 'down' and not price_rose):
        trade.status = 'won'
        trade.profit = round(trade.amount * trade.payout_pct / 100, 2)
    else:
        trade.status = 'lost'
        trade.profit = -trade.amount

    db.session.commit()
    return jsonify(trade.to_dict())


# ── Helper ────────────────────────────────────────────────────────────────────

def _parse_iso(s):
    s = str(s)
    if s.isdigit():
        return datetime.fromtimestamp(int(s))
    s = s.replace('Z', '+00:00')
    try:
        return datetime.fromisoformat(s).replace(tzinfo=None)
    except ValueError:
        return datetime.fromisoformat(s[:19])
