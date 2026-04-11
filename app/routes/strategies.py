from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from ..models.strategy import Strategy, Signal
from ..extensions import db

strategies_bp = Blueprint('strategies', __name__)

STRATEGY_TYPES = {
    'sma_crossover': 'SMA Crossover',
    'rsi':           'RSI',
    'bollinger':     'Bollinger Bands',
    'macd':          'MACD',
}


@strategies_bp.route('/')
@login_required
def index():
    strategies = (Strategy.query
                  .filter_by(user_id=current_user.id)
                  .order_by(Strategy.created_at.desc())
                  .all())
    return render_template('strategies/index.html', strategies=strategies,
                           strategy_types=STRATEGY_TYPES)


@strategies_bp.route('/create', methods=['POST'])
@login_required
def create():
    f = request.form
    strategy = Strategy(
        user_id=current_user.id,
        name=f['name'],
        type=f['type'],
        symbol=f['symbol'].upper().strip(),
        timeframe=f.get('timeframe', '1Day'),
        parameters=_extract_params(f),
        is_active=True,
    )
    db.session.add(strategy)
    db.session.commit()
    flash(f'Strategy "{strategy.name}" created.', 'success')
    return redirect(url_for('strategies.detail', id=strategy.id))


@strategies_bp.route('/<int:id>')
@login_required
def detail(id):
    strategy = Strategy.query.filter_by(id=id, user_id=current_user.id).first_or_404()
    signals = (Signal.query
               .filter_by(strategy_id=id)
               .order_by(Signal.signal_time.desc())
               .limit(50).all())
    return render_template('strategies/detail.html', strategy=strategy, signals=signals)


@strategies_bp.route('/<int:id>/update', methods=['POST'])
@login_required
def update(id):
    strategy = Strategy.query.filter_by(id=id, user_id=current_user.id).first_or_404()
    f = request.form
    strategy.name     = f.get('name', strategy.name)
    strategy.symbol   = f.get('symbol', strategy.symbol).upper().strip()
    strategy.timeframe = f.get('timeframe', strategy.timeframe)
    strategy.parameters = _extract_params(f, strategy.type)
    db.session.commit()
    flash('Strategy updated.', 'success')
    return redirect(url_for('strategies.detail', id=id))


@strategies_bp.route('/<int:id>/toggle', methods=['POST'])
@login_required
def toggle(id):
    strategy = Strategy.query.filter_by(id=id, user_id=current_user.id).first_or_404()
    strategy.is_active = not strategy.is_active
    db.session.commit()
    return jsonify({'is_active': strategy.is_active})


@strategies_bp.route('/<int:id>/delete', methods=['POST'])
@login_required
def delete(id):
    strategy = Strategy.query.filter_by(id=id, user_id=current_user.id).first_or_404()
    db.session.delete(strategy)
    db.session.commit()
    flash('Strategy deleted.', 'info')
    return redirect(url_for('strategies.index'))


def _extract_params(form, strategy_type=None):
    stype = strategy_type or form.get('type', '')
    if stype == 'sma_crossover':
        return {
            'fast_period': int(form.get('fast_period', 10)),
            'slow_period': int(form.get('slow_period', 20)),
        }
    if stype == 'rsi':
        return {
            'period':     int(form.get('period', 14)),
            'oversold':   int(form.get('oversold', 30)),
            'overbought': int(form.get('overbought', 70)),
        }
    if stype == 'bollinger':
        return {
            'period':  int(form.get('period', 20)),
            'std_dev': float(form.get('std_dev', 2.0)),
        }
    if stype == 'macd':
        return {
            'fast':   int(form.get('fast', 12)),
            'slow':   int(form.get('slow', 26)),
            'signal': int(form.get('signal', 9)),
        }
    return {}
