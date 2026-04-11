from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required, current_user
from ..services import bot_engine
from ..models.bot_trade import BotTrade

bot_bp = Blueprint('bot', __name__)


@bot_bp.route('/')
@login_required
def index():
    return render_template('bot/index.html')


@bot_bp.route('/api/start', methods=['POST'])
@login_required
def start():
    d          = request.json or {}
    symbol     = d.get('symbol', 'BTC/USD').upper()
    amount_usd = d.get('amount_usd')
    if not amount_usd or float(amount_usd) <= 0:
        return jsonify({'error': 'amount_usd must be positive'}), 400

    trade_id, err = bot_engine.start_bot_trade(
        current_app._get_current_object(),
        current_user.id,
        symbol,
        float(amount_usd),
    )
    if err:
        return jsonify({'error': err}), 500
    return jsonify({'trade_id': trade_id, 'message': f'Bot trade #{trade_id} started'})


@bot_bp.route('/api/cancel/<int:trade_id>', methods=['POST'])
@login_required
def cancel(trade_id):
    ok, err = bot_engine.cancel_bot_trade(
        current_app._get_current_object(),
        trade_id,
        current_user.id,
    )
    if not ok:
        return jsonify({'error': err}), 400
    return jsonify({'success': True})


@bot_bp.route('/api/trades')
@login_required
def trades():
    status = request.args.get('status', '')
    q = BotTrade.query.filter_by(user_id=current_user.id)
    if status:
        q = q.filter_by(status=status)
    rows = q.order_by(BotTrade.created_at.desc()).limit(50).all()
    return jsonify([r.to_dict() for r in rows])


@bot_bp.route('/api/trades/<int:trade_id>')
@login_required
def trade_detail(trade_id):
    t = BotTrade.query.filter_by(
        id=trade_id, user_id=current_user.id).first_or_404()
    return jsonify(t.to_dict())
