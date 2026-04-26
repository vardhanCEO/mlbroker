from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required, current_user
from ..services import bot_engine
from ..models.bot_trade import BotTrade

bot_bp = Blueprint('bot', __name__)


# ── Existing routes ───────────────────────────────────────────────────────────

@bot_bp.route('/')
@login_required
def index():
    return render_template('bot/index.html')


@bot_bp.route('/chart')
@login_required
def chart():
    return render_template('chart/index.html')


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


# ── ML routes ─────────────────────────────────────────────────────────────────

@bot_bp.route('/api/ml/train', methods=['POST'])
@login_required
def ml_train():
    """Launch background training for symbol + timeframe."""
    from ..services import ml_engine
    d         = request.json or {}
    symbol    = d.get('symbol', 'BTC/USD').upper()
    timeframe = d.get('timeframe', '15Min')

    started, err = ml_engine.start_training(
        current_app._get_current_object(), symbol, timeframe)
    if not started:
        return jsonify({'error': err or 'Could not start training'}), 400
    return jsonify({'status': 'training_started', 'symbol': symbol, 'timeframe': timeframe})


@bot_bp.route('/api/ml/status')
@login_required
def ml_status():
    """Training job status + DB model metadata."""
    from ..services import ml_engine
    from ..models.ml_model import MLModel

    symbol    = request.args.get('symbol', 'BTC/USD').upper()
    timeframe = request.args.get('timeframe', '15Min')

    job = ml_engine.get_training_status(symbol, timeframe)

    rec = MLModel.query.filter_by(
        symbol=symbol, timeframe=timeframe, is_active=True
    ).order_by(MLModel.trained_at.desc()).first()

    return jsonify({
        'job':   job,
        'model': rec.to_dict() if rec else None,
    })


@bot_bp.route('/api/ml/signal')
@login_required
def ml_signal():
    """Current ML signal for a symbol."""
    from ..services import ml_engine
    symbol    = request.args.get('symbol', 'BTC/USD').upper()
    timeframe = request.args.get('timeframe', '15Min')
    result    = ml_engine.predict_signal(
        current_app._get_current_object(), symbol, timeframe)
    return jsonify(result)


@bot_bp.route('/api/ml/regime')
@login_required
def ml_regime():
    """Current market regime for a symbol."""
    from ..services import regime_detector
    symbol    = request.args.get('symbol', 'BTC/USD').upper()
    timeframe = request.args.get('timeframe', '15Min')
    result    = regime_detector.detect_regime(
        current_app._get_current_object(), symbol, timeframe)
    return jsonify(result)


@bot_bp.route('/api/ml/risk')
@login_required
def ml_risk():
    """Current risk state for the logged-in user."""
    from ..services import risk_manager
    from ..services.alpaca import AlpacaClient

    try:
        acct     = AlpacaClient().get_account()
        bankroll = float(acct.get('equity') or acct.get('portfolio_value') or 10_000)
    except Exception:
        bankroll = 10_000.0

    state = risk_manager.get_risk_state(current_user.id, bankroll)
    return jsonify(state)


@bot_bp.route('/api/ml/confluence')
@login_required
def ml_confluence():
    """Full confluence decision for a symbol (signal + regime + risk + sizing)."""
    from ..services import confluence_engine
    from ..services.alpaca import AlpacaClient

    symbol         = request.args.get('symbol', 'BTC/USD').upper()
    timeframe      = request.args.get('timeframe', '15Min')
    min_confidence = float(request.args.get('min_confidence', 0.60))

    try:
        acct     = AlpacaClient().get_account()
        bankroll = float(acct.get('equity') or acct.get('portfolio_value') or 10_000)
    except Exception:
        bankroll = 10_000.0

    result = confluence_engine.get_confluence(
        current_app._get_current_object(),
        symbol, current_user.id, bankroll,
        min_confidence=min_confidence,
        timeframe=timeframe,
    )
    return jsonify(result)


@bot_bp.route('/api/ml/start', methods=['POST'])
@login_required
def ml_start():
    """
    Start a bot trade driven by the ML confluence engine.
    Optionally override amount_usd; if not provided uses Kelly sizing.
    """
    from ..services import confluence_engine
    from ..services.alpaca import AlpacaClient

    d              = request.json or {}
    symbol         = d.get('symbol', 'BTC/USD').upper()
    timeframe      = d.get('timeframe', '15Min')
    min_confidence = float(d.get('min_confidence', 0.60))
    override_usd   = d.get('amount_usd')   # optional manual override

    try:
        acct     = AlpacaClient().get_account()
        bankroll = float(acct.get('equity') or acct.get('portfolio_value') or 10_000)
    except Exception:
        bankroll = 10_000.0

    decision = confluence_engine.get_confluence(
        current_app._get_current_object(),
        symbol, current_user.id, bankroll,
        min_confidence=min_confidence,
        timeframe=timeframe,
    )

    if not decision['should_trade']:
        return jsonify({'error': decision['reason'], 'decision': decision}), 422

    if decision['direction'] != 'buy':
        return jsonify({
            'error': 'SELL signal — long-only bot skips downtrend entries',
            'decision': decision,
        }), 422

    amount = float(override_usd) if override_usd else decision['position_size_usd']
    if amount <= 0:
        return jsonify({'error': 'Computed position size is zero'}), 400

    direction = 'long'
    trade_id, err = bot_engine.start_bot_trade(
        current_app._get_current_object(),
        current_user.id,
        symbol,
        amount,
        direction=direction,
    )
    if err:
        return jsonify({'error': err}), 500

    # Record signal in DB
    try:
        from ..models.ml_model import MLModel, MLSignal
        from ..extensions import db
        model_rec = MLModel.query.filter_by(
            symbol=symbol, timeframe=timeframe, is_active=True
        ).order_by(MLModel.trained_at.desc()).first()

        sig = MLSignal(
            model_id      = model_rec.id if model_rec else None,
            symbol        = symbol,
            direction     = decision['direction'],
            confidence    = decision['confidence'],
            regime        = decision['regime_name'],
            kelly_frac    = decision['kelly_frac'],
            suggested_usd = decision['position_size_usd'],
            acted_on      = True,
            bot_trade_id  = trade_id,
        )
        db.session.add(sig)
        db.session.commit()
    except Exception:
        pass  # non-critical

    return jsonify({
        'trade_id': trade_id,
        'amount':   amount,
        'decision': decision,
        'message':  f'ML trade #{trade_id} started — {decision["reason"]}',
    })
