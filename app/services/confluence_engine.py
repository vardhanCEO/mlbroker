"""
Confluence Engine — combines ML signal + regime + risk gate into a single
trade decision with Kelly-sized position.

Usage:
    result = get_confluence(app, symbol, user_id, bankroll,
                            min_confidence=0.55, timeframe='15Min')

    if result['should_trade']:
        amount = result['position_size_usd']
        direction = result['direction']
        # → call bot_engine.start_bot_trade(app, user_id, symbol, amount)

Regime filter rules:
  bear + buy    → blocked (don't buy into downtrend)
  ranging       → confidence floor 0.60 (choppy market needs stronger signal)
  volatile      → confidence floor 0.65 (high spike risk)
  bull + buy    → confidence floor 0.55 (trend aligned — most lenient)
  bull + sell   → allowed (profit-taking is fine even in uptrend)
"""
import logging

logger = logging.getLogger('alphapilot.confluence')

# Regime ids (must match regime_detector.REGIME_NAMES)
REGIME_RANGING  = 0
REGIME_BULL     = 1
REGIME_BEAR     = 2
REGIME_VOLATILE = 3

# Per-regime minimum confidence floors (applied on top of user's min_confidence).
# Ranging: direction is unreliable in choppy markets → need stronger signal.
# Volatile: price can spike either way → require high conviction.
# Bull + BUY: trend alignment → respect user threshold without raising it.
REGIME_CONF_FLOOR = {
    REGIME_RANGING:  0.60,
    REGIME_VOLATILE: 0.65,
    REGIME_BEAR:     0.65,   # rarely reached since bear+buy is blocked upstream
    REGIME_BULL:     0.55,   # trend aligned for buys — standard bar
}


def get_confluence(
    app,
    symbol: str,
    user_id: int,
    bankroll: float,
    min_confidence: float = 0.60,
    timeframe: str = '15Min',
) -> dict:
    """
    Return a trade decision dict:
      should_trade   : bool
      direction      : 'buy' | 'sell' | None
      confidence     : float
      regime_name    : str
      regime_id      : int
      regime_color   : str
      kelly_frac     : float
      position_size_usd : float
      reason         : str (human-readable gate explanation)
      signal_features: dict (for UI display)
    """
    from .ml_engine       import predict_signal
    from .regime_detector import detect_regime
    from .risk_manager    import check_risk, size_position, kelly_fraction

    # 1 ── ML signal
    signal = predict_signal(app, symbol, timeframe)
    if signal.get('error'):
        return _blocked(signal['error'], 0.0, {}, 0, 'unknown', '#7d8590')

    direction  = signal['direction']
    confidence = signal['confidence']
    features   = signal.get('features', {})

    # 2 ── Regime
    regime = detect_regime(app, symbol, timeframe)
    regime_id    = regime.get('regime_id', REGIME_RANGING)
    regime_name  = regime.get('regime_name', 'unknown')
    regime_color = regime.get('color', '#7d8590')

    # 3 ── Regime filter
    if regime_id == REGIME_BEAR and direction == 'buy':
        return _blocked(
            'Bear regime — suppressing BUY signals',
            confidence, features, regime_id, regime_name, regime_color,
        )

    # 3.5 ── CSI multi-confirmation gate
    from .csi_engine import csi_score as _csi_score
    csi = _csi_score(app, symbol, timeframe)
    if csi.get('error'):
        logger.warning(f'[Confluence] CSI unavailable ({csi["error"]}), skipping gate')
        csi = None
    else:
        score = csi['buy_score'] if direction == 'buy' else csi['sell_score']
        min_s = csi['min_score']
        if score < min_s:
            comp_str = ' '.join(
                f"{k}={'Y' if (v['buy'] if direction == 'buy' else v['sell']) else 'N'}"
                for k, v in csi['components'].items()
            )
            logger.info(
                f'[Confluence] {symbol} CSI blocked: '
                f'{direction} {score}/6 < {min_s} [{comp_str}]'
            )
            return _blocked(
                f'CSI {score}/6 {direction.upper()} confirmations (need {min_s}) — {comp_str}',
                confidence, features, regime_id, regime_name, regime_color,
            )
        logger.info(
            f'[Confluence] {symbol} CSI passed: {direction} {score}/6'
        )

    # Effective threshold = max(user threshold, per-regime floor)
    # Bull + buy gets the lowest floor (trend aligned); volatile/ranging get higher bars.
    regime_floor  = REGIME_CONF_FLOOR.get(regime_id, min_confidence)
    effective_min = max(min_confidence, regime_floor)

    # 4 ── Confidence gate
    if confidence < effective_min:
        return _blocked(
            f'Confidence {confidence:.1%} < {regime_name} threshold {effective_min:.1%}',
            confidence, features, regime_id, regime_name, regime_color,
        )

    # 5 ── Risk gate
    risk = check_risk(user_id, bankroll)
    if not risk['can_trade']:
        return _blocked(
            risk['reason'],
            confidence, features, regime_id, regime_name, regime_color,
        )

    # 6 ── Position sizing (with dynamic payout + drawdown multiplier)
    raw_prob  = signal.get('raw_prob', 0.5)
    win_p     = raw_prob if direction == 'buy' else 1.0 - raw_prob
    kf        = kelly_fraction(win_p)
    pos_usd   = size_position(win_p, bankroll, user_id=user_id)

    logger.info(
        f'[Confluence] {symbol} {direction.upper()} conf={confidence:.2%} '
        f'regime={regime_name} kelly={kf:.3f} size=${pos_usd}'
    )

    csi_score_val = (
        (csi['buy_score'] if direction == 'buy' else csi['sell_score'])
        if csi else None
    )

    return {
        'should_trade':      True,
        'direction':         direction,
        'confidence':        confidence,
        'regime_id':         regime_id,
        'regime_name':       regime_name,
        'regime_color':      regime_color,
        'kelly_frac':        kf,
        'position_size_usd': pos_usd,
        'reason':            (
            f'{confidence:.1%} conf {direction.upper()} · '
            f'{regime_name} regime · CSI {csi_score_val}/6 · '
            f'Kelly {kf:.1%} · ${pos_usd}'
        ),
        'signal_features':   features,
        'risk':              risk,
        'csi':               csi,
    }


# ── Helper ────────────────────────────────────────────────────────────────────

def _blocked(reason, confidence, features, regime_id, regime_name, regime_color):
    return {
        'should_trade':      False,
        'direction':         None,
        'confidence':        confidence,
        'regime_id':         regime_id,
        'regime_name':       regime_name,
        'regime_color':      regime_color,
        'kelly_frac':        0.0,
        'position_size_usd': 0.0,
        'reason':            reason,
        'signal_features':   features,
        'risk':              {},
        'csi':               None,
    }
