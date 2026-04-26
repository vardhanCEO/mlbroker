"""
Risk Manager — Kelly Criterion sizing + anti-tilt protection.

Components
──────────
kelly_fraction()          Half-Kelly formula for binary options
check_risk()              Gate: daily loss limit + consecutive-loss cooldown
get_risk_state()          Current snapshot for the UI
record_trade_outcome()    Call after each completed trade to update in-memory state

Cooldown state is kept in-memory per user_id (single-process Flask).
Daily P&L and consecutive losses are computed live from the DB for accuracy.
"""
import logging
import threading
from datetime import datetime, timedelta

logger = logging.getLogger('alphapilot.risk')

# ── Config ────────────────────────────────────────────────────────────────────

MAX_DAILY_LOSS_PCT    = 0.05   # pause when daily loss ≥ 5 % of total capital
MAX_CONSECUTIVE_LOSS  = 3      # pause after 3 consecutive losses (tighter protection)
COOLDOWN_SECONDS      = 900    # 15-min cooldown after hitting consecutive limit
MAX_POSITION_USD      = 500.0  # hard cap per trade
MIN_POSITION_USD      = 10.0
DEFAULT_PAYOUT        = 0.8    # fallback Kelly payout when no trade history exists

# ── In-memory cooldown store ──────────────────────────────────────────────────

_cooldowns: dict  = {}   # user_id → cooldown_until (datetime)
_lock = threading.Lock()


# ── Kelly formula ─────────────────────────────────────────────────────────────

def kelly_fraction(win_prob: float, payout: float = DEFAULT_PAYOUT) -> float:
    """
    Half-Kelly for binary outcomes.
      f* = (p·b − q) / b
    where p = win_prob, q = 1−p, b = payout ratio.
    Returns half-Kelly, clamped to [0, 0.25].
    """
    if win_prob <= 0.5:
        return 0.0
    q = 1.0 - win_prob
    f = (win_prob * payout - q) / payout
    return round(max(0.0, min(f / 2.0, 0.25)), 4)


# ── Daily & consecutive stats from DB ────────────────────────────────────────

def _daily_pnl(user_id: int) -> float:
    """Sum of profit for all completed bot trades created today (UTC)."""
    from ..models.bot_trade import BotTrade
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    trades = BotTrade.query.filter(
        BotTrade.user_id == user_id,
        BotTrade.status  == 'completed',
        BotTrade.created_at >= today,
    ).all()
    return sum(t.profit or 0.0 for t in trades)


def _daily_invested(user_id: int) -> float:
    """Total USD invested in completed trades today."""
    from ..models.bot_trade import BotTrade
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    trades = BotTrade.query.filter(
        BotTrade.user_id == user_id,
        BotTrade.status  == 'completed',
        BotTrade.created_at >= today,
    ).all()
    return sum(t.amount_usd or 0.0 for t in trades)


def _consecutive_losses(user_id: int) -> int:
    """Count trailing consecutive losses across the last 10 completed trades."""
    from ..models.bot_trade import BotTrade
    recent = (BotTrade.query
              .filter_by(user_id=user_id, status='completed')
              .order_by(BotTrade.created_at.desc())
              .limit(10).all())
    count = 0
    for t in recent:
        if (t.profit or 0.0) < 0:
            count += 1
        else:
            break
    return count


# ── Public API ────────────────────────────────────────────────────────────────

def check_risk(user_id: int, bankroll: float) -> dict:
    """
    Returns {'can_trade': bool, 'reason': str, ...stats...}.
    Enforces:
      • Consecutive-loss cooldown (COOLDOWN_SECONDS after MAX_CONSECUTIVE_LOSS)
      • Daily loss limit (MAX_DAILY_LOSS_PCT of bankroll)
    """
    # Cooldown check
    with _lock:
        cooldown_until = _cooldowns.get(user_id)
    if cooldown_until and datetime.utcnow() < cooldown_until:
        remaining = int((cooldown_until - datetime.utcnow()).total_seconds())
        return {
            'can_trade':         False,
            'reason':            f'Cooling down — {remaining}s remaining',
            'consecutive_losses': MAX_CONSECUTIVE_LOSS,
            'daily_pnl':          _daily_pnl(user_id),
            'cooldown_remaining': remaining,
        }

    consec = _consecutive_losses(user_id)
    if consec >= MAX_CONSECUTIVE_LOSS:
        until = datetime.utcnow() + timedelta(seconds=COOLDOWN_SECONDS)
        with _lock:
            _cooldowns[user_id] = until
        logger.warning(
            f'[Risk] User {user_id}: {consec} consecutive losses — '
            f'cooling down until {until.strftime("%H:%M:%S")} UTC'
        )
        return {
            'can_trade':          False,
            'reason':             f'{consec} consecutive losses — '
                                   f'{COOLDOWN_SECONDS // 60}-min cooldown started',
            'consecutive_losses': consec,
            'daily_pnl':          _daily_pnl(user_id),
            'cooldown_remaining': COOLDOWN_SECONDS,
        }

    pnl = _daily_pnl(user_id)
    if bankroll > 0 and pnl < -(bankroll * MAX_DAILY_LOSS_PCT):
        return {
            'can_trade':          False,
            'reason':             f'Daily loss limit reached (−{MAX_DAILY_LOSS_PCT*100:.0f}% of portfolio)',
            'consecutive_losses': consec,
            'daily_pnl':          round(pnl, 4),
            'cooldown_remaining': 0,
        }

    return {
        'can_trade':          True,
        'reason':             'OK',
        'consecutive_losses': consec,
        'daily_pnl':          round(pnl, 4),
        'cooldown_remaining': 0,
    }


def get_risk_state(user_id: int, bankroll: float = 10_000.0) -> dict:
    """Full risk snapshot for the UI, always returns something."""
    try:
        state = check_risk(user_id, bankroll)
        state['daily_invested'] = round(_daily_invested(user_id), 2)
        state['daily_loss_limit_usd'] = round(bankroll * MAX_DAILY_LOSS_PCT, 2)
        return state
    except Exception as exc:
        logger.exception('[Risk] get_risk_state error')
        return {
            'can_trade': True, 'reason': 'state unavailable',
            'consecutive_losses': 0, 'daily_pnl': 0.0,
            'cooldown_remaining': 0, 'daily_invested': 0.0,
            'daily_loss_limit_usd': round(bankroll * MAX_DAILY_LOSS_PCT, 2),
        }


def _historical_payout(user_id: int, n: int = 20) -> float:
    """
    Estimate Kelly payout ratio (avg_win / avg_loss) from recent completed
    trades.  Falls back to DEFAULT_PAYOUT when fewer than 5 trades exist.
    Using real payout makes Kelly sizing much more accurate over time.
    """
    try:
        from ..models.bot_trade import BotTrade
        recent = (BotTrade.query
                  .filter_by(user_id=user_id, status='completed')
                  .filter(BotTrade.profit.isnot(None))
                  .order_by(BotTrade.created_at.desc())
                  .limit(n).all())
        if len(recent) < 5:
            return DEFAULT_PAYOUT
        wins   = [abs(t.profit) for t in recent if (t.profit or 0) > 0]
        losses = [abs(t.profit) for t in recent if (t.profit or 0) <= 0]
        if not wins or not losses:
            return DEFAULT_PAYOUT
        payout = (sum(wins) / len(wins)) / (sum(losses) / len(losses))
        return round(min(max(payout, 0.2), 3.0), 3)
    except Exception:
        return DEFAULT_PAYOUT


def size_position(win_prob: float, bankroll: float, user_id: int = None) -> float:
    """
    Kelly-sized position clamped to [MIN_POSITION_USD, MAX_POSITION_USD].
    Uses historical win/loss ratio for payout when user_id is provided.
    Applies drawdown-adaptive shrinkage: shrinks up to 50% as daily loss
    approaches the 5% daily limit.
    """
    payout = _historical_payout(user_id) if user_id else DEFAULT_PAYOUT
    frac   = kelly_fraction(win_prob, payout)
    if frac <= 0 or bankroll <= 0:
        return MIN_POSITION_USD
    raw = bankroll * frac

    # Drawdown multiplier: linearly shrink position as daily loss grows
    if user_id and bankroll > 0:
        try:
            pnl = _daily_pnl(user_id)
            if pnl < 0:
                loss_frac = abs(pnl) / bankroll
                shrink    = max(0.5, 1.0 - loss_frac / MAX_DAILY_LOSS_PCT * 0.5)
                raw      *= shrink
        except Exception:
            pass

    return round(max(MIN_POSITION_USD, min(raw, MAX_POSITION_USD)), 2)
