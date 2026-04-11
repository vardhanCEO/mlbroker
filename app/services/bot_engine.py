"""
15-minute trading bot engine.
Flow: place BUY → wait exactly 15 minutes → place SELL → record P&L.
All DB writes happen inside with app.app_context() blocks (background threads).
"""
import logging
import threading
import time
from datetime import datetime, timedelta

logger = logging.getLogger('alphapilot.bot')

SELL_DELAY   = 15 * 60   # 900 seconds
FILL_TIMEOUT = 45        # seconds to wait for an order to fill
FILL_POLL    = 2         # polling interval (seconds)

# In-memory timer store  trade_id → threading.Timer
_timers: dict = {}
_lock = threading.Lock()


# ── Public API ────────────────────────────────────────────────────────────────

def start_bot_trade(app, user_id: int, symbol: str, amount_usd: float):
    """
    Create a BotTrade record, place a market BUY via Alpaca SDK, then
    schedule the SELL exactly 15 minutes after the buy is confirmed.
    Returns (trade_id, error_str).  On success error_str is None.

    Price is fetched from Alpaca's latest-bar endpoint (not Binance) so the
    entire flow is driven by the Alpaca SDK.
    """
    from ..models.bot_trade import BotTrade
    from ..extensions import db
    from ..services.alpaca import AlpacaClient

    with app.app_context():
        # Use Alpaca's latest bar for the price estimate shown in logs/UI.
        # The actual BUY will be placed as a notional order so Alpaca
        # calculates the exact qty at execution time — no Binance needed.
        try:
            bar_data  = AlpacaClient().get_latest_bar(symbol)
            est_price = bar_data['bar']['c']
            est_qty   = round(amount_usd / est_price, 6)
        except Exception as e:
            return None, f'Cannot fetch Alpaca price for {symbol}: {e}'

        sell_at = datetime.utcnow() + timedelta(seconds=SELL_DELAY)
        trade   = BotTrade(
            user_id=user_id, symbol=symbol,
            qty=est_qty,          # estimated; updated to actual fill qty later
            amount_usd=amount_usd,
            sell_at=sell_at, status='buying',
        )
        db.session.add(trade)
        db.session.flush()
        _log(trade,
             f'Trade created — {symbol} ${amount_usd:.2f} USD '
             f'(Alpaca est. qty {est_qty:.6f} @ ${est_price:,.2f})')
        db.session.commit()
        trade_id = trade.id

    # Hand off to background thread immediately
    _spawn(_buy_then_schedule, app, trade_id)
    return trade_id, None


def cancel_bot_trade(app, trade_id: int, user_id: int):
    """Cancel an active trade and kill its pending timer."""
    from ..models.bot_trade import BotTrade
    from ..extensions import db
    from ..services.alpaca import AlpacaClient

    with app.app_context():
        trade = BotTrade.query.filter_by(id=trade_id, user_id=user_id).first()
        if not trade:
            return False, 'Trade not found'
        if trade.status not in ('active', 'buying'):
            return False, f'Cannot cancel — status is "{trade.status}"'

        _cancel_timer(trade_id)

        if trade.buy_order_id and trade.status == 'buying':
            try:
                AlpacaClient().cancel_order(trade.buy_order_id)
                _log(trade, 'Buy order cancelled on Alpaca')
            except Exception as e:
                _log(trade, f'Could not cancel buy order: {e}', 'WARN')

        trade.status    = 'cancelled'
        trade.error_msg = 'Cancelled by user'
        _log(trade, 'Trade cancelled by user')
        db.session.commit()

    return True, None


def recover_pending(app):
    """
    Called once at app startup.
    Reschedules sell timers for trades left in 'active' status from a previous run.
    """
    from ..models.bot_trade import BotTrade

    with app.app_context():
        active = BotTrade.query.filter_by(status='active').all()
        now    = datetime.utcnow()
        for t in active:
            delay = max(0.0, (t.sell_at - now).total_seconds())
            _schedule_sell(app, t.id, delay)
            logger.info(f'[Bot] Recovered trade #{t.id} ({t.symbol}) — sell in {delay:.0f}s')
        if active:
            logger.info(f'[Bot] {len(active)} active trade(s) recovered')


# ── Internal ──────────────────────────────────────────────────────────────────

def _spawn(fn, *args):
    t = threading.Thread(target=fn, args=args, daemon=True)
    t.start()


def _schedule_sell(app, trade_id: int, delay: float):
    def _run():
        time.sleep(0.05)
        _sell_trade(app, trade_id)

    timer = threading.Timer(delay, _run)
    timer.daemon = True
    with _lock:
        if trade_id in _timers:
            _timers[trade_id].cancel()
        _timers[trade_id] = timer
    timer.start()
    logger.info(f'[Bot] Sell timer #{trade_id} fires in {delay:.0f}s')


def _cancel_timer(trade_id: int):
    with _lock:
        timer = _timers.pop(trade_id, None)
    if timer:
        timer.cancel()


def _buy_then_schedule(app, trade_id: int):
    """
    Background thread:
      1. Place a NOTIONAL market BUY via Alpaca SDK (USD amount → Alpaca
         calculates qty from live order-book price at execution time).
      2. Poll the Alpaca SDK every FILL_POLL seconds until the order fills.
      3. Record actual filled_qty and filled_avg_price from Alpaca response.
      4. Reset sell_at to NOW + 15 min (clock starts from confirmed fill).
      5. Schedule the sell timer.
    """
    from ..models.bot_trade import BotTrade
    from ..extensions import db
    from ..services.alpaca import AlpacaClient

    with app.app_context():
        trade = db.session.get(BotTrade, trade_id)
        if not trade:
            return
        try:
            logger.info(
                f'[Bot] Submitting notional BUY ${trade.amount_usd:.2f} '
                f'of {trade.symbol} via Alpaca SDK')

            client = AlpacaClient()

            # ── Notional order: Alpaca resolves qty at execution time ──────
            order = client.place_order(
                symbol=trade.symbol,
                notional=trade.amount_usd,   # <── USD amount, not qty
                side='buy',
                order_type='market',
                time_in_force='gtc',
            )
            trade.buy_order_id = order['id']
            _log(trade,
                 f'BUY submitted via Alpaca SDK — '
                 f'order_id={order["id"]} initial_status={order.get("status")}')
            db.session.commit()

            # ── Poll Alpaca SDK until order is filled ─────────────────────
            filled = _wait_for_fill(client, order['id'], FILL_TIMEOUT)
            if filled:
                trade.buy_price  = _safe_float(filled.get('filled_avg_price'))
                trade.filled_qty = _safe_float(filled.get('filled_qty')) or trade.qty
                trade.qty        = trade.filled_qty   # overwrite estimate with real fill
                _log(trade,
                     f'BUY filled (Alpaca) — '
                     f'qty={trade.filled_qty:.6f} @ ${trade.buy_price:,.4f} '
                     f'| total=${trade.buy_price * trade.filled_qty:,.2f}')
            else:
                # Order placed but fill not confirmed — sell whatever qty we know
                _log(trade,
                     f'BUY order not confirmed within {FILL_TIMEOUT}s '
                     f'— will sell estimated qty {trade.qty:.6f}', 'WARN')

            # ── Start the 15-min clock from NOW (confirmed fill time) ─────
            trade.sell_at = datetime.utcnow() + timedelta(seconds=SELL_DELAY)
            trade.status  = 'active'
            _log(trade,
                 f'Bot trade ACTIVE — '
                 f'auto-SELL at {trade.sell_at.strftime("%Y-%m-%d %H:%M:%S")} UTC '
                 f'(+{SELL_DELAY//60} min)')
            db.session.commit()

        except Exception as e:
            logger.exception(f'[Bot] BUY error for trade #{trade_id}')
            trade.status    = 'error'
            trade.error_msg = str(e)
            _log(trade, f'BUY failed via Alpaca SDK: {e}', 'ERROR')
            db.session.commit()
            return   # Do NOT schedule sell if buy failed

    # Schedule sell SELL_DELAY seconds from now (outside app context)
    _schedule_sell(app, trade_id, SELL_DELAY)


def _sell_trade(app, trade_id: int):
    """Background thread: place sell order and record P&L."""
    from ..models.bot_trade import BotTrade
    from ..extensions import db
    from ..services.alpaca import AlpacaClient

    with app.app_context():
        trade = db.session.get(BotTrade, trade_id)
        if not trade or trade.status != 'active':
            logger.warning(f'[Bot] Skipping sell #{trade_id}: status={getattr(trade,"status","N/A")}')
            return

        sell_qty     = trade.filled_qty or trade.qty
        trade.status = 'selling'
        _log(trade,
             f'15-min timer fired — submitting SELL {sell_qty:.6f} {trade.symbol} '
             f'via Alpaca SDK')
        db.session.commit()

        try:
            client = AlpacaClient()
            # ── Qty-based SELL (we know the exact filled qty from the buy) ─
            order = client.place_order(
                symbol=trade.symbol,
                qty=sell_qty,          # exact qty from buy fill
                side='sell',
                order_type='market',
                time_in_force='gtc',
            )
            trade.sell_order_id = order['id']
            _log(trade,
                 f'SELL submitted via Alpaca SDK — '
                 f'order_id={order["id"]} initial_status={order.get("status")}')
            db.session.commit()

            filled = _wait_for_fill(client, order['id'], FILL_TIMEOUT)
            if filled:
                trade.sell_price = _safe_float(filled.get('filled_avg_price'))
                _log(trade,
                     f'SELL filled (Alpaca) — '
                     f'qty={sell_qty:.6f} @ ${trade.sell_price:,.4f} '
                     f'| total=${trade.sell_price * sell_qty:,.2f}')
            else:
                _log(trade, 'SELL order placed but fill not confirmed within timeout', 'WARN')

            trade.sold_at = datetime.utcnow()

            # P&L
            if trade.buy_price and trade.sell_price:
                trade.profit     = round((trade.sell_price - trade.buy_price) * sell_qty, 4)
                trade.profit_pct = round((trade.sell_price - trade.buy_price) / trade.buy_price * 100, 4)
                sign = '+' if trade.profit >= 0 else ''
                _log(trade,
                    f'P&L: {sign}${trade.profit:.4f} ({sign}{trade.profit_pct:.4f}%)')

            trade.status = 'completed'
            _log(trade, 'Trade completed ✓')
            db.session.commit()

        except Exception as e:
            logger.exception(f'[Bot] SELL error for trade #{trade_id}')
            trade.status    = 'error'
            trade.error_msg = str(e)
            _log(trade, f'SELL failed: {e}', 'ERROR')
            db.session.commit()

    with _lock:
        _timers.pop(trade_id, None)


def _wait_for_fill(client, order_id: str, timeout: int):
    """Poll until filled/cancelled/error or timeout. Returns order dict or None."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            o      = client.get_order(order_id)
            status = (o.get('status') or '').lower()
            logger.debug(f'[Bot] Poll {order_id}: {status}')
            if status in ('filled', 'partially_filled'):
                return o
            if status in ('canceled', 'cancelled', 'expired', 'rejected'):
                logger.warning(f'[Bot] Order {order_id} terminal status: {status}')
                return o
        except Exception as e:
            logger.warning(f'[Bot] Poll error {order_id}: {e}')
        time.sleep(FILL_POLL)
    logger.warning(f'[Bot] Fill timeout for {order_id}')
    return None


def _log(trade, message: str, level: str = 'INFO'):
    logs = list(trade.logs or [])
    logs.append({'ts': datetime.utcnow().isoformat(), 'level': level, 'msg': message})
    trade.logs = logs
    logger.info(f'[BotTrade #{trade.id}] {message}')


def _safe_float(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None
