"""
Pure-Python strategy calculations — no external dependencies required.
All functions operate on plain lists of floats.
"""


# ── Indicator helpers ───────────────────────────────────────────────────────

def _sma(prices, period):
    result = []
    for i in range(len(prices)):
        if i < period - 1:
            result.append(None)
        else:
            result.append(round(sum(prices[i - period + 1:i + 1]) / period, 4))
    return result


def _ema(prices, period):
    if len(prices) < period:
        return [None] * len(prices)
    result = [None] * (period - 1)
    result.append(round(sum(prices[:period]) / period, 4))
    k = 2 / (period + 1)
    for i in range(period, len(prices)):
        result.append(round(prices[i] * k + result[-1] * (1 - k), 4))
    return result


def _rsi(prices, period=14):
    if len(prices) <= period:
        return [None] * len(prices)
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [max(d, 0) for d in deltas]
    losses = [abs(min(d, 0)) for d in deltas]
    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period

    def _r(ag, al):
        return 100.0 if al == 0 else round(100 - 100 / (1 + ag / al), 2)

    result = [None] * period + [_r(avg_g, avg_l)]
    for i in range(period, len(deltas)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        result.append(_r(avg_g, avg_l))
    return result


def _bollinger(prices, period=20, std_mult=2):
    m = _sma(prices, period)
    result = []
    for i in range(len(prices)):
        if m[i] is None:
            result.append({'upper': None, 'middle': None, 'lower': None})
        else:
            window = prices[i - period + 1:i + 1]
            std = (sum((p - m[i]) ** 2 for p in window) / period) ** 0.5
            result.append({
                'upper': round(m[i] + std_mult * std, 4),
                'middle': round(m[i], 4),
                'lower': round(m[i] - std_mult * std, 4),
            })
    return result


def _macd(prices, fast=12, slow=26, signal_period=9):
    ef = _ema(prices, fast)
    es = _ema(prices, slow)
    macd_line = [
        round(f - s, 4) if (f is not None and s is not None) else None
        for f, s in zip(ef, es)
    ]
    indexed = [(i, v) for i, v in enumerate(macd_line) if v is not None]
    if len(indexed) < signal_period:
        none_arr = [None] * len(macd_line)
        return {'macd': macd_line, 'signal': none_arr, 'histogram': none_arr}
    raw_signal = _ema([v for _, v in indexed], signal_period)
    signal_line = [None] * len(macd_line)
    for idx, (orig_i, _) in enumerate(indexed):
        if idx < len(raw_signal) and raw_signal[idx] is not None:
            signal_line[orig_i] = raw_signal[idx]
    histogram = [
        round(m - s, 4) if (m is not None and s is not None) else None
        for m, s in zip(macd_line, signal_line)
    ]
    return {'macd': macd_line, 'signal': signal_line, 'histogram': histogram}


# ── Serialisation helper ────────────────────────────────────────────────────

def _series(timestamps, values):
    return [{'time': t, 'value': v} for t, v in zip(timestamps, values) if v is not None]


# ── Bar time formatter ──────────────────────────────────────────────────────

def _fmt_time(t, timeframe):
    """Return 'YYYY-MM-DD' for daily bars, Unix-seconds int for intraday."""
    if timeframe in ('1Day', '1Week', '1Month'):
        return str(t)[:10]   # works for both ISO strings and int timestamps
    if isinstance(t, int):
        return t             # already Unix seconds (Binance intraday)
    from datetime import datetime, timezone
    dt = datetime.fromisoformat(t.replace('Z', '+00:00'))
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


# ── Public entry point ──────────────────────────────────────────────────────

def run_strategy(strategy_type, bars, params, timeframe='1Day'):
    """
    Accepts the raw bars list from Alpaca (dicts with t/o/h/l/c/v).
    Returns { candles, volume, indicators, signals }.
    """
    ts = [_fmt_time(b['t'], timeframe) for b in bars]
    closes = [b['c'] for b in bars]

    candles = [
        {'time': ts[i], 'open': b['o'], 'high': b['h'], 'low': b['l'], 'close': b['c']}
        for i, b in enumerate(bars)
    ]
    volume = [
        {'time': ts[i], 'value': b['v'],
         'color': '#3fb950' if b['c'] >= b['o'] else '#f85149'}
        for i, b in enumerate(bars)
    ]

    indicators = {}
    signals = []

    # ── SMA Crossover ───────────────────────────────────────────────────────
    if strategy_type == 'sma_crossover':
        fast_p = int(params.get('fast_period', 10))
        slow_p = int(params.get('slow_period', 20))
        fast = _sma(closes, fast_p)
        slow = _sma(closes, slow_p)
        indicators['fast_sma'] = _series(ts, fast)
        indicators['slow_sma'] = _series(ts, slow)
        for i in range(1, len(closes)):
            if None in (fast[i], slow[i], fast[i - 1], slow[i - 1]):
                continue
            if fast[i - 1] <= slow[i - 1] and fast[i] > slow[i]:
                signals.append({'time': bars[i]['t'], 'action': 'buy', 'price': closes[i]})
            elif fast[i - 1] >= slow[i - 1] and fast[i] < slow[i]:
                signals.append({'time': bars[i]['t'], 'action': 'sell', 'price': closes[i]})

    # ── RSI ─────────────────────────────────────────────────────────────────
    elif strategy_type == 'rsi':
        period = int(params.get('period', 14))
        oversold = float(params.get('oversold', 30))
        overbought = float(params.get('overbought', 70))
        rsi_vals = _rsi(closes, period)
        indicators['rsi'] = _series(ts, rsi_vals)
        for i in range(1, len(rsi_vals)):
            if rsi_vals[i] is None or rsi_vals[i - 1] is None:
                continue
            if rsi_vals[i - 1] < oversold and rsi_vals[i] >= oversold:
                signals.append({'time': bars[i]['t'], 'action': 'buy', 'price': closes[i]})
            elif rsi_vals[i - 1] > overbought and rsi_vals[i] <= overbought:
                signals.append({'time': bars[i]['t'], 'action': 'sell', 'price': closes[i]})

    # ── Bollinger Bands ─────────────────────────────────────────────────────
    elif strategy_type == 'bollinger':
        period = int(params.get('period', 20))
        std_dev = float(params.get('std_dev', 2.0))
        bb = _bollinger(closes, period, std_dev)
        indicators['bb_upper'] = _series(ts, [b['upper'] for b in bb])
        indicators['bb_middle'] = _series(ts, [b['middle'] for b in bb])
        indicators['bb_lower'] = _series(ts, [b['lower'] for b in bb])
        for i in range(1, len(closes)):
            if bb[i]['upper'] is None or bb[i - 1]['upper'] is None:
                continue
            if closes[i - 1] > bb[i - 1]['lower'] and closes[i] <= bb[i]['lower']:
                signals.append({'time': bars[i]['t'], 'action': 'buy', 'price': closes[i]})
            elif closes[i - 1] < bb[i - 1]['upper'] and closes[i] >= bb[i]['upper']:
                signals.append({'time': bars[i]['t'], 'action': 'sell', 'price': closes[i]})

    # ── MACD ────────────────────────────────────────────────────────────────
    elif strategy_type == 'macd':
        fast_p = int(params.get('fast', 12))
        slow_p = int(params.get('slow', 26))
        sig_p = int(params.get('signal', 9))
        md = _macd(closes, fast_p, slow_p, sig_p)
        indicators['macd'] = _series(ts, md['macd'])
        indicators['macd_signal'] = _series(ts, md['signal'])
        indicators['macd_histogram'] = _series(ts, md['histogram'])
        m, s = md['macd'], md['signal']
        for i in range(1, len(m)):
            if None in (m[i], s[i], m[i - 1], s[i - 1]):
                continue
            if m[i - 1] <= s[i - 1] and m[i] > s[i]:
                signals.append({'time': bars[i]['t'], 'action': 'buy', 'price': closes[i]})
            elif m[i - 1] >= s[i - 1] and m[i] < s[i]:
                signals.append({'time': bars[i]['t'], 'action': 'sell', 'price': closes[i]})

    return {
        'candles': candles,
        'volume': volume,
        'indicators': indicators,
        'signals': signals,
    }
