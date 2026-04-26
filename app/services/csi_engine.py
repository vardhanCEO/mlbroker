"""
CSI Engine — Combined Signal Indicator (6-component multi-confirmation filter).

Mirrors the TradingView CSI by requiring N of 6 independent indicators to agree
before returning a confirmed direction. Default threshold: 4 of 6.

Components:
  1. PSMA     — % of a range of SMAs that price trades above (broad MA sentiment)
  2. SupOsc   — SuperTrend oscillator; buy when osc crosses up through -0.50
  3. Willy    — Williams %%R on the PREVIOUS candle (lookback confirmation)
  4. LCO      — Linear Correlation of price with time (reversal-zone detection)
  5. EvR      — Volume-weighted Effort vs Results (RROF directional oscillator)
  6. StochSup — SuperTrend applied directly to StochRSI; flip + position check

Usage:
    result = csi_score(app, 'BTC/USD', '15Min')
    # result['buy_score']  → int 0-6
    # result['sell_score'] → int 0-6
    # result['passes']     → bool (dominant score >= min_score)
    # result['direction']  → 'buy' | 'sell' | None
"""
import logging

logger = logging.getLogger('alphapilot.csi')

# ── Default parameters ─────────────────────────────────────────────────────────
DEFAULT_PARAMS = {
    # PSMA — both raw and smooth must be in extreme territory
    'psma_periods':    [10, 20, 30, 50, 100, 200],
    'psma_smooth':     5,
    'psma_buy_level':  20,   # below this → oversold (raise to 30-35 for more signals)
    'psma_sell_level': 80,   # above this → overbought (lower to 70-65 for more signals)

    # SupOsc — SuperTrend-based oscillator cross
    'suposc_atr_period':  10,
    'suposc_multiplier':  2.0,  # lower toward 1.5 for more frequent crosses
    'suposc_threshold':   0.50,

    # Willy — Williams %R, checked on PREVIOUS candle
    'willy_period':     14,
    'willy_oversold':   -80,   # raise to -70 for more buy eligibility
    'willy_overbought': -20,   # lower to -30 for more sell eligibility

    # LCO — linear correlation of price with time index
    'lco_period': 20,
    'lco_lower':  -0.8,  # raise toward -0.5 for more buy-eligible zones
    'lco_upper':   0.8,  # lower toward 0.5 for more sell-eligible zones

    # EvR — volume-weighted effort vs results (RROF)
    'evr_period': 14,
    'evr_smooth':  3,

    # StochSup — SuperTrend on StochRSI; checked over a 3-bar recency window
    'stochsup_rsi_period':   14,
    'stochsup_stoch_period': 14,
    'stochsup_k_smooth':      3,
    'stochsup_multiplier':   10,  # lower to 6-7 for more frequent flips
    'stochsup_atr_period':    3,

    # Gate threshold
    'min_score': 4,   # components required to agree (out of 6)
}


# ── Pure-Python math helpers ───────────────────────────────────────────────────

def _smean(values, period):
    """Rolling simple mean. None for warmup bars."""
    n = len(values)
    result = [None] * n
    for i in range(period - 1, n):
        window = [v for v in values[i - period + 1:i + 1] if v is not None]
        if len(window) == period:
            result[i] = sum(window) / period
    return result


def _ema_list(values, period):
    """EMA over a list that may contain None. Seeds from first `period` non-None values."""
    n = len(values)
    result = [None] * n
    k = 2.0 / (period + 1)

    seed_vals, seed_end = [], 0
    for i, v in enumerate(values):
        if v is not None:
            seed_vals.append(v)
            if len(seed_vals) == period:
                seed_end = i
                break
    if len(seed_vals) < period:
        return result

    result[seed_end] = sum(seed_vals) / period
    for j in range(seed_end + 1, n):
        if values[j] is not None and result[j - 1] is not None:
            result[j] = values[j] * k + result[j - 1] * (1 - k)
    return result


def _true_range(highs, lows, closes):
    tr = [highs[0] - lows[0]]
    for i in range(1, len(closes)):
        tr.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]),
        ))
    return tr


def _atr(highs, lows, closes, period):
    return _ema_list(_true_range(highs, lows, closes), period)


def _rsi_list(closes, period):
    n = len(closes)
    if n <= period:
        return [None] * n
    deltas = [closes[i] - closes[i - 1] for i in range(1, n)]
    gains  = [max(d, 0.0)   for d in deltas]
    losses = [abs(min(d, 0.0)) for d in deltas]
    ag = sum(gains[:period])  / period
    al = sum(losses[:period]) / period
    result = [None] * period
    result.append(100 - 100 / (1 + ag / (al + 1e-10)))
    for i in range(period, len(deltas)):
        ag = (ag * (period - 1) + gains[i])  / period
        al = (al * (period - 1) + losses[i]) / period
        result.append(100 - 100 / (1 + ag / (al + 1e-10)))
    return result


# ── SuperTrend (price-series, needs OHLC ATR) ─────────────────────────────────

def _supertrend(closes, highs, lows, atr_vals, multiplier):
    """
    Standard SuperTrend on OHLC data (uses pre-computed ATR).
    Returns (direction, upper_band, lower_band) — all as lists.
    direction: 1 = bullish, -1 = bearish, None = warmup.
    """
    n = len(closes)
    direction = [None] * n
    upper     = [None] * n
    lower     = [None] * n

    for i in range(n):
        if atr_vals[i] is None:
            continue
        hl2 = (highs[i] + lows[i]) / 2.0
        bu  = hl2 + multiplier * atr_vals[i]
        bl  = hl2 - multiplier * atr_vals[i]

        if upper[i - 1] is None:
            upper[i], lower[i] = bu, bl
            direction[i] = 1 if closes[i] > bu else -1
        else:
            upper[i] = min(bu, upper[i - 1]) if closes[i - 1] <= upper[i - 1] else bu
            lower[i] = max(bl, lower[i - 1]) if closes[i - 1] >= lower[i - 1] else bl
            if direction[i - 1] == -1:
                direction[i] = 1 if closes[i] > upper[i] else -1
            else:
                direction[i] = -1 if closes[i] < lower[i] else 1

    return direction, upper, lower


# ── SuperTrend on 1-D oscillator (no OHLC needed) ────────────────────────────

def _supertrend_1d(values, atr_period, multiplier):
    """
    SuperTrend applied to any scalar series (e.g. StochRSI).
    Uses |consecutive differences| as a proxy for ATR.
    """
    n = len(values)
    changes = [None]
    for i in range(1, n):
        if values[i] is not None and values[i - 1] is not None:
            changes.append(abs(values[i] - values[i - 1]))
        else:
            changes.append(None)
    atr_vals = _ema_list(changes, atr_period)

    direction = [None] * n
    upper     = [None] * n
    lower     = [None] * n

    for i in range(n):
        if values[i] is None or atr_vals[i] is None:
            continue
        bu = values[i] + multiplier * atr_vals[i]
        bl = values[i] - multiplier * atr_vals[i]

        if upper[i - 1] is None:
            upper[i], lower[i] = bu, bl
            direction[i] = 1
        else:
            prev_v = values[i - 1]
            upper[i] = min(bu, upper[i - 1]) if prev_v is not None and prev_v <= upper[i - 1] else bu
            lower[i] = max(bl, lower[i - 1]) if prev_v is not None and prev_v >= lower[i - 1] else bl
            if direction[i - 1] == -1:
                direction[i] = 1 if values[i] > upper[i] else -1
            else:
                direction[i] = -1 if values[i] < lower[i] else 1

    return direction, upper, lower


# ── Component checkers — each returns (buy: bool|None, sell: bool|None) ───────

def _check_psma(closes, p):
    """
    PSMA: % of [10,20,30,50,100,200]-bar SMAs that price is above.
    Buy  → raw < buy_level  AND smooth < buy_level
    Sell → raw > sell_level AND smooth > sell_level
    """
    periods = [per for per in p['psma_periods'] if per <= len(closes)]
    if not periods:
        return None, None

    # Pre-compute all SMAs
    smas = {}
    for per in periods:
        sma = [None] * (per - 1)
        for i in range(per - 1, len(closes)):
            sma.append(sum(closes[i - per + 1:i + 1]) / per)
        smas[per] = sma

    raw = []
    for i in range(len(closes)):
        valid   = [per for per in periods if smas[per][i] is not None]
        if not valid:
            raw.append(None)
        else:
            above = sum(1 for per in valid if closes[i] > smas[per][i])
            raw.append(above / len(valid) * 100.0)

    smooth = _ema_list(raw, p['psma_smooth'])
    if raw[-1] is None or smooth[-1] is None:
        return None, None

    buy  = raw[-1] < p['psma_buy_level']  and smooth[-1] < p['psma_buy_level']
    sell = raw[-1] > p['psma_sell_level'] and smooth[-1] > p['psma_sell_level']
    return buy, sell


def _check_suposc(closes, highs, lows, p):
    """
    SupOsc: SuperTrend-based oscillator.
    Oscillator = (close - band_midpoint) / (ATR * multiplier)
    Buy  → osc crosses up through -threshold (last 2 bars)
    Sell → osc crosses down through +threshold (last 2 bars)
    """
    atr_vals = _atr(highs, lows, closes, p['suposc_atr_period'])
    direction, upper, lower = _supertrend(closes, highs, lows, atr_vals, p['suposc_multiplier'])

    n = len(closes)
    osc = []
    for i in range(n):
        if direction[i] is None or atr_vals[i] is None:
            osc.append(None)
        else:
            mid   = (upper[i] + lower[i]) / 2.0
            denom = atr_vals[i] * p['suposc_multiplier'] + 1e-10
            osc.append((closes[i] - mid) / denom)

    if len(osc) < 2 or osc[-1] is None or osc[-2] is None:
        return None, None

    t = p['suposc_threshold']
    buy  = osc[-2] < -t and osc[-1] >= -t
    sell = osc[-2] >  t and osc[-1] <=  t
    return buy, sell


def _check_willy(closes, highs, lows, p):
    """
    Willy: Williams %%R on the PREVIOUS candle.
    Buy  → previous %R < oversold  (e.g. < -80)
    Sell → previous %R > overbought (e.g. > -20)
    """
    period = p['willy_period']
    n = len(closes)
    w = [None] * n
    for i in range(period - 1, n):
        hh = max(highs[i - period + 1:i + 1])
        ll = min(lows[i  - period + 1:i + 1])
        w[i] = (hh - closes[i]) / (hh - ll + 1e-10) * -100.0

    # "Previous candle" = w[-2]
    if len(w) < 2 or w[-2] is None:
        return None, None

    buy  = w[-2] < p['willy_oversold']
    sell = w[-2] > p['willy_overbought']
    return buy, sell


def _check_lco(closes, p):
    """
    LCO: Pearson correlation of price with a linear time index.
    +1 = perfect uptrend, -1 = perfect downtrend.
    Buy  → corr < lco_lower (strongly negative = potential reversal zone)
    Sell → corr > lco_upper (strongly positive = potential exhaustion zone)
    """
    period = p['lco_period']
    n      = len(closes)
    corr   = [None] * n

    x   = list(range(period))
    sx  = sum(x)
    sx2 = sum(xi ** 2 for xi in x)

    for i in range(period - 1, n):
        window = closes[i - period + 1:i + 1]
        sy  = sum(window)
        sxy = sum(xi * yi for xi, yi in zip(x, window))
        sy2 = sum(yi ** 2 for yi in window)
        num = period * sxy - sx * sy
        den = ((period * sx2 - sx ** 2) * (period * sy2 - sy ** 2)) ** 0.5
        corr[i] = num / den if den > 1e-10 else 0.0

    if corr[-1] is None:
        return None, None

    buy  = corr[-1] < p['lco_lower']
    sell = corr[-1] > p['lco_upper']
    return buy, sell


def _check_evr(closes, opens, highs, lows, volumes, p):
    """
    EvR (RROF): Volume-weighted directional effort vs results.
    RROF = body_direction * (vol / avg_vol) * (range / avg_range)
    Buy  → smoothed RROF is currently rising
    Sell → smoothed RROF is currently falling
    """
    n       = len(closes)
    ranges  = [h - l for h, l in zip(highs, lows)]
    avg_vol = _smean(volumes, p['evr_period'])
    avg_rng = _smean(ranges,  p['evr_period'])

    rrof = []
    for i in range(n):
        if avg_vol[i] is None or avg_rng[i] is None:
            rrof.append(None)
        else:
            body_dir   = (closes[i] - opens[i]) / (ranges[i] + 1e-10)
            vol_factor = volumes[i] / (avg_vol[i] + 1e-10)
            rng_factor = ranges[i]  / (avg_rng[i] + 1e-10)
            rrof.append(body_dir * vol_factor * rng_factor)

    rrof_s = _ema_list(rrof, p['evr_smooth'])

    if len(rrof_s) < 2 or rrof_s[-1] is None or rrof_s[-2] is None:
        return None, None

    buy  = rrof_s[-1] > rrof_s[-2]   # rising = bulls in control
    sell = rrof_s[-1] < rrof_s[-2]   # falling = bears in control
    return buy, sell


def _check_stochsup(closes, highs, lows, p):
    """
    StochSup: SuperTrend applied to Stochastic RSI.
    Buy  → trend flips bullish (within last 3 bars) AND StochRSI < 50
    Sell → trend flips bearish (within last 3 bars) AND StochRSI > 50
    """
    # Stochastic RSI %K
    rsi    = _rsi_list(closes, p['stochsup_rsi_period'])
    sp     = p['stochsup_stoch_period']
    n      = len(rsi)
    raw_k  = [None] * n
    for i in range(n):
        if rsi[i] is None:
            continue
        window = [rsi[j] for j in range(max(0, i - sp + 1), i + 1)
                  if rsi[j] is not None]
        if len(window) < sp:
            continue
        lo = min(window)
        hi = max(window)
        raw_k[i] = (rsi[i] - lo) / (hi - lo + 1e-10) * 100.0

    stoch_k = _smean(raw_k, p['stochsup_k_smooth'])

    # SuperTrend on StochRSI
    direction, _, _ = _supertrend_1d(stoch_k, p['stochsup_atr_period'],
                                     p['stochsup_multiplier'])

    # Look for a direction flip within the last 3 bars
    buy = sell = False
    for i in range(max(1, n - 3), n):
        if (direction[i] is None or direction[i - 1] is None
                or stoch_k[i] is None):
            continue
        if direction[i] == 1 and direction[i - 1] == -1 and stoch_k[i] < 50:
            buy = True
        if direction[i] == -1 and direction[i - 1] == 1 and stoch_k[i] > 50:
            sell = True

    return buy, sell


# ── Public API ─────────────────────────────────────────────────────────────────

def csi_score(app, symbol: str, timeframe: str, params: dict = None) -> dict:
    """
    Compute the CSI multi-confirmation score for the current bar.

    Returns:
        buy_score   : int  — components confirming a buy (0-6)
        sell_score  : int  — components confirming a sell (0-6)
        components  : dict — per-component {'buy': bool, 'sell': bool}
        direction   : 'buy' | 'sell' | None
        passes      : bool — True if dominant score >= min_score
        min_score   : int
        error       : None | str
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    try:
        with app.app_context():
            from .alpaca import AlpacaClient
            from datetime import datetime, timedelta

            # Need 300+ bars: PSMA-200 warmup + extra for other indicators
            tf_minutes = {'1Min': 1, '5Min': 5, '15Min': 15, '30Min': 30, '1Hour': 60}
            mins_back = tf_minutes.get(timeframe, 15) * 350
            start = datetime.utcnow() - timedelta(minutes=mins_back)

            bars = AlpacaClient().get_bars(
                symbol, timeframe, limit=350, start=start
            ).get('bars', [])

            if len(bars) < 220:
                return _err(f'CSI needs 220+ bars, got {len(bars)}')

            closes  = [b['c'] for b in bars]
            opens   = [b['o'] for b in bars]
            highs   = [b['h'] for b in bars]
            lows    = [b['l'] for b in bars]
            volumes = [b['v'] for b in bars]

            checks = {
                'psma':     _check_psma(closes, p),
                'suposc':   _check_suposc(closes, highs, lows, p),
                'willy':    _check_willy(closes, highs, lows, p),
                'lco':      _check_lco(closes, p),
                'evr':      _check_evr(closes, opens, highs, lows, volumes, p),
                'stochsup': _check_stochsup(closes, highs, lows, p),
            }

            buy_score  = sum(1 for b, _ in checks.values() if b is True)
            sell_score = sum(1 for _, s in checks.values() if s is True)
            min_s      = p['min_score']

            direction = None
            if buy_score >= min_s and buy_score > sell_score:
                direction = 'buy'
            elif sell_score >= min_s and sell_score > buy_score:
                direction = 'sell'

            logger.debug(
                f'[CSI] {symbol} {timeframe} '
                f'buy={buy_score}/6 sell={sell_score}/6 → {direction}'
            )

            return {
                'buy_score':  buy_score,
                'sell_score': sell_score,
                'components': {
                    k: {'buy': bool(b) if b is not None else None,
                        'sell': bool(s) if s is not None else None}
                    for k, (b, s) in checks.items()
                },
                'direction': direction,
                'passes':    direction is not None,
                'min_score': min_s,
                'error':     None,
            }

    except Exception as exc:
        logger.exception('[CSI] Error computing score')
        return _err(str(exc))


def _err(msg: str) -> dict:
    return {
        'buy_score': 0, 'sell_score': 0,
        'components': {}, 'direction': None,
        'passes': False, 'min_score': 0,
        'error': msg,
    }
