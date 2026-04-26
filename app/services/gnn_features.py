"""
GNN Feature Engine — Graph Attention Network for cross-asset signal aggregation.

Graph structure:
  Nodes : 8 Alpaca crypto pairs (BTC, ETH, SOL, BNB, AVAX, DOGE, XRP, LINK)
  Edges : 20-bar rolling return correlation between target and each neighbor
  Weights: clip(corr, 0) normalised per-row → positive attention only

For a target node (e.g. BTC/USD) at each bar, the layer produces:

  gnn_r1_nbr    attention-weighted avg 1-bar return of neighbors
                → "if ETH + SOL are rising, BTC likely follows"

  gnn_rsi_nbr   attention-weighted avg RSI-14 of neighbors
                → "are BTC's correlated peers overbought / oversold?"

  gnn_vol_nbr   attention-weighted avg volume-ratio of neighbors
                → "unusual volume propagating from correlated assets"

  gnn_dom_r1    BTC r1 minus weighted-avg neighbor r1 (relative strength)
                → positive = BTC leading peers; negative = BTC lagging

  gnn_breadth   fraction of all 8 assets with RSI-14 > 50
                → crypto market breadth (risk-on / risk-off signal)

  gnn_corr_mean mean edge weight across all neighbors
                → how tightly the whole market is moving together

  gnn_corr_disp std of edge weights (correlation dispersion)
                → rising dispersion = correlation breakdown / regime shift

  gnn_eth_lag1  ETH's 1-bar-lagged return
                → ETH historically leads BTC price discovery
"""
import logging
import numpy as np

logger = logging.getLogger('alphapilot.gnn')

GNN_FEATURES = [
    'gnn_r1_nbr',
    'gnn_rsi_nbr',
    'gnn_vol_nbr',
    'gnn_dom_r1',
    'gnn_breadth',
    'gnn_corr_mean',
    'gnn_corr_disp',
    'gnn_eth_lag1',
]

_CORR_WINDOW = 20
_VOL_WINDOW  = 10
_RSI_PERIOD  = 14


# ── Internal helpers ───────────────────────────────────────────────────────────

def _rsi_pd(series, period=14):
    """EWM RSI — matches ml_engine._rsi() exactly."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    al = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return 100 - (100 / (1 + ag / (al + 1e-10)))


def _bars_to_series(bars):
    """Convert an Alpaca bar list → (close Series, volume Series) with UTC index."""
    import pandas as pd
    rows = []
    for b in bars:
        t = b['t']
        if isinstance(t, (int, float)):
            ts = pd.Timestamp(t, unit='s', tz='UTC')
        else:
            ts = pd.Timestamp(t).tz_localize('UTC') if t.endswith('Z') or '+' not in t \
                 else pd.Timestamp(t)
        rows.append((ts, float(b['c']), float(b['v'])))
    df = (
        __import__('pandas').DataFrame(rows, columns=['ts', 'close', 'volume'])
        .set_index('ts')
        .sort_index()
    )
    return df['close'], df['volume']


def _build_multi_asset_df(bars_dict):
    """
    Merge per-symbol bar lists → aligned (close_df, volume_df).
    Forward-fills any minor gaps so the index is contiguous.
    """
    import pandas as pd
    close_parts, vol_parts = {}, {}
    for sym, bars in bars_dict.items():
        if bars:
            c, v = _bars_to_series(bars)
            close_parts[sym] = c
            vol_parts[sym]   = v

    if len(close_parts) < 2:
        return None, None

    close_df  = pd.DataFrame(close_parts).sort_index().ffill()
    volume_df = pd.DataFrame(vol_parts).sort_index().ffill()
    return close_df, volume_df


def _graph_attention(symbol, close_df, volume_df):
    """
    Core computation: build attention-weighted neighbor features.
    Returns a DataFrame of GNN_FEATURES indexed by timestamp.
    """
    import pandas as pd

    if symbol not in close_df.columns:
        return None

    all_syms  = list(close_df.columns)
    neighbors = [s for s in all_syms if s != symbol]
    idx       = close_df.index

    # ── Per-asset derived series ───────────────────────────────────────────────
    ret_s   = {s: close_df[s].pct_change(1)                        for s in all_syms}
    rsi_s   = {s: _rsi_pd(close_df[s], _RSI_PERIOD)                for s in all_syms}
    vol_ma  = {s: volume_df[s].rolling(_VOL_WINDOW).mean()         for s in all_syms}
    vol_r_s = {s: volume_df[s] / (vol_ma[s] + 1e-10)              for s in all_syms}

    # ── Rolling correlation → attention ────────────────────────────────────────
    tgt_ret = ret_s[symbol]
    corr_df = pd.DataFrame(
        {nbr: tgt_ret.rolling(_CORR_WINDOW).corr(ret_s[nbr]) for nbr in neighbors},
        index=idx,
    )
    attn_df  = corr_df.clip(lower=0)              # positive attention only
    attn_sum = attn_df.sum(axis=1).replace(0, np.nan)
    attn_norm = attn_df.div(attn_sum, axis=0)    # rows sum to 1

    # ── Neighbor feature matrices ──────────────────────────────────────────────
    ret_nbr = pd.DataFrame({n: ret_s[n]   for n in neighbors}, index=idx)
    rsi_nbr = pd.DataFrame({n: rsi_s[n]   for n in neighbors}, index=idx)
    vol_nbr = pd.DataFrame({n: vol_r_s[n] for n in neighbors}, index=idx)

    feats = pd.DataFrame(index=idx)

    # Attention-weighted aggregations
    feats['gnn_r1_nbr']  = (attn_norm * ret_nbr).sum(axis=1)
    feats['gnn_rsi_nbr'] = (attn_norm * rsi_nbr).sum(axis=1)
    feats['gnn_vol_nbr'] = (attn_norm * vol_nbr).sum(axis=1)

    # Relative strength: how much is BTC leading / lagging peers?
    feats['gnn_dom_r1']  = tgt_ret - feats['gnn_r1_nbr']

    # Crypto market breadth: fraction of all 8 assets with RSI > 50
    rsi_all = pd.DataFrame({s: rsi_s[s] for s in all_syms}, index=idx)
    feats['gnn_breadth'] = (rsi_all > 50).mean(axis=1)

    # Edge-weight statistics (how correlated and how stable)
    feats['gnn_corr_mean'] = corr_df.mean(axis=1)
    feats['gnn_corr_disp'] = corr_df.std(axis=1)

    # ETH leading indicator (1-bar lag: ETH leads BTC price discovery)
    eth = 'ETH/USD'
    feats['gnn_eth_lag1'] = ret_s[eth].shift(1) if eth in ret_s else 0.0

    return feats


# ── Public API ─────────────────────────────────────────────────────────────────

def build_graph_df(app, symbol: str, timeframe: str,
                   start=None, limit: int = 300):
    """
    Fetch bars for all 8 crypto pairs and return a DataFrame of GNN_FEATURES
    aligned to the target symbol's timestamp index.

    Called once during training so the GNN features can be merged into the
    main training DataFrame and used to train the 'graph' base model.
    """
    try:
        with app.app_context():
            from .alpaca import AlpacaClient, TICKER_PAIRS
            from datetime import datetime, timedelta

            if start is None:
                tf_minutes = {'1Min': 1, '5Min': 5, '15Min': 15,
                              '30Min': 30, '1Hour': 60}
                mins_back = tf_minutes.get(timeframe, 15) * (limit + 60)
                start = datetime.utcnow() - timedelta(minutes=mins_back)

            client    = AlpacaClient()
            bars_dict = {}
            for sym in TICKER_PAIRS:
                try:
                    b = client.get_bars(sym, timeframe, limit=limit,
                                        start=start).get('bars', [])
                    if b:
                        bars_dict[sym] = b
                except Exception as e:
                    logger.warning(f'[GNN] {sym} fetch failed: {e}')

            if len(bars_dict) < 2:
                logger.warning('[GNN] Fewer than 2 symbols fetched; skipping.')
                return None

            close_df, volume_df = _build_multi_asset_df(bars_dict)
            if close_df is None:
                return None

            feats = _graph_attention(symbol, close_df, volume_df)
            if feats is not None:
                logger.info(
                    f'[GNN] Graph features built: {len(feats)} bars, '
                    f'{len(bars_dict)} nodes, target={symbol}'
                )
            return feats

    except Exception:
        logger.exception('[GNN] build_graph_df failed')
        return None


def get_latest_gnn_row(app, symbol: str, timeframe: str,
                        limit: int = 250) -> dict:
    """
    Compute GNN features for prediction and return the latest bar as a dict.
    Falls back to zeros if anything fails so prediction always completes.
    """
    try:
        feats = build_graph_df(app, symbol, timeframe, limit=limit)
        if feats is None or feats.empty:
            return {f: 0.0 for f in GNN_FEATURES}

        # Take the last row that has at least the core aggregation features
        valid = feats.dropna(subset=['gnn_r1_nbr', 'gnn_corr_mean'])
        if valid.empty:
            return {f: 0.0 for f in GNN_FEATURES}

        last = valid.iloc[-1]
        return {f: float(last.get(f, 0.0)) for f in GNN_FEATURES}

    except Exception:
        logger.exception('[GNN] get_latest_gnn_row failed')
        return {f: 0.0 for f in GNN_FEATURES}
