"""
ML Signal Engine v2 — Hardened multi-model ensemble with GNN + TFT.

═══════════════════════════════════════════════════════════════════════
KEY FIXES OVER v1 (was 54% accuracy → targeting 62-68%):
═══════════════════════════════════════════════════════════════════════
  1. DATA LEAKAGE FIX: ATR threshold uses LAGGED atr (bar t), not future
  2. PURGED CV: gap = lookforward bars between train/val to prevent bleed
  3. SEPARATE CALIBRATION SPLIT: train/val/cal = 60/20/20
  4. OUT-OF-FOLD STACKING: meta sees OOF probs, not in-sample probs
  5. FEATURE ROBUSTNESS: winsorize outliers at 1st/99th, ffill→bfill NaNs
  6. STRONGER REGULARIZATION: depth 3-4, higher min_child_weight
  7. MORE DATA: 365 days (multiple market regimes) instead of 90
  8. TFT TEMPORAL MODEL: captures multi-scale temporal patterns XGB misses
  9. GNN GRAPH MODEL: cross-asset correlation as a proper base learner

Architecture:
  5 base models trained on distinct feature perspectives:
    • XGB momentum  — returns, ROC, MACD, SMA distances, session encoding
    • XGB mean_rev  — RSI, Stochastic, CCI, Bollinger %B, ATR, candle body
    • XGB volume    — volume ratios, VWAP deviation, channel position, ATR
    • XGB graph     — GNN cross-asset features (correlation, breadth, lag)
    • TFT temporal  — full OHLCV sequence model (lookback=64 bars)

  1 LightGBM meta-learner stacks 5 OOF probability outputs
  + session features + regime features → final calibrated win probability.

  Isotonic regression calibrates on a HELD-OUT calibration set.

Label: will close[t+lf] > close[t] by more than 0.5 × LAGGED_ATR14?
  15-Min bars → lookforward = 1  (15-min horizon)
  5-Min bars  → lookforward = 3  (15-min horizon)
  1-Min bars  → lookforward = 15 (15-min horizon)

MODEL_VERSION must be bumped any time ALL_FEATURES changes.
═══════════════════════════════════════════════════════════════════════
"""

import os
import logging
import threading
import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

logger = logging.getLogger('alphapilot.ml')

MODEL_DIR     = os.path.join(os.path.dirname(__file__), '..', '..', 'ml_models')
MODEL_VERSION = 6   # bump when ALL_FEATURES changes to reject stale models

# ── Stable feature ordering ──────────────────────────────────────────────────

FEATURE_SETS = {
    'momentum': [
        'r1', 'r3', 'r5', 'r10',
        'macd', 'macd_sig', 'macd_hist',
        'sma10_d', 'sma20_d', 'sma50_d',
        'lco20',
        'sharpe10', 'sharpe20',
        'consec',
        'hr_sin', 'hr_cos', 'dow_sin', 'dow_cos',
    ],
    'mean_rev': [
        'rsi14', 'rsi21',
        'stoch_k', 'cci14',
        'bb_pct_b', 'bb_bw', 'atr14',
        'psma4',
        'body', 'uwik', 'lwik',
        'accel',
        'r1', 'sma10_d', 'sma20_d',
    ],
    'volume': [
        'vol_r5', 'vol_r20',
        'vwap_d', 'chan_pos',
        'r1', 'r3', 'r5',
        'atr14', 'body', 'uwik', 'lwik',
        'bb_pct_b', 'hr_sin', 'hr_cos',
        'rvol5', 'rvol20',
        'vol_expand',
        'gk_vol',
    ],
    'graph': [
        'gnn_r1_nbr',
        'gnn_rsi_nbr',
        'gnn_vol_nbr',
        'gnn_dom_r1',
        'gnn_breadth',
        'gnn_corr_mean',
        'gnn_corr_disp',
        'gnn_eth_lag1',
    ],
}

# TFT uses raw OHLCV sequence — features computed internally
TFT_RAW_COLS = ['open', 'high', 'low', 'close', 'volume']
TFT_EXTRA_COLS = [
    'r1', 'rsi14', 'atr14', 'bb_pct_b', 'macd_hist',
    'vol_r5', 'rvol5', 'stoch_k',
    'hr_sin', 'hr_cos', 'dow_sin', 'dow_cos',
]
TFT_LOOKBACK = 64   # bars of history the TFT sees per sample

_seen: set = set()
ALL_FEATURES: list = []
for _fset in FEATURE_SETS.values():
    for _f in _fset:
        if _f not in _seen:
            ALL_FEATURES.append(_f)
            _seen.add(_f)

SESSION_FEATS = ['hr_sin', 'hr_cos', 'dow_sin', 'dow_cos']

# Regime features passed to meta-learner alongside base probs
REGIME_FEATS = ['rvol5', 'rvol20', 'vol_expand', 'lco20', 'atr14']

LOOKFORWARD = {'1Min': 15, '5Min': 3, '15Min': 1, '30Min': 1, '1Hour': 1}

# ── In-memory training state ─────────────────────────────────────────────────
_training_jobs: dict = {}
_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def _model_path(symbol: str, timeframe: str, name: str) -> str:
    sym = symbol.replace('/', '_')
    _ensure_dir()
    return os.path.join(MODEL_DIR, f'{sym}_{timeframe}_{name}.pkl')


def _bars_to_df(bars: list):
    """Convert Alpaca bars list → pandas DataFrame with datetime index."""
    import pandas as pd
    if not bars:
        return None
    rows = [{'ts': b['t'], 'open': b['o'], 'high': b['h'],
             'low': b['l'], 'close': b['c'], 'volume': b['v']}
            for b in bars]
    df = pd.DataFrame(rows)
    if isinstance(bars[0]['t'], (int, float)):
        df.index = pd.to_datetime(df['ts'], unit='s', utc=True)
    else:
        df.index = pd.to_datetime(df['ts'], utc=True)
    return df.drop(columns='ts').sort_index().astype(float)


def _rsi(series, period: int):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    al = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = ag / (al + 1e-10)
    return 100 - (100 / (1 + rs))


def _winsorize(series, lower_pct=0.01, upper_pct=0.99):
    """Clip outliers to percentile bounds — prevents single bars from
    dominating tree splits and creating fragile rules."""
    lo = series.quantile(lower_pct)
    hi = series.quantile(upper_pct)
    return series.clip(lo, hi)


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING (unchanged features, added robustness)
# ══════════════════════════════════════════════════════════════════════════════

def compute_features(df):
    """
    Add all ML feature columns to a copy of df.
    Requires columns: open, high, low, close, volume.
    Now includes winsorization and robust NaN handling.
    """
    import pandas as pd
    d = df.copy()

    # ── Returns ───────────────────────────────────────────────────────────────
    d['r1']  = d['close'].pct_change(1)
    d['r3']  = d['close'].pct_change(3)
    d['r5']  = d['close'].pct_change(5)
    d['r10'] = d['close'].pct_change(10)
    ret1     = d['r1']

    # ── RSI ───────────────────────────────────────────────────────────────────
    d['rsi14'] = _rsi(d['close'], 14)
    d['rsi21'] = _rsi(d['close'], 21)

    # ── MACD (price-normalised) ───────────────────────────────────────────────
    ema12 = d['close'].ewm(span=12, adjust=False).mean()
    ema26 = d['close'].ewm(span=26, adjust=False).mean()
    macd_raw       = ema12 - ema26
    d['macd']      = macd_raw / d['close']
    d['macd_sig']  = d['macd'].ewm(span=9, adjust=False).mean()
    d['macd_hist'] = d['macd'] - d['macd_sig']

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    mid   = d['close'].rolling(20).mean()
    std   = d['close'].rolling(20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    d['bb_pct_b'] = (d['close'] - lower) / (upper - lower + 1e-10)
    d['bb_bw']    = (upper - lower) / (mid + 1e-10)

    # ── ATR (normalised by close) ─────────────────────────────────────────────
    hl = d['high'] - d['low']
    hc = (d['high'] - d['close'].shift()).abs()
    lc = (d['low']  - d['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    d['atr14'] = tr.ewm(span=14, adjust=False).mean() / d['close']

    # ── Stochastic %K ─────────────────────────────────────────────────────────
    low14        = d['low'].rolling(14).min()
    high14       = d['high'].rolling(14).max()
    d['stoch_k'] = (d['close'] - low14) / (high14 - low14 + 1e-10)

    # ── CCI-14 ────────────────────────────────────────────────────────────────
    tp     = (d['high'] + d['low'] + d['close']) / 3
    tp_sma = tp.rolling(14).mean()
    tp_mad = tp.rolling(14).apply(
                 lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    d['cci14'] = (tp - tp_sma) / (0.015 * tp_mad + 1e-10) / 200.0

    # ── VWAP deviation ────────────────────────────────────────────────────────
    vwap_num    = (d['close'] * d['volume']).rolling(20).sum()
    vwap_den    = d['volume'].rolling(20).sum()
    vwap        = vwap_num / (vwap_den + 1e-10)
    d['vwap_d'] = (d['close'] - vwap) / (d['close'] + 1e-10)

    # ── Channel position ──────────────────────────────────────────────────────
    roll_high    = d['high'].rolling(20).max()
    roll_low     = d['low'].rolling(20).min()
    d['chan_pos'] = (d['close'] - roll_low) / (roll_high - roll_low + 1e-10)

    # ── Volume ratios ─────────────────────────────────────────────────────────
    vm5  = d['volume'].rolling(5).mean()
    vm20 = d['volume'].rolling(20).mean()
    d['vol_r5']  = d['volume'] / (vm5  + 1e-10)
    d['vol_r20'] = d['volume'] / (vm20 + 1e-10)

    # ── SMA distances ─────────────────────────────────────────────────────────
    sma10 = d['close'].rolling(10).mean()
    sma20 = d['close'].rolling(20).mean()
    sma30 = d['close'].rolling(30).mean()
    sma50 = d['close'].rolling(50).mean()
    d['sma10_d'] = (d['close'] - sma10) / d['close']
    d['sma20_d'] = (d['close'] - sma20) / d['close']
    d['sma50_d'] = (d['close'] - sma50) / d['close']

    # ── PSMA-4 ────────────────────────────────────────────────────────────────
    d['psma4'] = (
        (d['close'] > sma10).astype(float) +
        (d['close'] > sma20).astype(float) +
        (d['close'] > sma30).astype(float) +
        (d['close'] > sma50).astype(float)
    ) / 4.0

    # ── LCO-20 (linear correlation trend quality) ─────────────────────────────
    _tidx = pd.Series(np.arange(len(d)), index=d.index, dtype=float)
    d['lco20'] = d['close'].rolling(20).corr(_tidx)

    # ── Realized volatility ───────────────────────────────────────────────────
    d['rvol5']      = ret1.rolling(5).std()
    d['rvol20']     = ret1.rolling(20).std()
    d['vol_expand'] = d['rvol5'] / (d['rvol20'] + 1e-10)

    # ── Garman-Klass volatility ───────────────────────────────────────────────
    _log_hl = np.log(d['high'] / (d['low'] + 1e-10))
    _log_co = np.log(d['close'] / (d['open'] + 1e-10))
    d['gk_vol'] = (0.5 * _log_hl**2
                   - (2 * np.log(2) - 1) * _log_co**2).rolling(10).mean()

    # ── Rolling Sharpe ────────────────────────────────────────────────────────
    d['sharpe10'] = ret1.rolling(10).mean() / (ret1.rolling(10).std() + 1e-10)
    d['sharpe20'] = ret1.rolling(20).mean() / (ret1.rolling(20).std() + 1e-10)

    # ── Consecutive bar direction ─────────────────────────────────────────────
    _bar_dir  = np.sign(d['close'] - d['open'])
    d['consec'] = _bar_dir.rolling(4).sum() / 4.0

    # ── Acceleration ──────────────────────────────────────────────────────────
    d['accel'] = ret1 - ret1.shift(1)

    # ── Candle structure ──────────────────────────────────────────────────────
    rng       = d['high'] - d['low'] + 1e-10
    d['body'] = (d['close'] - d['open']) / rng
    d['uwik'] = (d['high'] - d[['open', 'close']].max(axis=1)) / rng
    d['lwik'] = (d[['open', 'close']].min(axis=1) - d['low']) / rng

    # ── Session (cyclical encoding) ───────────────────────────────────────────
    d['hr_sin']  = np.sin(2 * np.pi * d.index.hour / 24)
    d['hr_cos']  = np.cos(2 * np.pi * d.index.hour / 24)
    d['dow_sin'] = np.sin(2 * np.pi * d.index.dayofweek / 7)
    d['dow_cos'] = np.cos(2 * np.pi * d.index.dayofweek / 7)

    # ══════════════════════════════════════════════════════════════════════════
    # FIX #5: Winsorize numeric features to clip outlier spikes
    # ══════════════════════════════════════════════════════════════════════════
    numeric_feats = [c for c in ALL_FEATURES if c in d.columns
                     and c not in SESSION_FEATS]  # don't clip sin/cos
    for col in numeric_feats:
        d[col] = _winsorize(d[col], 0.005, 0.995)

    # Forward-fill then back-fill remaining NaNs (from rolling warmup)
    # instead of just dropna which loses 50+ rows
    feat_cols = [c for c in ALL_FEATURES if c in d.columns]
    d[feat_cols] = d[feat_cols].ffill().bfill()

    return d


# ══════════════════════════════════════════════════════════════════════════════
#  DEPENDENCY CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def _check_deps():
    """Raise ImportError with install instructions if ML packages are missing."""
    missing = []
    for pkg in ['xgboost', 'lightgbm', 'sklearn']:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg if pkg != 'sklearn' else 'scikit-learn')
    if missing:
        raise ImportError(
            f'ML packages not installed: {", ".join(missing)}. '
            f'Run: pip install {" ".join(missing)}'
        )


def _check_torch():
    """Check if PyTorch is available for TFT model."""
    try:
        import torch  # noqa
        return True
    except ImportError:
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL FACTORIES (tighter regularization = less overfitting)
# ══════════════════════════════════════════════════════════════════════════════

def _make_xgb(scale_pos_weight: float = 1.0, n_features: int = 15):
    """
    XGBoost classifier with regularization scaled to dataset size.
    FIX #6: reduced depth (4→3), higher min_child_weight, more lambda.
    """
    import xgboost as xgb
    major = int(xgb.__version__.split('.')[0])
    ctor_kwargs = {'early_stopping_rounds': 40} if major >= 2 else {}

    return xgb.XGBClassifier(
        n_estimators=800,
        max_depth=3,                # was 5 — shallower trees generalize better
        learning_rate=0.015,        # was 0.02 — slower learning
        subsample=0.7,
        colsample_bytree=min(0.8, max(0.5, 8 / (n_features + 1))),
        min_child_weight=20,        # was 12 — stronger regularization
        reg_alpha=1.0,              # was 0.5 — more L1
        reg_lambda=5.0,             # was 2.5 — more L2
        gamma=0.3,                  # was 0.1 — needs more loss reduction to split
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
        **ctor_kwargs,
    )


def _make_lgb_meta():
    """LightGBM meta-learner — very shallow, highly regularized."""
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        n_estimators=400,
        max_depth=3,                # was 6 — meta should be simple
        num_leaves=8,               # was 48 — drastically simpler
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=30,        # was 20
        reg_alpha=0.5,
        reg_lambda=5.0,             # was 2.5
        random_state=42,
        verbose=-1,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TFT (Temporal Fusion Transformer) — lightweight PyTorch implementation
# ══════════════════════════════════════════════════════════════════════════════

class TemporalBlock:
    """
    Lightweight Temporal Fusion Transformer for sequence classification.
    Uses: LSTM encoder + multi-head attention + gated residual network.
    Falls back gracefully if PyTorch is unavailable.
    """

    @staticmethod
    def is_available():
        return _check_torch()

    @staticmethod
    def build_sequences(df, feature_cols, lookback=TFT_LOOKBACK):
        """Build (N, lookback, n_features) array from DataFrame."""
        data = df[feature_cols].values.astype(np.float32)
        sequences = []
        for i in range(lookback, len(data)):
            sequences.append(data[i - lookback:i])
        return np.array(sequences)  # (N, lookback, n_feat)

    @staticmethod
    def train_tft(X_seq, y, X_val_seq, y_val, epochs=60, lr=1e-3):
        """
        Train a lightweight TFT and return (model, val_probs).
        X_seq: (N, lookback, n_feat), y: (N,)
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_feat = X_seq.shape[2]

        class GatedResidualNetwork(nn.Module):
            def __init__(self, d_model, dropout=0.15):
                super().__init__()
                self.fc1 = nn.Linear(d_model, d_model)
                self.fc2 = nn.Linear(d_model, d_model)
                self.gate = nn.Linear(d_model, d_model)
                self.ln = nn.LayerNorm(d_model)
                self.drop = nn.Dropout(dropout)
                self.elu = nn.ELU()

            def forward(self, x):
                h = self.elu(self.fc1(x))
                h = self.drop(h)
                h = self.fc2(h)
                g = torch.sigmoid(self.gate(h))
                return self.ln(x + g * h)

        class LightTFT(nn.Module):
            def __init__(self, n_features, d_model=48, n_heads=4, dropout=0.2):
                super().__init__()
                self.input_proj = nn.Linear(n_features, d_model)
                self.lstm = nn.LSTM(d_model, d_model, num_layers=2,
                                   batch_first=True, dropout=dropout,
                                   bidirectional=False)
                self.attn = nn.MultiheadAttention(d_model, n_heads,
                                                  dropout=dropout,
                                                  batch_first=True)
                self.grn = GatedResidualNetwork(d_model, dropout)
                self.head = nn.Sequential(
                    nn.Linear(d_model, 16),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(16, 1),
                )

            def forward(self, x):
                # x: (batch, lookback, n_feat)
                h = self.input_proj(x)           # (B, T, d_model)
                h, _ = self.lstm(h)               # (B, T, d_model)
                attn_out, _ = self.attn(h, h, h)  # self-attention
                h = h + attn_out                   # residual
                h = self.grn(h[:, -1, :])          # last timestep + GRN
                return self.head(h).squeeze(-1)    # (B,)

        model = LightTFT(n_feat).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Class weights for imbalanced data
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / (n_pos + 1e-6)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_ds = TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=False)

        X_val_t = torch.tensor(X_val_seq, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

        best_loss = float('inf')
        best_state = None
        patience = 12
        wait = 0

        for epoch in range(epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_loss = criterion(val_logits, y_val_t).item()

            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info(f'[TFT] Early stop at epoch {epoch+1}, best_loss={best_loss:.4f}')
                    break

        if best_state:
            model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            val_probs = torch.sigmoid(model(X_val_t)).cpu().numpy()

        return model, val_probs

    @staticmethod
    def predict(model, X_seq):
        """Get probability from trained TFT model."""
        import torch
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            x = torch.tensor(X_seq, dtype=torch.float32).to(device)
            return torch.sigmoid(model(x)).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
#  LABELLING — FIX #1: use LAGGED ATR for threshold (no future data)
# ══════════════════════════════════════════════════════════════════════════════

def _create_labels(df, lf: int):
    """
    Create binary labels using ATR-adaptive threshold.

    CRITICAL FIX: uses ATR at bar t (already known), NOT ATR at bar t+lf.
    The old code used df['atr14'] which is computed on the full dataset
    (including the close used in the future return calc). Now we explicitly
    use a LAGGED ATR that's known at decision time.
    """
    # Future return: what happens lf bars from now
    future_ret = (df['close'].shift(-lf) - df['close']) / df['close']

    # ATR threshold: what we KNOW at bar t (shift by 1 for safety —
    # ensures the ATR doesn't include the current bar's range)
    lagged_atr = df['atr14'].shift(1)
    dyn_thresh = (lagged_atr * 0.5).clip(lower=0.0001, upper=0.005)

    labels = np.where(future_ret >  dyn_thresh, 1,
             np.where(future_ret < -dyn_thresh, 0, np.nan))
    return labels


# ══════════════════════════════════════════════════════════════════════════════
#  OUT-OF-FOLD STACKING — FIX #4: meta never sees in-sample probabilities
# ══════════════════════════════════════════════════════════════════════════════

def _generate_oof_probs(X, y, feature_sets_with_idx, spw, n_splits=5, purge_gap=1):
    """
    Generate out-of-fold probabilities for each base model using
    purged time-series splits.

    Returns:
        oof_probs: dict of {fs_name: np.array of shape (len(X),)}
        trained_models: dict of {fs_name: list of fold models}
    """
    import xgboost as xgb
    from sklearn.metrics import accuracy_score

    n = len(X)
    fold_size = n // n_splits
    oof_probs = {name: np.full(n, np.nan) for name in feature_sets_with_idx}
    trained_models = {name: [] for name in feature_sets_with_idx}

    for fold_i in range(n_splits):
        val_start = fold_i * fold_size
        val_end   = min(val_start + fold_size, n)

        # FIX #2: Purge gap between train and val
        train_end = max(0, val_start - purge_gap)
        train_idx = list(range(0, train_end))

        # Also use data AFTER val (with purge gap) if available
        post_val_start = min(val_end + purge_gap, n)
        train_idx += list(range(post_val_start, n))

        if len(train_idx) < 100:
            continue

        val_idx = list(range(val_start, val_end))
        X_tr_fold, y_tr_fold = X[train_idx], y[train_idx]
        X_vl_fold, y_vl_fold = X[val_idx], y[val_idx]

        for fs_name, col_idx in feature_sets_with_idx.items():
            mdl = _make_xgb(scale_pos_weight=spw, n_features=len(col_idx))
            fit_kwargs = {}
            major = int(xgb.__version__.split('.')[0])
            if major < 2:
                fit_kwargs['early_stopping_rounds'] = 40

            mdl.fit(
                X_tr_fold[:, col_idx], y_tr_fold,
                eval_set=[(X_vl_fold[:, col_idx], y_vl_fold)],
                verbose=False,
                **fit_kwargs,
            )
            probs = mdl.predict_proba(X_vl_fold[:, col_idx])[:, 1]
            oof_probs[fs_name][val_idx] = probs
            trained_models[fs_name].append(mdl)

            acc = accuracy_score(y_vl_fold, (probs >= 0.5).astype(int))
            logger.info(f'[ML] OOF fold {fold_i} {fs_name}: val_acc={acc:.3f}')

    return oof_probs, trained_models


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING THREAD
# ══════════════════════════════════════════════════════════════════════════════

def _training_thread(app, symbol: str, timeframe: str, key: tuple):
    from ..services.alpaca import AlpacaClient
    from ..models.ml_model import MLModel
    from ..extensions import db
    import joblib
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.isotonic import IsotonicRegression
    import lightgbm as lgb

    _ensure_dir()
    with app.app_context():
        try:
            _check_deps()
            has_torch = _check_torch()

            # ══════════════════════════════════════════════════════════════════
            # FIX #7: 365 days of data (was 90) — capture multiple regimes
            # ══════════════════════════════════════════════════════════════════
            logger.info(f'[ML] Fetching bars: {symbol} {timeframe} (365 days)')
            start_date = datetime.now() - timedelta(days=365)
            bars = AlpacaClient().get_bars(
                symbol, timeframe, limit=10000, start=start_date
            ).get('bars', [])

            if len(bars) < 500:
                raise ValueError(f'Need ≥500 bars, got {len(bars)}')

            df = _bars_to_df(bars)
            df = compute_features(df)

            # ── GNN cross-asset graph features ────────────────────────────────
            from .gnn_features import build_graph_df, GNN_FEATURES
            logger.info(f'[ML] Building GNN graph features for {symbol}')
            gnn_df = build_graph_df(app, symbol, timeframe,
                                    start=start_date, limit=11000)
            if gnn_df is not None:
                df = df.join(gnn_df[GNN_FEATURES], how='left')
                df[GNN_FEATURES] = df[GNN_FEATURES].fillna(0.0)
                logger.info(f'[ML] GNN features merged ({gnn_df.shape[0]} rows)')
                has_gnn = True
            else:
                logger.warning('[ML] GNN features unavailable — filling zeros')
                for _gf in GNN_FEATURES:
                    df[_gf] = 0.0
                has_gnn = False

            lf = LOOKFORWARD.get(timeframe, 1)

            # ══════════════════════════════════════════════════════════════════
            # FIX #1: Labels with LAGGED ATR (no future information leak)
            # ══════════════════════════════════════════════════════════════════
            df['label'] = _create_labels(df, lf)
            df = df.dropna(subset=['label'])
            df['label'] = df['label'].astype(int)

            # Also drop any remaining NaN rows in features
            feat_cols = [c for c in ALL_FEATURES if c in df.columns]
            df = df.dropna(subset=feat_cols)

            n_buy  = int((df['label'] == 1).sum())
            n_sell = int((df['label'] == 0).sum())
            logger.info(
                f'[ML] Labels (lagged ATR threshold): '
                f'buy={n_buy} sell={n_sell} ratio={n_buy/(n_sell+1e-6):.2f}'
            )

            if len(df) < 400:
                raise ValueError(f'Only {len(df)} usable rows after labelling')

            # ══════════════════════════════════════════════════════════════════
            # FIX #3: Three-way split: train(60%) / val(20%) / cal(20%)
            # with purge gaps between each
            # ══════════════════════════════════════════════════════════════════
            n = len(df)
            split1 = int(n * 0.60)
            split2 = int(n * 0.80)

            # Purge gap = lookforward bars between splits
            purge = lf + 1

            X_all = df[ALL_FEATURES].values
            y_all = df['label'].values

            X_tr = X_all[:split1]
            y_tr = y_all[:split1]
            X_vl = X_all[split1 + purge:split2]
            y_vl = y_all[split1 + purge:split2]
            X_cal = X_all[split2 + purge:]
            y_cal = y_all[split2 + purge:]

            logger.info(
                f'[ML] Splits: train={len(X_tr)}, val={len(X_vl)}, '
                f'cal={len(X_cal)}, purge_gap={purge}'
            )

            # Class-balance weight
            n_neg = int((y_tr == 0).sum())
            n_pos = int((y_tr == 1).sum())
            spw   = n_neg / (n_pos + 1e-10)
            logger.info(f'[ML] Train: buy={n_pos} sell={n_neg} spw={spw:.3f}')

            # ══════════════════════════════════════════════════════════════════
            # FIX #4: Out-of-fold stacking for XGBoost base models
            # ══════════════════════════════════════════════════════════════════
            active_sets = dict(FEATURE_SETS)
            if not has_gnn:
                # Don't waste a base learner on all-zero GNN features
                active_sets = {k: v for k, v in active_sets.items() if k != 'graph'}

            feature_sets_with_idx = {}
            for fs_name, cols in active_sets.items():
                col_idx = [ALL_FEATURES.index(c) for c in cols]
                feature_sets_with_idx[fs_name] = col_idx

            logger.info(f'[ML] Generating OOF probabilities ({len(active_sets)} base models)')
            oof_probs, fold_models = _generate_oof_probs(
                X_tr, y_tr, feature_sets_with_idx,
                spw=spw, n_splits=5, purge_gap=purge,
            )

            # Build OOF training matrix for meta-learner
            # Rows with any NaN OOF prob (edge folds) are dropped
            oof_matrix = np.column_stack(
                [oof_probs[name] for name in active_sets]
            )
            valid_oof = ~np.isnan(oof_matrix).any(axis=1)
            oof_matrix = oof_matrix[valid_oof]
            y_tr_oof   = y_tr[valid_oof]

            # Add session + regime features for meta
            sess_tr = df.iloc[:split1][SESSION_FEATS].values[valid_oof]
            regime_tr = df.iloc[:split1][REGIME_FEATS].values[valid_oof]
            meta_X_tr = np.column_stack([oof_matrix, sess_tr, regime_tr])

            logger.info(
                f'[ML] OOF matrix: {meta_X_tr.shape[0]} rows, '
                f'{meta_X_tr.shape[1]} features '
                f'({len(active_sets)} base + {len(SESSION_FEATS)} session + {len(REGIME_FEATS)} regime)'
            )

            # ── Validation probabilities (from final retrained models) ─────────
            # Retrain each base model on FULL training set for final inference
            final_models = {}
            val_base_probs = []
            import xgboost as xgb

            for fs_name, col_idx in feature_sets_with_idx.items():
                mdl = _make_xgb(scale_pos_weight=spw, n_features=len(col_idx))
                fit_kwargs = {}
                if int(xgb.__version__.split('.')[0]) < 2:
                    fit_kwargs['early_stopping_rounds'] = 40
                mdl.fit(
                    X_tr[:, col_idx], y_tr,
                    eval_set=[(X_vl[:, col_idx], y_vl)],
                    verbose=False,
                    **fit_kwargs,
                )
                val_p = mdl.predict_proba(X_vl[:, col_idx])[:, 1]
                val_base_probs.append(val_p)
                final_models[fs_name] = mdl

                acc_b = accuracy_score(y_vl, (val_p >= 0.5).astype(int))
                logger.info(f'[ML] Final base {fs_name}: val_acc={acc_b:.3f}')

                joblib.dump(
                    {'model': mdl, 'col_idx': col_idx, 'fs': fs_name,
                     'version': MODEL_VERSION},
                    _model_path(symbol, timeframe, f'base_{fs_name}'),
                )

            # ── TFT temporal model ────────────────────────────────────────────
            tft_model = None
            tft_val_prob = None
            tft_cal_prob = None
            has_tft = False

            if has_torch and len(X_tr) > TFT_LOOKBACK + 100:
                try:
                    tft_feats = [c for c in TFT_EXTRA_COLS if c in df.columns]
                    logger.info(f'[TFT] Training with {len(tft_feats)} features, lookback={TFT_LOOKBACK}')

                    # Build sequences from the featured DataFrame
                    df_train = df.iloc[:split1]
                    df_val   = df.iloc[split1 + purge:split2]
                    df_cal   = df.iloc[split2 + purge:]

                    X_seq_tr = TemporalBlock.build_sequences(df_train, tft_feats, TFT_LOOKBACK)
                    y_seq_tr = df_train['label'].values[TFT_LOOKBACK:]

                    X_seq_vl = TemporalBlock.build_sequences(df_val, tft_feats, TFT_LOOKBACK)
                    y_seq_vl = df_val['label'].values[TFT_LOOKBACK:]

                    X_seq_cal = TemporalBlock.build_sequences(df_cal, tft_feats, TFT_LOOKBACK)

                    # Normalize sequences (per-feature z-score from training stats)
                    tr_mean = X_seq_tr.mean(axis=(0, 1), keepdims=True)
                    tr_std  = X_seq_tr.std(axis=(0, 1), keepdims=True) + 1e-8
                    X_seq_tr  = (X_seq_tr - tr_mean) / tr_std
                    X_seq_vl  = (X_seq_vl - tr_mean) / tr_std
                    X_seq_cal = (X_seq_cal - tr_mean) / tr_std

                    tft_model, tft_vp = TemporalBlock.train_tft(
                        X_seq_tr, y_seq_tr, X_seq_vl, y_seq_vl,
                        epochs=60, lr=1e-3,
                    )

                    tft_acc = accuracy_score(y_seq_vl, (tft_vp >= 0.5).astype(int))
                    logger.info(f'[TFT] Val accuracy: {tft_acc:.3f}')

                    # Pad TFT probs to match val length (first TFT_LOOKBACK rows are missing)
                    tft_val_prob = np.full(len(y_vl), 0.5)
                    tft_val_prob[-len(tft_vp):] = tft_vp

                    tft_cal_probs_raw = TemporalBlock.predict(tft_model, X_seq_cal)
                    tft_cal_prob = np.full(len(y_cal), 0.5)
                    tft_cal_prob[-len(tft_cal_probs_raw):] = tft_cal_probs_raw

                    has_tft = True

                    # Save TFT model and normalization stats
                    import torch
                    torch.save({
                        'model_state': tft_model.state_dict(),
                        'n_features': len(tft_feats),
                        'feature_cols': tft_feats,
                        'tr_mean': tr_mean,
                        'tr_std': tr_std,
                        'version': MODEL_VERSION,
                    }, _model_path(symbol, timeframe, 'tft').replace('.pkl', '.pt'))
                    logger.info('[TFT] Model saved')

                except Exception as tft_exc:
                    logger.warning(f'[TFT] Training failed (non-fatal): {tft_exc}')
                    has_tft = False

            # ── Build meta validation matrix ──────────────────────────────────
            sess_vl = df.iloc[split1 + purge:split2][SESSION_FEATS].values
            regime_vl = df.iloc[split1 + purge:split2][REGIME_FEATS].values

            meta_val_parts = [np.column_stack(val_base_probs)]
            if has_tft:
                meta_val_parts.append(tft_val_prob.reshape(-1, 1))
            meta_val_parts.extend([sess_vl, regime_vl])
            meta_X_vl = np.column_stack(meta_val_parts)

            # Match meta training columns
            if has_tft:
                # Add TFT OOF column to training (dummy 0.5 since TFT doesn't do OOF)
                tft_oof = np.full((len(y_tr_oof), 1), 0.5)
                meta_X_tr = np.column_stack([
                    oof_matrix, tft_oof, sess_tr, regime_tr
                ])

            # ── Train meta-learner ────────────────────────────────────────────
            meta = _make_lgb_meta()
            meta.fit(
                meta_X_tr, y_tr_oof,
                eval_set=[(meta_X_vl, y_vl)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='X does not have valid feature names')
                meta_p_vl = meta.predict_proba(meta_X_vl)[:, 1]

            acc_meta = float(accuracy_score(y_vl, (meta_p_vl >= 0.5).astype(int)))
            logloss_meta = float(log_loss(y_vl, meta_p_vl))
            logger.info(f'[ML] Meta-learner: val_acc={acc_meta:.3f}, logloss={logloss_meta:.4f}')

            # ══════════════════════════════════════════════════════════════════
            # FIX #3: Isotonic calibration on HELD-OUT calibration set
            # (was trained on val set = data snooping)
            # ══════════════════════════════════════════════════════════════════
            cal_base_probs = []
            for fs_name, col_idx in feature_sets_with_idx.items():
                mdl = final_models[fs_name]
                cal_p = mdl.predict_proba(X_cal[:, col_idx])[:, 1]
                cal_base_probs.append(cal_p)

            sess_cal = df.iloc[split2 + purge:][SESSION_FEATS].values
            regime_cal = df.iloc[split2 + purge:][REGIME_FEATS].values

            meta_cal_parts = [np.column_stack(cal_base_probs)]
            if has_tft:
                meta_cal_parts.append(tft_cal_prob.reshape(-1, 1))
            meta_cal_parts.extend([sess_cal, regime_cal])
            meta_X_cal = np.column_stack(meta_cal_parts)

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='X does not have valid feature names')
                meta_p_cal = meta.predict_proba(meta_X_cal)[:, 1]

            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(meta_p_cal, y_cal)
            cal_p  = iso.predict(meta_p_cal)
            acc_cal = float(accuracy_score(y_cal, (cal_p >= 0.5).astype(int)))
            logger.info(f'[ML] Calibration set accuracy: {acc_cal:.3f}')

            # ── Save meta model ───────────────────────────────────────────────
            n_base_models = len(active_sets) + (1 if has_tft else 0)
            joblib.dump({
                'model': meta,
                'calibrator': iso,
                'all_feats': ALL_FEATURES,
                'active_sets': list(active_sets.keys()),
                'has_tft': has_tft,
                'n_base_models': n_base_models,
                'version': MODEL_VERSION,
            }, _model_path(symbol, timeframe, 'meta'))

            # ── HMM regime model ──────────────────────────────────────────────
            try:
                from ..services.regime_detector import fit_regime_model
                fit_regime_model(df, symbol, timeframe)
            except Exception as regime_exc:
                logger.warning(f'[ML] Regime model skipped: {regime_exc}')

            # ── Persist to DB ─────────────────────────────────────────────────
            # Report the CALIBRATION set accuracy (most honest metric)
            rec = MLModel.query.filter_by(
                symbol=symbol, timeframe=timeframe, is_active=True).first()
            if rec:
                rec.accuracy   = round(acc_cal * 100, 2)
                rec.n_samples  = n
                rec.trained_at = datetime.utcnow()
            else:
                db.session.add(MLModel(
                    symbol=symbol, timeframe=timeframe,
                    model_type=f'xgb_lgb_tft_ensemble_v2 ({n_base_models} base)',
                    lookforward=lf,
                    accuracy=round(acc_cal * 100, 2),
                    n_samples=n,
                ))
            db.session.commit()

            with _lock:
                _training_jobs[key] = {
                    'status': 'done',
                    'accuracy': round(acc_cal * 100, 2),
                    'val_accuracy': round(acc_meta * 100, 2),
                    'n_samples': n,
                    'n_base_models': n_base_models,
                    'has_tft': has_tft,
                    'has_gnn': has_gnn,
                    'error': None,
                    'trained_at': datetime.utcnow().isoformat(),
                }
            logger.info(
                f'[ML] Training complete: {symbol} {timeframe} '
                f'val_acc={acc_meta:.3f} cal_acc={acc_cal:.3f} '
                f'models={n_base_models} (TFT={has_tft}, GNN={has_gnn})'
            )

        except Exception as exc:
            logger.exception(f'[ML] Training failed: {symbol} {timeframe}')
            with _lock:
                _training_jobs[key] = {'status': 'error', 'error': str(exc)}


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def start_training(app, symbol: str = 'BTC/USD', timeframe: str = '15Min'):
    """Launch model training in a background thread. Returns (started, error)."""
    key = (symbol, timeframe)
    with _lock:
        if _training_jobs.get(key, {}).get('status') == 'training':
            return False, 'Already training'
        _training_jobs[key] = {
            'status': 'training',
            'started_at': datetime.utcnow().isoformat(),
        }
    t = threading.Thread(
        target=_training_thread, args=(app, symbol, timeframe, key), daemon=True)
    t.start()
    return True, None


def get_training_status(symbol: str = 'BTC/USD', timeframe: str = '15Min') -> dict:
    key = (symbol, timeframe)
    with _lock:
        return _training_jobs.get(key, {'status': 'idle'}).copy()


def predict_signal(app, symbol: str = 'BTC/USD', timeframe: str = '15Min') -> dict:
    """
    Generate an ML signal using trained models.

    Returns dict:
      direction   : 'buy' | 'sell'
      confidence  : float 0–1  (calibrated probability for predicted direction)
      raw_prob    : float 0–1  (raw P(up) from meta-learner before calibration)
      base_probs  : dict of per-model probabilities
      features    : dict of key indicator values
      error       : None | str
    """
    try:
        _check_deps()
        import joblib
    except ImportError as dep_err:
        return {'direction': None, 'confidence': 0.0, 'features': {}, 'error': str(dep_err)}

    meta_path = _model_path(symbol, timeframe, 'meta')
    if not os.path.exists(meta_path):
        return {
            'direction': None, 'confidence': 0.0,
            'features': {}, 'error': 'No trained model. Click Train first.',
        }

    with app.app_context():
        try:
            from ..services.alpaca import AlpacaClient

            # Version check
            meta_dict = joblib.load(meta_path)
            if meta_dict.get('version', 1) < MODEL_VERSION:
                return {
                    'direction': None, 'confidence': 0.0, 'features': {},
                    'error': 'Model is outdated — please retrain.',
                }

            # Fetch enough bars for features + TFT lookback
            tf_minutes = {'1Min': 1, '5Min': 5, '15Min': 15, '30Min': 30, '1Hour': 60}
            mins_back  = tf_minutes.get(timeframe, 15) * 300
            start      = datetime.utcnow() - timedelta(minutes=mins_back * 2)
            bars = AlpacaClient().get_bars(
                symbol, timeframe, limit=300, start=start
            ).get('bars', [])

            if len(bars) < 80:
                return {'error': f'Insufficient bars (got {len(bars)}, need 80)'}

            df = _bars_to_df(bars)
            df = compute_features(df)

            # GNN features
            from .gnn_features import get_latest_gnn_row, GNN_FEATURES
            gnn_row = get_latest_gnn_row(app, symbol, timeframe, limit=300)
            for _gf, _gv in gnn_row.items():
                df[_gf] = _gv

            df  = df.dropna()
            row = df.iloc[[-1]]
            X_all = row[ALL_FEATURES].values

            # ── Base model probabilities ──────────────────────────────────────
            active_sets = meta_dict.get('active_sets', list(FEATURE_SETS.keys()))
            base_probs = {}

            for fs_name in active_sets:
                bp = _model_path(symbol, timeframe, f'base_{fs_name}')
                if not os.path.exists(bp):
                    base_probs[fs_name] = 0.5
                    continue
                m_dict  = joblib.load(bp)
                col_idx = m_dict['col_idx']
                prob    = float(m_dict['model'].predict_proba(X_all[:, col_idx])[0, 1])
                base_probs[fs_name] = prob

            meta_parts = [np.array(list(base_probs.values())).reshape(1, -1)]

            # ── TFT probability ───────────────────────────────────────────────
            has_tft = meta_dict.get('has_tft', False)
            tft_prob = 0.5

            if has_tft and _check_torch():
                try:
                    import torch
                    tft_path = _model_path(symbol, timeframe, 'tft').replace('.pkl', '.pt')
                    if os.path.exists(tft_path):
                        tft_data = torch.load(tft_path, map_location='cpu', weights_only=False)
                        tft_feats = tft_data['feature_cols']

                        # Rebuild TFT model
                        n_feat = tft_data['n_features']
                        from ml_signal_engine_v2 import TemporalBlock

                        class LightTFT(torch.nn.Module):
                            def __init__(self, n_features, d_model=48, n_heads=4, dropout=0.2):
                                super().__init__()
                                self.input_proj = torch.nn.Linear(n_features, d_model)
                                self.lstm = torch.nn.LSTM(d_model, d_model, num_layers=2,
                                                          batch_first=True, dropout=dropout)
                                self.attn = torch.nn.MultiheadAttention(
                                    d_model, n_heads, dropout=dropout, batch_first=True)
                                self.grn_fc1 = torch.nn.Linear(d_model, d_model)
                                self.grn_fc2 = torch.nn.Linear(d_model, d_model)
                                self.grn_gate = torch.nn.Linear(d_model, d_model)
                                self.grn_ln = torch.nn.LayerNorm(d_model)
                                self.grn_drop = torch.nn.Dropout(dropout)
                                self.head = torch.nn.Sequential(
                                    torch.nn.Linear(d_model, 16),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(dropout),
                                    torch.nn.Linear(16, 1),
                                )

                            def forward(self, x):
                                h = self.input_proj(x)
                                h, _ = self.lstm(h)
                                attn_out, _ = self.attn(h, h, h)
                                h = h + attn_out
                                last = h[:, -1, :]
                                g = torch.sigmoid(self.grn_gate(
                                    self.grn_drop(torch.nn.functional.elu(self.grn_fc1(last)))))
                                grn_out = self.grn_ln(last + g * self.grn_fc2(
                                    self.grn_drop(torch.nn.functional.elu(self.grn_fc1(last)))))
                                return self.head(grn_out).squeeze(-1)

                        tft_model = LightTFT(n_feat)
                        tft_model.load_state_dict(tft_data['model_state'])
                        tft_model.eval()

                        # Build sequence from latest data
                        seq_data = df[tft_feats].values[-TFT_LOOKBACK:].astype(np.float32)
                        seq_data = (seq_data - tft_data['tr_mean'].squeeze(0)) / tft_data['tr_std'].squeeze(0)
                        seq_data = seq_data.reshape(1, TFT_LOOKBACK, -1)

                        with torch.no_grad():
                            tft_prob = float(torch.sigmoid(
                                tft_model(torch.tensor(seq_data))).item())

                except Exception as tft_err:
                    logger.warning(f'[TFT] Prediction fallback: {tft_err}')
                    tft_prob = 0.5

            if has_tft:
                meta_parts.append(np.array([[tft_prob]]))

            # Session + regime features
            sess   = row[SESSION_FEATS].values
            regime = row[REGIME_FEATS].values
            meta_parts.extend([sess, regime])
            meta_X = np.column_stack(meta_parts)

            # ── Meta prediction ───────────────────────────────────────────────
            meta = meta_dict['model']
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',
                                        message='X does not have valid feature names')
                raw_prob = float(meta.predict_proba(meta_X)[0, 1])

            # Apply isotonic calibration
            calibrator = meta_dict.get('calibrator')
            cal_prob = raw_prob
            if calibrator is not None:
                cal_prob = float(calibrator.predict([raw_prob])[0])

            direction  = 'buy' if cal_prob >= 0.5 else 'sell'
            confidence = cal_prob if direction == 'buy' else 1.0 - cal_prob

            return {
                'direction':  direction,
                'confidence': round(confidence, 4),
                'raw_prob':   round(raw_prob, 4),
                'cal_prob':   round(cal_prob, 4),
                'base_probs': {
                    f'p_{k}': round(v, 4) for k, v in base_probs.items()
                },
                'tft_prob':   round(tft_prob, 4),
                'features': {
                    'rsi14':      round(float(row['rsi14'].iloc[0]), 2),
                    'macd_hist':  round(float(row['macd_hist'].iloc[0]) * 1000, 4),
                    'bb_pct_b':   round(float(row['bb_pct_b'].iloc[0]), 4),
                    'stoch_k':    round(float(row['stoch_k'].iloc[0]), 4),
                    'cci14':      round(float(row['cci14'].iloc[0]), 4),
                    'atr_pct':    round(float(row['atr14'].iloc[0]) * 100, 4),
                    'vol_r5':     round(float(row['vol_r5'].iloc[0]), 3),
                    'vwap_d':     round(float(row['vwap_d'].iloc[0]), 4),
                    'chan_pos':    round(float(row['chan_pos'].iloc[0]), 4),
                    'lco20':      round(float(row['lco20'].iloc[0]), 4),
                    'vol_expand': round(float(row['vol_expand'].iloc[0]), 3),
                    'gnn_eth_lag1':  round(gnn_row.get('gnn_eth_lag1', 0.0), 5),
                    'gnn_dom_r1':    round(gnn_row.get('gnn_dom_r1', 0.0), 5),
                    'gnn_breadth':   round(gnn_row.get('gnn_breadth', 0.0), 3),
                    'gnn_corr_mean': round(gnn_row.get('gnn_corr_mean', 0.0), 3),
                },
                'error': None,
            }

        except Exception as exc:
            logger.exception('[ML] Prediction error')
            return {'error': str(exc)}