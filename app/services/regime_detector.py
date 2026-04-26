"""
HMM Regime Detector — 4-state Hidden Markov Model on log-returns + volatility.

States:
  0 → ranging   (low vol, near-zero mean return)
  1 → bull      (low-medium vol, positive mean return)
  2 → bear      (low-medium vol, negative mean return)
  3 → volatile  (high vol, any direction)

The HMM is re-fitted each time a full training run is triggered, and the
state mapping is inferred from fitted Gaussian means post-training.
"""
import os
import logging
import numpy as np

logger = logging.getLogger('alphapilot.regime')

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'ml_models')

REGIME_NAMES  = {0: 'ranging', 1: 'bull', 2: 'bear', 3: 'volatile'}
REGIME_COLORS = {0: '#7d8590', 1: '#3fb950', 2: '#f85149', 3: '#e3b341'}


def _hmm_path(symbol: str, timeframe: str) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    sym = symbol.replace('/', '_')
    return os.path.join(MODEL_DIR, f'{sym}_{timeframe}_hmm.pkl')


def _build_hmm_features(df) -> np.ndarray:
    """
    Returns (n, 3) array: [log_return, rolling_10_std, log_volume_ratio].
    Adding volume breaks regime ties when return/vol look similar but
    volume fingerprint differs (e.g., bear + capitulation spike).
    Rows with NaN are dropped.
    """
    log_ret   = np.log(df['close'] / df['close'].shift(1))
    roll_std  = log_ret.rolling(10).std()
    vol_ratio = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)
    log_vol   = np.log1p(vol_ratio)
    arr  = np.column_stack([log_ret.values, roll_std.values, log_vol.values])
    mask = ~np.isnan(arr).any(axis=1)
    return arr[mask]


def _label_states(model) -> dict:
    """
    Assign regime labels to HMM states based on fitted Gaussian means.
    means_ shape: (n_components, 3) → col-0=log_ret, col-1=rolling_std, col-2=log_vol
    Only cols 0 and 1 are used for labelling.
    """
    means   = model.means_           # (4, 3)
    returns = means[:, 0]
    vols    = means[:, 1]

    vol_thresh = np.percentile(vols, 75)
    labels: dict = {}
    for i, (ret, vol) in enumerate(zip(returns, vols)):
        if vol >= vol_thresh:
            labels[i] = 3  # volatile
        elif ret > 1e-4:
            labels[i] = 1  # bull
        elif ret < -1e-4:
            labels[i] = 2  # bear
        else:
            labels[i] = 0  # ranging
    return labels


def fit_regime_model(df, symbol: str, timeframe: str) -> bool:
    """
    Train a GaussianHMM on df and save to disk.
    Called from ml_engine._training_thread after the XGBoost training.
    Returns True on success.
    """
    try:
        from hmmlearn import hmm
        import joblib

        X = _build_hmm_features(df)
        if len(X) < 50:
            logger.warning('[Regime] Not enough data for HMM fit')
            return False

        model = hmm.GaussianHMM(
            n_components=4,
            covariance_type='diag',
            n_iter=100,
            random_state=42,
            min_covar=1e-3,
        )
        model.fit(X)
        state_labels = _label_states(model)

        import joblib
        joblib.dump({'model': model, 'state_labels': state_labels}, _hmm_path(symbol, timeframe))
        logger.info(f'[Regime] HMM trained for {symbol} {timeframe}. Labels: {state_labels}')
        return True

    except ImportError:
        logger.warning('[Regime] hmmlearn not installed — regime detection disabled')
        return False
    except Exception as exc:
        logger.exception('[Regime] HMM fit error')
        return False


def detect_regime(app, symbol: str = 'BTC/USD', timeframe: str = '15Min') -> dict:
    """
    Predict current market regime.

    Returns dict:
      regime_id   : int 0–3
      regime_name : str
      color       : hex colour for UI
      error       : None | str
    """
    path = _hmm_path(symbol, timeframe)
    if not os.path.exists(path):
        return {
            'regime_id': 0, 'regime_name': 'unknown',
            'color': '#7d8590', 'error': 'HMM not trained yet',
        }

    with app.app_context():
        try:
            import joblib
            from ..services.alpaca import AlpacaClient
            from ..services.ml_engine import _bars_to_df

            # 150 bars gives the HMM more context for accurate state inference
            bars = AlpacaClient().get_bars(symbol, timeframe, limit=150).get('bars', [])
            if len(bars) < 25:
                raise ValueError('Insufficient bars')

            df = _bars_to_df(bars)
            X  = _build_hmm_features(df)
            if len(X) < 10:
                raise ValueError('Not enough feature rows')

            data         = joblib.load(path)
            model        = data['model']
            state_labels = data['state_labels']

            # Validity guard: reject models that are stale (wrong feature count)
            # or corrupted (NaN in any learned parameter from a bad training run).
            def _model_invalid(m):
                if m.means_.shape[1] != X.shape[1]:
                    return f'feature count {m.means_.shape[1]} != {X.shape[1]}'
                for attr in ('startprob_', 'transmat_', 'means_', '_covars_'):
                    val = getattr(m, attr, None)
                    if val is not None and np.isnan(val).any():
                        return f'NaN in {attr}'
                return None

            invalid_reason = _model_invalid(model)
            if invalid_reason:
                logger.warning(
                    f'[Regime] Invalid HMM for {symbol} ({invalid_reason}). '
                    'Deleting — will retrain on next training run.'
                )
                try:
                    os.remove(path)
                except OSError:
                    pass
                return {
                    'regime_id': 0, 'regime_name': 'ranging',
                    'color': REGIME_COLORS[0],
                    'error': f'HMM invalid ({invalid_reason}) — retrain to fix',
                }

            hidden_states = model.predict(X)
            # Consensus of last 5 states — prevents single-bar regime flips
            from collections import Counter
            window        = hidden_states[-5:]
            current_state = int(Counter(window).most_common(1)[0][0])
            regime_id     = state_labels.get(current_state, 0)

            return {
                'regime_id':   regime_id,
                'regime_name': REGIME_NAMES.get(regime_id, 'unknown'),
                'color':       REGIME_COLORS.get(regime_id, '#7d8590'),
                'error':       None,
            }

        except ImportError:
            return {
                'regime_id': 0, 'regime_name': 'unknown',
                'color': '#7d8590', 'error': 'hmmlearn not installed',
            }
        except Exception as exc:
            logger.exception('[Regime] Prediction error')
            return {
                'regime_id': 0, 'regime_name': 'error',
                'color': '#7d8590', 'error': str(exc),
            }
