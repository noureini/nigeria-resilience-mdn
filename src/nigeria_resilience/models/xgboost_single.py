"""
XGBoostLSS -- single-component Gaussian distributional model.

Uses the local xgboostlss package with Gaussian(stabilization="MAD").
Optuna HPO via xgblss.hyper_opt, then final training on full train set.
"""

import xgboost as xgb

from nigeria_resilience.models.seed import setup_seed
from xgboostlss.distributions.Gaussian import Gaussian
from xgboostlss.model import XGBoostLSS

# ============================================================================
# Public API
# ============================================================================

def train(
    train_df,
    val_df,
    full_train_df,
    features,
    target,
    n_trials=50,
    num_boost_round=100,
    early_stopping_rounds=20,
    max_minutes=10,
    seed=42,
    hparams=None,
):
    """
    Full XGBoostLSS single-Gaussian training pipeline.

    Parameters
    ----------
    train_df : DataFrame
        Training split.
    val_df : DataFrame
        Validation split.
    full_train_df : DataFrame
        Combined train + val for final refit.
    features : list of str
    target : str
    n_trials : int
        Number of Optuna trials.
    num_boost_round : int
    early_stopping_rounds : int
    max_minutes : int
    seed : int
    hparams : dict or None
        Selected hyperparameters (incl. ``opt_rounds``).  If given, the Optuna
        search is skipped and the final model is fitted with them directly.

    Returns
    -------
    dict with keys:
        'model'      -- trained XGBoostLSS object
        'opt_params' -- optimised hyperparameters
        'n_rounds'   -- optimal boosting rounds
        'features', 'target'
    """
    setup_seed(seed)

    # ---- DMatrices ----
    X_train_xgb = train_df[features].copy()
    X_valid_xgb = val_df[features].copy()
    X_full_train_xgb = full_train_df[features].copy()

    y_train_xgb = train_df[target]
    y_valid_xgb = val_df[target]
    y_full_train_xgb = full_train_df[target]

    dtrain = xgb.DMatrix(X_train_xgb, label=y_train_xgb)
    dvalid = xgb.DMatrix(X_valid_xgb, label=y_valid_xgb)
    dtrain_full = xgb.DMatrix(X_full_train_xgb, label=y_full_train_xgb)

    # ---- Model init ----
    setup_seed(seed)
    xgblss = XGBoostLSS(
        Gaussian(
            stabilization="MAD",
            response_fn="softplus",
            loss_fn="nll",
        )
    )

    # ---- Hyper-parameter optimisation ----
    param_dict = {
        "eta":              ["float", {"low": 1e-5,   "high": 1,     "log": True}],
        "max_depth":        ["int",   {"low": 1,      "high": 10,    "log": False}],
        "gamma":            ["float", {"low": 1e-8,   "high": 40,    "log": True}],
        "subsample":        ["float", {"low": 0.2,    "high": 1.0,   "log": False}],
        "colsample_bytree": ["float", {"low": 0.2,    "high": 1.0,   "log": False}],
        "min_child_weight": ["float", {"low": 1e-8,   "high": 500,   "log": True}],
        "booster":          ["categorical", ["gbtree"]],
    }

    if hparams is None:
        opt_param = xgblss.hyper_opt(
            param_dict,
            dtrain,
            dvalid,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            max_minutes=max_minutes,
            n_trials=n_trials,
            silence=True,
            seed=seed,
            hp_seed=seed,
        )
    else:
        opt_param = dict(hparams)

    # ---- Final training ----
    setup_seed(seed)
    opt_params = opt_param.copy()
    n_rounds = opt_params["opt_rounds"]
    del opt_params["opt_rounds"]

    xgblss.train(
        opt_params,
        dtrain_full,
        num_boost_round=n_rounds,
    )

    return {
        'model': xgblss,
        'opt_params': opt_params,
        'n_rounds': n_rounds,
        'hparams': {**opt_params, 'opt_rounds': n_rounds},
        'features': features,
        'target': target,
    }


def predict(artifacts, test_df):
    """
    Predict distributional parameters on new data.

    Parameters
    ----------
    artifacts : dict
        Output of ``train()``.
    test_df : DataFrame

    Returns
    -------
    pred_params : DataFrame
        Columns depend on the distribution (typically 'loc' and 'scale'
        for single Gaussian).
    """
    xgblss = artifacts['model']
    features = artifacts['features']

    X_test_xgb = test_df[features].copy()
    dtest = xgb.DMatrix(X_test_xgb)

    pred_params = xgblss.predict(dtest)
    return pred_params
