"""
XGBoostLSS -- Mixture of Gaussians distributional model.

Distribution selection among M=2,3,4 Gaussian mixtures, then
Optuna HPO and final training using the local xgboostlss package.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import xgboost as xgb

from nigeria_resilience.models.seed import setup_seed
from xgboostlss.distributions.Gaussian import Gaussian
from xgboostlss.distributions.Mixture import Mixture
from xgboostlss.distributions.mixture_distribution_utils import MixtureDistributionClass
from xgboostlss.model import XGBoostLSS

tfd = tfp.distributions


# ============================================================================
# Helper functions
# ============================================================================

def slice_parameter_vectors(parameter_vector):
    """Returns unpacked (alpha, mu, sigma) from [N, 3K] parameter matrix."""
    k = parameter_vector.shape[1] // 3
    a = parameter_vector[:, :k]
    m = parameter_vector[:, k:2*k]
    s = parameter_vector[:, 2*k:3*k]
    return a, m, s


def gnll_loss(y_true, parameter_vector):
    """Computes the mean negative log-likelihood loss of y given the mixture parameters."""
    alpha, mu, sigma = slice_parameter_vectors(parameter_vector)

    dist = tfd.MixtureSameFamily(
        tfd.Categorical(probs=tf.convert_to_tensor(alpha, tf.float32)),
        tfd.Normal(loc=tf.convert_to_tensor(mu, tf.float32),
                   scale=tf.convert_to_tensor(sigma, tf.float32))
    )
    y = tf.convert_to_tensor(y_true.reshape(-1))
    log_likelihood = dist.log_prob(tf.transpose(y))
    return -tf.reduce_mean(log_likelihood, axis=-1)


def compute_national_poverty_line(
    ppp_conversion_factor=101.91,
    national_poverty_line_2019_daily=137430 / 365,
):
    inflation_rate_2019_2016 = (200.30 - 291.4) / 291.4
    national_poverty_line_2016 = national_poverty_line_2019_daily * (1 + inflation_rate_2019_2016)
    national_poverty_line_2016 = round(national_poverty_line_2016 / ppp_conversion_factor, 2)
    national_poverty_line = national_poverty_line_2016
    return national_poverty_line


# ============================================================================
# Public API
# ============================================================================

def train(
    train_df,
    val_df,
    full_train_df,
    features,
    target,
    M=3,
    n_trials=50,
    num_boost_round=100,
    early_stopping_rounds=20,
    max_minutes=10,
    seed=42,
    hparams=None,
    run_dist_select=False,
):
    """
    Full XGBoostLSS mixture-of-Gaussians training pipeline.

    Parameters
    ----------
    train_df : DataFrame
    val_df : DataFrame
    full_train_df : DataFrame
        Combined train + val for final refit.
    features : list of str
    target : str
    M : int
        Number of mixture components (default 3).
    n_trials : int
    num_boost_round : int
    early_stopping_rounds : int
    max_minutes : int
    seed : int
    hparams : dict or None
        Selected hyperparameters (incl. ``opt_rounds``).  If given, the Optuna
        search is skipped and the final model is fitted with them directly.
    run_dist_select : bool
        If True, run distribution selection among M=2,3,4 first.

    Returns
    -------
    dict with keys:
        'model'      -- trained XGBoostLSS mixture object
        'opt_params' -- optimised hyperparameters
        'n_rounds'   -- optimal boosting rounds
        'M'          -- number of mixture components
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

    # ---- Optional: distribution selection ----
    if run_dist_select:
        mix_dist_class = MixtureDistributionClass()
        candidate_distributions = [
            Mixture(Gaussian(stabilization="MAD", response_fn="softplus", loss_fn="nll"), M=2),
            Mixture(Gaussian(stabilization="MAD", response_fn="softplus", loss_fn="nll"), M=3),
            Mixture(Gaussian(stabilization="MAD", response_fn="softplus", loss_fn="nll"), M=4),
        ]
        dist_nll = mix_dist_class.dist_select(
            target=y_train_xgb,
            candidate_distributions=candidate_distributions,
            max_iter=50,
            plot=False,
        )
        print("Distribution selection NLL:", dist_nll)

    # ---- Model init ----
    setup_seed(seed)
    xgblss_mix = XGBoostLSS(
        Mixture(
            Gaussian(
                stabilization="MAD",
                response_fn="softplus",
                loss_fn="nll",
            ),
            M=M,
            tau=1.0,
            hessian_mode="individual",
        )
    )

    # ---- HPO ----
    setup_seed(seed)
    param_dic_mix = {
        "eta":              ["float", {"low": 1e-5,   "high": 1,     "log": True}],
        "max_depth":        ["int",   {"low": 1,      "high": 10,    "log": False}],
        "gamma":            ["float", {"low": 1e-8,   "high": 40,    "log": True}],
        "subsample":        ["float", {"low": 0.2,    "high": 1.0,   "log": False}],
        "colsample_bytree": ["float", {"low": 0.2,    "high": 1.0,   "log": False}],
        "min_child_weight": ["float", {"low": 1e-8,   "high": 500,   "log": True}],
        "booster":          ["categorical", ["gbtree"]],
    }

    if hparams is None:
        opt_param_mix = xgblss_mix.hyper_opt(
            param_dic_mix,
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
        opt_param_mix = dict(hparams)

    # ---- Final training ----
    setup_seed(seed)
    opt_params_mix = opt_param_mix.copy()
    n_rounds = opt_params_mix["opt_rounds"]
    del opt_params_mix["opt_rounds"]

    xgblss_mix.train(
        opt_params_mix,
        dtrain_full,
        num_boost_round=n_rounds,
    )

    return {
        'model': xgblss_mix,
        'opt_params': opt_params_mix,
        'n_rounds': n_rounds,
        'hparams': {**opt_params_mix, 'opt_rounds': n_rounds},
        'M': M,
        'features': features,
        'target': target,
    }


def predict(artifacts, test_df):
    """
    Predict mixture distributional parameters on new data.

    Parameters
    ----------
    artifacts : dict
        Output of ``train()``.
    test_df : DataFrame

    Returns
    -------
    pred_params : DataFrame
        Columns for each mixture component's weight, loc, and scale.
    """
    xgblss_mix = artifacts['model']
    features = artifacts['features']

    X_test_xgb = test_df[features].copy()
    dtest = xgb.DMatrix(X_test_xgb)

    pred_params = xgblss_mix.predict(dtest)
    return pred_params
