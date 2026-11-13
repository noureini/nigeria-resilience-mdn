"""
Neural network with a single Gaussian output (NN-1D) and QRT loss.

Same architecture, search space and training procedure as the MDN
(``mdn.py``) with the number of mixture components fixed to one, i.e. a
heteroskedastic Gaussian network.
"""

from nigeria_resilience.models import mdn


def train(train_df, val_df, features, numeric_features, target,
          n_trials=50, seed=42, hparams=None):
    """Optuna search (unless ``hparams`` is given) and final refit; see ``mdn.train``."""
    return mdn.train(train_df, val_df, features=features,
                     numeric_features=numeric_features, target=target,
                     n_trials=n_trials, seed=seed, hparams=hparams,
                     n_components_range=(1, 1))


def predict(artifacts, test_df):
    """Return ``[alpha (=1), mu, sigma]`` for each row of ``test_df``."""
    return mdn.predict(artifacts, test_df)
