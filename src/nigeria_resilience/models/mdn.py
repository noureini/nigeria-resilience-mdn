"""
Mixture Density Network (MDN) with QRT loss and Optuna HPO.

Uses TensorFlow / Keras / TensorFlow Probability.
Multi-component Gaussian mixture; recalibrated NLL (QRT) training.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import optuna
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import uniform as unif
from sklearn.preprocessing import MinMaxScaler
from statsmodels.graphics.gofplots import qqplot
from tensorflow import keras

from nigeria_resilience.models.seed import setup_seed

tfd = tfp.distributions


# ============================================================================
# Utility functions: PIT Q-Q + MMD (numpy)
# ============================================================================

def show_pit_qq(params_np, y_true, title):
    """params_np: [N, 3K] -> (alpha, mu, sigma); y_true: [N,]"""
    k = params_np.shape[1] // 3
    a = params_np[:, :k]
    m = params_np[:, k:2*k]
    s = params_np[:, 2*k:3*k]
    dist = tfd.MixtureSameFamily(
        tfd.Categorical(probs=tf.convert_to_tensor(a, tf.float32)),
        tfd.Normal(loc=tf.convert_to_tensor(m, tf.float32),
                   scale=tf.convert_to_tensor(s, tf.float32))
    )
    u = dist.cdf(tf.convert_to_tensor(y_true.reshape(-1), tf.float32)).numpy()
    plt.figure(figsize=(4.2, 4.2))
    qqplot(u, dist=unif, line="45", fit=False)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def uniform_grid_like_np(u):
    n = u.shape[0]
    return ((np.arange(n) + 0.5) / n).reshape(-1, 1).astype(np.float32)


def _rbf_np(A, B, bw):
    diff = A[:, None, :] - B[None, :, :]
    sq = (diff ** 2).sum(-1)
    return np.exp(-sq / (2.0 * bw * bw + 1e-12))


def mmd2_biased_np(u):
    """Biased MMD^2 between u (PITs) and Uniform(0,1) using 3 bandwidths."""
    u = u.reshape(-1, 1).astype(np.float32)
    ref = uniform_grid_like_np(u)
    val = 0.0
    for bw in (0.03, 0.1, 0.3):
        val += _rbf_np(u, u, bw).mean() + _rbf_np(ref, ref, bw).mean() - 2 * _rbf_np(u, ref, bw).mean()
    return float(max(val, 0.0))


# ============================================================================
# MDN architecture
# ============================================================================

def nnelu(x):
    """1 + ELU -> strictly positive (min value approaches 0+)."""
    return 1.0 + tf.nn.elu(x)


def build_mdn(
    n_features,
    n_hidden=2,
    neurons=256,
    n_components=3,
    drop_rate=0.1,
    l2_reg_v1=1e-4,
    l2_reg_v2=1e-4,
    learning_rate=3e-4,
):
    inp = keras.Input(shape=(n_features,), name="x")
    x = inp
    for i in range(n_hidden):
        x = keras.layers.Dense(
            neurons, activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_reg_v1),
            name=f"dense_{i}"
        )(x)
        if drop_rate > 0.0:
            x = keras.layers.Dropout(drop_rate, name=f"drop_{i}")(x)

    alpha = keras.layers.Dense(n_components, activation='softmax', name='alpha')(x)
    mu = keras.layers.Dense(
        n_components,
        kernel_regularizer=keras.regularizers.l2(l2_reg_v2),
        name='mu'
    )(x)
    sigma = keras.layers.Dense(
        n_components, activation=nnelu,
        kernel_regularizer=keras.regularizers.l2(l2_reg_v2),
        name='sigma'
    )(x)

    params = keras.layers.Concatenate(name='params')([alpha, mu, sigma])
    model = keras.Model(inputs=inp, outputs=params, name="mdn")

    opt = keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=opt, loss=lambda y_true, y_pred: 0.0)  # placeholder
    return model


# ============================================================================
# QRT loss (recalibrated NLL)
# ============================================================================

@tf.function
def _logistic_pdf(x, loc, scale):
    t = (x - loc) / (scale + 1e-12)
    s = tf.math.sigmoid(t)
    return (s * (1.0 - s)) / (scale + 1e-12)


def qrt_loss_factory(kde_b=0.3, eps=1e-8):
    """
    loss(y, params) = -mean[ log f(y|x) + log phi(F(y|x)) ],
    where phi is the pdf of a smooth empirical CDF on [0,1] built via
    logistic KDE with reflection from the current batch PITs.
    """
    @tf.function
    def qrt_loss(y_true, params):
        y_true = tf.cast(y_true, tf.float32)
        k = tf.shape(params)[1] // 3
        a = params[:, :k]
        m = params[:, k:2*k]
        s = params[:, 2*k:3*k]

        dist = tfd.MixtureSameFamily(
            tfd.Categorical(probs=a),
            tfd.Normal(loc=m, scale=s)
        )

        y = tf.squeeze(y_true, axis=-1)
        logf = dist.log_prob(y)
        u = dist.cdf(y)
        u = tf.clip_by_value(u, 1e-6, 1.0 - 1e-6)
        u = tf.reshape(u, [-1, 1])
        B = tf.cast(tf.shape(u)[0], tf.float32)

        var = (kde_b ** 2.0) * tf.pow(B, -2.0 / 5.0)
        scale = tf.sqrt(var) * (tf.sqrt(3.0) / tf.constant(math.pi, tf.float32))

        uT = tf.transpose(u)
        pdf_main = _logistic_pdf(u, uT, scale)
        pdf_left = _logistic_pdf(u, -uT, scale)
        pdf_right = _logistic_pdf(u, 2.0 - uT, scale)
        pdf_mix = (pdf_main + pdf_left + pdf_right) / (3.0 * B)
        phi_u = tf.reduce_sum(pdf_mix, axis=1)

        loss = -tf.reduce_mean(logf + tf.math.log(phi_u + eps))
        return loss
    return qrt_loss


def compile_with_qrt(model, learning_rate, kde_b=0.3):
    model.optimizer = keras.optimizers.Adam(learning_rate)
    model.loss = qrt_loss_factory(kde_b=kde_b)
    model.compile(optimizer=model.optimizer, loss=model.loss)
    return model


# ============================================================================
# Callbacks
# ============================================================================

class RecordBestValNLL(keras.callbacks.Callback):
    """Record best validation NLL (recalibrated) and the training NLL at that epoch."""
    def __init__(self):
        super().__init__()
        self.best_val = np.inf
        self.best_epoch = None
        self.train_nll_at_val_best = None

    def on_epoch_end(self, epoch, logs=None):
        v = float(logs.get("val_loss", np.inf))
        if v < self.best_val:
            self.best_val = v
            self.best_epoch = epoch + 1
            self.train_nll_at_val_best = float(logs.get("loss", np.nan))


class StopWhenTrainMatches(keras.callbacks.Callback):
    def __init__(self, target, tol=0.0):
        super().__init__()
        self.target = target
        self.tol = tol
        self.hit = None

    def on_epoch_end(self, epoch, logs=None):
        tr = float(logs.get("loss"))
        if tr <= self.target + self.tol:
            self.hit = epoch + 1
            self.model.stop_training = True
            print(f"[Stopping Criteria] epoch={self.hit} train={tr:.6f} target={self.target:.6f}")


# ============================================================================
# Public API
# ============================================================================

def _prepare_xy(train_df, val_df, features, numeric_features, target):
    """MinMax-scale the numeric features (fit on the training split only)."""
    scaler = MinMaxScaler()
    xtr = train_df[features].copy()
    xva = val_df[features].copy()
    xtr[numeric_features] = scaler.fit_transform(xtr[numeric_features])
    xva[numeric_features] = scaler.transform(xva[numeric_features])
    return (scaler,
            xtr.to_numpy().astype('float32'), xva.to_numpy().astype('float32'),
            train_df[target].to_numpy().astype('float32'),
            val_df[target].to_numpy().astype('float32'))


def fit_final(X_train, y_train, X_valid, y_valid, params, kde_b, target_nll, seed=42):
    """Refit the selected architecture on train + validation.

    Training stops at the first epoch whose training loss (QRT) reaches
    ``target_nll``, the training loss the selected Optuna trial had at its best
    validation epoch.  Returns ``(model, stop_epoch)``.
    """
    setup_seed(seed)
    print("[Final] Retraining from scratch with winner hyperparameters (QRT).")
    final_model = build_mdn(
        n_features=X_train.shape[1],
        n_hidden=params['n_hidden'],
        neurons=params['neurons'],
        n_components=params['n_components'],
        drop_rate=params['dropout'],
        l2_reg_v1=params['l2_reg_v1'],
        l2_reg_v2=params['l2_reg_v2'],
        learning_rate=params['learning_rate'],
    )
    final_model = compile_with_qrt(final_model, learning_rate=params['learning_rate'],
                                   kde_b=kde_b)
    X_trval = np.vstack([X_train, X_valid]).astype('float32')
    y_trval = np.concatenate([y_train, y_valid]).astype('float32')
    stopping_cb = StopWhenTrainMatches(float(target_nll))
    final_model.fit(
        X_trval, y_trval,
        epochs=600, batch_size=params['batch_size'],
        shuffle=False,
        callbacks=[stopping_cb],
        verbose=2,
    )
    return final_model, stopping_cb.hit


def train(
    train_df,
    val_df,
    features,
    numeric_features,
    target,
    n_trials=50,
    seed=42,
    hparams=None,
    n_components_range=(2, 4),
):
    """
    Full MDN training pipeline: data scaling, Optuna HPO (QRT loss),
    then final retraining on train+val.

    Parameters
    ----------
    train_df, val_df : DataFrames
    features : list of str
        All feature column names.
    numeric_features : list of str
        Subset to MinMax-scale.
    target : str
    n_trials : int
    seed : int
    hparams : dict or None
        If given (keys ``params``, ``kde_b``, ``target_nll``), skip the Optuna
        search and refit these hyperparameters directly.
    n_components_range : (int, int)
        Search range for the number of mixture components; ``(1, 1)`` gives
        the single-Gaussian network (NN-1D).

    Returns
    -------
    dict with keys:
        'model'   -- final trained Keras model
        'scaler'  -- fitted MinMaxScaler
        'features', 'numeric_features', 'target'
        'best_params' -- dict of best Optuna hyperparameters
    """
    setup_seed(seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ---- Data prep ----
    scaler, X_train, X_valid, y_train, y_valid = _prepare_xy(
        train_df, val_df, features, numeric_features, target)

    if hparams is None:
        # ---- Optuna objective ----
        def objective(trial):
            setup_seed(seed)

            n_hidden = trial.suggest_int('n_hidden', 1, 4)
            neurons = trial.suggest_int('neurons', 64, 768, step=64)
            drop_rate = trial.suggest_float('dropout', 0.0, 0.3)
            n_components = trial.suggest_int('n_components', *n_components_range)
            l2_reg_v1 = trial.suggest_float('l2_reg_v1', 1e-5, 1e-2, log=True)
            l2_reg_v2 = trial.suggest_float('l2_reg_v2', 1e-5, 1e-1, log=True)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
            kde_b = trial.suggest_float('kde_b', 0.2, 0.5)
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

            model = build_mdn(
                n_features=X_train.shape[1],
                n_hidden=n_hidden, neurons=neurons, n_components=n_components,
                drop_rate=drop_rate, l2_reg_v1=l2_reg_v1, l2_reg_v2=l2_reg_v2,
                learning_rate=learning_rate,
            )
            model = compile_with_qrt(model, learning_rate=learning_rate, kde_b=kde_b)

            rec_cb = RecordBestValNLL()
            es = keras.callbacks.EarlyStopping(
                monitor="val_loss", mode="min",
                patience=20, restore_best_weights=True, verbose=0,
            )

            hist = model.fit(
                X_train, y_train,
                validation_data=(X_valid, y_valid),
                epochs=200, batch_size=batch_size,
                shuffle=False,
                callbacks=[rec_cb, es],
                verbose=0,
            )

            val_hist = hist.history.get("val_loss", [])
            best_idx = int(np.argmin(val_hist)) if len(val_hist) else -1
            best_val_nll = float(val_hist[best_idx]) if best_idx >= 0 else float("nan")

            # Diagnostics: MMD on validation
            params_val = model.predict(X_valid, verbose=0)
            k = params_val.shape[1] // 3
            a = params_val[:, :k]; m = params_val[:, k:2*k]; s = params_val[:, 2*k:3*k]
            dist_val = tfd.MixtureSameFamily(
                tfd.Categorical(probs=tf.convert_to_tensor(a, tf.float32)),
                tfd.Normal(loc=tf.convert_to_tensor(m, tf.float32),
                           scale=tf.convert_to_tensor(s, tf.float32))
            )
            u_val = np.clip(
                dist_val.cdf(tf.convert_to_tensor(y_valid.reshape(-1), tf.float32)).numpy(),
                1e-6, 1 - 1e-6,
            )
            val_mmd = mmd2_biased_np(u_val)

            trial.set_user_attr("best_epoch", int(rec_cb.best_epoch) if rec_cb.best_epoch else None)
            trial.set_user_attr("train_nll_at_val_best", float(rec_cb.train_nll_at_val_best))
            trial.set_user_attr("kde_b", float(kde_b))

            print(f"[Trial {trial.number}] best_epoch={rec_cb.best_epoch}")
            print(f"[Trial {trial.number}] best_val_NLL(QRT)={best_val_nll:.6g}   val_MMD={val_mmd:.6g}")
            print(f"[Trial {trial.number}] train_NLL_at_val_best(QRT)={rec_cb.train_nll_at_val_best:.6g}")
            print(f"[Trial {trial.number}] params={trial.params}")

            return best_val_nll

        # ---- Run HPO ----
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_trial
        print(f" [HPO] Best trial #{best.number}: val_NLL(QRT)={best.value:.6g}")
        print("Best params:", best.params)

        hparams = {
            "params": dict(best.params),
            "kde_b": float(best.user_attrs.get("kde_b", 0.3)),
            "target_nll": float(best.user_attrs["train_nll_at_val_best"]),
            "best_trial": int(best.number),
            "best_val_nll": float(best.value),
        }

    # ---- Final model: retrain on train+val ----
    final_model, stop_epoch = fit_final(
        X_train, y_train, X_valid, y_valid,
        hparams["params"], hparams["kde_b"], hparams["target_nll"], seed=seed)

    artifacts = {
        'model': final_model,
        'scaler': scaler,
        'features': features,
        'numeric_features': numeric_features,
        'target': target,
        'best_params': hparams["params"],
        'hparams': {**hparams, "stop_epoch": stop_epoch},
    }
    return artifacts


def predict(artifacts, test_df):
    """
    Predict mixture parameters on new data.

    Parameters
    ----------
    artifacts : dict
        Output of ``train()``.
    test_df : DataFrame

    Returns
    -------
    params : np.ndarray, shape (n, 3*K)
        Concatenation of [alpha, mu, sigma] for K components.
    """
    model = artifacts['model']
    scaler = artifacts['scaler']
    features = artifacts['features']
    numeric_features = artifacts['numeric_features']

    xte = test_df[features].copy()
    xte[numeric_features] = scaler.transform(xte[numeric_features])
    X_test = xte.to_numpy().astype('float32')

    params = model.predict(X_test, verbose=0)
    return params
