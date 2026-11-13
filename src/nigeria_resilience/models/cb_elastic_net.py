"""
Cisse & Barret (C&B) two-stage model with Elastic Net regularisation.

Stage 1: OLS (optionally Elastic-Net-penalised) for the conditional mean.
Stage 2: Poisson/Gamma GLM (optionally regularised) on squared residuals
          for the conditional variance.
Grid search over regularisation strengths; selection by validation NLL.
Final refit on train+val.
"""

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from scipy.stats import uniform as unif
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.gofplots import qqplot

from nigeria_resilience.models.seed import setup_seed

# ============================================================================
# Internal helpers
# ============================================================================

def _apply_numeric_scaling(df, cols_order, numeric_cols, scaler=None, fit=False):
    """Return X in cols_order, scaling ONLY numeric_cols that are present."""
    Xdf = df[cols_order].astype(float).copy()
    if numeric_cols:
        present = [c for c in numeric_cols if c in cols_order]
        if fit:
            scaler = StandardScaler(with_mean=True, with_std=True).fit(Xdf[present].values)
        if scaler is not None and len(present):
            Xdf.loc[:, present] = scaler.transform(Xdf[present].values)
    return Xdf.values, scaler


def _add_const(X, add_constant):
    return sm.add_constant(X, has_constant='add') if add_constant else X


def _to_1d(a):
    """Coerce any Series/array/column vector to flat 1-D (n,)."""
    return np.asarray(a, dtype=float).reshape(-1)


def normal_nll(y, mu, sigma2, reduction="mean", eps=1e-8):
    y = np.asarray(y).reshape(-1)
    mu = np.asarray(mu).reshape(-1)
    s2 = np.clip(np.asarray(sigma2).reshape(-1), eps, None)
    v = 0.5 * (np.log(2 * np.pi * s2) + (y - mu) ** 2 / s2)
    return float(v.mean() if reduction == "mean" else v.sum())


# ============================================================================
# Core training routine (with grid search)
# ============================================================================

def _train_mean_variance_model(
    train_df,
    val_df,
    explanatory_vars,
    numeric_vars,
    target,
    mean_alpha_grid=(0.0, 0.01, 0.05, 0.1),
    mean_L1_wt_grid=(0.0, 0.1),
    var_alpha_grid=(0.0, 0.01, 0.05, 0.1),
    var_L1_wt_grid=(0.0, 0.1),
    var_family="poisson",
    add_constant=True,
    fgls_refit=False,
    eps=1e-8,
):
    # --- design matrices; fit scaler on TRAIN numeric only ---
    X_tr, scaler = _apply_numeric_scaling(train_df, explanatory_vars, numeric_vars, scaler=None, fit=True)
    X_va, _ = _apply_numeric_scaling(val_df, explanatory_vars, numeric_vars, scaler=scaler, fit=False)
    X_tr_pen = _add_const(X_tr, add_constant)
    X_va_pen = _add_const(X_va, add_constant)

    # unscaled for unpenalized fits
    X_tr_raw = train_df[explanatory_vars].astype(float).values
    X_va_raw = val_df[explanatory_vars].astype(float).values
    X_tr_ols = _add_const(X_tr_raw, add_constant)
    X_va_ols = _add_const(X_va_raw, add_constant)

    y_tr = _to_1d(train_df[target].values)
    y_va = _to_1d(val_df[target].values)

    fam = (sm.families.Poisson() if var_family.lower() == "poisson"
           else sm.families.Gamma(sm.families.links.log()))

    # --- grid search on VAL ---
    best = dict(score=np.inf, mean_alpha=None, mean_L1_wt=None, var_alpha=None, var_L1_wt=None)
    best_mean_res = best_var_res = None
    for a_mean, l1m, a_var, l1v in product(mean_alpha_grid, mean_L1_wt_grid,
                                            var_alpha_grid, var_L1_wt_grid):
        # Stage 1 (mean)
        if a_mean > 0.0:
            mean_model = sm.OLS(y_tr, X_tr_pen, missing='drop')
            mean_res = mean_model.fit_regularized(method='elastic_net', alpha=a_mean, L1_wt=l1m, refit=False)
            mu_tr = _to_1d(mean_res.predict(X_tr_pen))
            mu_va = _to_1d(mean_res.predict(X_va_pen))
        else:
            mean_model = sm.OLS(y_tr, X_tr_ols, missing='drop')
            mean_res = mean_model.fit(cov_type='HC1')
            mu_tr = _to_1d(mean_res.predict(X_tr_ols))
            mu_va = _to_1d(mean_res.predict(X_va_ols))

        # Stage 2 (variance) on TRAIN residuals^2
        resid2_tr = _to_1d((y_tr - mu_tr) ** 2)
        Xv_tr = X_tr_pen if a_var > 0.0 else X_tr_ols
        var_model = sm.GLM(resid2_tr, Xv_tr, family=fam, missing='drop')
        var_res = (var_model.fit_regularized(alpha=a_var, L1_wt=l1v)
                   if a_var > 0.0 else var_model.fit())

        # Predict variance on VAL
        Xv_va = X_va_pen if a_var > 0.0 else X_va_ols
        sigma2_va = _to_1d(var_res.predict(Xv_va))
        sigma2_va = np.maximum(sigma2_va, eps)

        # Select by validation NLL
        nll_va = normal_nll(y_va, mu_va, sigma2_va, reduction="mean", eps=eps)
        if nll_va < best['score']:
            best.update(score=nll_va, mean_alpha=a_mean, mean_L1_wt=l1m,
                        var_alpha=a_var, var_L1_wt=l1v)
            best_mean_res, best_var_res = mean_res, var_res

    # --- optional FGLS refit on TRAIN ---
    if fgls_refit:
        mu_tr = _to_1d(best_mean_res.predict(X_tr_pen if best['mean_alpha'] > 0 else X_tr_ols))
        sigma2_tr = _to_1d(best_var_res.predict(X_tr_pen if best['var_alpha'] > 0 else X_tr_ols))
        sigma2_tr = np.maximum(sigma2_tr, eps)
        w = 1.0 / sigma2_tr
        if best['mean_alpha'] > 0.0:
            sw = np.sqrt(w)[:, None]
            Xw, yw = (X_tr_pen * sw), (y_tr * np.sqrt(w))
            best_mean_res = sm.OLS(yw, Xw).fit_regularized(
                method='elastic_net', alpha=best['mean_alpha'],
                L1_wt=best['mean_L1_wt'], refit=False,
            )
        else:
            best_mean_res = sm.WLS(y_tr, X_tr_ols, weights=w).fit(cov_type='HC1')

    # --- FINAL REFIT on TRAIN + VAL ---
    dev_df = pd.concat([train_df, val_df], axis=0)
    X_dev, _ = _apply_numeric_scaling(dev_df, explanatory_vars, numeric_vars,
                                      scaler=scaler, fit=False)
    X_dev_pen = _add_const(X_dev, add_constant)

    X_dev_raw = dev_df[explanatory_vars].astype(float).values
    X_dev_ols = _add_const(X_dev_raw, add_constant)
    y_dev = _to_1d(dev_df[target].values)

    # Stage 1 final
    if best['mean_alpha'] > 0.0:
        mean_final = sm.OLS(y_dev, X_dev_pen).fit_regularized(
            method='elastic_net', alpha=best['mean_alpha'],
            L1_wt=best['mean_L1_wt'], refit=False,
        )
        mu_dev = _to_1d(mean_final.predict(X_dev_pen))
    else:
        mean_final = sm.OLS(y_dev, X_dev_ols).fit(cov_type='HC1')
        mu_dev = _to_1d(mean_final.predict(X_dev_ols))

    # Stage 2 final
    resid2_dev = _to_1d((y_dev - mu_dev) ** 2)
    Xv_dev = X_dev_pen if best['var_alpha'] > 0.0 else X_dev_ols
    fam = (sm.families.Poisson() if var_family.lower() == "poisson"
           else sm.families.Gamma(sm.families.links.log()))
    var_final = (
        sm.GLM(resid2_dev, Xv_dev, family=fam).fit_regularized(
            alpha=best['var_alpha'], L1_wt=best['var_L1_wt'])
        if best['var_alpha'] > 0.0
        else sm.GLM(resid2_dev, Xv_dev, family=fam).fit()
    )

    artifacts = dict(
        scaler=scaler,
        explanatory_vars=explanatory_vars,
        numeric_vars=numeric_vars,
        add_constant=add_constant,
        var_family=var_family,
        best_hparams=best,
    )
    return (best_mean_res, best_var_res, best), (mean_final, var_final), artifacts


# ============================================================================
# Prediction helper
# ============================================================================

def _predict_two_stage(df, models, artifacts, eps=1e-8):
    mean_final, var_final = models
    cols = artifacts['explanatory_vars']
    num = artifacts['numeric_vars']
    scaler = artifacts['scaler']
    add_constant = artifacts['add_constant']
    best = artifacts['best_hparams']

    X, _ = _apply_numeric_scaling(df, cols, num, scaler=scaler, fit=False)
    X_pen = _add_const(X, add_constant)

    X_raw = df[cols].astype(float).values
    X_ols = _add_const(X_raw, add_constant)

    # Use scaled features when alpha > 0 (penalized), unscaled when alpha == 0
    mu = _to_1d(
        mean_final.predict(X_pen)
        if best['mean_alpha'] > 0
        else mean_final.predict(X_ols)
    )
    sigma2 = _to_1d(
        var_final.predict(X_pen)
        if best['var_alpha'] > 0
        else var_final.predict(X_ols)
    )
    return mu, np.maximum(sigma2, eps)


# ============================================================================
# PIT + QQ
# ============================================================================

def pit_from_normal(y, mu, sigma2, eps=1e-12):
    return norm.cdf(_to_1d(y), loc=_to_1d(mu),
                    scale=np.sqrt(np.clip(_to_1d(sigma2), eps, None)))


def plot_pit_qq(u, title="PIT Q-Q (Test)"):
    u = np.clip(np.asarray(u).ravel(), 1e-9, 1 - 1e-9)
    fig = qqplot(u, dist=unif, line="45", fit=False)
    ax = fig.axes[0]
    ax.set_title(title)
    ax.set_xlabel("Theoretical quantiles (Uniform)")
    ax.set_ylabel("Empirical PIT quantiles")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Coefficient tables
# ============================================================================

def coefficient_tables(mean_model, var_model, artifacts):
    """
    Returns two DataFrames (mean, variance) each with columns:
      ['term', 'coef_stdscale', 'coef_original_scale']
    """
    cols = artifacts['explanatory_vars']
    add_constant = artifacts['add_constant']
    names = (['const'] + cols) if add_constant else cols

    def _series(model):
        p = model.params
        return pd.Series(np.asarray(p), index=names[:len(p)])

    def _to_original_scale(coef_std):
        coef_orig = coef_std.copy()
        scaler = artifacts['scaler']
        num = set(artifacts['numeric_vars'])
        if scaler is None:
            return coef_orig

        means = dict(zip(artifacts['numeric_vars'], scaler.mean_))
        scales = dict(zip(artifacts['numeric_vars'], scaler.scale_))

        for j in coef_orig.index:
            if j in num:
                s = scales[j]
                if s and s > 0:
                    coef_orig[j] = coef_std[j] / s

        if add_constant and 'const' in coef_orig.index:
            adj = 0.0
            for j in coef_std.index:
                if j in num:
                    s = scales[j]; m = means[j]
                    if s and s > 0:
                        adj += coef_std[j] * m / s
            coef_orig['const'] = coef_std['const'] - adj
        return coef_orig

    mean_std = _series(mean_model)
    var_std = _series(var_model)

    mean_orig = _to_original_scale(mean_std)
    var_orig = _to_original_scale(var_std)

    mean_df = pd.DataFrame({
        'term': mean_std.index,
        'coef_stdscale': mean_std.values,
        'coef_original_scale': mean_orig.values,
    })
    var_df = pd.DataFrame({
        'term': var_std.index,
        'coef_stdscale': var_std.values,
        'coef_original_scale': var_orig.values,
    })
    return mean_df, var_df


# ============================================================================
# Public API
# ============================================================================

def train(
    train_df,
    val_df,
    explanatory_vars,
    numeric_vars,
    target,
    var_family="poisson",
    fgls_refit=False,
    seed=42,
):
    """
    Train the Elastic-Net C&B two-stage model with grid-search HPO.

    Parameters
    ----------
    train_df, val_df : DataFrames
    explanatory_vars : list of str
    numeric_vars : list of str
        Numeric features to standardize.
    target : str
    var_family : str
        "poisson" or "gamma".
    fgls_refit : bool
    seed : int

    Returns
    -------
    dict with keys:
        'mean_final', 'var_final' -- final statsmodels results (train+val)
        'mean_dev', 'var_dev'     -- dev results (train only, best HP)
        'artifacts'               -- internal metadata (scaler, hparams, etc.)
    """
    setup_seed(seed)

    (mean_dev, var_dev, best_hp), (mean_final, var_final), arts = _train_mean_variance_model(
        train_df, val_df,
        explanatory_vars=explanatory_vars,
        numeric_vars=numeric_vars,
        target=target,
        var_family=var_family,
        fgls_refit=fgls_refit,
    )

    print("Chosen hyperparams (from validation):", arts['best_hparams'])

    return {
        'mean_final': mean_final,
        'var_final': var_final,
        'mean_dev': mean_dev,
        'var_dev': var_dev,
        'artifacts': arts,
    }


def predict(result, test_df):
    """
    Predict mu and sigma^2 on new data.

    Parameters
    ----------
    result : dict
        Output of ``train()``.
    test_df : DataFrame

    Returns
    -------
    mu : np.ndarray, shape (n,)
    sigma2 : np.ndarray, shape (n,)
    """
    mu, sigma2 = _predict_two_stage(
        test_df,
        (result['mean_final'], result['var_final']),
        result['artifacts'],
    )
    return mu, sigma2
