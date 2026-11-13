"""Scoring metrics: CRPS, GNLL, EMD accuracy, bootstrap comparison."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import norm
from sklearn.metrics import mean_squared_error

from nigeria_resilience.visualization.save import save_and_show

tfd = tfp.distributions


# =====================================================================
# 1) CRPS utilities
# =====================================================================

def crps_normal(y, mu, sigma, eps=1e-12):
    """Vectorized CRPS for Normal(mu, sigma) at observations *y*.  Returns array [N]."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma = np.clip(np.asarray(sigma, dtype=np.float64).reshape(-1), eps, None)
    z = (y - mu) / sigma
    return sigma * (2.0 * norm.pdf(z) + z * (2.0 * norm.cdf(z) - 1.0) - 1.0 / np.sqrt(np.pi))


def crps_mixture_gauss_mean_tf(y, weights, means, scales, eps=1e-8):
    """Mean CRPS for Gaussian mixture using TF/TFP.  Returns scalar float."""
    y = tf.cast(y, tf.float32)
    weights = tf.cast(weights, tf.float32)
    means = tf.cast(means, tf.float32)
    scales = tf.cast(scales, tf.float32)

    w = weights
    s = scales

    std_normal = tfd.Normal(loc=0.0, scale=1.0)

    def A(delta, sigma):
        z = delta / sigma
        pdf = std_normal.prob(z)
        cdf = std_normal.cdf(z)
        return sigma * (2.0 * pdf + z * (2.0 * cdf - 1.0))

    delta = tf.expand_dims(y, -1) - means
    term1 = tf.reduce_sum(w * A(delta, s), axis=1)

    mu_i = tf.expand_dims(means, 2)
    mu_j = tf.expand_dims(means, 1)
    w_i = tf.expand_dims(w, 2)
    w_j = tf.expand_dims(w, 1)
    s_i = tf.expand_dims(s, 2)
    s_j = tf.expand_dims(s, 1)
    delta_mu = mu_i - mu_j
    sigma_sum = tf.sqrt(s_i ** 2 + s_j ** 2)
    term2 = 0.5 * tf.reduce_sum(w_i * w_j * A(delta_mu, sigma_sum), axis=[1, 2])

    crps = term1 - term2
    return float(tf.reduce_mean(crps).numpy())


# =====================================================================
# 2) GNLL evaluator
# =====================================================================

def gnll_eval(y, alpha, mu, sigma, reduction="mean", eps=1e-12):
    """Mean (or sum) negative log-likelihood under a Gaussian mixture."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    alpha = np.asarray(alpha, dtype=np.float32)
    mu = np.asarray(mu, dtype=np.float32)
    sigma = np.clip(np.asarray(sigma, dtype=np.float32), eps, None)
    if alpha.ndim == 1:
        alpha = alpha[:, None]
    if mu.ndim == 1:
        mu = mu[:, None]
    if sigma.ndim == 1:
        sigma = sigma[:, None]
    row_sums = np.sum(alpha, axis=1, keepdims=True)
    alpha = alpha / np.maximum(row_sums, eps)
    cat = tfd.Categorical(probs=tf.convert_to_tensor(alpha, tf.float32))
    comps = tfd.Normal(
        loc=tf.convert_to_tensor(mu, tf.float32),
        scale=tf.convert_to_tensor(sigma, tf.float32),
    )
    gm = tfd.MixtureSameFamily(mixture_distribution=cat, components_distribution=comps)
    lp = gm.log_prob(tf.convert_to_tensor(y, tf.float32))
    nll = -tf.reduce_mean(lp) if reduction == "mean" else -tf.reduce_sum(lp)
    return float(nll.numpy())


# =====================================================================
# 3) PIT -> EMD accuracy proxy
# =====================================================================

def wasserstein_discrete(p, q):
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return float(np.mean(np.abs(cdf_p - cdf_q)) * 2.0)


def cdf_accuracy(cdf_vals, bins=100):
    """Accuracy proxy in [0,1]: 1 - EMD between empirical PIT histogram and Uniform."""
    u = np.clip(np.asarray(cdf_vals, dtype=float).ravel(), 0.0, 1.0)
    counts, _ = np.histogram(u, bins=bins, range=(0, 1))
    pmf = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts, dtype=float)
    unif = np.full_like(pmf, 1.0 / bins, dtype=float)
    div = wasserstein_discrete(unif, pmf)
    return float(1.0 - np.clip(div, 0.0, 1.0))


# =====================================================================
# 4) Model metric collectors (per bootstrap sample)
# =====================================================================

def metrics_for_models(
    test_df, y_col,
    sg_specs=(
        ("OLS+GLM", "mean_predicted_ols_glm", "std_predicted_ols_glm", "cdf_ols_glm"),
        ("Elastic Net", "mean_predicted_elastic_net", "std_predicted_elastic_net", "cdf_elastic_net"),
        ("XGB-1D", "mean_predicted_xgb_1D", "std_predicted_xgb_1D", "cdf_xgb_1d"),
        ("NN-1D", "mean_predicted_nn", "std_predicted_nn", "cdf_nn"),
    ),
    mdn_arrays=None,
    xgbmix_arrays=None,
):
    """Return ``{model: {"MSE","NLL","CRPS","EMDacc"}}``."""
    y = test_df[y_col].to_numpy(np.float64)
    out = {}

    for name, m_col, s_col, cdf_col in sg_specs:
        mu = test_df[m_col].to_numpy(np.float64)
        sd = np.clip(test_df[s_col].to_numpy(np.float64), 1e-12, None)
        mse = mean_squared_error(y, mu)
        gnll = gnll_eval(y, alpha=np.ones_like(mu), mu=mu, sigma=sd, reduction="mean")
        crps = float(np.mean(crps_normal(y, mu, sd)))
        emdacc = cdf_accuracy(test_df[cdf_col].to_numpy())
        out[name] = {"MSE": mse, "NLL": gnll, "CRPS": crps, "EMDacc": emdacc}

    if mdn_arrays is not None:
        w = mdn_arrays["alpha"]
        mu = mdn_arrays["mu"]
        sd = mdn_arrays["sigma"]
        w_norm = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        m_hat = np.sum(w_norm * mu, axis=1)
        mse = mean_squared_error(y, m_hat)
        gnll = gnll_eval(y, alpha=w, mu=mu, sigma=sd, reduction="mean")
        crps = crps_mixture_gauss_mean_tf(
            tf.convert_to_tensor(y),
            tf.convert_to_tensor(w),
            tf.convert_to_tensor(mu),
            tf.convert_to_tensor(sd),
        )
        emdacc = cdf_accuracy(test_df[mdn_arrays["cdf_col"]].to_numpy())
        out["MDN"] = {"MSE": mse, "NLL": gnll, "CRPS": crps, "EMDacc": emdacc}

    if xgbmix_arrays is not None:
        w = xgbmix_arrays["alpha"]
        mu = xgbmix_arrays["mu"]
        sd = xgbmix_arrays["sigma"]
        w_norm = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        m_hat = np.sum(w_norm * mu, axis=1)
        mse = mean_squared_error(y, m_hat)
        gnll = gnll_eval(y, alpha=w, mu=mu, sigma=sd, reduction="mean")
        crps = crps_mixture_gauss_mean_tf(
            tf.convert_to_tensor(y),
            tf.convert_to_tensor(w),
            tf.convert_to_tensor(mu),
            tf.convert_to_tensor(sd),
        )
        emdacc = cdf_accuracy(test_df[xgbmix_arrays["cdf_col"]].to_numpy())
        out["XGB-Mix"] = {"MSE": mse, "NLL": gnll, "CRPS": crps, "EMDacc": emdacc}

    return out


# =====================================================================
# 5) Bootstrap driver
# =====================================================================

def bootstrap_compare(test_df_full, y_col, mdn_arrays=None, xgbmix_arrays=None, B=400, rng_seed=123):
    rng = np.random.default_rng(rng_seed)
    N = len(test_df_full)
    metrics = ["MSE", "NLL", "CRPS", "EMDacc"]
    store = {m: {} for m in metrics}

    for b in range(B):
        idx = rng.integers(0, N, size=N)
        dfb = test_df_full.iloc[idx].reset_index(drop=True)

        mdn_b = None
        if mdn_arrays is not None:
            mdn_b = {
                "alpha": mdn_arrays["alpha"][idx, :],
                "mu": mdn_arrays["mu"][idx, :],
                "sigma": mdn_arrays["sigma"][idx, :],
                "cdf_col": mdn_arrays["cdf_col"],
            }
        xgb_b = None
        if xgbmix_arrays is not None:
            xgb_b = {
                "alpha": xgbmix_arrays["alpha"][idx, :],
                "mu": xgbmix_arrays["mu"][idx, :],
                "sigma": xgbmix_arrays["sigma"][idx, :],
                "cdf_col": xgbmix_arrays["cdf_col"],
            }

        res = metrics_for_models(dfb, y_col, mdn_arrays=mdn_b, xgbmix_arrays=xgb_b)
        for model, vals in res.items():
            for m in metrics:
                store[m].setdefault(model, []).append(vals[m])

    summ = {m: {} for m in metrics}
    for m in metrics:
        for model, arr in store[m].items():
            arr = np.asarray(arr, dtype=float)
            summ[m][model] = {
                "mean": float(np.mean(arr)),
                "ci_low": float(np.quantile(arr, 0.025)),
                "ci_high": float(np.quantile(arr, 0.975)),
                "boot": arr,
            }
    return summ


# =====================================================================
# 6) 2x2 bar plots with 95 % CI + mean labels
# =====================================================================

def plot_all_metrics_bar(summ, out_path=None):
    metrics = ["MSE", "NLL", "CRPS", "EMDacc"]
    sort_dirs = {"MSE": True, "NLL": True, "CRPS": True, "EMDacc": False}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        data = summ[metric]
        models = sorted(
            data.keys(),
            key=lambda m: data[m]["mean"],
            reverse=not sort_dirs[metric],
        )

        means = np.array([data[m]["mean"] for m in models])
        lows = np.array([data[m]["ci_low"] for m in models])
        highs = np.array([data[m]["ci_high"] for m in models])

        x = np.arange(len(models))
        ax.bar(x, means, 0.65, edgecolor="black", alpha=0.85)
        yerr = np.vstack([means - lows, highs - means])
        ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="black", capsize=5, linewidth=1.2)

        for i, m in enumerate(means):
            ax.text(
                x[i],
                highs[i] + 0.02 * (highs[i] - lows[i] + 1e-9),
                f"{m:.2f}",
                ha="center", va="bottom", fontsize=9,
            )

        ax.set_xticks(x, models, rotation=25, ha="right")
        ax.set_title(f"{metric} (mean +/- 95% CI)")
        ax.set_ylabel(metric)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7)

    fig.tight_layout()

    if out_path:
        save_and_show(fig, out_path)
    else:
        plt.show()

    return fig
