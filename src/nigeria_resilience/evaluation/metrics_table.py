"""Build results tables: relative-performance grid and model-summary table."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import mean_squared_error

from nigeria_resilience.visualization.save import save_and_show

tfd = tfp.distributions


# =====================================================================
# Relative summary vs reference
# =====================================================================

def _rel_summary_vs_ref(summ_metric_dict, ref_model="MDN", model_order=None, mode="ratio", eps=1e-12):
    """Build per-model bootstrap summaries relative to a reference model.

    Returns list of tuples ``(label, mean, ci_low, ci_high)`` excluding *ref*.
    """
    assert mode in ("ratio", "diff")
    if model_order is None:
        model_order = [m for m in summ_metric_dict.keys() if m != ref_model]
    ref_boot = np.asarray(summ_metric_dict[ref_model]["boot"], float)
    rows = []
    for m in model_order:
        if m == ref_model or m not in summ_metric_dict:
            continue
        mb = np.asarray(summ_metric_dict[m]["boot"], float)
        if mode == "ratio":
            arr = mb / np.maximum(ref_boot, eps)
        else:
            arr = mb - ref_boot
        rows.append((
            m,
            float(np.mean(arr)),
            float(np.quantile(arr, 0.025)),
            float(np.quantile(arr, 0.975)),
        ))
    return rows


def plot_vs_mdn_grid(summ, ref_model="MDN", mode="ratio", model_order=None, out_path=None):
    """2x2 comparison vs MDN for MSE, NLL, CRPS, EMDacc."""
    metrics = ["MSE", "NLL", "CRPS", "EMDacc"]

    if model_order is None:
        first = next(m for m in metrics if m in summ)
        model_order = [k for k in summ[first].keys() if k != ref_model]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        if metric not in summ:
            ax.axis("off")
            continue

        rows = _rel_summary_vs_ref(
            summ[metric], ref_model=ref_model, model_order=model_order, mode=mode,
        )

        labels = [r[0] for r in rows]
        means = np.array([r[1] for r in rows])
        lo = np.array([r[2] for r in rows])
        hi = np.array([r[3] for r in rows])
        y = np.arange(len(labels))

        if mode == "ratio":
            ref_line = 1.0
            xlabel = f"Relative to {ref_model}  (model / {ref_model})"
            better_note = (
                "< 1 is better than ref"
                if metric in ["MSE", "NLL", "CRPS"]
                else "> 1 is better than ref"
            )
        else:
            ref_line = 0.0
            xlabel = f"Difference vs {ref_model}  (model - {ref_model})"
            better_note = (
                "Negative = better than ref"
                if metric in ["MSE", "NLL", "CRPS"]
                else "Positive = better than ref"
            )

        ax.axvline(ref_line, color="black", linewidth=1.7, linestyle="--", zorder=5)
        ax.hlines(y, lo, hi, colors="tab:blue", linewidth=6, alpha=0.6)
        ax.errorbar(
            means, y, xerr=[means - lo, hi - means],
            fmt="o", capsize=5, color="black", linewidth=1.2, zorder=6,
        )
        for yi, m in zip(y, means):
            ax.text(m, yi + 0.05, f"{m:.2f}", va="bottom", ha="center", fontsize=9)

        ax.set_yticks(y, labels)

        if mode == "ratio":
            hint = "(<1 better)" if metric in ["MSE", "NLL", "CRPS"] else "(>1 better)"
            ax.set_title(f"{metric}: relative performance {hint}")
        else:
            hint = "(neg. better)" if metric in ["MSE", "NLL", "CRPS"] else "(pos. better)"
            ax.set_title(f"{metric}: difference {hint}")

        ax.set_xlabel(xlabel)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.7)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        margin = 0.1
        xmin = np.nanmin(lo)
        xmax = np.nanmax(hi)
        if mode == "ratio":
            left = min(xmin, ref_line) * (1 - margin)
            right = max(xmax, ref_line) * (1 + margin)
            ax.set_xlim(left, right)
        else:
            span = xmax - xmin
            ax.set_xlim(xmin - margin * span, xmax + margin * span)

        ax.text(
            0.98, 0.02, better_note,
            ha="right", va="bottom", transform=ax.transAxes,
            fontsize=8, color="dimgray",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if out_path:
        save_and_show(fig, out_path)
    else:
        plt.show()

    return fig


# =====================================================================
# Build results table
# =====================================================================

def _gnll_eval(y, alpha, mu, sigma, reduction="mean", eps=1e-12):
    """Thin wrapper kept private to this module."""
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


def _wasserstein_discrete(p, q):
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return float(np.mean(np.abs(cdf_p - cdf_q)) * 2.0)


def _cdf_accuracy(cdf_vals, bins=100):
    u = np.clip(np.asarray(cdf_vals, dtype=float).ravel(), 0.0, 1.0)
    counts, _ = np.histogram(u, bins=bins, range=(0, 1))
    pmf = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts, dtype=float)
    unif = np.full_like(pmf, 1.0 / bins, dtype=float)
    div = _wasserstein_discrete(unif, pmf)
    return float(1.0 - np.clip(div, 0.0, 1.0))


def eval_linear_model_cb(test_df, feature_cols, y_col, mean_model, var_model):
    exog = sm.add_constant(test_df[feature_cols], has_constant="add")
    mu = mean_model.predict(exog).astype(np.float32)
    var = var_model.predict(exog).astype(np.float32)
    sigma = np.sqrt(np.clip(var, 1e-12, None)).astype(np.float32)
    y = test_df[y_col].to_numpy(np.float32)
    mse = mean_squared_error(y, mu)
    nll = _gnll_eval(y, alpha=np.ones_like(mu), mu=mu, sigma=sigma, reduction="mean")
    return mse, nll


def slice_parameter_vectors(parameter_vector):
    """Unpack MDN parameter vectors of shape (N, 3K) into (alpha, mu, sigma)."""
    k = parameter_vector.shape[1] // 3
    a = parameter_vector[:, :k]
    m = parameter_vector[:, k : 2 * k]
    s = parameter_vector[:, 2 * k : 3 * k]
    return a, m, s


def eval_mdn_model(x_test, y_test, mdn_model):
    y_pred = mdn_model.predict(x_test, verbose=0)
    alpha, mu, sigma = slice_parameter_vectors(y_pred)
    y = np.asarray(y_test, dtype=np.float32).reshape(-1)
    m_hat = np.sum(alpha * mu, axis=1)
    mse = mean_squared_error(y, m_hat)
    nll = _gnll_eval(y, alpha=alpha, mu=mu, sigma=sigma, reduction="mean")
    return mse, nll


def eval_xgb_1d(y, loc, scale):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    mu = np.asarray(loc, dtype=np.float32).reshape(-1)
    sd = np.clip(np.asarray(scale, dtype=np.float32).reshape(-1), 1e-12, None)
    mse = mean_squared_error(y, mu)
    nll = _gnll_eval(y, alpha=np.ones_like(mu), mu=mu, sigma=sd, reduction="mean")
    return mse, nll


def eval_xgb_md(y, mix_probs, locs, scales):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    ps = np.asarray(mix_probs, dtype=np.float32)
    mu = np.asarray(locs, dtype=np.float32)
    sd = np.clip(np.asarray(scales, dtype=np.float32), 1e-12, None)
    if ps.ndim == 1:
        ps = ps[:, None]
    if mu.ndim == 1:
        mu = mu[:, None]
    if sd.ndim == 1:
        sd = sd[:, None]
    ps = ps / np.maximum(ps.sum(axis=1, keepdims=True), 1e-12)
    m_hat = np.sum(ps * mu, axis=1)
    mse = mean_squared_error(y, m_hat)
    nll = _gnll_eval(y, alpha=ps, mu=mu, sigma=sd, reduction="mean")
    return mse, nll


def build_results_table(
    test_df, y_col, feature_cols,
    ols_results, poisson_results,
    mdn_model=None, x_test=None, y_test=None,
    pred_params_1d=None,
    xgb_mix_probs=None, xgb_mix_locs=None, xgb_mix_scales=None,
):
    """Build a summary DataFrame with MSE, NLL, EMD per model."""
    rows = []

    # OLS & GLM
    mse_cb, nll_cb = eval_linear_model_cb(test_df, feature_cols, y_col, ols_results, poisson_results)
    emd_cb = _cdf_accuracy(test_df["cdf_ols_glm"])
    rows.append(("OLS & GLM", mse_cb, nll_cb, emd_cb))

    # MDN
    if mdn_model is not None and x_test is not None and y_test is not None:
        mse_mdn, nll_mdn = eval_mdn_model(x_test, y_test, mdn_model)
        emd_mdn = _cdf_accuracy(test_df["cdf_mdn"])
        rows.append(("MDN", mse_mdn, nll_mdn, emd_mdn))

    # XGB-LSS (Gaussian)
    if pred_params_1d is not None:
        if isinstance(pred_params_1d, dict):
            loc = pred_params_1d["loc"]
            scale = pred_params_1d["scale"]
        elif isinstance(pred_params_1d, pd.DataFrame):
            loc = pred_params_1d["loc"].to_numpy()
            scale = pred_params_1d["scale"].to_numpy()
        else:
            arr = np.asarray(pred_params_1d)
            assert arr.shape[1] >= 2
            loc, scale = arr[:, 0], arr[:, 1]
        mse_x1d, nll_x1d = eval_xgb_1d(test_df[y_col].to_numpy(), loc, scale)
        emd_x1d = _cdf_accuracy(test_df["cdf_xgb_1d"])
        rows.append(("XGB-LSS (One Gaussian)", mse_x1d, nll_x1d, emd_x1d))

    # XGB-LSS (Mixture)
    if xgb_mix_probs is not None and xgb_mix_locs is not None and xgb_mix_scales is not None:
        mse_xmd, nll_xmd = eval_xgb_md(
            test_df[y_col].to_numpy(), xgb_mix_probs, xgb_mix_locs, xgb_mix_scales,
        )
        emd_xmd = _cdf_accuracy(test_df["cdf_xgb_md"])
        rows.append(("XGB-LSS (Mixture of Gaussians)", mse_xmd, nll_xmd, emd_xmd))

    results = pd.DataFrame(rows, columns=["Model", "MSE", "NLL", "EMD"])
    display = results.copy()
    display["MSE"] = display["MSE"].round(2)
    display["NLL"] = display["NLL"].round(2)
    display["EMD"] = display["EMD"].round(2)
    return results, display
