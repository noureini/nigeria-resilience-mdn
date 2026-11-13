"""Q-Q plots and inverse-quantile diagnostics for PIT values."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta
from scipy.stats import uniform as unif
from statsmodels.graphics.gofplots import qqplot

from nigeria_resilience.visualization.save import save_and_show

# ---------------------------------------------------------------------------
# Internal helper: sanitize duplicate matplotlib keyword arguments
# ---------------------------------------------------------------------------

def _sanitize_plot_kwargs(d):
    d = dict(d)
    for long_k, short_k in [
        ("markerfacecolor", "mfc"),
        ("markeredgecolor", "mec"),
        ("markersize", "ms"),
        ("markeredgewidth", "mew"),
        ("linewidth", "lw"),
        ("linestyle", "ls"),
        ("color", "c"),
    ]:
        if long_k in d and short_k in d:
            d.pop(short_k)
    return d


# ---------------------------------------------------------------------------
# Q-Q panel of PIT vs Uniform(0,1)
# ---------------------------------------------------------------------------

def plot_pit_qq_panel(
    df, cols, titles, *,
    same_limits=True,
    show_band=True,
    conf=0.95,
    marker_kwargs=None,
    out_path=None,
    return_fig=False,
    ncols=None,
    nrows=None,
):
    """QQ panel of PIT vs Uniform(0,1) with optional Beta order-stat band.

    Parameters
    ----------
    same_limits : bool
        Force each axis to [0, 1] x [0, 1].
    show_band : bool
        Draw pointwise Beta order-stat band at ``1 - conf`` level.
    out_path : str, optional
        If given the figure is saved **and** displayed via ``save_and_show``.
    """
    if len(cols) != len(titles):
        raise ValueError("`cols` and `titles` must have the same length.")
    nplots = len(cols)
    if nplots == 0:
        raise ValueError("`cols` is empty.")

    base_kwargs = dict(
        marker="o", markersize=4, alpha=0.5,
        markerfacecolor="C0", markeredgecolor="none",
    )
    mk = _sanitize_plot_kwargs({**base_kwargs, **(marker_kwargs or {})})

    # Layout
    if ncols is None and nrows is None:
        ncols = 3
        nrows = math.ceil(nplots / ncols)
    elif ncols is not None and nrows is None:
        nrows = math.ceil(nplots / ncols)
    elif ncols is None and nrows is not None:
        ncols = math.ceil(nplots / nrows)

    if ncols * nrows < nplots:
        raise ValueError(
            f"Grid too small: ncols*nrows={ncols * nrows} < {nplots} plots."
        )

    fig, ax = plt.subplots(
        nrows, ncols, sharex=True, sharey=True, figsize=(14, 10)
    )
    ax = np.atleast_1d(ax).ravel()

    alpha = 1.0 - conf

    for i, (c, title) in enumerate(zip(cols, titles)):
        u = np.clip(df[c].to_numpy(dtype=float), 1e-12, 1 - 1e-12)
        n = len(u)

        qqplot(u, dist=unif, fit=False, line=None, ax=ax[i], **mk)

        ax[i].plot([0, 1], [0, 1], "k--", lw=1.2, label="45\u00b0 line (Uniform)")

        if show_band and n > 1:
            ixs = np.arange(1, n + 1)
            x = (ixs - 0.5) / n
            lo = beta.ppf(alpha / 2.0, ixs, n - ixs + 1)
            hi = beta.ppf(1.0 - alpha / 2.0, ixs, n - ixs + 1)
            ax[i].fill_between(
                x, lo, hi, color="gray", alpha=0.18,
                label=f"{int(conf * 100)}% band",
            )

        if same_limits:
            ax[i].set_xlim(0, 1)
            ax[i].set_ylim(0, 1)

        ax[i].set_title(title)
        ax[i].set_xlabel("Theoretical quantiles (Uniform)")
        ax[i].set_ylabel("Empirical PIT quantiles")
        ax[i].grid(True, alpha=0.3)

        handles, labels = ax[i].get_legend_handles_labels()
        if handles:
            ax[i].legend(loc="lower right")

    for j in range(nplots, len(ax)):
        fig.delaxes(ax[j])

    fig.tight_layout()

    if out_path:
        save_and_show(fig, out_path)
    elif not return_fig:
        plt.show()

    if return_fig:
        return fig, ax


# ---------------------------------------------------------------------------
# Inverse-quantile panel
# ---------------------------------------------------------------------------

def _invquant_stats_per_bin(u, x, qs, *, categorical=False, bins=100):
    """Return bin_centers, {q: (p_hat, se)} dict.  NaN where a bin has no data."""
    u = np.asarray(u, dtype=float)
    mask = np.isfinite(u)
    if categorical:
        cats = pd.Categorical(x)
        groups = cats.codes
        labels = list(cats.categories)
        mask &= groups >= 0
        groups = groups[mask]
        u = u[mask]
        n_bins = len(labels)
        bin_centers = np.arange(n_bins, dtype=float)
        xticks = (bin_centers, labels)
    else:
        x = np.asarray(x, dtype=float)
        mask &= np.isfinite(x)
        x = x[mask]
        u = u[mask]
        edges = np.linspace(np.nanmin(x), np.nanmax(x), bins + 1)
        groups = np.clip(np.digitize(x, edges) - 1, 0, bins - 1)
        n_bins = bins
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        xticks = None

    counts = np.bincount(groups, minlength=n_bins).astype(float)
    counts[counts == 0] = np.nan

    stats = {}
    for q in qs:
        ind = (u <= q).astype(float)
        sums = np.bincount(groups, weights=ind, minlength=n_bins)
        p_hat = sums / counts
        se = np.sqrt(p_hat * (1.0 - p_hat) / counts)
        stats[q] = (p_hat, se)
    return bin_centers, stats, xticks


def plot_invquants_panel(
    df, variable, *,
    suffixes=("mdn", "ols_glm", "xgb_1d", "xgb_md"),
    titles=("MDN", "OLS & GLM", "XGB-LSS (Gaussian)", "XGB-LSS (Mixture)"),
    quantiles=(0.10, 0.30, 0.50, 0.70, 0.90, 0.97),
    categorical=False, bins=100,
    out_path=None, return_fig=False,
):
    """Panel plot: for each model, plot P(PIT <= q | x-bin) with binomial SE bars."""
    assert len(suffixes) == len(titles), (
        f"suffixes ({len(suffixes)}) and titles ({len(titles)}) must have same length"
    )

    pit_col_map = {
        "mdn": "cdf_mdn",
        "ols_glm": "cdf_ols_glm",
        "xgb_1d": "cdf_xgb_1d",
        "xgb_md": "cdf_xgb_md",
        "elastic_net": "cdf_elastic_net",
        "nn": "cdf_nn",
    }
    colors = ["b", "g", "k", "r", "y", "m"][: len(quantiles)]
    markers = ["v", "<", "o", ">", "^", "s"][: len(quantiles)]

    nplots = len(suffixes)
    ncols = min(nplots, 2)
    nrows = math.ceil(nplots / ncols)
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True,
                           figsize=(14, 5 * nrows + 1))
    axs = np.atleast_1d(ax).ravel()

    for i, (suf, title) in enumerate(zip(suffixes, titles)):
        if suf not in pit_col_map:
            raise ValueError(
                f"Unknown suffix '{suf}'. Choose from {list(pit_col_map)}"
            )
        pit_col = pit_col_map[suf]

        u = np.clip(df[pit_col].to_numpy(dtype=float), 1e-12, 1 - 1e-12)
        x = df[variable]

        centers, stats, xticks = _invquant_stats_per_bin(
            u, x, quantiles, categorical=categorical, bins=bins
        )

        for q, c, m in zip(quantiles, colors, markers):
            p_hat, se = stats[q]
            axs[i].errorbar(
                centers, p_hat, yerr=se, linestyle="none",
                marker=m, c=c, label=f"q={q:g}", alpha=0.9,
            )
            if len(centers) > 0:
                axs[i].hlines(
                    y=q, xmin=centers[0], xmax=centers[-1],
                    colors=c, linestyles="dashed", linewidth=1.0,
                )

        axs[i].set_title(title, fontsize=13)
        axs[i].set_ylabel("Quantiles")
        axs[i].grid(True, alpha=0.3)
        axs[i].set_ylim(0, 1)

        if categorical and xticks is not None:
            locs, labels = xticks
            axs[i].set_xticks(locs)
            axs[i].set_xticklabels(labels, rotation=0)
            axs[i].set_xlabel(variable)
        else:
            axs[i].set_xlabel(variable)

    # Hide unused axes
    for j in range(nplots, len(axs)):
        fig.delaxes(axs[j])

    fig.tight_layout(rect=(0, 0.04, 1, 1))

    if out_path:
        save_and_show(fig, out_path)
    elif not return_fig:
        plt.show()

    if return_fig:
        return fig, ax
