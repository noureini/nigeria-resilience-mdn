"""ECDF and KDE distribution plots for resilience scores."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from nigeria_resilience.visualization.save import save_and_show

# =====================================================================
# Shared constants
# =====================================================================

COLMAP = {
    "MDN":         "resilience_score_mdn",
    "NN-1D":       "resilience_score_nn",
    "XGB-1D":      "resilience_score_xgb_1d",
    "XGB-Mix":     "resilience_score_xgb_md",
    "Elastic Net": "resilience_score_elastic_net",
    "OLS+GLM":     "resilience_score_ols_glm",
}

PANELS = [
    ("$1.90/day", "_pl_1_90"),
    ("$3.20/day", "_pl_3_20"),
    ("$5.50/day", "_pl_5_50"),
    ("National line", "_at_nat"),
]

LS = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1))]
MK = ["", "o", "s", "D", "^", "v"]


# =====================================================================
# ECDF helper
# =====================================================================

def _ecdf(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    x = np.clip(x, 0.0, 1.0)
    if x.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    x = np.sort(x)
    y = np.arange(1, x.size + 1) / x.size
    return x, y


# =====================================================================
# ECDF grid
# =====================================================================

def plot_resilience_ecdf_grid_pretty(
    df: pd.DataFrame, *,
    figsize=(13, 10), dpi=160,
    legend_fontsize=9, annotate_quartiles=True,
    out_path=None,
):
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    axes = axes.flatten()

    for ax, (title, suffix) in zip(axes, PANELS):
        for i, (model, base) in enumerate(COLMAP.items()):
            col = f"{base}{suffix}"
            if suffix == "_pl_1_90" and col not in df.columns and base in df.columns:
                col = base
            if col not in df.columns:
                continue

            vals = np.asarray(df[col], float)
            x, y = _ecdf(vals)
            y_surv = 1.0 - y
            ax.step(
                x, y_surv, where="post",
                linewidth=2.0,
                linestyle=LS[i % len(LS)],
                marker=MK[i % len(MK)],
                markevery=max(1, len(x) // 30),
                label=model,
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Resilience probability", fontsize=11)
        ax.set_ylabel("1 - Empirical CDF", fontsize=11)
        ax.set_title(f"Distribution of resilience scores -- {title}", fontsize=13)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
        ax.legend(
            loc="center left", bbox_to_anchor=(1.02, 0.5),
            frameon=False, fontsize=legend_fontsize,
        )

    plt.tight_layout()

    if out_path:
        save_and_show(fig, out_path)
    else:
        plt.show()

    return fig


# =====================================================================
# KDE helpers
# =====================================================================

def _kde_on_01(x, grid):
    """Simple boundary-aware KDE on [0,1]: reflect at edges."""
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return None
    x_reflect = np.concatenate([-x, x, 2 - x])
    kde = gaussian_kde(x_reflect)
    dens = kde(grid)
    area = np.trapz(dens, grid)
    return dens / area if area > 0 else dens


def plot_resilience_kde_grid_pretty(
    df: pd.DataFrame, *,
    figsize=(13, 10), dpi=160,
    legend_fontsize=9, grid_points=400,
    out_path=None,
):
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    axes = axes.flatten()
    grid = np.linspace(0, 1, grid_points)
    ymax = 0.0
    curves = []

    for title, suffix in PANELS:
        panel_curves = []
        for i, (model, base) in enumerate(COLMAP.items()):
            col = f"{base}{suffix}"
            if suffix == "_pl_1_90" and col not in df.columns and base in df.columns:
                col = base
            if col not in df.columns:
                continue
            vals = df[col].to_numpy(float)
            dens = _kde_on_01(vals, grid)
            if dens is None:
                continue
            ymax = max(ymax, float(np.max(dens)))
            panel_curves.append((i, model, dens))
        curves.append((title, panel_curves))

    ymax = max(ymax, 1e-3) * 1.08

    for ax, (title, panel_curves) in zip(axes, curves):
        for i, model, dens in panel_curves:
            ax.plot(
                grid, dens,
                linewidth=2.0,
                linestyle=LS[i % len(LS)],
                marker=MK[i % len(MK)],
                markevery=max(1, grid_points // 30),
                label=model,
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, ymax)
        ax.set_xlabel("Resilience probability", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"Distribution of resilience scores -- {title}", fontsize=13)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
        ax.legend(
            loc="center left", bbox_to_anchor=(1.02, 0.5),
            frameon=False, fontsize=legend_fontsize,
        )

    plt.tight_layout()

    if out_path:
        save_and_show(fig, out_path)
    else:
        plt.show()

    return fig
