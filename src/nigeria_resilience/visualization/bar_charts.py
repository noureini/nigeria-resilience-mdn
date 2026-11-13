"""Mean resilience bars, ratio-vs-MDN grids, and category-share stacked bars."""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, PercentFormatter

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
    ("$1.90 Poverty Line", "_pl_1_90"),
    ("$3.20 Poverty Line", "_pl_3_20"),
    ("$5.50 Poverty Line", "_pl_5_50"),
    ("National Poverty Line", "_at_nat"),
]

MODEL_ORDER_MEAN = ("OLS+GLM", "Elastic Net", "XGB-1D", "XGB-Mix", "NN-1D", "MDN")
MODEL_ORDER_RATIO = ("XGB-Mix", "NN-1D", "XGB-1D", "Elastic Net", "OLS+GLM")

# Categories (name, lower_inclusive, upper_exclusive)
CATS = [
    ("Non-resilient (<0.5)", 0.0, 0.5),
    ("0.5-0.8", 0.5, 0.8),
    ("Most resilient (>=0.8)", 0.8, 1.01),
]

MODEL_ORDER_CAT_RATIO = ("MDN", "XGB-Mix", "NN-1D", "XGB-1D", "Elastic Net", "OLS+GLM")


# =====================================================================
# Internal helpers
# =====================================================================

def _resolve_colname(df: pd.DataFrame, base: str, suffix: str) -> Optional[str]:
    cand = f"{base}{suffix}"
    if cand in df.columns:
        return cand
    if suffix == "_pl_1_90" and base in df.columns:
        return base
    return None


def _stack_panel_matrix(df: pd.DataFrame, suffix: str, model_order=MODEL_ORDER_MEAN) -> Tuple[np.ndarray, List[str]]:
    cols, labels = [], []
    for model in model_order:
        base = COLMAP[model]
        colname = _resolve_colname(df, base, suffix)
        if colname is None:
            continue
        cols.append(np.asarray(df[colname], float))
        labels.append(model)
    if not cols:
        return np.empty((0, 0)), []
    mat = np.vstack(cols).T
    return np.clip(mat, 0.0, 1.0), labels


def _paired_bootstrap_means(matrix: np.ndarray, B: int = 400, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    N, M = matrix.shape
    boot = np.empty((B, M), dtype=float)
    for b in range(B):
        idx = rng.integers(0, N, size=N)
        boot[b, :] = matrix[idx, :].mean(axis=0)
    return boot


def _summ_from_boot(boot: np.ndarray) -> List[dict]:
    out = []
    for j in range(boot.shape[1]):
        col = boot[:, j]
        out.append({
            "mean": float(np.mean(col)),
            "ci_low": float(np.quantile(col, 0.025)),
            "ci_high": float(np.quantile(col, 0.975)),
            "boot": col,
        })
    return out


# =====================================================================
# Mean resilience bar grid
# =====================================================================

def plot_mean_resilience_bar_grid(
    test_df_temporal: pd.DataFrame, *,
    out_path: str = None,
    B: int = 400, seed: int = 42,
    figsize=(14, 10), dpi: int = 160,
    y_formatter="%.2f",
    font_scale: float = 1.1,
):
    """Mean resilience bar grid with independent y-axes and bootstrap CI."""
    panel_summaries = []
    for title, suffix in PANELS:
        mat, labels = _stack_panel_matrix(test_df_temporal, suffix, model_order=MODEL_ORDER_MEAN)
        if mat.size == 0:
            panel_summaries.append((title, [], np.array([]), np.array([]), np.array([])))
            continue
        boot = _paired_bootstrap_means(mat, B=B, seed=seed)
        summ = _summ_from_boot(boot)
        means = np.array([s["mean"] for s in summ])
        lows = np.array([s["ci_low"] for s in summ])
        highs = np.array([s["ci_high"] for s in summ])
        panel_summaries.append((title, labels, means, lows, highs))

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    axes = axes.flatten()

    for ax, (title, labels, means, lows, highs) in zip(axes, panel_summaries):
        if len(labels) == 0:
            ax.axis("off")
            continue

        x = np.arange(len(labels))
        barw = 0.65
        ax.bar(x, means, barw, edgecolor="black", alpha=0.9)

        if highs.size:
            yerr = np.vstack([means - lows, highs - means])
            ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="black", capsize=5, linewidth=1.2)
            local_top = float(np.nanmax(highs)) * 1.10
            if not np.isfinite(local_top) or local_top <= 0:
                local_top = 1.0
        else:
            local_top = 1.0

        pad = 0.02 * local_top
        for i, m in enumerate(means):
            y_pos = (highs[i] if highs.size else m) + pad
            ax.text(x[i], y_pos, f"{m:.2f}", ha="center", va="bottom", fontsize=10 * font_scale)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9 * font_scale)
        ax.set_ylabel("Mean resilience probability", fontsize=10 * font_scale)
        ax.set_title(f"Mean resilience at {title} (+/-95% CI)", fontsize=11 * font_scale)
        ax.yaxis.set_major_formatter(FormatStrFormatter(y_formatter))
        ax.tick_params(axis="y", labelsize=9 * font_scale)
        ax.tick_params(axis="x", labelsize=9 * font_scale)
        ax.set_ylim(0, local_top)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    for k in range(len(PANELS), 4):
        axes[k].axis("off")

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
    fig.subplots_adjust(hspace=0.25, wspace=0.15)

    if out_path:
        save_and_show(fig, out_path)
    else:
        plt.show()

    return fig


# =====================================================================
# Ratio vs MDN difference-style grid
# =====================================================================

def _paired_bootstrap_ratio_grid(mat: np.ndarray, labels: List[str], B=400, seed=42):
    rng = np.random.default_rng(seed)
    N, M = mat.shape
    if "MDN" not in labels:
        return {}
    ref_idx = labels.index("MDN")

    disp_models = [m for m in MODEL_ORDER_RATIO if m in labels]
    disp_indices = [labels.index(m) for m in disp_models]

    boots = {m: [] for m in disp_models}
    for _ in range(B):
        idx = rng.integers(0, N, size=N)
        means = mat[idx, :].mean(axis=0)
        ref = means[ref_idx]
        ref = max(1e-12, ref)
        for m, j in zip(disp_models, disp_indices):
            boots[m].append(means[j] / ref)

    out = {}
    for m in disp_models:
        arr = np.asarray(boots[m], float)
        out[m] = {
            "mean": float(np.mean(arr)),
            "ci_low": float(np.quantile(arr, 0.025)),
            "ci_high": float(np.quantile(arr, 0.975)),
            "boot": arr,
        }
    return out


def plot_ratio_vs_mdn_diff_grid(
    test_df_temporal: pd.DataFrame, *,
    out_path: str = None,
    B: int = 400, seed: int = 42,
    figsize=(14, 10), dpi: int = 160,
    x_formatter="%.2f",
    capsize: float = 6.0,
    title_template: str = "Relative difference at {pl_label} Poverty line",
):
    """2x2 horizontal difference-style plots: ratio = mean_model / mean_MDN."""
    # Include MDN in the stacking so we have the reference
    full_order = ["MDN"] + list(MODEL_ORDER_RATIO)
    panel_results = []
    for title, suffix in PANELS:
        mat, labels = _stack_panel_matrix(test_df_temporal, suffix, model_order=full_order)
        if mat.size == 0 or "MDN" not in labels:
            panel_results.append((title, {}))
            continue
        res = _paired_bootstrap_ratio_grid(mat, labels, B=B, seed=seed)
        panel_results.append((title, res))

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    axes = axes.flatten()

    for ax, (pl_label, res) in zip(axes, panel_results):
        if not res:
            ax.axis("off")
            continue

        models = [m for m in MODEL_ORDER_RATIO if m in res]
        y = np.arange(len(models))

        means = np.array([res[m]["mean"] for m in models])
        lows = np.array([res[m]["ci_low"] for m in models])
        highs = np.array([res[m]["ci_high"] for m in models])

        ax.axvline(1.0, color="black", lw=1.8, ls="--", alpha=0.9)
        ax.hlines(y, lows, highs, color="tab:blue", lw=6, alpha=0.7)
        ax.errorbar(
            means, y, xerr=[means - lows, highs - means],
            fmt="o", capsize=5, color="black", linewidth=1.2, zorder=3,
        )

        for yi, m in zip(y, means):
            ax.text(m, yi + 0.1, f"{m:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_yticks(y)
        ax.set_yticklabels(models, fontsize=10)
        ax.set_ylim(-0.5, len(models) - 0.5)

        ax.set_xlabel("Resilience score ratio (model / MDN) [>1 higher]", fontsize=10)
        ax.set_title(title_template.format(pl_label=pl_label), fontsize=12, pad=8)
        ax.xaxis.set_major_formatter(FormatStrFormatter(x_formatter))
        ax.grid(True, axis="x", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

        xmin = np.nanmin(lows)
        xmax = np.nanmax(highs)
        if np.isfinite(xmin) and np.isfinite(xmax):
            span = xmax - xmin
            pad_x = 0.05 * span if span > 0 else 0.02
            ax.set_xlim(xmin - pad_x, xmax + pad_x)
        else:
            ax.set_xlim(0.9, 1.1)

    for k in range(len(panel_results), 4):
        axes[k].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if out_path:
        save_and_show(fig, out_path)
    else:
        plt.show()

    fig.exhibit_data = pd.DataFrame(
        [{"poverty_line": pl, "model": m, "mean": r["mean"], "ci_low": r["ci_low"],
          "ci_high": r["ci_high"]}
         for pl, res in panel_results for m, r in res.items() if isinstance(r, dict)])
    return fig


# =====================================================================
# Category-share stacked bars -- plotting only
# =====================================================================

def plot_category_shares_grid(
    results_by_line: dict, *,
    out_path: str = None,
    figsize=(14, 10), dpi: int = 160,
    annotate_threshold: float = 0.08,
    ci_gap: float = 0.03,
    edge_padding: float = 0.015,
):
    """2x2 stacked-bar grid of category shares (non-resilient / resilient / most-resilient)."""
    cats = [
        "Non-resilient (<0.5)",
        "Resilient (0.5-0.8)",
        "Most-resilient (>=0.8)",
    ]
    colors = {
        "Non-resilient (<0.5)": "#d62728",
        "Resilient (0.5-0.8)": "#ff7f0e",
        "Most-resilient (>=0.8)": "#2ca02c",
    }

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    axes = axes.flatten()

    first_handles, first_labels = None, None

    for ax, (panel_title, panel_dict) in zip(axes, results_by_line.items()):
        if not panel_dict:
            ax.axis("off")
            continue

        models = [m for m in MODEL_ORDER_MEAN if m in panel_dict]
        x = np.arange(len(models))
        barw = 0.65

        cum = np.zeros(len(models), dtype=float)
        for cat in cats:
            means = np.array([panel_dict[m][cat]["mean"] for m in models])
            lows = np.array([panel_dict[m][cat]["ci_low"] for m in models])
            highs = np.array([panel_dict[m][cat]["ci_high"] for m in models])

            ax.bar(
                x, means, barw, bottom=cum,
                color=colors[cat], edgecolor="white", linewidth=0.8, label=cat,
            )

            seg_bottom = cum
            seg_top = cum + means
            mid = (seg_bottom + seg_top) / 2.0

            for i, mval in enumerate(means):
                if mval < annotate_threshold:
                    continue
                main_y = np.clip(mid[i], seg_bottom[i] + edge_padding, seg_top[i] - edge_padding)
                ci_y = main_y - ci_gap
                ci_y = np.clip(ci_y, seg_bottom[i] + edge_padding, seg_top[i] - edge_padding)
                ax.text(
                    x[i], main_y, f"{mval * 100:.0f}%",
                    ha="center", va="center", fontsize=9, color="white", weight="bold",
                )
                ax.text(
                    x[i], ci_y, f"[{lows[i] * 100:.0f}, {highs[i] * 100:.0f}]",
                    ha="center", va="center", fontsize=7, color="white",
                )

            cum += means

        ax.set_xticks(x, models, rotation=25, ha="right")
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_ylabel("Share of households")
        ax.set_title(f"At the {panel_title}")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.6)

        if first_handles is None:
            first_handles, first_labels = ax.get_legend_handles_labels()
        ax.legend().remove()

    for k in range(len(results_by_line), 4):
        axes[k].axis("off")

    if first_handles:
        fig.legend(
            first_handles, first_labels,
            loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02),
        )

    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.98])

    if out_path:
        save_and_show(fig, out_path)
    else:
        plt.show()

    return fig


# =====================================================================
# Category-share ratio vs MDN
# =====================================================================

def _get_probs_by_model(df: pd.DataFrame, suffix: str) -> Dict[str, np.ndarray]:
    out = {}
    for m in MODEL_ORDER_CAT_RATIO:
        col = _resolve_colname(df, COLMAP[m], suffix)
        if col is None:
            continue
        out[m] = np.clip(np.asarray(df[col], float), 0.0, 1.0)
    return out


def _shares_in_bins(p: np.ndarray) -> np.ndarray:
    shares = []
    for _, lo, hi in CATS:
        shares.append(np.mean((p >= lo) & (p < hi)))
    return np.asarray(shares, float)


def bootstrap_ratio_vs_ref_for_line(
    probs_by_model: Dict[str, np.ndarray],
    ref: str = "MDN", B: int = 400, seed: int = 42, eps: float = 1e-9,
) -> Dict[str, Dict[str, Dict]]:
    """Category-share ratios (model / ref) per bootstrap draw."""
    assert ref in probs_by_model, f"Reference {ref} not in provided models."

    rng = np.random.default_rng(seed)
    N = len(next(iter(probs_by_model.values())))
    models = [m for m in MODEL_ORDER_CAT_RATIO if m in probs_by_model and m != ref]

    boots = {cat[0]: {m: [] for m in models} for cat in CATS}

    for _ in range(B):
        idx = rng.integers(0, N, size=N)
        ref_shares = _shares_in_bins(probs_by_model[ref][idx]) + eps
        for m in models:
            m_shares = _shares_in_bins(probs_by_model[m][idx]) + eps
            ratios = m_shares / ref_shares
            for (cat, _, _), r in zip(CATS, ratios):
                boots[cat][m].append(r)

    out = {}
    for cat_name in boots:
        out[cat_name] = {}
        for m, arr in boots[cat_name].items():
            arr = np.asarray(arr, float)
            log_arr = np.log(arr)
            p_boot = 2 * min((log_arr <= 0).mean(), (log_arr >= 0).mean())
            out[cat_name][m] = {
                "mean": float(np.mean(arr)),
                "ci_low": float(np.quantile(arr, 0.025)),
                "ci_high": float(np.quantile(arr, 0.975)),
                "p_boot": float(p_boot),
                "boot": arr,
            }
    return out


def ratios_vs_mdn_all_lines(
    df: pd.DataFrame, B: int = 400, seed: int = 42, ref: str = "MDN",
) -> Dict[str, Dict[str, Dict]]:
    """Compute category-share ratios for every poverty line."""
    results = {}
    for panel_title, suffix in PANELS:
        probs_by_model = _get_probs_by_model(df, suffix)
        if ref not in probs_by_model:
            results[panel_title] = {}
            continue
        results[panel_title] = bootstrap_ratio_vs_ref_for_line(
            probs_by_model, ref=ref, B=B, seed=seed,
        )
    return results


def plot_ratio_grid_for_category(
    results_by_line: Dict[str, Dict[str, Dict]],
    category: str, *,
    ref: str = "MDN",
    model_order=None,
    figsize=(14, 10), dpi: int = 160,
    xfmt="%.2f",
    out_path: Optional[str] = None,
):
    """2x2 grid: ratio = model/MDN for a single resilience category."""
    if model_order is None:
        model_order = [m for m in MODEL_ORDER_CAT_RATIO if m != ref]

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    axes = axes.flatten()

    for ax, (panel_title, cat_dict) in zip(axes, results_by_line.items()):
        if category not in cat_dict or len(cat_dict[category]) == 0:
            ax.axis("off")
            continue

        ratios = cat_dict[category]
        models = [m for m in model_order if m in ratios]
        y = np.arange(len(models))
        means = np.array([ratios[m]["mean"] for m in models])
        lows = np.array([ratios[m]["ci_low"] for m in models])
        highs = np.array([ratios[m]["ci_high"] for m in models])

        ax.axvline(1.0, color="black", lw=1.6, ls="--", alpha=0.9)
        ax.hlines(y, lows, highs, color="tab:blue", lw=6, alpha=0.7)
        ax.errorbar(
            means, y, xerr=[means - lows, highs - means],
            fmt="o", color="black", capsize=5, elinewidth=1.5, zorder=3,
        )
        for yi, mval in zip(y, means):
            if np.isfinite(mval):
                ax.text(mval, yi + 0.10, f"{mval:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_yticks(y, models)
        ax.set_xlabel(f"Ratio of shares (model / {ref})  [>1 = higher than {ref}]")
        ax.set_title(panel_title)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.7)
        ax.xaxis.set_major_formatter(FormatStrFormatter(xfmt))
        ax.margins(y=0.1)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

        finite = np.isfinite(np.concatenate([lows, highs]))
        if finite.any():
            xmin = np.nanmin(lows)
            xmax = np.nanmax(highs)
            span = xmax - xmin
            pad_v = 0.1 * span if span > 0 else 0.05
            ax.set_xlim(max(0, xmin - pad_v), xmax + pad_v)
        else:
            ax.set_xlim(0.8, 1.2)

    for k in range(len(results_by_line), 4):
        axes[k].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if out_path:
        save_and_show(fig, out_path)
    else:
        plt.show()

    return fig
