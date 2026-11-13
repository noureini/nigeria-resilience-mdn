"""Effort / resilience-gap analysis: consensus households (MDN baseline)
and score-gap computations with bootstrap inference."""

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from tqdm import tqdm

from nigeria_resilience.analysis.smd import COLMAP, PANELS, _resolve_colname
from nigeria_resilience.visualization.save import save_and_show

# Effort/score-gap analysis compares each model against the MDN reference.
# Unlike smd.MODEL_ORDER (scoped to 4 models for the disagreement analysis),
# the effort suite includes NN-1D so every non-MDN model appears in the figure.
MODEL_ORDER = ("XGB-1D", "XGB-Mix", "NN-1D", "Elastic Net", "OLS+GLM")

THRESHOLDS = (0.5, 0.8)


# =====================================================================
# Core helpers
# =====================================================================

def _effort(score: np.ndarray, t: float) -> np.ndarray:
    """Compute effort needed to reach threshold *t*."""
    s = np.asarray(score, float)
    return np.maximum(0.0, t - s)


def _bootstrap_se_ci(
    vals: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """Bootstrap SE and 95 % CI for the mean of *vals*."""
    vals = np.asarray(vals, float)
    n = len(vals)
    if n < 5:
        return (np.nan, np.nan, np.nan)
    boots = np.empty(n_boot, float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = np.nanmean(vals[idx])
    se = float(np.nanstd(boots, ddof=1))
    ci_low = float(np.nanpercentile(boots, 2.5))
    ci_high = float(np.nanpercentile(boots, 97.5))
    return se, ci_low, ci_high


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


# =====================================================================
# Effort for agree-in-need households
# =====================================================================

def compute_effort_agree_vs_mdn(
    df: pd.DataFrame,
    id_col: str = "household_id",
    thresholds: Tuple[float, ...] = (0.5, 0.8),
    n_boot: int = 400,
    seed: int = 123,
    min_n: int = 50,
    verbose: bool = True,
) -> Dict[str, Dict[float, pd.DataFrame]]:
    """Effort for households where MDN and model agree on being below threshold.

    Parameters
    ----------
    df : DataFrame
        Dataset with resilience scores.
    id_col : str
        Household ID column.
    thresholds : tuple
        Thresholds to analyse.
    n_boot : int
        Bootstrap replications.
    seed : int
        Random seed.
    min_n : int
        Minimum agree-in-need sample size.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        results[poverty_title][threshold] = DataFrame indexed by model.
    """
    rng = np.random.default_rng(seed)
    out: Dict[str, Dict[float, pd.DataFrame]] = {}

    total_jobs = len(PANELS) * len(thresholds) * len(MODEL_ORDER)
    pbar = tqdm(total=total_jobs, desc="Computing effort (agree-in-need)", leave=True)

    for (title, suffix) in PANELS:
        if verbose:
            print(f"\n{'='*60}\nProcessing: {title}\n{'='*60}")
        out[title] = {}
        mdn_col = _resolve_colname(df, COLMAP["MDN"], suffix)

        if mdn_col is None:
            if verbose:
                print("  ERROR: MDN column not found!")
            for t in thresholds:
                pbar.update(len(MODEL_ORDER))
                out[title][t] = pd.DataFrame()
            continue

        for t in thresholds:
            if verbose:
                print(f"\n  Threshold: {t}")
            rows = []
            for model in MODEL_ORDER:
                mod_col = _resolve_colname(df, COLMAP[model], suffix)
                if mod_col is None:
                    if verbose:
                        print(f"    - {model:15s}: Column not found")
                    pbar.update(1)
                    continue

                base = df[[id_col, mdn_col, mod_col]].dropna()
                base = base[
                    np.isfinite(base[mdn_col].to_numpy())
                    & np.isfinite(base[mod_col].to_numpy())
                ]
                if base.empty:
                    pbar.update(1)
                    continue

                mdn = base[mdn_col].to_numpy(float)
                mod = base[mod_col].to_numpy(float)

                agree_need = (mdn < t) & (mod < t)
                n_agree = agree_need.sum()
                if n_agree < min_n:
                    if verbose:
                        print(f"    - {model:15s}: Only {n_agree} consensus households (min={min_n})")
                    pbar.update(1)
                    continue

                e_mdn = _effort(mdn[agree_need], t)
                e_mod = _effort(mod[agree_need], t)
                delta = e_mod - e_mdn

                m_mdn = float(np.mean(e_mdn))
                m_mod = float(np.mean(e_mod))
                m_del = float(np.mean(delta))

                se_mdn, ciL_mdn, ciH_mdn = _bootstrap_se_ci(e_mdn, n_boot, rng)
                se_mod, ciL_mod, ciH_mod = _bootstrap_se_ci(e_mod, n_boot, rng)
                se_del, ciL_del, ciH_del = _bootstrap_se_ci(delta, n_boot, rng)

                if verbose:
                    print(f"    + {model:15s}: n={n_agree:,}, effort_diff={m_del:+.3f}")

                rows.append({
                    "model": model,
                    "n_agree_need": int(n_agree),
                    "mean_eff_mdn": m_mdn, "se_eff_mdn": se_mdn,
                    "ciL_eff_mdn": ciL_mdn, "ciH_eff_mdn": ciH_mdn,
                    "mean_eff_model": m_mod, "se_eff_model": se_mod,
                    "ciL_eff_model": ciL_mod, "ciH_eff_model": ciH_mod,
                    "mean_delta_model_minus_mdn": m_del, "se_delta": se_del,
                    "ciL_delta": ciL_del, "ciH_delta": ciH_del,
                })
                pbar.update(1)

            tab = pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()
            out[title][t] = tab

    pbar.close()
    return out


# =====================================================================
# Bar chart grid of effort differences
# =====================================================================

def plot_effort_difference_bars_grid(
    results: Dict[str, Dict[float, pd.DataFrame]],
    thresholds: Tuple[float, ...] = (0.5, 0.8),
    figsize: Tuple[float, float] = (18, 12),
    dpi: int = 300,
    outfile: Optional[str] = None,
    out_path: Optional[str] = None,
) -> plt.Figure:
    """2x2 grid of bar charts showing effort difference (model - MDN).

    Parameters
    ----------
    results : dict
        Output from compute_effort_agree_vs_mdn().
    out_path : str, optional
        Save path (uses save_and_show).
    """
    threshold_colors = {
        0.5: '#2E86AB',
        0.8: '#A23B72',
    }
    threshold_labels = {
        0.5: 'To Resilient (rho=0.5)',
        0.8: 'To Most-resilient (rho=0.8)',
    }
    panel_titles = {
        "$1.90/day Poverty Line": "$1.90/day",
        "$3.20/day Poverty Line": "$3.20/day",
        "$5.50/day Poverty Line": "$5.50/day",
        "National Poverty Line": "National",
    }

    poverty_titles = list(results.keys())[:4]

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    axes = axes.flatten()

    all_models_set = set()
    for poverty_title in poverty_titles:
        by_t = results[poverty_title]
        for t in thresholds:
            tab = by_t.get(t, pd.DataFrame())
            if not tab.empty:
                all_models_set.update([m for m in tab.index if m != "MDN"])
    all_models = [m for m in MODEL_ORDER if m in all_models_set and m != "MDN"]

    if not all_models:
        return fig

    for panel_idx, poverty_title in enumerate(poverty_titles):
        ax = axes[panel_idx]
        if poverty_title not in results:
            ax.axis('off')
            continue

        by_t = results[poverty_title]
        x = np.arange(len(all_models))
        n_thresh = len(thresholds)
        width = 0.35

        ax.axhline(0, color="black", lw=1.5, ls="--", alpha=0.5, zorder=1)

        for i, t in enumerate(thresholds):
            tab = by_t.get(t, pd.DataFrame())
            if tab.empty:
                continue

            vals = []
            se_vals = []
            for model in all_models:
                if model in tab.index:
                    vals.append(tab.loc[model, "mean_delta_model_minus_mdn"])
                    se_vals.append(tab.loc[model, "se_delta"])
                else:
                    vals.append(np.nan)
                    se_vals.append(np.nan)

            vals = np.array(vals)
            se_vals = np.array(se_vals)
            offset = (i - n_thresh / 2 + 0.5) * width

            color = threshold_colors.get(t, '#1f77b4')
            label = threshold_labels.get(t, f'Threshold {t}')

            if panel_idx == 0:
                ax.bar(x + offset, vals, width, label=label, alpha=0.85,
                       color=color, edgecolor='black', linewidth=0.8)
            else:
                ax.bar(x + offset, vals, width, alpha=0.85, color=color,
                       edgecolor='black', linewidth=0.8)

            valid = np.isfinite(vals) & np.isfinite(se_vals)
            if valid.any():
                ax.errorbar(x[valid] + offset, vals[valid], yerr=se_vals[valid],
                            fmt="none", ecolor="black", elinewidth=1.4,
                            capsize=3.5, capthick=1.4, zorder=10)

            for xi, v, s in zip(x, vals, se_vals):
                if not np.isfinite(v):
                    continue
                label_text = f'{v:+.3f}'
                if v >= 0:
                    y_offset = (s if np.isfinite(s) else 0) + 0.004
                    y_pos = v + y_offset
                    va = 'bottom'
                else:
                    y_offset = (s if np.isfinite(s) else 0) + 0.003
                    y_pos = v - y_offset
                    va = 'top'
                ax.text(xi + offset, y_pos, label_text, ha='center', va=va,
                        fontsize=7, fontweight='bold', color='black', zorder=20)

        ax.set_xticks(x)
        ax.set_xticklabels(all_models, rotation=35, ha='right', fontsize=10, fontweight='bold')
        if panel_idx % 2 == 0:
            ax.set_ylabel('Effort Difference: Model - MDN\n(Negative = Less Effort)',
                          fontsize=11, fontweight='bold')
        ax.set_title(panel_titles.get(poverty_title, poverty_title),
                     fontsize=13, fontweight='bold', pad=12)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25, linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if len(axes) > 0:
        axes[0].legend(loc='lower left', fontsize=10, frameon=True, fancybox=True, shadow=True)

    plt.tight_layout(rect=[0, 0.045, 1, 0.96])

    _out = out_path or outfile
    if _out:
        save_and_show(fig, _out)

    return fig


# =====================================================================
# Score-gap analysis (bootstrap)
# =====================================================================

def _gap_stats_from_scores(s: np.ndarray, tau: float) -> Dict[str, float]:
    s = _finite(s)
    if s.size == 0:
        return dict(n=0, share_ge=np.nan, gap_mean=np.nan, gap_median=np.nan)
    ge = (s >= tau)
    below = ~ge
    gap = np.maximum(0.0, tau - s)
    return dict(
        n=int(s.size),
        share_ge=float(np.mean(ge)),
        gap_mean=float(np.mean(gap[below])) if np.any(below) else 0.0,
        gap_median=float(np.median(gap[below])) if np.any(below) else 0.0,
    )


def bootstrap_gap(
    scores: np.ndarray,
    tau: float,
    n_boot: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Bootstrap share >= tau and conditional gap stats."""
    s = _finite(scores)
    n = s.size
    if n == 0:
        return dict(
            share_ge=np.nan, gap_mean=np.nan, gap_median=np.nan,
            share_ge_se=np.nan, gap_mean_se=np.nan, gap_median_se=np.nan,
            share_ge_lo=np.nan, share_ge_hi=np.nan,
            gap_mean_lo=np.nan, gap_mean_hi=np.nan,
            gap_median_lo=np.nan, gap_median_hi=np.nan,
            n=0,
        )

    pt = _gap_stats_from_scores(s, tau)
    boot_share, boot_gmean, boot_gmed = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        st = _gap_stats_from_scores(s[idx], tau)
        boot_share.append(st["share_ge"])
        boot_gmean.append(st["gap_mean"])
        boot_gmed.append(st["gap_median"])

    def _summ(arr):
        arr = np.asarray(arr, float)
        if arr.size < max(30, int(0.2 * n_boot)):
            return (np.nan, np.nan, np.nan)
        return (
            float(np.std(arr, ddof=1)),
            float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)),
        )

    share_se, share_lo, share_hi = _summ(boot_share)
    gmean_se, gmean_lo, gmean_hi = _summ(boot_gmean)
    gmed_se, gmed_lo, gmed_hi = _summ(boot_gmed)

    return dict(
        n=pt["n"],
        share_ge=pt["share_ge"], share_ge_se=share_se,
        share_ge_lo=share_lo, share_ge_hi=share_hi,
        gap_mean=pt["gap_mean"], gap_mean_se=gmean_se,
        gap_mean_lo=gmean_lo, gap_mean_hi=gmean_hi,
        gap_median=pt["gap_median"], gap_median_se=gmed_se,
        gap_median_lo=gmed_lo, gap_median_hi=gmed_hi,
    )


def compute_score_gap_all(
    df: pd.DataFrame,
    id_col: str,
    n_boot: int = 400,
    seed: int = 123,
    include_mdn: bool = True,
) -> pd.DataFrame:
    """Long table with share >= tau and conditional gap for every
    poverty_line x model x threshold."""
    rng = np.random.default_rng(seed)
    models = (["MDN"] + list(MODEL_ORDER)) if include_mdn else list(MODEL_ORDER)
    rows = []

    total = len(PANELS) * len(models) * len(THRESHOLDS)
    pbar = tqdm(total=total, desc="Score-gap (bootstrap)", leave=False)

    for (title, suffix) in PANELS:
        for m in models:
            sc_col = _resolve_colname(df, COLMAP[m], suffix)
            if sc_col is None or sc_col not in df.columns:
                for tau in THRESHOLDS:
                    rows.append({
                        "poverty_line": title, "suffix": suffix, "model": m,
                        "tau": tau, "n": 0,
                        "share_ge": np.nan, "share_ge_se": np.nan,
                        "share_ge_lo": np.nan, "share_ge_hi": np.nan,
                        "gap_mean": np.nan, "gap_mean_se": np.nan,
                        "gap_mean_lo": np.nan, "gap_mean_hi": np.nan,
                        "gap_median": np.nan, "gap_median_se": np.nan,
                        "gap_median_lo": np.nan, "gap_median_hi": np.nan,
                        "score_col": sc_col,
                    })
                    pbar.update(1)
                continue

            base = df[[id_col, sc_col]].replace([np.inf, -np.inf], np.nan).dropna()
            scores = base[sc_col].to_numpy(float)

            for tau in THRESHOLDS:
                st = bootstrap_gap(scores, tau=tau, n_boot=n_boot, rng=rng)
                rows.append({
                    "poverty_line": title, "suffix": suffix, "model": m,
                    "tau": tau, "n": st["n"],
                    "share_ge": st["share_ge"], "share_ge_se": st["share_ge_se"],
                    "share_ge_lo": st["share_ge_lo"], "share_ge_hi": st["share_ge_hi"],
                    "gap_mean": st["gap_mean"], "gap_mean_se": st["gap_mean_se"],
                    "gap_mean_lo": st["gap_mean_lo"], "gap_mean_hi": st["gap_mean_hi"],
                    "gap_median": st["gap_median"], "gap_median_se": st["gap_median_se"],
                    "gap_median_lo": st["gap_median_lo"], "gap_median_hi": st["gap_median_hi"],
                    "score_col": sc_col,
                })
                pbar.update(1)

    pbar.close()
    return pd.DataFrame(rows)


# =====================================================================
# KDE of agree-in-need delta
# =====================================================================

def _agree_mask(mdn: np.ndarray, mod: np.ndarray, thr: float) -> np.ndarray:
    mdn = np.asarray(mdn, float)
    mod = np.asarray(mod, float)
    return np.isfinite(mdn) & np.isfinite(mod) & (mdn < thr) & (mod < thr)


def _kde_xy(values: np.ndarray, grid: np.ndarray, bw="scott"):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 5:
        return None
    kde = gaussian_kde(values, bw_method=bw)
    return kde(grid)


def plot_kde_agree_vs_mdn(
    df: pd.DataFrame,
    poverty_lines=None,
    models=("XGB-1D", "XGB-Mix", "Elastic Net", "OLS+GLM"),
    thresholds=(0.5, 0.8),
    n_grid=400,
    clip=(-0.6, 0.6),
    bw="scott",
    min_n=50,
    figsize=(14, 6),
    dpi=160,
    title_prefix="Agree-in-need KDE: delta = MDN - model",
    out_path=None,
):
    """KDE overlay of delta = MDN - model for agree-in-need households."""
    if poverty_lines is None:
        poverty_lines = PANELS

    n_rows = len(poverty_lines)
    n_cols = len(thresholds)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi, squeeze=False)

    x_min, x_max = clip
    grid = np.linspace(x_min, x_max, n_grid)

    for r, (pline_name, suffix) in enumerate(poverty_lines):
        mdn_col = _resolve_colname(df, COLMAP["MDN"], suffix)
        if mdn_col is None:
            for c in range(n_cols):
                axes[r, c].set_axis_off()
            continue

        for c, thr in enumerate(thresholds):
            ax = axes[r, c]
            ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.6)
            ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.25)
            ax.set_xlim(x_min, x_max)
            ax.set_ylabel("Density" if c == 0 else "")
            ax.set_xlabel("delta = s(MDN) - s(model)")
            ax.set_title(f"{pline_name} -- agree: score<{thr}")

            y_max = 0.0
            for m in models:
                mod_col = _resolve_colname(df, COLMAP[m], suffix)
                if mod_col is None:
                    continue
                sub = df[[mdn_col, mod_col]].dropna()
                mdn = sub[mdn_col].to_numpy(dtype=float)
                mod = sub[mod_col].to_numpy(dtype=float)
                msk = _agree_mask(mdn, mod, thr)
                if msk.sum() < min_n:
                    continue
                delta = (mdn[msk] - mod[msk]).astype(float)
                y_vals = _kde_xy(delta, grid, bw=bw)
                if y_vals is None:
                    continue
                y_max = max(y_max, float(np.nanmax(y_vals)))
                ax.plot(grid, y_vals, linewidth=2.0, label=f"{m} (n={msk.sum()})")
                ax.axvline(float(np.mean(delta)), lw=1.2, alpha=0.35)

            ax.text(0.02, 0.98,
                    "delta<0: model more optimistic\ndelta>0: model more pessimistic",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))
            if y_max > 0:
                ax.set_ylim(0, y_max * 1.15)
            if r == 0 and c == n_cols - 1:
                ax.legend(loc="upper left", fontsize=8, frameon=False)

    fig.suptitle(title_prefix, y=1.02, fontsize=13)
    fig.tight_layout()

    if out_path:
        save_and_show(fig, out_path)

    return fig
