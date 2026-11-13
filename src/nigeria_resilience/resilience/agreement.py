"""Band-wise agreement analysis: bootstrap how other models classify
households that MDN places in each resilience band."""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from nigeria_resilience.visualization.save import save_and_show

# =====================================================================
# Configuration
# =====================================================================

BANDS = [(0.0, 0.5), (0.5, 0.8), (0.8, 1.0)]
BAND_LABELS = ["Non-resilient (<0.5)", "Resilient (0.5-0.8)", "Most-resilient (>=0.8)"]

PANELS = [
    ("$1.90/day Poverty Line", "_pl_1_90"),
    ("$3.20/day Poverty Line", "_pl_3_20"),
    ("$5.50/day Poverty Line", "_pl_5_50"),
    ("National Poverty Line", "_at_nat"),
]

COLMAP = {
    "MDN":         "resilience_score_mdn",
    "NN-1D":       "resilience_score_nn",
    "XGB-1D":      "resilience_score_xgb_1d",
    "XGB-Mix":     "resilience_score_xgb_md",
    "Elastic Net": "resilience_score_elastic_net",
    "OLS+GLM":     "resilience_score_ols_glm",
}

MODEL_ORDER = ("OLS+GLM", "Elastic Net", "XGB-1D", "NN-1D", "XGB-Mix")


# =====================================================================
# Helpers
# =====================================================================

def _band_index(p):
    if p < 0.5:
        return 0
    if p < 0.8:
        return 1
    return 2


def _summ_from_vec(vec):
    return {
        "mean": float(np.mean(vec)),
        "ci_low": float(np.quantile(vec, 0.025)),
        "ci_high": float(np.quantile(vec, 0.975)),
        "boot": vec,
    }


def _resolve_colname(df, base, suffix):
    cand = f"{base}{suffix}"
    if cand in df.columns:
        return cand
    if suffix == "_pl_1_90" and base in df.columns:
        return base
    return None


def _stack_panel_matrix(df, suffix):
    """Stack resilience scores into (N, M) matrix including MDN."""
    labels_full = ["MDN"] + list(MODEL_ORDER)
    cols, labels = [], []
    for model in labels_full:
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


# =====================================================================
# Bootstrap band-wise agreement vs MDN
# =====================================================================

def bootstrap_bandwise_agreement_vs_mdn(test_df_temporal, B=400, seed=42):
    """For each poverty line and MDN band, compute how other models classify
    the same households, with paired bootstrap CIs.

    Returns
    -------
    results : dict
        ``results[line_title][band_label][model][target_band_label]`` ->
        ``{mean, ci_low, ci_high, boot}``
    """
    rng = np.random.default_rng(seed)
    out = {}

    for title, suffix in PANELS:
        mat, labels = _stack_panel_matrix(test_df_temporal, suffix)
        if mat.size == 0 or "MDN" not in labels:
            out[title] = {}
            continue

        ref_idx = labels.index("MDN")
        N, M = mat.shape
        mdn_bands_full = np.array([_band_index(p) for p in mat[:, ref_idx]], dtype=int)

        store = {
            b: {m: {t: [] for t in range(3)} for m in labels if m != "MDN"}
            for b in range(3)
        }

        for _ in range(B):
            idx = rng.integers(0, N, size=N)
            mdn_b = mdn_bands_full[idx]
            mat_b = mat[idx, :]

            for b in range(3):
                sel = np.where(mdn_b == b)[0]
                if sel.size == 0:
                    for m in labels:
                        if m == "MDN":
                            continue
                        for t in range(3):
                            store[b][m][t].append(np.nan)
                    continue

                for m_i, m in enumerate(labels):
                    if m == "MDN":
                        continue
                    tgt = np.array([_band_index(p) for p in mat_b[sel, m_i]], dtype=int)
                    for t in range(3):
                        store[b][m][t].append(np.mean(tgt == t))

        out_line = {}
        for b in range(3):
            band_lab = BAND_LABELS[b]
            out_line[band_lab] = {}
            for m in labels:
                if m == "MDN":
                    continue
                out_line[band_lab][m] = {}
                for t in range(3):
                    arr = np.asarray(store[b][m][t], float)
                    arr = arr[np.isfinite(arr)]
                    if arr.size == 0:
                        out_line[band_lab][m][BAND_LABELS[t]] = {
                            "mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "boot": arr,
                        }
                    else:
                        out_line[band_lab][m][BAND_LABELS[t]] = _summ_from_vec(arr)
        out[title] = out_line

    return out


# =====================================================================
# Visualization
# =====================================================================

def plot_bandwise_agreement_grid(
    results_by_line_bandwise, *,
    col_width=5.4, row_height=3.4, dpi=180,
    annotate_threshold=0.08,
    out_path=None,
    font_scale=1.0,
    bar_width=0.82,
    x_margin=0.12,
    hspace=0.45, wspace=0.25,
):
    """Grid: rows = poverty lines, cols = MDN bands (L/M/H).
    Stacked bars show where each non-MDN model places MDN's households."""
    colors = {
        "Non-resilient (<0.5)": "#d62728",
        "Resilient (0.5-0.8)": "#ff7f0e",
        "Most-resilient (>=0.8)": "#2ca02c",
    }

    ordered_titles = [t for t, _ in PANELS if t in results_by_line_bandwise]
    n_rows, n_cols = len(ordered_titles), 3
    figsize = (col_width * n_cols, row_height * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi, squeeze=False)

    for r, title in enumerate(ordered_titles):
        panel = results_by_line_bandwise[title]
        for c, band_lab in enumerate(BAND_LABELS):
            ax = axes[r, c]
            if band_lab not in panel or not panel[band_lab]:
                ax.axis("off")
                continue

            models = [m for m in MODEL_ORDER if m in panel[band_lab] and m != "MDN"]
            x = np.arange(len(models))

            cum = np.zeros(len(models), dtype=float)
            for seg_lab in BAND_LABELS:
                means = np.array([panel[band_lab][m][seg_lab]["mean"] for m in models])
                lows = np.array([panel[band_lab][m][seg_lab]["ci_low"] for m in models])
                highs = np.array([panel[band_lab][m][seg_lab]["ci_high"] for m in models])

                ax.bar(
                    x, means, bar_width, bottom=cum,
                    color=colors[seg_lab], edgecolor="white", linewidth=0.8,
                    label=seg_lab if (r == 0 and c == 0) else None,
                )

                mid = cum + means / 2.0
                for i, mval in enumerate(means):
                    if mval < annotate_threshold:
                        continue
                    gap = min(0.025, 0.25 * mval)
                    y_mean = mid[i] + 0.5 * gap
                    y_ci = mid[i] - 0.5 * gap
                    ax.text(
                        x[i], y_mean, f"{mval * 100:.0f}%",
                        ha="center", va="center", color="white",
                        fontsize=10 * font_scale, fontweight="bold",
                    )
                    ax.text(
                        x[i], y_ci, f"[{lows[i] * 100:.0f}, {highs[i] * 100:.0f}]",
                        ha="center", va="center", color="white",
                        fontsize=8 * font_scale,
                    )

                cum += means

            if r == 0:
                ax.set_title(f"Within MDN band: {band_lab}", fontsize=12 * font_scale)

            ax.set_xticks(x, models, rotation=25, ha="right", fontsize=10 * font_scale)
            ax.set_ylim(0, 1.0)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
            ax.margins(x=x_margin)

            if c == 0:
                ax.text(
                    -0.22, 0.5, title,
                    transform=ax.transAxes, rotation=90, va="center", ha="right",
                    fontsize=12 * font_scale,
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 0.06), fontsize=11 * font_scale,
    )

    fig.text(
        0.015, 0.5, "Share of households", rotation=90, va="center",
        ha="left", fontsize=12 * font_scale,
    )

    fig.tight_layout(rect=[0.05, 0.08, 1.0, 1.0])
    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    if out_path:
        save_and_show(fig, out_path)
    else:
        plt.show()

    return fig


# =====================================================================
# Band-based SMD profiling (one-vs-rest and pairwise)
# =====================================================================

from nigeria_resilience.analysis.smd import smd_simple as smd

_BANDS_OVR = [
    ("Non-resilient (<0.5)", 0),
    ("Resilient (0.5-0.8)", 1),
    ("Most resilient (>=0.8)", 2),
]

PAIRWISE3 = [
    ("Resilient vs Non-resilient", 1, 0),
    ("Most resilient vs Non-resilient", 2, 0),
    ("Most resilient vs Resilient", 2, 1),
]


def _band_from_score(s: np.ndarray) -> np.ndarray:
    """Assign band codes: 0=NR, 1=R, 2=MR, -1=invalid."""
    s = np.asarray(s, float)
    b = np.full_like(s, -1, dtype=int)
    ok = np.isfinite(s)
    b[ok & (s < 0.5)] = 0
    b[ok & (s >= 0.5) & (s < 0.8)] = 1
    b[ok & (s >= 0.8)] = 2
    return b


def compute_smd_by_band_onevsrest(
    df: pd.DataFrame,
    id_col: str,
    X_cols: List[str],
    models: Tuple[str, ...] = None,
    n_boot: int = 400,
    seed: int = 123,
    min_in: int = 30,
    min_out: int = 30,
) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    """One-vs-rest SMD profiles per band.

    Returns out[panel_title][model][band_title] -> DataFrame.
    """
    from tqdm import tqdm

    if models is None:
        models = ("MDN",) + MODEL_ORDER

    rng = np.random.default_rng(seed)
    out: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}

    total = len(PANELS) * len(models) * len(_BANDS_OVR) * n_boot
    pbar = tqdm(total=total, desc="Band profiles (plain bootstrap)", leave=False)

    for (panel_title, suffix) in PANELS:
        out[panel_title] = {}
        for model in models:
            out[panel_title][model] = {}

            score_base = COLMAP.get(model, None)
            if score_base is None:
                for band_title, _ in _BANDS_OVR:
                    out[panel_title][model][band_title] = pd.DataFrame()
                pbar.update(len(_BANDS_OVR) * n_boot)
                continue

            from nigeria_resilience.analysis.smd import _resolve_colname as _rc
            score_col = _rc(df, score_base, suffix)
            if score_col is None:
                for band_title, _ in _BANDS_OVR:
                    out[panel_title][model][band_title] = pd.DataFrame()
                pbar.update(len(_BANDS_OVR) * n_boot)
                continue

            base = df[[id_col, score_col] + X_cols].copy()
            base = base.replace([np.inf, -np.inf], np.nan).dropna(subset=[id_col, score_col])
            if base.empty:
                for band_title, _ in _BANDS_OVR:
                    out[panel_title][model][band_title] = pd.DataFrame()
                pbar.update(len(_BANDS_OVR) * n_boot)
                continue

            score = base[score_col].astype(float).to_numpy()
            band = _band_from_score(score)
            keep_all = (band >= 0)
            sub_all = base.loc[keep_all, X_cols].copy()
            band_all = band[keep_all]

            for band_title, band_code in _BANDS_OVR:
                m_in = (band_all == band_code)
                m_out = (band_all != band_code)
                n_in = int(m_in.sum())
                n_out = int(m_out.sum())

                if (n_in < min_in) or (n_out < min_out):
                    out[panel_title][model][band_title] = pd.DataFrame()
                    pbar.update(n_boot)
                    continue

                D = m_in.astype(int)
                rows_smd = [(xcol, smd(sub_all[xcol].to_numpy(), D)) for xcol in X_cols]
                tab = pd.DataFrame(rows_smd, columns=["var", "smd"]).set_index("var")
                tab["abs_smd"] = np.abs(tab["smd"])
                tab["n_in"] = n_in
                tab["n_out"] = n_out

                sub2 = sub_all.copy()
                sub2["_D"] = D
                boot_vals = {v: [] for v in X_cols}
                ok_draws = 0

                for _ in range(n_boot):
                    b = sub2.sample(n=len(sub2), replace=True, random_state=rng)
                    Db = b["_D"].to_numpy().astype(int)
                    if (int((Db == 1).sum()) < min_in) or (int((Db == 0).sum()) < min_out):
                        pbar.update(1)
                        continue
                    ok_draws += 1
                    for xcol in X_cols:
                        sb = smd(b[xcol].to_numpy(), Db)
                        if np.isfinite(sb):
                            boot_vals[xcol].append(float(sb))
                    pbar.update(1)

                min_kept = max(30, int(0.2 * n_boot))
                boot_mean, boot_se, ci_low, ci_high, ok_rate = [], [], [], [], []
                for xcol in X_cols:
                    arr = np.asarray(boot_vals[xcol], float)
                    ok_rate.append(len(arr) / max(1, n_boot))
                    if len(arr) < min_kept:
                        boot_mean.append(np.nan); boot_se.append(np.nan)
                        ci_low.append(np.nan); ci_high.append(np.nan)
                    else:
                        boot_mean.append(float(np.nanmean(arr)))
                        boot_se.append(float(np.nanstd(arr, ddof=1)))
                        ci_low.append(float(np.nanpercentile(arr, 2.5)))
                        ci_high.append(float(np.nanpercentile(arr, 97.5)))

                tab["boot_mean"] = boot_mean
                tab["boot_se"] = boot_se
                tab["ci_low"] = ci_low
                tab["ci_high"] = ci_high
                tab["boot_ok_rate"] = ok_rate
                tab["boot_bias"] = tab["boot_mean"] - tab["smd"]
                tab.attrs["boot_ok_draws"] = ok_draws
                out[panel_title][model][band_title] = tab

    pbar.close()
    return out


def compute_smd_by_band_pairwise(
    df: pd.DataFrame,
    id_col: str,
    X_cols: List[str],
    models: Tuple[str, ...] = None,
    n_boot: int = 400,
    seed: int = 123,
    min_g0: int = 30,
    min_g1: int = 30,
) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    """Pairwise SMD profiles between bands.

    Returns out[panel_title][model][contrast_title] -> DataFrame.
    """
    from tqdm import tqdm

    if models is None:
        models = ("MDN",) + MODEL_ORDER

    rng = np.random.default_rng(seed)
    out: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}

    total = len(PANELS) * len(models) * len(PAIRWISE3) * n_boot
    pbar = tqdm(total=total, desc="Pairwise band SMD (plain bootstrap)", leave=False)

    for (panel_title, suffix) in PANELS:
        out[panel_title] = {}
        for model in models:
            out[panel_title][model] = {}

            base_col = COLMAP.get(model, None)
            from nigeria_resilience.analysis.smd import _resolve_colname as _rc
            score_col = _rc(df, base_col, suffix) if base_col is not None else None

            if score_col is None:
                for cname, _, _ in PAIRWISE3:
                    out[panel_title][model][cname] = pd.DataFrame()
                pbar.update(len(PAIRWISE3) * n_boot)
                continue

            base = df[[id_col, score_col] + X_cols].copy()
            base = base.replace([np.inf, -np.inf], np.nan).dropna(subset=[id_col, score_col])
            if base.empty:
                for cname, _, _ in PAIRWISE3:
                    out[panel_title][model][cname] = pd.DataFrame()
                pbar.update(len(PAIRWISE3) * n_boot)
                continue

            score = base[score_col].astype(float).to_numpy()
            band = _band_from_score(score)

            for cname, g1, g0 in PAIRWISE3:
                m = (band == g1) | (band == g0)
                if not np.any(m):
                    out[panel_title][model][cname] = pd.DataFrame()
                    pbar.update(n_boot)
                    continue

                sub = base.loc[m, X_cols].copy()
                D = (band[m] == g1).astype(int)
                n1 = int((D == 1).sum())
                n0 = int((D == 0).sum())

                if (n1 < min_g1) or (n0 < min_g0):
                    out[panel_title][model][cname] = pd.DataFrame()
                    pbar.update(n_boot)
                    continue

                rows_smd = [(xcol, smd(sub[xcol].to_numpy(), D)) for xcol in X_cols]
                tab = pd.DataFrame(rows_smd, columns=["var", "smd"]).set_index("var")
                tab["abs_smd"] = np.abs(tab["smd"])
                tab["n0"] = n0
                tab["n1"] = n1

                sub2 = sub.copy()
                sub2["_D"] = D
                boot_vals = {v: [] for v in X_cols}
                ok_draws = 0

                for _ in range(n_boot):
                    b_df = sub2.sample(n=len(sub2), replace=True, random_state=rng)
                    Db = b_df["_D"].to_numpy().astype(int)
                    if (int((Db == 1).sum()) < min_g1) or (int((Db == 0).sum()) < min_g0):
                        pbar.update(1)
                        continue
                    ok_draws += 1
                    for xcol in X_cols:
                        sb = smd(b_df[xcol].to_numpy(), Db)
                        if np.isfinite(sb):
                            boot_vals[xcol].append(float(sb))
                    pbar.update(1)

                min_kept = max(30, int(0.2 * n_boot))
                boot_mean, boot_se, ci_low, ci_high, ok_rate = [], [], [], [], []
                for xcol in X_cols:
                    arr = np.asarray(boot_vals[xcol], float)
                    ok_rate.append(len(arr) / max(1, n_boot))
                    if len(arr) < min_kept:
                        boot_mean.append(np.nan); boot_se.append(np.nan)
                        ci_low.append(np.nan); ci_high.append(np.nan)
                    else:
                        boot_mean.append(float(np.nanmean(arr)))
                        boot_se.append(float(np.nanstd(arr, ddof=1)))
                        ci_low.append(float(np.nanpercentile(arr, 2.5)))
                        ci_high.append(float(np.nanpercentile(arr, 97.5)))

                tab["boot_mean"] = boot_mean
                tab["boot_se"] = boot_se
                tab["ci_low"] = ci_low
                tab["ci_high"] = ci_high
                tab["boot_ok_rate"] = ok_rate
                tab["boot_bias"] = tab["boot_mean"] - tab["smd"]
                tab.attrs["boot_ok_draws"] = ok_draws
                out[panel_title][model][cname] = tab

    pbar.close()
    return out


# =====================================================================
# Plotting: band profiles grid per model
# =====================================================================

def plot_band_profiles_grid_per_model(
    band_results: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    labels_map: Dict[str, str],
    models: Tuple[str, ...] = None,
    top_k: int = 12,
    select_by: str = "abs_smd",
    smd_thresh: float = 0.10,
    err: str = "se",
    se_mult: float = 1.96,
    figsize=(16, 9),
    dpi: int = 170,
    annotate: bool = True,
    out_path: Optional[str] = None,
):
    """One figure per model: rows=PANELS, cols=BANDS (one-vs-rest).

    Returns figs[model] = Figure.
    """
    if models is None:
        models = ("MDN",) + MODEL_ORDER

    figs = {}
    band_titles = [bt for bt, _ in _BANDS_OVR]
    panel_titles = [pt for pt, _ in PANELS]

    for model in models:
        max_abs = 0.0
        for panel_title in panel_titles:
            for band_title in band_titles:
                dfm = band_results.get(panel_title, {}).get(model, {}).get(band_title, pd.DataFrame())
                if dfm is None or dfm.empty or "smd" not in dfm.columns:
                    continue
                dfx = dfm.copy()
                dfx["abs_smd"] = np.abs(dfx["smd"])
                score = dfx["abs_smd"]
                dfx = dfx[score >= smd_thresh]
                if dfx.empty:
                    continue
                top_vars = score.loc[dfx.index].sort_values(ascending=False).head(top_k).index
                dfx = dfx.reindex(top_vars)
                vals = dfx["smd"].to_numpy(float)
                lo, hi = vals.copy(), vals.copy()
                if err == "se" and "boot_se" in dfx.columns:
                    d = np.abs(dfx["boot_se"].to_numpy(float)) * se_mult
                    lo, hi = vals - d, vals + d
                elif err == "ci" and ("ci_low" in dfx.columns) and ("ci_high" in dfx.columns):
                    lo, hi = dfx["ci_low"].to_numpy(float), dfx["ci_high"].to_numpy(float)
                cur = np.nanmax(np.abs(np.r_[lo, hi]))
                if np.isfinite(cur):
                    max_abs = max(max_abs, float(cur))

        if max_abs <= 0:
            continue
        xpad = 0.18 * max_abs
        xlim = (-max_abs - xpad, max_abs + xpad)
        dx = 0.02 * (xlim[1] - xlim[0])

        fig, axes = plt.subplots(len(PANELS), len(_BANDS_OVR), figsize=figsize, dpi=dpi,
                                  squeeze=False, sharex=True)

        for r, (panel_title, _) in enumerate(PANELS):
            for c, (band_title, _) in enumerate(_BANDS_OVR):
                ax = axes[r, c]
                dfm = band_results.get(panel_title, {}).get(model, {}).get(band_title, pd.DataFrame())

                ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.6)
                ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.set_xlim(*xlim)

                if r == 0:
                    ax.set_title(band_title, fontsize=11)
                if c == 0:
                    ax.set_ylabel(panel_title, fontsize=11)

                if dfm is None or dfm.empty or "smd" not in dfm.columns:
                    ax.text(0.5, 0.5, "No data / too few n", ha="center", va="center",
                            transform=ax.transAxes, fontsize=10)
                    ax.set_yticks([])
                    continue

                dfx = dfm.copy()
                dfx["abs_smd"] = np.abs(dfx["smd"])
                score = dfx["abs_smd"]
                dfx = dfx[score >= smd_thresh]
                if dfx.empty:
                    ax.text(0.5, 0.5, f"No vars |SMD|>={smd_thresh}", ha="center", va="center",
                            transform=ax.transAxes, fontsize=10)
                    ax.set_yticks([])
                    continue

                top_vars = score.loc[dfx.index].sort_values(ascending=False).head(top_k).index.tolist()
                dfx = dfx.reindex(top_vars)
                vals = dfx["smd"].to_numpy(float)
                y_pos = np.arange(len(vals))
                labels_list = [labels_map.get(v, v) for v in dfx.index.tolist()]

                lo, hi = vals.copy(), vals.copy()
                if err == "se" and "boot_se" in dfx.columns:
                    d = np.abs(dfx["boot_se"].to_numpy(float)) * se_mult
                    lo, hi = vals - d, vals + d
                elif err == "ci" and ("ci_low" in dfx.columns) and ("ci_high" in dfx.columns):
                    lo, hi = dfx["ci_low"].to_numpy(float), dfx["ci_high"].to_numpy(float)

                ax.barh(y_pos, vals, alpha=0.85, zorder=2)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels_list, fontsize=8)
                ax.invert_yaxis()

                if err in ("se", "ci"):
                    ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(vals)
                    ax.hlines(y_pos[ok], lo[ok], hi[ok], colors="black", linewidth=1.2, zorder=6)
                    ax.vlines(lo[ok], y_pos[ok] - 0.18, y_pos[ok] + 0.18,
                              colors="black", linewidth=1.2, zorder=6)
                    ax.vlines(hi[ok], y_pos[ok] - 0.18, y_pos[ok] + 0.18,
                              colors="black", linewidth=1.2, zorder=6)

                if annotate:
                    for i, vv in enumerate(vals):
                        if not np.isfinite(vv):
                            continue
                        l_i = lo[i] if np.isfinite(lo[i]) else vv
                        h_i = hi[i] if np.isfinite(hi[i]) else vv
                        xt = (h_i + dx) if vv >= 0 else (l_i - dx)
                        ha = "left" if vv >= 0 else "right"
                        ax.text(xt, i, f"{vv:+.2f}", ha=ha, va="center", fontsize=8,
                                bbox=dict(facecolor="white", alpha=0.80, edgecolor="none", pad=0.6),
                                clip_on=False, zorder=7)

                if r == len(PANELS) - 1:
                    ax.set_xlabel("SMD (SD units)", fontsize=10)

        fig.suptitle(f"Band profiles -- {model} -- Top {top_k} covariates per cell (one-vs-rest), err={err}",
                     y=1.01, fontsize=14)
        fig.tight_layout()

        if out_path:
            save_and_show(fig, out_path.replace(".png", f"_{model}.png"))

        figs[model] = fig

    return figs


def plot_pairwise_profiles_grid_per_model(
    pairwise_results: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    labels_map: Dict[str, str],
    models: Tuple[str, ...] = None,
    contrasts: Tuple[str, ...] = None,
    top_k: int = 12,
    select_by: str = "abs_smd",
    smd_thresh: float = 0.10,
    err: str = "se",
    se_mult: float = 1.96,
    figsize=(18, 9),
    dpi: int = 170,
    annotate: bool = True,
    out_path: Optional[str] = None,
):
    """One figure per model: rows=PANELS, cols=pairwise contrasts.

    Returns figs[model] = Figure.
    """
    if models is None:
        models = ("MDN",) + MODEL_ORDER
    if contrasts is None:
        contrasts = tuple([c for c, _, _ in PAIRWISE3])

    figs = {}
    panel_titles = [pt for pt, _ in PANELS]

    for model in models:
        max_abs = 0.0
        for panel_title in panel_titles:
            for cname in contrasts:
                dfc = pairwise_results.get(panel_title, {}).get(model, {}).get(cname, pd.DataFrame())
                if dfc is None or dfc.empty or "smd" not in dfc.columns:
                    continue
                dfx = dfc.copy()
                dfx["abs_smd"] = np.abs(dfx["smd"])
                score = dfx["abs_smd"]
                dfx = dfx[score >= smd_thresh]
                if dfx.empty:
                    continue
                top_vars = score.loc[dfx.index].sort_values(ascending=False).head(top_k).index
                dfx = dfx.reindex(top_vars)
                vals = dfx["smd"].to_numpy(float)
                lo, hi = vals.copy(), vals.copy()
                if err == "se" and "boot_se" in dfx.columns:
                    d = np.abs(dfx["boot_se"].to_numpy(float)) * se_mult
                    lo, hi = vals - d, vals + d
                elif err == "ci" and ("ci_low" in dfx.columns) and ("ci_high" in dfx.columns):
                    lo, hi = dfx["ci_low"].to_numpy(float), dfx["ci_high"].to_numpy(float)
                cur = np.nanmax(np.abs(np.r_[lo, hi]))
                if np.isfinite(cur):
                    max_abs = max(max_abs, float(cur))

        if max_abs <= 0:
            continue
        xpad = 0.18 * max_abs
        xlim = (-max_abs - xpad, max_abs + xpad)
        dx = 0.02 * (xlim[1] - xlim[0])

        fig, axes = plt.subplots(len(PANELS), len(contrasts), figsize=figsize, dpi=dpi,
                                  squeeze=False, sharex=True)

        for r, (panel_title, _) in enumerate(PANELS):
            for c_idx, cname in enumerate(contrasts):
                ax = axes[r, c_idx]
                dfc = pairwise_results.get(panel_title, {}).get(model, {}).get(cname, pd.DataFrame())

                ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.6)
                ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.set_xlim(*xlim)

                if r == 0:
                    ax.set_title(cname, fontsize=11)
                if c_idx == 0:
                    ax.set_ylabel(panel_title, fontsize=11)

                if dfc is None or dfc.empty or "smd" not in dfc.columns:
                    ax.text(0.5, 0.5, "No data / too few n", ha="center", va="center",
                            transform=ax.transAxes, fontsize=10)
                    ax.set_yticks([])
                    continue

                dfx = dfc.copy()
                dfx["abs_smd"] = np.abs(dfx["smd"])
                score = dfx["abs_smd"]
                dfx = dfx[score >= smd_thresh]
                if dfx.empty:
                    ax.text(0.5, 0.5, f"No vars |SMD|>={smd_thresh}", ha="center", va="center",
                            transform=ax.transAxes, fontsize=10)
                    ax.set_yticks([])
                    continue

                top_vars = score.loc[dfx.index].sort_values(ascending=False).head(top_k).index.tolist()
                dfx = dfx.reindex(top_vars)
                vals = dfx["smd"].to_numpy(float)
                y_pos = np.arange(len(vals))
                labels_list = [labels_map.get(v, v) for v in dfx.index.tolist()]

                lo, hi = vals.copy(), vals.copy()
                if err == "se" and "boot_se" in dfx.columns:
                    d = np.abs(dfx["boot_se"].to_numpy(float)) * se_mult
                    lo, hi = vals - d, vals + d
                elif err == "ci" and ("ci_low" in dfx.columns) and ("ci_high" in dfx.columns):
                    lo, hi = dfx["ci_low"].to_numpy(float), dfx["ci_high"].to_numpy(float)

                ax.barh(y_pos, vals, alpha=0.85, zorder=2)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels_list, fontsize=8)
                ax.invert_yaxis()

                if err in ("se", "ci"):
                    ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(vals)
                    ax.hlines(y_pos[ok], lo[ok], hi[ok], colors="black", linewidth=1.2, zorder=6)
                    ax.vlines(lo[ok], y_pos[ok] - 0.18, y_pos[ok] + 0.18,
                              colors="black", linewidth=1.2, zorder=6)
                    ax.vlines(hi[ok], y_pos[ok] - 0.18, y_pos[ok] + 0.18,
                              colors="black", linewidth=1.2, zorder=6)

                if annotate:
                    for i, vv in enumerate(vals):
                        if not np.isfinite(vv):
                            continue
                        l_i = lo[i] if np.isfinite(lo[i]) else vv
                        h_i = hi[i] if np.isfinite(hi[i]) else vv
                        xt = (h_i + dx) if vv >= 0 else (l_i - dx)
                        ha = "left" if vv >= 0 else "right"
                        ax.text(xt, i, f"{vv:+.2f}", ha=ha, va="center", fontsize=8,
                                bbox=dict(facecolor="white", alpha=0.80, edgecolor="none", pad=0.6),
                                clip_on=False, zorder=7)

                if r == len(PANELS) - 1:
                    ax.set_xlabel("SMD (SD units)", fontsize=10)

        fig.suptitle(f"Pairwise band profiles -- {model} -- Top {top_k} per cell, err={err}",
                     y=1.01, fontsize=14)
        fig.tight_layout()

        if out_path:
            save_and_show(fig, out_path.replace(".png", f"_{model}.png"))

        figs[model] = fig

    return figs
