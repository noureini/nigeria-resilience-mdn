import pandas as pd

"""Brier Skill Score analysis: fixed-ref BSS, reliability-resolution decomposition,
paired-bootstrap BSS ratios."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from nigeria_resilience.visualization.save import save_and_show

# =====================================================================
# Column map (resilience probability columns)
# =====================================================================

COLMAP = {
    "MDN":         "resilience_score_mdn",
    "NN-1D":       "resilience_score_nn",
    "XGB-1D":      "resilience_score_xgb_1d",
    "XGB-Mix":     "resilience_score_xgb_md",
    "Elastic Net": "resilience_score_elastic_net",
    "OLS+GLM":     "resilience_score_ols_glm",
}


# =====================================================================
# Basic helpers
# =====================================================================

def make_binary_outcome(log_y, pl_value_usd):
    thr = np.log(pl_value_usd)
    return (np.asarray(log_y) > thr).astype(int)


def brier_score(y_true, p_pred):
    y = np.asarray(y_true, float)
    p = np.asarray(p_pred, float)
    return float(np.mean((p - y) ** 2))


def _expand_pref(p_ref, n):
    if np.isscalar(p_ref):
        return np.full(n, float(p_ref), dtype=float)
    p_ref = np.asarray(p_ref, float)
    assert p_ref.shape[0] == n
    return p_ref


# =====================================================================
# BSS with fixed reference
# =====================================================================

def brier_skill_bootstrap_fixed_ref(y_true, preds_dict, p_ref, B=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, int)
    N = y_true.shape[0]
    p_ref_vec = _expand_pref(p_ref, N)
    P = {m: np.asarray(p, float) for m, p in preds_dict.items()}
    boots = {m: [] for m in preds_dict}
    bs_ref_point = brier_score(y_true, p_ref_vec)
    for _ in range(B):
        idx = rng.integers(0, N, size=N)
        yb = y_true[idx]
        pr_b = p_ref_vec[idx]
        bs_ref = brier_score(yb, pr_b)
        if bs_ref <= 1e-15:
            for m in boots:
                boots[m].append(np.nan)
            continue
        for m, p in P.items():
            pb = p[idx]
            bs = brier_score(yb, pb)
            boots[m].append(1.0 - bs / bs_ref)
    out = {}
    for m, arr in boots.items():
        arr = np.asarray(arr, float)
        out[m] = {
            "mean": float(np.nanmean(arr)),
            "ci_low": float(np.nanquantile(arr, 0.025)),
            "ci_high": float(np.nanquantile(arr, 0.975)),
            "boot": arr,
            "bs_ref_point": float(bs_ref_point),
        }
    return out


def _get_preds_for_line(df, suffix, strict=True):
    preds = {}
    for model, base in COLMAP.items():
        candidates = [f"{base}{suffix}"]
        if suffix == "_pl_1_90":
            candidates.append(base)
        found = next((c for c in candidates if c in df.columns), None)
        if found is None:
            msg = f"Missing column for {model}: tried {candidates}"
            if strict:
                raise KeyError(msg)
            print("WARN:", msg)
            preds[model] = np.full(len(df), np.nan)
            continue
        preds[model] = np.asarray(df[found], float)
    return preds


def bss_bar_grid_fixedref(
    test_df_temporal,
    log_y_baseline,
    baseline_label="prev-year prevalence on test",
    intl_lines=None,
    nat_line_value=None,
    B=1000, seed=42,
    out_path=None,
):
    if intl_lines is None:
        intl_lines = [("1.90", 1.90), ("3.20", 3.20), ("5.50", 5.50)]

    log_y_test = test_df_temporal["lead_ltotcons_per_capita_per_day_2016cst"].to_numpy()

    panels = [(lab, val, f"_pl_{lab.replace('.', '_')}") for lab, val in intl_lines]
    if nat_line_value is not None:
        panels.append(("National", nat_line_value, "_at_nat"))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (lab, pl_val, suffix) in zip(axes, panels):
        y_true = make_binary_outcome(log_y_test, pl_val)
        y_base = make_binary_outcome(log_y_baseline, pl_val)
        p_ref_val = float(np.mean(y_base))

        preds = _get_preds_for_line(test_df_temporal, suffix)
        summ = brier_skill_bootstrap_fixed_ref(y_true, preds, p_ref=p_ref_val, B=B, seed=seed)

        models = sorted(summ.keys(), key=lambda m: summ[m]["mean"], reverse=True)
        means = np.array([summ[m]["mean"] for m in models])
        lows = np.array([summ[m]["ci_low"] for m in models])
        highs = np.array([summ[m]["ci_high"] for m in models])

        x = np.arange(len(models))
        barw = 0.65
        ax.bar(x, means, barw, edgecolor="black", alpha=0.85)
        yerr = np.vstack([means - lows, highs - means])
        ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="black", capsize=5, linewidth=1.2)

        for i, m in enumerate(means):
            ax.text(
                x[i], highs[i] + 0.02 * (highs[i] - lows[i] + 1e-9),
                f"{m:.2f}", ha="center", va="bottom", fontsize=9,
            )

        ax.set_xticks(x, models, rotation=25, ha="right")
        ax.set_ylabel("Brier Skill Score (higher is better)")
        ax.set_title(f"BSS at {lab} Poverty Line ({baseline_label})")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.axhline(0.0, color="black", lw=2.0)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.6)

    for k in range(len(panels), 4):
        axes[k].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if out_path:
        save_and_show(fig, out_path)
    else:
        plt.show()

    return fig


# =====================================================================
# Murphy decomposition: reliability & resolution
# =====================================================================

def murphy_components(y_true, p_pred, n_bins=10, strategy="quantile", fixed_edges=None):
    y = np.asarray(y_true, int)
    p = np.asarray(p_pred, float)
    N = len(y)

    if fixed_edges is None:
        if strategy == "quantile":
            qs = np.linspace(0, 1, n_bins + 1)
            edges = np.unique(np.quantile(p, qs))
            if len(edges) - 1 < n_bins:
                edges = np.linspace(0, 1, n_bins + 1)
        else:
            edges = np.linspace(0, 1, n_bins + 1)
    else:
        edges = fixed_edges

    inds = np.digitize(p, edges[1:-1], right=True)
    n_k, f_k, o_k = [], [], []
    for b in range(len(edges) - 1):
        m = inds == b
        if not np.any(m):
            continue
        n_k.append(np.sum(m))
        f_k.append(np.mean(p[m]))
        o_k.append(np.mean(y[m]))
    n_k = np.asarray(n_k, float)
    f_k = np.asarray(f_k, float)
    o_k = np.asarray(o_k, float)

    o_bar = float(np.mean(y))
    REL = float((n_k * (f_k - o_k) ** 2).sum() / N)
    RES = float((n_k * (o_k - o_bar) ** 2).sum() / N)
    UNC = float(o_bar * (1.0 - o_bar))
    return REL, RES, UNC, edges


def rel_res_scatter_grid_normalized(
    test_df_temporal,
    intl_lines=None,
    nat_line_value=None,
    n_bins=10, strategy="quantile",
    out_path=None,
):
    """2x2 scatter of normalized Murphy components (REL/UNC vs RES/UNC)."""
    if intl_lines is None:
        intl_lines = [("1.90", 1.90), ("3.20", 3.20), ("5.50", 5.50)]

    log_y = test_df_temporal["lead_ltotcons_per_capita_per_day_2016cst"].to_numpy()
    panels = [(lab, val, f"_pl_{lab.replace('.', '_')}") for lab, val in intl_lines]
    if nat_line_value is not None:
        panels.append(("National", nat_line_value, "_at_nat"))

    per_panel = []
    all_r, all_s = [], []
    for lab, pl_val, suffix in panels:
        y_true = make_binary_outcome(log_y, pl_val)

        preds_all = []
        for base in COLMAP.values():
            col = f"{base}{suffix}"
            if col not in test_df_temporal.columns and suffix == "_pl_1_90":
                col = base
            preds_all.append(np.asarray(test_df_temporal[col], float))
        P_mean = np.mean(np.column_stack(preds_all), axis=1)
        _, _, UNC, edges = murphy_components(y_true, P_mean, n_bins=n_bins, strategy=strategy)

        models, Rnorm, Snorm, BSS_vals = [], [], [], []
        for model, base in COLMAP.items():
            col = f"{base}{suffix}"
            if col not in test_df_temporal.columns and suffix == "_pl_1_90":
                col = base
            p = np.asarray(test_df_temporal[col], float)
            REL, RES, UNC_chk, _ = murphy_components(
                y_true, p, n_bins=n_bins, strategy=strategy, fixed_edges=edges,
            )
            r = REL / UNC_chk if UNC_chk > 0 else np.nan
            s = RES / UNC_chk if UNC_chk > 0 else np.nan
            models.append(model)
            Rnorm.append(r)
            Snorm.append(s)
            BSS_vals.append(s - r)

        Rnorm = np.array(Rnorm)
        Snorm = np.array(Snorm)
        BSS_vals = np.array(BSS_vals)
        per_panel.append((lab, UNC, models, Rnorm, Snorm, BSS_vals))
        all_r.append(Rnorm)
        all_s.append(Snorm)

    all_r = np.concatenate(all_r)
    all_s = np.concatenate(all_s)
    pad = 0.15
    xmin, xmax = np.nanmin(all_r), np.nanmax(all_r)
    xr = xmax - xmin or 1e-6
    ymin, ymax = np.nanmin(all_s), np.nanmax(all_s)
    yr = ymax - ymin or 1e-6
    xmin, xmax = max(0, xmin - pad * xr), xmax + pad * xr
    ymin, ymax = max(0, ymin - pad * yr), ymax + pad * yr

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for ax, (lab, UNC, models, Rnorm, Snorm, BSS_vals) in zip(axes, per_panel):
        ax.plot([xmin, xmax], [xmin, xmax], color="black", lw=1.3, alpha=0.8)
        ax.scatter(Rnorm, Snorm, c="black", s=80, zorder=3)
        for m, x, y in zip(models, Rnorm, Snorm):
            ax.text(x, y, "  " + m, va="center", ha="left", fontsize=9)

        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_xlabel("Normalized Reliability (lower is better)")
        ax.set_ylabel("Normalized Resolution (higher is better)")
        ax.set_title(f"Reliability vs Resolution at {lab} Poverty Line")
        ax.grid(True, ls="--", alpha=0.35)

    for k in range(len(per_panel), 4):
        axes[k].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if out_path:
        save_and_show(fig, out_path)
    else:
        plt.show()

    fig.exhibit_data = pd.DataFrame(
        [{"poverty_line": lab, "model": m, "reliability_norm": r, "resolution_norm": sv, "bss": b}
         for lab, _, models, Rn, Sn, Bv in per_panel
         for m, r, sv, b in zip(models, Rn, Sn, Bv)])
    return fig


# =====================================================================
# Paired-bootstrap BSS ratios with fixed reference
# =====================================================================

def bss_fixed_ref(y_true, p_pred, p_ref):
    """BSS vs fixed reference probabilities *p_ref* (scalar or vector)."""
    y = np.asarray(y_true, int)
    p = np.asarray(p_pred, float)
    pr = _expand_pref(p_ref, len(y))
    bs_model = brier_score(y, p)
    bs_ref = brier_score(y, pr)
    if bs_ref <= 1e-15:
        return np.nan
    return 1.0 - bs_model / bs_ref


def paired_bootstrap_bss_ratio_fixedref(y_true, preds_by_model, p_ref, ref="MDN", B=400, seed=42):
    """Returns ratios ``BSS_fixed(model) / BSS_fixed(ref)`` with 95 % CI."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true, int)
    N = len(y)
    pref_vec = _expand_pref(p_ref, N)
    ref_pred = np.asarray(preds_by_model[ref], float)
    ratios = {m: [] for m in preds_by_model if m != ref}

    for _ in range(B):
        idx = rng.integers(0, N, size=N)
        yb = y[idx]
        pr_b = pref_vec[idx]
        bss_r = bss_fixed_ref(yb, ref_pred[idx], pr_b)
        if not np.isfinite(bss_r) or bss_r == 0.0:
            continue
        for m, p in preds_by_model.items():
            if m == ref:
                continue
            bss_m = bss_fixed_ref(yb, np.asarray(p, float)[idx], pr_b)
            if np.isfinite(bss_m):
                ratios[m].append(bss_m / bss_r)

    out = {}
    for m, arr in ratios.items():
        arr = np.asarray(arr, float)
        out[m] = {
            "mean": float(np.nanmean(arr)) if arr.size else np.nan,
            "ci_low": float(np.nanquantile(arr, 0.025)) if arr.size else np.nan,
            "ci_high": float(np.nanquantile(arr, 0.975)) if arr.size else np.nan,
            "boot": arr,
        }
    return out


def _preds_for_suffix(df, suffix, strict=True):
    preds = {}
    for model, base in COLMAP.items():
        candidates = [f"{base}{suffix}"]
        if suffix == "_pl_1_90":
            candidates.append(base)
        found = next((c for c in candidates if c in df.columns), None)
        if found is None:
            msg = f"Missing column for {model}: tried {candidates}"
            if strict:
                raise KeyError(msg)
            preds[model] = np.full(len(df), np.nan)
            continue
        preds[model] = np.asarray(df[found], float)
    return preds


def bss_ratio_by_line_fixedref(
    test_df_temporal, log_y_baseline,
    intl_lines=None, nat_line_value=None,
    B=400, seed=42, ref="MDN",
):
    """BSS(model)/BSS(ref) per poverty line using a fixed baseline."""
    if intl_lines is None:
        intl_lines = [("1.90", 1.90), ("3.20", 3.20), ("5.50", 5.50)]

    log_y_t1 = test_df_temporal["lead_ltotcons_per_capita_per_day_2016cst"].to_numpy()

    panels = [(lab, val, f"_pl_{lab.replace('.', '_')}") for lab, val in intl_lines]
    if nat_line_value is not None:
        panels.append(("National", nat_line_value, "_at_nat"))

    results = {}
    for lab, pl_val, suffix in panels:
        y_true = make_binary_outcome(log_y_t1, pl_val)
        y_base = make_binary_outcome(log_y_baseline, pl_val)
        p_ref = float(np.mean(y_base))
        preds = _preds_for_suffix(test_df_temporal, suffix)
        results[lab] = paired_bootstrap_bss_ratio_fixedref(
            y_true, preds, p_ref=p_ref, ref=ref, B=B, seed=seed,
        )
    return results


def plot_bss_ratio_grid(results_by_line, ref="MDN", model_order=None, out_path=None):
    """2x2 grid: BSS ratio = BSS_fixed(model) / BSS_fixed(ref) with 95 % CI."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    if model_order is None:
        model_order = [m for m in COLMAP if m != ref]

    for ax, (lab, ratios) in zip(axes, results_by_line.items()):
        models = [m for m in model_order if m in ratios]
        means = np.array([ratios[m]["mean"] for m in models])
        lows = np.array([ratios[m]["ci_low"] for m in models])
        highs = np.array([ratios[m]["ci_high"] for m in models])
        y = np.arange(len(models))

        ax.axvline(1.0, color="black", lw=1.8, ls="--", alpha=0.9)
        ax.hlines(y, lows, highs, color="tab:blue", lw=6, alpha=0.7)
        ax.errorbar(
            means, y, xerr=[means - lows, highs - means],
            fmt="o", capsize=5, color="black", linewidth=1.2, zorder=3,
        )
        for yi, m in enumerate(means):
            if np.isfinite(m):
                ax.text(m, yi + 0.08, f"{m:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_yticks(y, models)
        ax.set_xlabel(f"BSS ratio (model / {ref}) [>1 better]")
        ax.set_title(f"BSS: Relative Performance at {lab} Poverty Line")
        ax.grid(True, axis="x", linestyle="--", linewidth=0.7)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.margins(y=0.1)

        finite = np.isfinite(np.concatenate([lows, highs]))
        if finite.any():
            xmin = np.nanmin(lows)
            xmax = np.nanmax(highs)
            span = xmax - xmin
            if span == 0 or not np.isfinite(span):
                xmin, xmax = 0.8, 1.2
            margin = 0.1
            ax.set_xlim(min(xmin, 1.0) * (1 - margin), max(xmax, 1.0) * (1 + margin))
        else:
            ax.set_xlim(0.8, 1.2)

    for k in range(len(results_by_line), 4):
        axes[k].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if out_path:
        save_and_show(fig, out_path)
    else:
        plt.show()

    return fig
