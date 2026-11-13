"""Bootstrap computation of resilience category shares (non-resilient / resilient / most-resilient)."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
    ("$1.90/day Poverty Line", "_pl_1_90"),
    ("$3.20/day Poverty Line", "_pl_3_20"),
    ("$5.50/day Poverty Line", "_pl_5_50"),
    ("National Poverty Line", "_at_nat"),
]

MODEL_ORDER = ("OLS+GLM", "Elastic Net", "XGB-1D", "NN-1D", "XGB-Mix", "MDN")


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


def _stack_panel_matrix(df: pd.DataFrame, suffix: str, model_order=MODEL_ORDER) -> Tuple[np.ndarray, List[str]]:
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


def _bin_counts(p: np.ndarray) -> np.ndarray:
    """Return counts for categories [non_resil, mid, most]."""
    non_res = np.sum(p < 0.5)
    mid = np.sum((p >= 0.5) & (p < 0.8))
    most = np.sum(p >= 0.8)
    return np.array([non_res, mid, most], dtype=float)


def _paired_bootstrap_category_props(P: np.ndarray, B: int = 400, seed: int = 42) -> np.ndarray:
    """Return bootstrap array of shape (B, M, 3) with category proportions."""
    rng = np.random.default_rng(seed)
    N, M = P.shape
    boot = np.empty((B, M, 3), dtype=float)
    for b in range(B):
        idx = rng.integers(0, N, size=N)
        Pb = P[idx, :]
        for m in range(M):
            counts = _bin_counts(Pb[:, m])
            boot[b, m, :] = counts / N
    return boot


def _summ_from_boot_3cat(boot: np.ndarray) -> List[Dict[str, dict]]:
    """Return list (per model) of dicts for the 3 categories with mean & 95% CI."""
    B, M, K = boot.shape
    out = []
    for m in range(M):
        model_summ = {}
        for k, cat in enumerate([
            "Non-resilient (<0.5)",
            "Resilient (0.5-0.8)",
            "Most-resilient (>=0.8)",
        ]):
            col = boot[:, m, k]
            model_summ[cat] = {
                "mean": float(np.mean(col)),
                "ci_low": float(np.quantile(col, 0.025)),
                "ci_high": float(np.quantile(col, 0.975)),
                "boot": col,
            }
        any_res = boot[:, m, 1] + boot[:, m, 2]
        model_summ["Resilient (>=0.5)"] = {
            "mean": float(np.mean(any_res)),
            "ci_low": float(np.quantile(any_res, 0.025)),
            "ci_high": float(np.quantile(any_res, 0.975)),
            "boot": any_res,
        }
        out.append(model_summ)
    return out


# =====================================================================
# Public API
# =====================================================================

def bootstrap_resilience_category_shares(
    test_df_temporal: pd.DataFrame,
    B: int = 400, seed: int = 42,
) -> Dict[str, dict]:
    """Compute bootstrap summaries of category proportions for each panel.

    Returns
    -------
    results : dict
        ``results[panel_title][model]`` is a dict mapping category names to
        ``{mean, ci_low, ci_high, boot}`` summaries.
    """
    results = {}
    for title, suffix in PANELS:
        P, labels = _stack_panel_matrix(test_df_temporal, suffix)
        if P.size == 0:
            results[title] = {}
            continue
        boot = _paired_bootstrap_category_props(P, B=B, seed=seed)
        summ_list = _summ_from_boot_3cat(boot)
        results[title] = {lbl: summ for lbl, summ in zip(labels, summ_list)}
    return results
