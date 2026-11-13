"""Publication-ready SMD (Standardized Mean Difference) disagreement analysis.

Analyses household characteristics associated with model disagreement in
poverty resilience predictions.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from nigeria_resilience.visualization.save import save_and_show

# =====================================================================
# Configuration
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

MAIN_PANELS = [
    ("$1.90/day Poverty Line", "_pl_1_90"),
    ("National Poverty Line", "_at_nat"),
]

MODEL_ORDER = ("XGB-1D", "XGB-Mix", "Elastic Net", "OLS+GLM")

MODEL_GROUPS = {
    "Distributional": ["XGB-1D", "XGB-Mix"],
    "Two-Stage": ["Elastic Net", "OLS+GLM"],
}

# =====================================================================
# Publication-quality variable labels
# =====================================================================

PUBLICATION_LABELS = {
    "fhh": "Female Household Head",
    "rural": "Rural Location",
    "marital_status_head_hh_married_monogamous": "Married (Monogamous)",
    "marital_status_head_hh_married_polygamous": "Married (Polygamous)",
    "marital_status_head_hh_widowed": "Widowed",
    "age_head_hh": "Age of Household Head",
    "agactivities_hh": "Agricultural Household",
    "prop_female_age_sup_66": "Share Elderly Females (>66)",
    "prop_female_age_16_65": "Share Adult Females (16-65)",
    "prop_male_age_16_65": "Share Adult Males (16-65)",
    "prop_female_age_5_15": "Share Girls (5-15)",
    "prop_male_age_5_15": "Share Boys (5-15)",
    "prop_all_age_0_4": "Share Young Children (0-4)",
    "education_level_head_hh_higher": "Higher Education",
    "education_level_head_hh_others": "Other Education",
    "education_level_head_hh_primary": "Primary Education",
    "education_level_head_hh_secondary": "Secondary Education",
    "own_car": "Car Ownership",
    "own_computer": "Computer Ownership",
    "own_motorcycle": "Motorcycle Ownership",
    "own_tv": "Television Ownership",
    "own_house": "House Ownership",
    "number_rooms": "Number of Rooms",
    "connection_electricity": "Electricity Connection",
    "saving": "Has Savings",
    "loan": "Has Outstanding Loan",
    "main_cook_fuel_firewood": "Cooking Fuel: Firewood",
    "main_cook_fuel_kerosene": "Cooking Fuel: Kerosene",
    "main_source_water_borehole": "Water Source: Borehole",
    "main_source_water_pipe borne": "Water Source: Piped",
    "main_source_water_river": "Water Source: River",
    "main_source_water_well": "Water Source: Well",
    "main_toilet_facility_flush to septic tank": "Toilet: Flush Septic",
    "main_toilet_facility_none": "Toilet: None",
    "main_toilet_facility_pit latrine": "Toilet: Pit Latrine",
    "main_toilet_facility_v.i.p latrine": "Toilet: VIP Latrine",
    "farm_area": "Farm Area (hectares)",
    "death_adultworking": "Death of Working Adult",
    "illness_incomearningsmbers": "Illness of Income Earner",
    "event_drought": "Drought (Community)",
    "event_flood": "Flood (Community)",
    "event_crop_disease": "Crop Disease (Community)",
    "event_livestock_disease": "Livestock Disease (Community)",
    "SPI_3_end": "Precipitation Index (3-month)",
    "NDVI_z_mean": "Vegetation Index (NDVI)",
    "battle_dist": "Distance to Battle",
    "protest_dist": "Distance to Protest",
    "riot_dist": "Distance to Riot",
    "vac_dist": "Distance to Violence vs Civilians",
    "battle_time_pre_1_3": "Battle Events (1-3 months prior)",
    "battle_time_pre_4_6": "Battle Events (4-6 months prior)",
    "protest_time_pre_1_3": "Protest Events (1-3 months prior)",
    "protest_time_pre_4_6": "Protest Events (4-6 months prior)",
    "riot_time_pre_1_3": "Riot Events (1-3 months prior)",
    "riot_time_pre_4_6": "Riot Events (4-6 months prior)",
    "vac_time_pre_1_3": "Violence vs Civilians (1-3 months prior)",
    "vac_time_pre_4_6": "Violence vs Civilians (4-6 months prior)",
}

VARIABLE_CATEGORIES = {
    "Demographics": [
        "fhh", "rural", "age_head_hh", "agactivities_hh",
        "marital_status_head_hh_married_monogamous",
        "marital_status_head_hh_married_polygamous",
        "marital_status_head_hh_widowed",
    ],
    "Household Composition": [
        "prop_female_age_sup_66", "prop_female_age_16_65",
        "prop_male_age_16_65", "prop_female_age_5_15",
        "prop_male_age_5_15", "prop_all_age_0_4",
    ],
    "Education": [
        "education_level_head_hh_higher", "education_level_head_hh_others",
        "education_level_head_hh_primary", "education_level_head_hh_secondary",
    ],
    "Assets & Amenities": [
        "own_car", "own_computer", "own_motorcycle", "own_tv",
        "own_house", "number_rooms", "connection_electricity",
        "saving", "loan",
    ],
    "Housing Conditions": [
        "main_cook_fuel_firewood", "main_cook_fuel_kerosene",
        "main_source_water_borehole", "main_source_water_pipe borne",
        "main_source_water_river", "main_source_water_well",
        "main_toilet_facility_flush to septic tank",
        "main_toilet_facility_none", "main_toilet_facility_pit latrine",
        "main_toilet_facility_v.i.p latrine", "farm_area",
    ],
    "Shocks": [
        "death_adultworking", "illness_incomearningsmbers",
        "event_drought", "event_flood", "event_crop_disease",
        "event_livestock_disease",
    ],
    "Environment": ["SPI_3_end", "NDVI_z_mean"],
    "Conflict & Violence": [
        "battle_dist", "protest_dist", "riot_dist", "vac_dist",
        "battle_time_pre_1_3", "battle_time_pre_4_6",
        "protest_time_pre_1_3", "protest_time_pre_4_6",
        "riot_time_pre_1_3", "riot_time_pre_4_6",
        "vac_time_pre_1_3", "vac_time_pre_4_6",
    ],
}


# =====================================================================
# Helpers
# =====================================================================

def _resolve_colname(df: pd.DataFrame, base: str, suffix: str) -> Optional[str]:
    cand = f"{base}{suffix}"
    if cand in df.columns:
        return cand
    if suffix == "_pl_1_90" and base in df.columns:
        return base
    return None


def _type1_mask(mdn: np.ndarray) -> np.ndarray:
    """MDN says non-resilient (< 0.5)."""
    return np.isfinite(mdn) & (mdn < 0.5)


def _type2_mask(mdn: np.ndarray) -> np.ndarray:
    """MDN says resilient (0.5-0.8)."""
    return np.isfinite(mdn) & (mdn >= 0.5) & (mdn < 0.8)


def _D_type1(mod: np.ndarray) -> np.ndarray:
    """Disagreement Type 1: model upgrades to resilient/most-resilient."""
    return (np.asarray(mod, float) >= 0.5).astype(int)


def _D_type2(mod: np.ndarray) -> np.ndarray:
    """Disagreement Type 2: model upgrades to most-resilient."""
    return (np.asarray(mod, float) >= 0.8).astype(int)


# =====================================================================
# SMD calculation
# =====================================================================

def smd_weighted(x: np.ndarray, d: np.ndarray) -> float:
    """SMD with weighted pooled standard deviation."""
    x = np.asarray(x, float)
    d = np.asarray(d, int)
    m1 = x[d == 1]
    m0 = x[d == 0]
    if len(m1) < 2 or len(m0) < 2:
        return np.nan
    n1, n0 = len(m1), len(m0)
    mu1, mu0 = np.nanmean(m1), np.nanmean(m0)
    s1, s0 = np.nanstd(m1, ddof=1), np.nanstd(m0, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n0 - 1) * s0 ** 2) / (n1 + n0 - 2))
    if not np.isfinite(sp) or sp == 0:
        return np.nan
    return (mu1 - mu0) / sp


def smd_simple(x: np.ndarray, d: np.ndarray) -> float:
    """Simple pooled SD."""
    x = np.asarray(x, float)
    d = np.asarray(d, int)
    m1 = x[d == 1]
    m0 = x[d == 0]
    if len(m1) < 2 or len(m0) < 2:
        return np.nan
    mu1, mu0 = np.nanmean(m1), np.nanmean(m0)
    s1, s0 = np.nanstd(m1, ddof=1), np.nanstd(m0, ddof=1)
    sp = np.sqrt((s1 ** 2 + s0 ** 2) / 2.0)
    if not np.isfinite(sp) or sp == 0:
        return np.nan
    return (mu1 - mu0) / sp


# =====================================================================
# Bootstrap SMD analysis
# =====================================================================

def compute_smd_disagreement(
    df: pd.DataFrame,
    id_col: str,
    X_cols: List[str],
    n_boot: int = 400,
    seed: int = 123,
    min_d0: int = 30,
    min_d1: int = 30,
    topk_sel: int = 15,
    asmd_thr: float = 0.10,
    use_weighted_smd: bool = True,
) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    """Bootstrap SMD analysis for disagreement patterns.

    Returns nested dict: ``out[poverty_line][model]["type1"/"type2"]`` -> DataFrame
    """
    rng = np.random.default_rng(seed)
    smd_func = smd_weighted if use_weighted_smd else smd_simple

    out: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}

    total_draws = len(MAIN_PANELS) * len(MODEL_ORDER) * 2 * n_boot
    pbar = tqdm(total=total_draws, desc="SMD bootstrap analysis", leave=True)

    for title, suffix in MAIN_PANELS:
        out[title] = {}
        mdn_col = _resolve_colname(df, COLMAP["MDN"], suffix)

        if mdn_col is None:
            for m in MODEL_ORDER:
                out[title][m] = {"type1": pd.DataFrame(), "type2": pd.DataFrame()}
            pbar.update(len(MODEL_ORDER) * 2 * n_boot)
            continue

        for model in MODEL_ORDER:
            mod_col = _resolve_colname(df, COLMAP[model], suffix)

            if mod_col is None:
                out[title][model] = {"type1": pd.DataFrame(), "type2": pd.DataFrame()}
                pbar.update(2 * n_boot)
                continue

            base = df[[id_col, mdn_col, mod_col] + X_cols].copy()
            base = base.replace([np.inf, -np.inf], np.nan)
            base = base.dropna(subset=[id_col, mdn_col, mod_col])

            if base.empty:
                out[title][model] = {"type1": pd.DataFrame(), "type2": pd.DataFrame()}
                pbar.update(2 * n_boot)
                continue

            mdn = base[mdn_col].astype(float).to_numpy()
            mod = base[mod_col].astype(float).to_numpy()

            def _analyze_one_type(which: str) -> pd.DataFrame:
                if which == "type1":
                    msk = _type1_mask(mdn)
                    D = _D_type1(mod)[msk]
                else:
                    msk = _type2_mask(mdn)
                    D = _D_type2(mod)[msk]

                sub = base.loc[msk, [id_col] + X_cols].copy()

                if sub.empty:
                    pbar.update(n_boot)
                    return pd.DataFrame()

                n1, n0 = int((D == 1).sum()), int((D == 0).sum())

                if n1 < min_d1 or n0 < min_d0:
                    pbar.update(n_boot)
                    return pd.DataFrame()

                rows = []
                for xcol in X_cols:
                    s_obs = smd_func(sub[xcol].to_numpy(), D)
                    rows.append((xcol, s_obs))

                tab = pd.DataFrame(rows, columns=["var", "smd"]).set_index("var")
                tab["abs_smd"] = np.abs(tab["smd"])

                sub2 = sub.copy()
                sub2["_D"] = D.astype(int)

                boot_vals = {v: [] for v in X_cols}
                count_topk = {v: 0 for v in X_cols}
                count_thr = {v: 0 for v in X_cols}
                ok_draws = 0

                for _ in range(n_boot):
                    b = sub2.sample(n=len(sub2), replace=True, random_state=rng)
                    Db = b["_D"].to_numpy().astype(int)
                    n1b, n0b = int((Db == 1).sum()), int((Db == 0).sum())

                    if n1b < min_d1 or n0b < min_d0:
                        pbar.update(1)
                        continue

                    ok_draws += 1
                    abs_smd_draw = {}

                    for xcol in X_cols:
                        sb = smd_func(b[xcol].to_numpy(), Db)
                        if np.isfinite(sb):
                            sb = float(sb)
                            boot_vals[xcol].append(sb)
                            abs_smd_draw[xcol] = abs(sb)

                    for xcol, av in abs_smd_draw.items():
                        if av >= asmd_thr:
                            count_thr[xcol] += 1

                    if abs_smd_draw:
                        k = min(topk_sel, len(abs_smd_draw))
                        top_vars = sorted(abs_smd_draw.items(), key=lambda kv: kv[1], reverse=True)[:k]
                        for xcol, _ in top_vars:
                            count_topk[xcol] += 1

                    pbar.update(1)

                denom = max(1, ok_draws)
                min_kept = max(30, int(0.2 * n_boot))

                ci_low, ci_high, boot_mean, boot_se, boot_ok_rate = [], [], [], [], []
                freq_topk, freq_thr = [], []

                for xcol in X_cols:
                    arr = np.asarray(boot_vals[xcol], float)
                    boot_ok_rate.append(len(arr) / max(1, n_boot))
                    freq_topk.append(count_topk[xcol] / denom)
                    freq_thr.append(count_thr[xcol] / denom)

                    if len(arr) < min_kept:
                        ci_low.append(np.nan)
                        ci_high.append(np.nan)
                        boot_mean.append(np.nan)
                        boot_se.append(np.nan)
                    else:
                        boot_mean.append(float(np.nanmean(arr)))
                        boot_se.append(float(np.nanstd(arr, ddof=1)))
                        ci_low.append(float(np.nanpercentile(arr, 2.5)))
                        ci_high.append(float(np.nanpercentile(arr, 97.5)))

                tab["boot_mean"] = boot_mean
                tab["boot_se"] = boot_se
                tab["boot_bias"] = tab["boot_mean"] - tab["smd"]
                tab["ci_low"] = ci_low
                tab["ci_high"] = ci_high
                tab["n0"] = n0
                tab["n1"] = n1
                tab["boot_ok_rate"] = boot_ok_rate
                tab["freq_topk"] = freq_topk
                tab["freq_thr"] = freq_thr
                tab.attrs["boot_ok_draws"] = ok_draws

                return tab

            out[title][model] = {
                "type1": _analyze_one_type("type1"),
                "type2": _analyze_one_type("type2"),
            }

    pbar.close()
    return out


# =====================================================================
# Publication-quality visualization
# =====================================================================

def plot_smd_publication(
    results: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    labels_map: Dict[str, str] = None,
    which: str = "type1",
    poverty_lines: List[str] = None,
    models: List[str] = None,
    smd_thresh: float = 0.15,
    top_k: int = 10,
    figsize: Tuple[float, float] = (14, 8),
    dpi: int = 300,
    err: str = "se",
    se_mult: float = 1.96,
    annotate_values: bool = True,
    annotate_freq: bool = False,
    show_interpretation: bool = False,
    out_path: str = None,
) -> plt.Figure:
    """Publication-quality SMD plot with professional styling."""
    if labels_map is None:
        labels_map = PUBLICATION_LABELS
    if poverty_lines is None:
        poverty_lines = list(results.keys())
    if models is None:
        models = MODEL_ORDER

    n_lines = len(poverty_lines)
    n_models = len(models)

    fig, axes = plt.subplots(n_lines, n_models, figsize=figsize, dpi=dpi, squeeze=False)

    # Collect top-K variables
    var_scores = {}
    for pov_line in poverty_lines:
        for model in models:
            dfm = results.get(pov_line, {}).get(model, {}).get(which, pd.DataFrame())
            if dfm is None or dfm.empty or "smd" not in dfm.columns:
                continue
            dfx = dfm.copy()
            dfx["abs_smd"] = np.abs(dfx["smd"])
            dfx = dfx[dfx["abs_smd"] >= smd_thresh]
            if dfx.empty:
                continue
            score = dfx["freq_topk"].astype(float) if "freq_topk" in dfx.columns else dfx["abs_smd"].astype(float)
            for v, sc in score.items():
                if np.isfinite(sc):
                    var_scores[v] = max(var_scores.get(v, 0.0), float(sc))

    if not var_scores:
        print("No variables meet SMD threshold")
        return fig

    top_vars = [v for v, _ in sorted(var_scores.items(), key=lambda kv: kv[1], reverse=True)][:top_k]
    pretty_labels = [labels_map.get(v, v) for v in top_vars]

    # Compute symmetric x-limits
    max_abs = 0.0
    for pov_line in poverty_lines:
        for model in models:
            dfm = results.get(pov_line, {}).get(model, {}).get(which, pd.DataFrame())
            if dfm is None or dfm.empty:
                continue
            dfa = dfm.reindex(top_vars)
            for vname in top_vars:
                if vname not in dfa.index:
                    continue
                s = dfa.loc[vname, "smd"]
                if not np.isfinite(s):
                    continue
                s = float(s)
                lo, hi = s, s
                if err == "se" and "boot_se" in dfa.columns:
                    se_val = dfa.loc[vname, "boot_se"]
                    if np.isfinite(se_val):
                        d = abs(float(se_val)) * se_mult
                        lo, hi = s - d, s + d
                elif err == "ci" and "ci_low" in dfa.columns:
                    cil, cih = dfa.loc[vname, "ci_low"], dfa.loc[vname, "ci_high"]
                    if np.isfinite(cil) and np.isfinite(cih):
                        lo, hi = float(cil), float(cih)
                max_abs = max(max_abs, abs(lo), abs(hi))

    if max_abs <= 0:
        max_abs = 0.5
    xpad = 0.15 * max_abs
    xlim = (-max_abs - xpad, max_abs + xpad)

    # Plot each panel
    for r, pov_line in enumerate(poverty_lines):
        for c, model in enumerate(models):
            ax = axes[r, c]
            ax.axvline(0, color="black", lw=1.2, ls="--", alpha=0.5, zorder=1)
            ax.grid(True, axis="x", linestyle=":", linewidth=0.5, alpha=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.8)
            ax.spines["bottom"].set_linewidth(0.8)

            ax.set_yticks(np.arange(len(top_vars)))
            if c == 0:
                ax.set_yticklabels(pretty_labels, fontsize=9)
            else:
                ax.set_yticklabels([""] * len(top_vars))

            ax.invert_yaxis()

            if r == 0:
                ax.set_title(model, fontsize=11, fontweight="bold", pad=8)
            if r == n_lines - 1:
                ax.set_xlabel("SMD (Standard Deviation Units)", fontsize=10)

            ax.set_xlim(*xlim)

            dfm = results.get(pov_line, {}).get(model, {}).get(which, pd.DataFrame())
            if dfm is None or dfm.empty or "smd" not in dfm.columns:
                continue

            dfa = dfm.reindex(top_vars).copy()
            dfa["abs_smd"] = np.abs(dfa["smd"])
            keep = np.isfinite(dfa["smd"]) & (dfa["abs_smd"] >= smd_thresh)

            if keep.sum() == 0:
                continue

            y = np.where(keep.to_numpy())[0].astype(int)
            vals = dfa.loc[keep, "smd"].astype(float).to_numpy()

            lo_arr = vals.copy()
            hi_arr = vals.copy()

            if err == "se" and "boot_se" in dfa.columns:
                se = dfa.loc[keep, "boot_se"].astype(float).to_numpy()
                d = np.abs(se) * se_mult
                lo_arr = vals - d
                hi_arr = vals + d
            elif err == "ci" and "ci_low" in dfa.columns:
                lo_arr = dfa.loc[keep, "ci_low"].astype(float).to_numpy()
                hi_arr = dfa.loc[keep, "ci_high"].astype(float).to_numpy()

            colors = ["#1f77b4" if v >= 0 else "#d62728" for v in vals]

            ax.barh(y, vals, height=0.7, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5, zorder=3)

            if err in ("se", "ci"):
                cap_height = 0.15
                ax.hlines(y, lo_arr, hi_arr, colors="black", linewidth=1.2, zorder=4)
                ax.vlines(lo_arr, y - cap_height, y + cap_height, colors="black", linewidth=1.2, zorder=4)
                ax.vlines(hi_arr, y - cap_height, y + cap_height, colors="black", linewidth=1.2, zorder=4)

            if annotate_values:
                fq = None
                if annotate_freq and "freq_topk" in dfa.columns:
                    fq = dfa.loc[keep, "freq_topk"].astype(float).to_numpy()

                dx = 0.02 * (xlim[1] - xlim[0])
                for j, (yi, vv, l_j, h_j) in enumerate(zip(y, vals, lo_arr, hi_arr)):
                    if vv >= 0:
                        xt = (h_j if np.isfinite(h_j) else vv) + dx
                        ha = "left"
                    else:
                        xt = (l_j if np.isfinite(l_j) else vv) - dx
                        ha = "right"

                    if annotate_freq and fq is not None and j < len(fq):
                        txt = f"{vv:+.2f}\n({fq[j]:.0%})"
                        fontsize = 7
                    else:
                        txt = f"{vv:+.2f}"
                        fontsize = 8

                    ax.text(
                        xt, yi, txt, ha=ha, va="center", fontsize=fontsize,
                        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5),
                        clip_on=False, zorder=5,
                    )

            if show_interpretation and r == 0 and c == 0:
                interp_text = (
                    "Interpretation:\n"
                    "  Positive SMD: Higher values\n"
                    "  predict model upgrade\n"
                    "  Negative SMD: Higher values\n"
                    "  predict agreement\n"
                    f"  Whiskers: +/-{se_mult:.2f} SE"
                )
                ax.text(
                    0.02, 0.98, interp_text,
                    transform=ax.transAxes, fontsize=7, va="top", ha="left",
                    bbox=dict(
                        boxstyle="round,pad=0.5", facecolor="lightyellow",
                        alpha=0.85, edgecolor="gray", linewidth=0.8,
                    ),
                    zorder=10,
                )

            if c == 0:
                ax.text(
                    -1.1, 0.5, pov_line,
                    transform=ax.transAxes, rotation=90, va="center", ha="center",
                    fontsize=11, fontweight="bold",
                )

    plt.tight_layout(rect=[0.08, 0.04, 1.0, 0.96])

    if out_path:
        save_and_show(fig, out_path)
    else:
        plt.show()

    return fig


# =====================================================================
# Alternative: grouped by category
# =====================================================================

def plot_smd_by_category(
    results: Dict,
    categories: Dict[str, List[str]],
    labels_map: Dict[str, str] = None,
    which: str = "type1",
    poverty_line: str = "$1.90/day Poverty Line",
    models: List[str] = None,
    smd_thresh: float = 0.15,
    vars_per_cat: int = 3,
    figsize: Tuple[float, float] = (12, 10),
    dpi: int = 300,
    out_path: str = None,
) -> plt.Figure:
    """Plot SMD grouped by variable category."""
    if labels_map is None:
        labels_map = PUBLICATION_LABELS
    if models is None:
        models = MODEL_ORDER

    n_cats = len(categories)
    n_models = len(models)

    fig, axes = plt.subplots(n_cats, n_models, figsize=figsize, dpi=dpi, squeeze=False)

    for r, (cat_name, cat_vars) in enumerate(categories.items()):
        var_scores = {}
        for model in models:
            dfm = results.get(poverty_line, {}).get(model, {}).get(which, pd.DataFrame())
            if dfm is None or dfm.empty:
                continue
            dfx = dfm.reindex([v for v in cat_vars if v in dfm.index]).copy()
            dfx["abs_smd"] = np.abs(dfx["smd"])
            dfx = dfx[dfx["abs_smd"] >= smd_thresh]
            for v in dfx.index:
                var_scores[v] = max(var_scores.get(v, 0.0), float(dfx.loc[v, "abs_smd"]))

        if not var_scores:
            for c in range(n_models):
                axes[r, c].axis("off")
            continue

        top_in_cat = [v for v, _ in sorted(var_scores.items(), key=lambda kv: kv[1], reverse=True)][:vars_per_cat]
        pretty_labels_cat = [labels_map.get(v, v) for v in top_in_cat]

        for c, model in enumerate(models):
            ax = axes[r, c]
            ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.5)
            ax.grid(True, axis="x", linestyle=":", linewidth=0.5, alpha=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.set_yticks(np.arange(len(top_in_cat)))
            if c == 0:
                ax.set_yticklabels(pretty_labels_cat, fontsize=9)
            else:
                ax.set_yticklabels([""] * len(top_in_cat))

            ax.invert_yaxis()

            if r == 0:
                ax.set_title(model, fontsize=11, fontweight="bold")
            if r == n_cats - 1:
                ax.set_xlabel("SMD (SD units)", fontsize=10)

            if c == 0:
                ax.text(
                    -0.4, 0.5, cat_name,
                    transform=ax.transAxes, rotation=90, va="center", ha="right",
                    fontsize=10, fontweight="bold", color="#1f77b4",
                )

            dfm = results.get(poverty_line, {}).get(model, {}).get(which, pd.DataFrame())
            if dfm is None or dfm.empty:
                continue

            dfa = dfm.reindex(top_in_cat).copy()
            dfa["abs_smd"] = np.abs(dfa["smd"])
            keep_mask = np.isfinite(dfa["smd"]) & (dfa["abs_smd"] >= smd_thresh)

            if keep_mask.sum() == 0:
                continue

            y = np.where(keep_mask.to_numpy())[0]
            vals = dfa.loc[keep_mask, "smd"].astype(float).to_numpy()

            colors = ["#1f77b4" if v >= 0 else "#d62728" for v in vals]
            ax.barh(y, vals, height=0.6, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
            ax.set_xlim(-0.8, 0.8)

    fig.suptitle(
        f"SMD by Variable Category -- {poverty_line} -- Type 1 Disagreement",
        fontsize=13, fontweight="bold", y=0.98,
    )

    plt.tight_layout(rect=[0.10, 0.02, 1.0, 0.96])

    if out_path:
        save_and_show(fig, out_path)
    else:
        plt.show()

    return fig


# =====================================================================
# Summary table
# =====================================================================

def create_smd_summary_table(
    results: Dict,
    which: str = "type1",
    top_k: int = 15,
    smd_thresh: float = 0.15,
    outfile: str = None,
) -> pd.DataFrame:
    """Create summary table of SMD results for manuscript."""
    rows = []
    for pov_line in results:
        for model in MODEL_ORDER:
            dfm = results.get(pov_line, {}).get(model, {}).get(which, pd.DataFrame())
            if dfm is None or dfm.empty:
                continue
            dfx = dfm.copy()
            dfx["abs_smd"] = np.abs(dfx["smd"])
            dfx = dfx[dfx["abs_smd"] >= smd_thresh].copy()
            dfx = dfx.sort_values("abs_smd", ascending=False).head(top_k)
            for var in dfx.index:
                rows.append({
                    "Variable": PUBLICATION_LABELS.get(var, var),
                    "Variable_Code": var,
                    "Poverty_Line": pov_line,
                    "Model": model,
                    "SMD": dfx.loc[var, "smd"],
                    "Abs_SMD": dfx.loc[var, "abs_smd"],
                    "Boot_SE": dfx.loc[var, "boot_se"],
                    "CI_Low": dfx.loc[var, "ci_low"],
                    "CI_High": dfx.loc[var, "ci_high"],
                    "N_Disagree": dfx.loc[var, "n1"],
                    "N_Agree": dfx.loc[var, "n0"],
                    "Freq_TopK": dfx.loc[var, "freq_topk"],
                })

    summary = pd.DataFrame(rows)
    if outfile:
        summary.to_csv(outfile, index=False, float_format="%.4f")
    return summary
