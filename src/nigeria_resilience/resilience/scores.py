"""Resilience score calculation: P(consumption > poverty line)."""

import numpy as np
from scipy import stats

# =====================================================================
# Poverty line dictionary (2011 PPP)
# =====================================================================

POVERTY_LINES_USD = {
    "1_90": 1.90,   # extreme poverty
    "3_20": 3.20,   # lower-middle-income
    "5_50": 5.50,   # upper-middle-income
}


# =====================================================================
# Core CDF helpers
# =====================================================================

def cdf_above_threshold_normal(thresh_log, mu, sigma, eps=1e-12):
    """P(Y > thresh) under a Normal(mu, sigma) for each observation."""
    sd = np.clip(np.asarray(sigma, float), eps, None)
    return 1.0 - stats.norm.cdf(thresh_log, loc=mu, scale=sd)


def cdf_above_threshold_mixture(dist_obj, thresh_log):
    """P(Y > thresh) under a TFP MixtureSameFamily distribution object."""
    p = 1.0 - dist_obj.cdf(thresh_log).numpy()
    return np.clip(p, 0.0, 1.0)


# =====================================================================
# Add resilience scores for a single poverty line
# =====================================================================

def add_scores_for_line(
    df, label, pl_value_usd, *,
    gm_mdn=None, dist_xgb_md=None, gm_nn=None,
    ppp_mode="logged-usd",
):
    """Compute resilience scores for all models and store them in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``mean_predicted_*`` and ``std_predicted_*`` columns.
    label : str
        Poverty-line label, e.g. ``"1_90"`` or ``"3_20"``.
    pl_value_usd : float
        Poverty line in USD/day (PPP-consistent with model).
    gm_mdn, dist_xgb_md, gm_nn : TFP distribution objects, optional
        Mixture distributions for MDN, XGB-MD, NN.
    ppp_mode : str
        Currently only ``"logged-usd"`` is supported.
    """
    if ppp_mode == "logged-usd":
        thr = np.log(pl_value_usd)
    else:
        raise ValueError("Only logged-usd mode implemented here")

    # Mixtures
    if gm_mdn is not None:
        df[f"resilience_score_mdn_pl_{label}"] = cdf_above_threshold_mixture(gm_mdn, thr)
    if dist_xgb_md is not None:
        df[f"resilience_score_xgb_md_pl_{label}"] = cdf_above_threshold_mixture(dist_xgb_md, thr)
    if gm_nn is not None:
        df[f"resilience_score_nn_pl_{label}"] = cdf_above_threshold_mixture(gm_nn, thr)

    # Single-Gaussians
    df[f"resilience_score_ols_glm_pl_{label}"] = cdf_above_threshold_normal(
        thr, df["mean_predicted_ols_glm"], df["std_predicted_ols_glm"],
    )
    df[f"resilience_score_elastic_net_pl_{label}"] = cdf_above_threshold_normal(
        thr, df["mean_predicted_elastic_net"], df["std_predicted_elastic_net"],
    )
    df[f"resilience_score_xgb_1d_pl_{label}"] = cdf_above_threshold_normal(
        thr, df["mean_predicted_xgb_1D"], df["std_predicted_xgb_1D"],
    )


def add_scores_for_national_line(
    df, nat_pl, *,
    gm_mdn=None, dist_xgb_md=None, gm_nn=None,
):
    """Add resilience scores at the national poverty line."""
    thr_nat = np.log(nat_pl)

    if gm_mdn is not None:
        df["resilience_score_mdn_at_nat"] = cdf_above_threshold_mixture(gm_mdn, thr_nat)
    if dist_xgb_md is not None:
        df["resilience_score_xgb_md_at_nat"] = cdf_above_threshold_mixture(dist_xgb_md, thr_nat)
    if gm_nn is not None:
        df["resilience_score_nn_at_nat"] = cdf_above_threshold_mixture(gm_nn, thr_nat)

    df["resilience_score_ols_glm_at_nat"] = cdf_above_threshold_normal(
        thr_nat, df["mean_predicted_ols_glm"], df["std_predicted_ols_glm"],
    )
    df["resilience_score_elastic_net_at_nat"] = cdf_above_threshold_normal(
        thr_nat, df["mean_predicted_elastic_net"], df["std_predicted_elastic_net"],
    )
    df["resilience_score_xgb_1d_at_nat"] = cdf_above_threshold_normal(
        thr_nat, df["mean_predicted_xgb_1D"], df["std_predicted_xgb_1D"],
    )
