"""Fig. 5 (ratios to the MDN) and Fig. A.1 (levels) -- MSE, NLL, CRPS, EMD accuracy.

Paired bootstrap, 400 replications.  The same functions applied to the LGA
scenario give Fig. 13 (script 10).

Reads   outputs/predictions/nigeria_temporal/{test_scored.parquet,mixture_params.npz}
Writes  outputs/figures/Fig5_relative_performance_vs_mdn.{png,pdf}
        outputs/figures/FigA1_forecast_metrics.{png,pdf}
        outputs/tables/Fig5_FigA1_forecast_metrics.csv
"""
import numpy as np
import pandas as pd

from nigeria_resilience import exhibits as ex
from nigeria_resilience.config import TARGET_COL
from nigeria_resilience.evaluation.metrics import bootstrap_compare, plot_all_metrics_bar
from nigeria_resilience.evaluation.metrics_table import plot_vs_mdn_grid

MODEL_ORDER = ["OLS+GLM", "Elastic Net", "XGB-1D", "NN-1D", "XGB-Mix"][::-1]


def metrics_summary(scenario):
    """Bootstrap distribution of MSE, NLL, CRPS and EMD accuracy for the six models."""
    df = ex.load_scored(scenario)
    z = ex.load_mixtures(scenario)
    mdn = {"alpha": z["mdn_alpha"], "mu": z["mdn_mu"], "sigma": z["mdn_sigma"],
           "cdf_col": "cdf_mdn"}
    mix = {"alpha": z["xgbmix_alpha"], "mu": z["xgbmix_mu"], "sigma": z["xgbmix_sigma"],
           "cdf_col": "cdf_xgb_md"}
    return bootstrap_compare(df, TARGET_COL, mdn_arrays=mdn, xgbmix_arrays=mix,
                             B=ex.BOOT_B, rng_seed=ex.SEED)


def summary_table(summ):
    """Levels and paired ratios to the MDN, with 95% bootstrap intervals."""
    rows = []
    for metric, by_model in summ.items():
        ref = np.asarray(by_model["MDN"]["boot"])
        for model, s in by_model.items():
            ratio = np.asarray(s["boot"]) / ref
            rows.append({"metric": metric, "model": model, "mean": s["mean"],
                         "ci_low": s["ci_low"], "ci_high": s["ci_high"],
                         "ratio_to_mdn_mean": float(np.mean(ratio)),
                         "ratio_to_mdn_ci_low": float(np.quantile(ratio, 0.025)),
                         "ratio_to_mdn_ci_high": float(np.quantile(ratio, 0.975))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ex.setup()
    summ = metrics_summary("nigeria_temporal")
    plot_vs_mdn_grid(summ, ref_model="MDN", mode="ratio", model_order=MODEL_ORDER,
                     out_path=str(ex.FIG_DIR / "Fig5_relative_performance_vs_mdn.png"))
    plot_all_metrics_bar(summ, out_path=str(ex.FIG_DIR / "FigA1_forecast_metrics.png"))
    ex.save_table(summary_table(summ), "Fig5_FigA1_forecast_metrics")
