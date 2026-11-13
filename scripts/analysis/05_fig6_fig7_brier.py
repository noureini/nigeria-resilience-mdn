"""Fig. 6 (Brier skill score ratios to the MDN) and Fig. 7 (Murphy decomposition).

BSS reference probability: share of test households above each line at t (2015).
Paired bootstrap, 400 replications (seed 1).

Reads   outputs/predictions/nigeria_temporal/test_scored.parquet
Writes  outputs/figures/Fig6_brier_skill_ratio_vs_mdn.{png,pdf}
        outputs/figures/Fig7_murphy_decomposition.{png,pdf}
        outputs/tables/Fig6_brier_skill_ratio.csv, Fig7_murphy_decomposition.csv
"""
from nigeria_resilience import exhibits as ex
from nigeria_resilience.resilience.brier import (
    bss_ratio_by_line_fixedref,
    plot_bss_ratio_grid,
    rel_res_scatter_grid_normalized,
)

ex.setup()
df = ex.load_scored("nigeria_temporal")
nat = ex.national_line()
log_y_baseline = df["ltotcons_per_capita_per_day_2016cst"].to_numpy()

ratios = bss_ratio_by_line_fixedref(test_df_temporal=df, log_y_baseline=log_y_baseline,
                                    nat_line_value=nat, B=ex.BOOT_B, seed=1, ref="MDN")
plot_bss_ratio_grid(ratios, ref="MDN",
                    model_order=["XGB-Mix", "NN-1D", "XGB-1D", "Elastic Net", "OLS+GLM"],
                    out_path=str(ex.FIG_DIR / "Fig6_brier_skill_ratio_vs_mdn.png"))
ex.save_table(ex.flatten(ratios), "Fig6_brier_skill_ratio")

fig = rel_res_scatter_grid_normalized(df, nat_line_value=nat, n_bins=10)
ex.save_table(fig.exhibit_data, "Fig7_murphy_decomposition")
ex.save_figure(fig, "Fig7_murphy_decomposition")
