"""Figs. 10 and A.4-A.6 -- GLM (logit link) regressions of each model's resilience scores on covariates.

Fig. 10: $1.90/day line; Figs. A.4, A.5, A.6: $3.20/day, $5.50/day and national lines.
Features selected by significance (p < 0.05); bootstrap SE (400 reps, seed 123).
The four figures share rows and colour scale.

Reads   outputs/predictions/nigeria_temporal/test_scored.parquet
Writes  outputs/figures/Fig10_resilience_correlates.{png,pdf}            ($1.90/day)
        outputs/figures/FigA4_resilience_correlates_320.{png,pdf}         ($3.20/day)
        outputs/figures/FigA5_resilience_correlates_550.{png,pdf}         ($5.50/day)
        outputs/figures/FigA6_resilience_correlates_national.{png,pdf}    (national line)
        outputs/figures/Fig10_FigA4_A6_resilience_correlates_grid.{png,pdf} (all four, one page)
        outputs/tables/Fig10_resilience_correlates.csv
"""
from nigeria_resilience import exhibits as ex
from nigeria_resilience.analysis.meta_regression import (
    plot_coefficient_heatmap_grid,
    plot_coefficient_heatmap_single,
    run_all_poverty_lines,
)

ex.setup()
df = ex.load_scored("nigeria_temporal")
meta = run_all_poverty_lines(
    df=df, X_cols=ex.COVARS,
    models=["MDN", "NN-1D", "XGB-1D", "XGB-Mix", "Elastic Net", "OLS+GLM"],
    method="glm", feature_selection="significant", significance_level=0.05, top_k=30,
    n_boot=ex.BOOT_B, seed=123, verbose=False)
for line, name in (("$1.90/day Poverty Line", "Fig10_resilience_correlates"),
                   ("$3.20/day Poverty Line", "FigA4_resilience_correlates_320"),
                   ("$5.50/day Poverty Line", "FigA5_resilience_correlates_550"),
                   ("National Poverty Line", "FigA6_resilience_correlates_national")):
    plot_coefficient_heatmap_single(meta, line, outfile=str(ex.FIG_DIR / f"{name}.png"))
plot_coefficient_heatmap_grid(results_dict=meta, show_se=False,
                              outfile=str(ex.FIG_DIR / "Fig10_FigA4_A6_resilience_correlates_grid.png"))
ex.save_table(ex.flatten(meta), "Fig10_resilience_correlates")
