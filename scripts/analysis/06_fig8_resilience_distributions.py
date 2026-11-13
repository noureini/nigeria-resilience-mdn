"""Fig. 8 -- Survival functions (1 - ECDF) of predicted resilience scores, four poverty lines.

Reads   outputs/predictions/nigeria_temporal/test_scored.parquet
Writes  outputs/figures/Fig8_resilience_score_distributions.{png,pdf}
        outputs/tables/Fig8_resilience_scores_summary.csv  (mean and quantiles by model x line)
"""
from nigeria_resilience import exhibits as ex
from nigeria_resilience.visualization.distributions import plot_resilience_ecdf_grid_pretty

ex.setup()
df = ex.load_scored("nigeria_temporal")
plot_resilience_ecdf_grid_pretty(
    df, figsize=(13, 10), dpi=160, legend_fontsize=9, annotate_quartiles=False,
    out_path=str(ex.FIG_DIR / "Fig8_resilience_score_distributions.png"))
cols = [c for c in df.columns if c.startswith("resilience_score_")]
summ = df[cols].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T
summ.index.name = "score"
ex.save_table(summ.reset_index(), "Fig8_resilience_scores_summary")
