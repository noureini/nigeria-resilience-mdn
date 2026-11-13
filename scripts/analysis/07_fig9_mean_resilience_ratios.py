"""Fig. 9 -- Mean predicted resilience relative to the MDN (paired bootstrap, 400 reps).

Reads   outputs/predictions/nigeria_temporal/test_scored.parquet
Writes  outputs/figures/Fig9_mean_resilience_ratio_vs_mdn.{png,pdf}
        outputs/tables/Fig9_mean_resilience_ratio.csv
"""
from nigeria_resilience import exhibits as ex
from nigeria_resilience.visualization.bar_charts import plot_ratio_vs_mdn_diff_grid

ex.setup()
df = ex.load_scored("nigeria_temporal")
fig = plot_ratio_vs_mdn_diff_grid(df, B=ex.BOOT_B, seed=ex.SEED)
ex.save_table(fig.exhibit_data, "Fig9_mean_resilience_ratio")
ex.save_figure(fig, "Fig9_mean_resilience_ratio_vs_mdn")
