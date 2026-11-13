"""Fig. B.6 -- Share of households in each resilience category, by model and poverty line.

Reads   outputs/predictions/nigeria_temporal/test_scored.parquet
Writes  outputs/figures/FigB6_resilience_categories.{png,pdf}
        outputs/tables/FigB6_resilience_categories.csv
"""
from nigeria_resilience import exhibits as ex
from nigeria_resilience.resilience.categories import bootstrap_resilience_category_shares
from nigeria_resilience.visualization.bar_charts import plot_category_shares_grid

ex.setup()
df = ex.load_scored("nigeria_temporal")
cat = bootstrap_resilience_category_shares(df, B=ex.BOOT_B, seed=ex.SEED)
plot_category_shares_grid(cat, figsize=(14, 10), dpi=200,
                          out_path=str(ex.FIG_DIR / "FigB6_resilience_categories.png"))
ex.save_table(ex.flatten(cat), "FigB6_resilience_categories")
