"""Fig. B.8 -- Effort needed to reach the resilience thresholds (0.5, 0.8), relative to the MDN.

Reads   outputs/predictions/nigeria_temporal/test_scored.parquet
Writes  outputs/figures/FigB8_effort_vs_mdn.{png,pdf}
        outputs/tables/FigB8_effort_vs_mdn.csv
"""
from nigeria_resilience import exhibits as ex
from nigeria_resilience.resilience.effort import (
    compute_effort_agree_vs_mdn,
    plot_effort_difference_bars_grid,
)

ex.setup()
df = ex.load_scored("nigeria_temporal")
res = compute_effort_agree_vs_mdn(df, id_col="hhid", thresholds=(0.5, 0.8), n_boot=ex.BOOT_B,
                                  seed=123, min_n=50, verbose=False)
plot_effort_difference_bars_grid(res, thresholds=(0.5, 0.8), figsize=(18, 12), dpi=300,
                                 outfile=str(ex.FIG_DIR / "FigB8_effort_vs_mdn.png"))
ex.save_table(ex.flatten(res), "FigB8_effort_vs_mdn")
