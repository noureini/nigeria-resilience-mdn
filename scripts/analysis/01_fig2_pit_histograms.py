"""Fig. 2 -- PIT histograms of the six models (2015 test set, temporal split).

Reads   outputs/predictions/nigeria_temporal/test_scored.parquet
Writes  outputs/figures/Fig2_pit_histograms.{png,pdf}
        outputs/tables/Fig2_pit_values.csv   (PIT value of every test household, per model)
"""
from nigeria_resilience import exhibits as ex
from nigeria_resilience.evaluation.pit import plot_pit_panel

ex.setup()
df = ex.load_scored("nigeria_temporal")
plot_pit_panel(df, ex.PIT_COLS, ex.PIT_TITLES, bins=20, stat="probability", same_ylim=True,
               ncols=2, out_path=str(ex.FIG_DIR / "Fig2_pit_histograms.png"))
ex.save_table(df[["hhid"] + ex.PIT_COLS], "Fig2_pit_values")
