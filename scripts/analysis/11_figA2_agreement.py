"""Fig. A.2 -- Agreement of each model with the MDN resilience bands (paired bootstrap).

Reads   outputs/predictions/nigeria_temporal/test_scored.parquet
Writes  outputs/figures/FigA2_agreement_with_mdn.{png,pdf}
        outputs/tables/FigA2_agreement_with_mdn.csv
"""
from nigeria_resilience import exhibits as ex
from nigeria_resilience.resilience.agreement import (
    bootstrap_bandwise_agreement_vs_mdn,
    plot_bandwise_agreement_grid,
)

ex.setup()
df = ex.load_scored("nigeria_temporal")
band = bootstrap_bandwise_agreement_vs_mdn(df, B=ex.BOOT_B, seed=ex.SEED)
plot_bandwise_agreement_grid(band, col_width=7.0, row_height=8.0, dpi=200, font_scale=1.15,
                             bar_width=0.88, x_margin=0.14, hspace=0.25, wspace=0.18,
                             out_path=str(ex.FIG_DIR / "FigA2_agreement_with_mdn.png"))
ex.save_table(ex.flatten(band), "FigA2_agreement_with_mdn")
