"""Fig. 3 -- Q-Q plots of the PIT values: (a) all, (b) rural, (c) urban households.

Reads   outputs/predictions/nigeria_temporal/test_scored.parquet
Writes  outputs/figures/Fig3{a_all,b_rural,c_urban}_pit_qq.{png,pdf}
"""
from nigeria_resilience import exhibits as ex
from nigeria_resilience.evaluation.diagnostics import plot_pit_qq_panel

ex.setup()
df = ex.load_scored("nigeria_temporal")
for tag, sub in [("a_all", df), ("b_rural", df[df.rural == 1]), ("c_urban", df[df.rural == 0])]:
    plot_pit_qq_panel(sub, ex.PIT_COLS, ex.PIT_TITLES, same_limits=True, show_band=False,
                      conf=0.95, ncols=2, nrows=3,
                      out_path=str(ex.FIG_DIR / f"Fig3{tag}_pit_qq.png"))
