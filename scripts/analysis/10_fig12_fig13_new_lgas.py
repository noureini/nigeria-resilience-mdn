"""Figs. 12-13 -- Spatio-temporal validation: new LGAs at t+1 (LGA-temporal split).

Fig. 12  Q-Q plots of PIT values for the test LGAs
Fig. 13  forecast metrics relative to the MDN (paired bootstrap, 400 reps)

Reads   outputs/predictions/nigeria_lga_temporal/{test_scored.parquet,mixture_params.npz}
Writes  outputs/figures/Fig12_new_lgas_pit_qq.{png,pdf}
        outputs/figures/Fig13_new_lgas_relative_performance.{png,pdf}
        outputs/tables/Fig13_new_lgas_forecast_metrics.csv
"""
import importlib.util
from pathlib import Path

from nigeria_resilience import exhibits as ex
from nigeria_resilience.evaluation.diagnostics import plot_pit_qq_panel
from nigeria_resilience.evaluation.metrics_table import plot_vs_mdn_grid

_spec = importlib.util.spec_from_file_location(
    "forecast_metrics", Path(__file__).with_name("04_fig5_figA1_forecast_metrics.py"))
fm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fm)

ex.setup()
df = ex.load_scored("nigeria_lga_temporal")
plot_pit_qq_panel(df, ex.PIT_COLS, ex.PIT_TITLES,
                  out_path=str(ex.FIG_DIR / "Fig12_new_lgas_pit_qq.png"))

summ = fm.metrics_summary("nigeria_lga_temporal")
plot_vs_mdn_grid(summ, ref_model="MDN", mode="ratio", model_order=fm.MODEL_ORDER,
                 out_path=str(ex.FIG_DIR / "Fig13_new_lgas_relative_performance.png"))
ex.save_table(fm.summary_table(summ), "Fig13_new_lgas_forecast_metrics")
