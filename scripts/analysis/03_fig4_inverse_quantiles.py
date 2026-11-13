"""Fig. 4 -- Inverse-quantile plots by household type (rural/urban x agricultural).

Reads   outputs/predictions/nigeria_temporal/test_scored.parquet
Writes  outputs/figures/Fig4_inverse_quantiles.{png,pdf}
        outputs/tables/Fig4_inverse_quantiles.csv   (P(PIT <= q | type), per model)
"""
import numpy as np
import pandas as pd

from nigeria_resilience import exhibits as ex
from nigeria_resilience.evaluation.diagnostics import plot_invquants_panel

ex.setup()
df = ex.load_scored("nigeria_temporal")
plot_invquants_panel(df, variable="ag_rural_hh", suffixes=ex.SUFFIXES_6,
                     titles=ex.PIT_TITLES, categorical=True,
                     out_path=str(ex.FIG_DIR / "Fig4_inverse_quantiles.png"))

rows = []
for suf in ex.SUFFIXES_6:
    for typ, g in df.groupby("ag_rural_hh", observed=True):
        u = g[f"cdf_{suf}"].to_numpy()
        for q in (0.10, 0.30, 0.50, 0.70, 0.90, 0.97):
            rows.append({"model": suf, "household_type": typ, "q": q, "n": len(u),
                         "share_pit_le_q": float(np.mean(u <= q))})
ex.save_table(pd.DataFrame(rows), "Fig4_inverse_quantiles")
