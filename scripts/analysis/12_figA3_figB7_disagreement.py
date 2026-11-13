"""Fig. A.3 (type 2) and Fig. B.7 (type 1) -- Characteristics of households on which
the models disagree with the MDN: weighted standardized mean differences (400 reps).

Reads   outputs/predictions/nigeria_temporal/test_scored.parquet
Writes  outputs/figures/FigA3_disagreement_type2.{png,pdf}
        outputs/figures/FigB7_disagreement_type1.{png,pdf}
        outputs/tables/FigA3_FigB7_disagreement_smd.csv
"""
from nigeria_resilience import exhibits as ex
from nigeria_resilience.analysis.smd import (
    PUBLICATION_LABELS,
    compute_smd_disagreement,
    plot_smd_publication,
)

ex.setup()
df = ex.load_scored("nigeria_temporal")
smd = compute_smd_disagreement(df=df, id_col="hhid", X_cols=ex.COVARS, n_boot=ex.BOOT_B,
                               seed=123, min_d0=30, min_d1=30, topk_sel=15, asmd_thr=0.10,
                               use_weighted_smd=True)
for which, name in (("type2", "FigA3_disagreement_type2"),
                    ("type1", "FigB7_disagreement_type1")):
    plot_smd_publication(results=smd, labels_map=PUBLICATION_LABELS, which=which,
                         poverty_lines=["$1.90/day Poverty Line", "National Poverty Line"],
                         models=["XGB-1D", "XGB-Mix", "Elastic Net", "OLS+GLM"],
                         smd_thresh=0.15, top_k=15, figsize=(14, 8), dpi=300, err="se",
                         se_mult=1.96, annotate_values=True,
                         out_path=str(ex.FIG_DIR / f"{name}.png"))
ex.save_table(ex.flatten(smd), "FigA3_FigB7_disagreement_smd")
