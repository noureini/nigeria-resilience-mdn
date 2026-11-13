"""Fig. 11 and Figs. B.1-B.5 -- Targeting performance of the resilience scores.

Fig. 11   coverage gap, exclusion / inclusion error, F1 at tau = 0.5 (paired bootstrap, 400 reps)
Fig. B.1  coverage gap by eligibility threshold
Fig. B.2  exclusion error by threshold
Fig. B.3  inclusion error by threshold
Fig. B.4  precision-recall curves
Fig. B.5  fixed-budget targeting

Reads   outputs/predictions/nigeria_temporal/test_scored.parquet
Writes  outputs/figures/{Fig11,FigB1..FigB5}_*.{png,pdf}
        outputs/tables/Fig11_targeting_summary.csv, Fig11_targeting_bootstrap_tau05.csv
"""
import shutil
import tempfile
from pathlib import Path

from nigeria_resilience import exhibits as ex
from nigeria_resilience.analysis.targeting import run_complete_targeting_analysis

NAMES = {
    "bootstrap_comparison_threshold_05": "Fig11_targeting_tau05",
    "coverage_gap_by_threshold": "FigB1_coverage_gap",
    "exclusion_error_by_threshold": "FigB2_exclusion_error",
    "inclusion_error_by_threshold": "FigB3_inclusion_error",
    "precision_recall_curves": "FigB4_precision_recall",
    "fixed_budget_targeting": "FigB5_fixed_budget_targeting",
}

ex.setup()
df = ex.load_scored("nigeria_temporal")
with tempfile.TemporaryDirectory() as tmp:
    summary, boot05, _boot08 = run_complete_targeting_analysis(
        test_df=df, national_poverty_line_value=ex.national_line(),
        output_dir=tmp + "/", B=ex.BOOT_B, seed=ex.SEED)
    for src, dst in NAMES.items():
        for ext in ("png", "pdf"):
            f = Path(tmp) / f"{src}.{ext}"
            if f.exists():
                shutil.copyfile(f, ex.FIG_DIR / f"{dst}.{ext}")
        print(f"  {src} -> figures/{dst}")
ex.save_table(summary, "Fig11_targeting_summary")
ex.save_table(ex.flatten(boot05), "Fig11_targeting_bootstrap_tau05")
