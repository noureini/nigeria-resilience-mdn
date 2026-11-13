"""
Predict the 2015 test households with the fitted models (inference only).

    python scripts/model/predict.py --scenario nigeria_temporal
    python scripts/model/predict.py --scenario nigeria_lga_temporal

Loads the six fitted models from <models-dir>/<scenario>/model/ (default: the
shipped models/), predicts the conditional distribution of log consumption at t+1
for every test household, and computes the PIT values and resilience scores. The
analysis scripts read these predictions; verify compares them with the predictions
saved when the models were fitted (models/<scenario>/test_scored.parquet).

Reads   the analysis panel (config.PANEL_FILE: outputs/data/analysis_panel.xlsx from the data stage)
        <models-dir>/<scenario>/model/   (see src/nigeria_resilience/models/store.py)
Writes  outputs/predictions/<scenario>/test_scored.parquet
        outputs/predictions/<scenario>/mixture_params.npz
"""
from __future__ import annotations

import argparse

from nigeria_resilience.config import PATHS, SCENARIOS, SEED
from nigeria_resilience.models import store
from nigeria_resilience.scoring import load_split, score_test_set, write_predictions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=SCENARIOS, required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    model_dir = PATHS["models"] / args.scenario / "model"
    _, _, _, test_df = load_split(args.scenario)
    scored, mixtures = score_test_set(store.load_models(model_dir), test_df, seed=args.seed)
    out = PATHS["predictions"] / args.scenario
    write_predictions(out, scored, mixtures)
    print(f"Predicted {len(scored)} test households with the models in {model_dir}; "
          f"wrote {out}")


if __name__ == "__main__":
    main()
