"""
Fit the six distributional models and score the 2015 test households.

    python scripts/model/train_models.py --scenario nigeria_temporal
    python scripts/model/train_models.py --scenario nigeria_lga_temporal
    python scripts/model/train_models.py --scenario nigeria_temporal --hpo   # rerun Optuna (hours)

Scenarios (paper Section 2.4-2.5)
---------------------------------
nigeria_temporal      train 2010-2012, validate 2013, test 2015, all LGAs (main results)
nigeria_lga_temporal  same years, disjoint train / validation / test LGAs (40/40/20 %)
                      -> Figs. 12-13 (new LGAs at t+1)

Models: OLS+GLM (Cisse & Barrett two-stage), Elastic Net two-stage, XGBoostLSS
single Gaussian (XGB-1D) and 3-component mixture (XGB-Mix), single-Gaussian
neural network (NN-1D) and mixture density network (MDN).

By default the hyperparameters selected in the published Optuna run are read
from config/hparams/<scenario>.json and each model is refitted once on
train + validation.  With --hpo the 50-trial searches are rerun first.

Reads   the analysis panel (config.PANEL_FILE: outputs/data/analysis_panel.xlsx from the data stage)
        config/hparams/<scenario>.json
Writes  <out-dir>/<scenario>/model/                 the fitted models (see models/store.py)
        <out-dir>/<scenario>/test_scored.parquet   test households + predicted
                                                   mean, sd, PIT, resilience scores
        <out-dir>/<scenario>/mixture_params.npz     (alpha, mu, sigma) of MDN, NN-1D, XGB-Mix
        <out-dir>/<scenario>/hparams.json           hyperparameters actually used
        <out-dir>/<scenario>/training_info.json     seed, data hash, versions, platform
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import tensorflow as tf

from nigeria_resilience.config import (
    PANEL_FILE,
    PATHS,
    SCENARIOS,
    SEED,
    TARGET_COL,
    features_numeric,
    list_all,
)
from nigeria_resilience.data.loading import load_nigeria_data
from nigeria_resilience.data.splitting import splitting_data
from nigeria_resilience.models import (
    cb_elastic_net,
    cb_ols_glm,
    mdn,
    nn_gaussian,
    store,
    xgboost_mixture,
    xgboost_single,
)
from nigeria_resilience.models.xgboost_mixture import compute_national_poverty_line
from nigeria_resilience.scoring import SPLIT_TYPE, score_test_set, write_predictions

ROOT = Path(__file__).resolve().parents[2]


def _cpu_name() -> str:
    """CPU model (the refit is bit-identical only for the same instruction set)."""
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "win32":
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except OSError:
        pass
    return platform.processor()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - not a git checkout
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=SCENARIOS, required=True)
    ap.add_argument("--hpo", action="store_true",
                    help="rerun the Optuna searches instead of using config/hparams")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--hparams-path", type=Path, default=None,
                    help="hyperparameter file (default config/hparams/<scenario>.json)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "models_retrained",
                    help="where to write the scored test set (default models_retrained/)")
    ap.add_argument("--overwrite-shipped", action="store_true",
                    help="allow writing into models/ (the published outputs)")
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    if out_dir == (ROOT / "models").resolve() and not args.overwrite_shipped:
        sys.exit("Refusing to overwrite the shipped models in models/; "
                 "pass --overwrite-shipped to do so.")
    out = out_dir / args.scenario
    out.mkdir(parents=True, exist_ok=True)

    hp_path = args.hparams_path or PATHS["config"] / "hparams" / f"{args.scenario}.json"
    H = None if args.hpo else json.loads(Path(hp_path).read_text(encoding="utf-8"))
    seed, n_trials = args.seed, args.n_trials

    # ---- Data and split ------------------------------------------------------
    data = load_nigeria_data()
    full_train, train_df, val_df, test_df = splitting_data(
        data, split_type=SPLIT_TYPE[args.scenario])
    print(f"Full train {full_train.shape}  train {train_df.shape}  "
          f"validation {val_df.shape}  test {test_df.shape}", flush=True)

    used: dict = {}

    # ---- 1. OLS + GLM (benchmark) ------------------------------------------
    ols_art = cb_ols_glm.train(train_df, val_df, explanatory_vars=list_all,
                               target=TARGET_COL, seed=seed)

    # ---- 2. Elastic Net two-stage (grid search on validation; fast) --------
    enet_art = cb_elastic_net.train(train_df, val_df, explanatory_vars=list_all,
                                    numeric_vars=features_numeric, target=TARGET_COL,
                                    var_family="poisson", fgls_refit=False, seed=seed)
    used["elastic_net"] = {k: (float(v) if v is not None else None)
                           for k, v in enet_art["artifacts"]["best_hparams"].items()}

    # ---- 3. XGBoostLSS, single Gaussian ------------------------------------
    xgb1d_art = xgboost_single.train(
        train_df, val_df, full_train, features=list_all, target=TARGET_COL,
        n_trials=n_trials, seed=seed, hparams=None if H is None else H["xgb_1d"])
    used["xgb_1d"] = xgb1d_art["hparams"]

    # ---- 4. XGBoostLSS, mixture of M = 3 Gaussians -------------------------
    mix_hp = None if H is None else {k: v for k, v in H["xgb_mix"].items() if k != "M"}
    xgbmix_art = xgboost_mixture.train(
        train_df, val_df, full_train, features=list_all, target=TARGET_COL, M=3,
        n_trials=n_trials, seed=seed, hparams=mix_hp)
    used["xgb_mix"] = {"M": 3, **xgbmix_art["hparams"]}

    # ---- 5. MDN -------------------------------------------------------------
    mdn_art = mdn.train(train_df, val_df, features=list_all,
                        numeric_features=features_numeric, target=TARGET_COL,
                        n_trials=n_trials, seed=seed,
                        hparams=None if H is None else H["mdn"])
    used["mdn"] = mdn_art["hparams"]

    # ---- 6. NN-1D -----------------------------------------------------------
    nn_art = nn_gaussian.train(train_df, val_df, features=list_all,
                               numeric_features=features_numeric, target=TARGET_COL,
                               n_trials=n_trials, seed=seed,
                               hparams=None if H is None else H["nn_1d"])
    used["nn_1d"] = nn_art["hparams"]

    # ---- Save the fitted models, reload them, and score the test set -------
    # The predictions are computed from the saved files, exactly as
    # scripts/model/predict.py does from the shipped models.
    store.save_models(out / "model", ols_art, enet_art, xgb1d_art, xgbmix_art, mdn_art, nn_art)
    scored, mixtures = score_test_set(store.load_models(out / "model"), test_df, seed=seed)
    write_predictions(out, scored, mixtures)
    nat = compute_national_poverty_line()
    (out / "hparams.json").write_text(json.dumps(used, indent=2, default=float) + "\n",
                                      encoding="utf-8")

    import importlib.metadata as md
    pkgs = ["tensorflow", "tensorflow-probability", "keras", "xgboost", "torch", "numpy",
            "pandas", "scipy", "statsmodels", "scikit-learn", "optuna"]
    info = {
        "scenario": args.scenario,
        "hyperparameters": "optuna search (--hpo)" if args.hpo else str(Path(hp_path).as_posix()),
        "n_trials": n_trials if args.hpo else None,
        "seed": seed,
        "stop_epochs": {"mdn": used["mdn"].get("stop_epoch"),
                        "nn_1d": used["nn_1d"].get("stop_epoch")},
        "n_test": len(test_df),
        "national_poverty_line_ppp": nat,
        "panel_sha256": _sha256(PANEL_FILE),
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "cpu": _cpu_name(),
        "determinism_env": {k: os.environ.get(k) for k in (
            "PYTHONHASHSEED", "TF_DETERMINISTIC_OPS", "TF_ENABLE_ONEDNN_OPTS",
            "ONEDNN_MAX_CPU_ISA", "OMP_NUM_THREADS")},
        "tf_threads": {"intra_op": tf.config.threading.get_intra_op_parallelism_threads(),
                       "inter_op": tf.config.threading.get_inter_op_parallelism_threads()},
        "python": sys.version.split()[0],
        "packages": {p: md.version(p) for p in pkgs},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "training_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"MDN stopped at epoch {info['stop_epochs']['mdn']}, "
          f"NN-1D at epoch {info['stop_epochs']['nn_1d']}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
