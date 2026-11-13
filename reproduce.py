"""
Reproduce every figure and table of the paper and its appendix.

    python reproduce.py                    # data -> predict -> analysis -> verify
    python reproduce.py --retrain          # refit the six models first, then predict with them
    python reproduce.py --stage data       # run a single stage
    python reproduce.py --stage analysis --from 9    # resume the analysis at step 9
    python reproduce.py --stage verify     # only compare outputs/ with reference/

Stages
------
data      Build the household panel from data/raw (GHS-Panel extract, deflators,
          CHIRPS/MODIS, ACLED) into outputs/data/analysis_panel.xlsx; verify compares
          it with the shipped analysis file data/processed/.
model     (only with --retrain) Refit OLS+GLM, Elastic Net, XGB-1D, XGB-Mix, NN-1D and
          MDN with the selected hyperparameters (config/hparams/) for both scenarios
          into models_retrained/. By default the shipped fitted models in models/ are used.
predict   Load the fitted models (models/<scenario>/model/, or models_retrained/) and
          predict the 2015 test households: predicted distributions, PIT values and
          resilience scores, written to outputs/predictions/.
analysis  Run scripts/analysis/01..16 on outputs/predictions/.  Every exhibit is
          written under its paper
          number: outputs/figures/Fig2_*.pdf ... FigB8_*.pdf, outputs/tables/TableA1_* ...
verify    Compare the predictions with those saved when the models were fitted, and
          the regenerated exhibits with reference/; writes
          outputs/verification_report.md.

Every step runs in a fresh Python process with the determinism settings below;
its log is written to outputs/logs/<step>.log.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nigeria_resilience.paper import ANALYSIS_STEPS, DATA_STEPS, MODEL_STEPS, PREDICT_STEPS

# Read by TensorFlow / BLAS at import time, so they are set in the environment of
# each child process rather than inside the scripts.
DETERMINISM_ENV = {
    "PYTHONHASHSEED": "42",
    "TF_DETERMINISTIC_OPS": "1",
    "TF_CUDNN_DETERMINISTIC": "1",
    "TF_ENABLE_ONEDNN_OPTS": "0",
    # Same matrix kernels on every CPU of a vendor (AVX-512 machines otherwise
    # take another code path). The shipped models were fitted with this setting.
    "ONEDNN_MAX_CPU_ISA": "AVX2",
    "NIGERIA_RESILIENCE_TF_THREADS": "0",   # TensorFlow's default (models/seed.py)
    # NumPy and OpenBLAS: the same (AVX2) kernels on CPUs with and without AVX-512.
    # On a CPU without AVX-512, NumPy warns that there is nothing to disable.
    "OPENBLAS_CORETYPE": "Haswell",
    "ATEN_CPU_CAPABILITY": "avx2",          # PyTorch (used by XGBoostLSS)
    "NPY_DISABLE_CPU_FEATURES": "AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL",
    "DNNL_MAX_CPU_ISA": "AVX2",
    "TF_CPP_MIN_LOG_LEVEL": "2",
    "CUDA_VISIBLE_DEVICES": "-1",       # CPU only: the published results are CPU results
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "MPLBACKEND": "Agg",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
    "PYTHONUNBUFFERED": "1",
}
EXTRA_ENV: dict[str, str] = {}


def run_step(step: str, log_dir: Path) -> float:
    script, *args = step.split()
    log_path = log_dir / (Path(script).stem + "".join(f"_{a}" for a in args[1:2]) + ".log")
    env = {**os.environ, **DETERMINISM_ENV, "NIGERIA_RESILIENCE_ROOT": str(ROOT), **EXTRA_ENV}
    if "NIGERIA_RESILIENCE_TF_THREADS_OVERRIDE" in os.environ:  # sensitivity checks only
        env["NIGERIA_RESILIENCE_TF_THREADS"] = os.environ["NIGERIA_RESILIENCE_TF_THREADS_OVERRIDE"]
    print(f"  > {script} {' '.join(args)}".rstrip(), flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run([sys.executable, str(ROOT / script), *args],
                              cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
        print("\n".join("      " + line for line in tail))
        raise SystemExit(f"Step failed ({script}, exit {proc.returncode}); "
                         f"see {log_path.relative_to(ROOT)}")
    print(f"    done in {dt:,.0f} s", flush=True)
    return dt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["all", "data", "model", "predict", "analysis", "verify"],
                    default="all")
    ap.add_argument("--retrain", action="store_true",
                    help="refit the six models into models_retrained/ and predict with them")
    ap.add_argument("--from", dest="start", type=int, default=1,
                    help="first analysis step to run (e.g. 9)")
    args = ap.parse_args()

    if args.retrain:
        EXTRA_ENV["NIGERIA_RESILIENCE_MODELS_DIR"] = str(ROOT / "models_retrained")
        os.environ.update(EXTRA_ENV)

    log_dir = ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stages = (["data"] + (["model"] if args.retrain else []) + ["predict", "analysis", "verify"]
              if args.stage == "all" else [args.stage])

    t0 = time.time()
    for stage in stages:
        print(f"\n== {stage} ==", flush=True)
        if stage == "data":
            for s in DATA_STEPS:
                run_step(s, log_dir)
        elif stage == "model":
            for s in MODEL_STEPS:
                run_step(s, log_dir)
        elif stage == "predict":
            for s in PREDICT_STEPS:
                run_step(s, log_dir)
        elif stage == "analysis":
            for s in ANALYSIS_STEPS:
                if int(Path(s.split()[0]).name[:2]) >= args.start:
                    run_step(s, log_dir)
        elif stage == "verify":
            from nigeria_resilience.verify import verify
            ok = verify(ROOT)
            if not ok and args.stage == "verify":
                raise SystemExit(1)
    print(f"\nFinished in {(time.time() - t0) / 60:,.1f} min.")


if __name__ == "__main__":
    main()
