# Replication package: *Predicting Poverty-Resilient Households using Machine Learning in Sub-Saharan Africa*

Nouréini Souleymane Sayouti and Edouard Romeo Mensah.
*Oxford Bulletin of Economics and Statistics* (2026).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24147304.svg)](https://doi.org/10.5281/zenodo.24147304)
Archived on Zenodo, version 1.0.2: [10.5281/zenodo.24200611](https://doi.org/10.5281/zenodo.24200611)
(the record's DOI, [10.5281/zenodo.24147304](https://doi.org/10.5281/zenodo.24147304), always opens the latest version).

This repository contains the code, the six fitted models and the shareable data
needed to reproduce every figure and table in the paper and its appendix, with one
command, on Windows or Linux (macOS through Docker). The household survey (World
Bank GHS-Panel) is free to download but may not be redistributed: download it first
([Data availability](#data-availability), about 30 minutes plus a Stata run for the
EPAR indicators), then

```bash
uv sync                      # create the pinned Python environment (once)
uv run python reproduce.py
```

The run takes about 20 minutes on a laptop CPU (AMD Ryzen 5 6600H). It builds the analysis panel
from the survey files, loads the fitted models and predicts the 2015 test households,
regenerates all exhibits under their paper numbers (`outputs/figures/Fig2_*.pdf`,
`outputs/tables/TableA4_*.tex`, ...) and checks them against the published
versions (`outputs/verification_report.md`). Refitting the models is optional
(`--retrain`, about 5 to 15 minutes); see
[Refitting the models](#refitting-the-models-and-rerunning-the-hyperparameter-search).

---

## Contents

1. [What the code does](#what-the-code-does)
2. [Requirements](#requirements)
3. [Reproducing the results](#reproducing-the-results)
4. [Exhibits and the scripts that produce them](#exhibits-and-the-scripts-that-produce-them)
5. [Verification against the published results](#verification-against-the-published-results)
6. [Repository layout](#repository-layout)
7. [Data availability](#data-availability)
8. [Refitting the models and rerunning the hyperparameter search](#refitting-the-models-and-rerunning-the-hyperparameter-search)
9. [Reproducibility notes](#reproducibility-notes)
10. [Citation and license](#citation-and-license)

## What the code does

1. **Data** (`scripts/data/00_build_extract.py`, `01_build_panel.py`). Builds, from
   the World Bank survey files, a balanced panel of 3,441
   Nigerian households observed in six rounds of the General Household
   Survey-Panel (2010/11, 2012/13, 2015/16; post-planting and post-harvest):
   real per-capita consumption in 2016 PPP dollars, its value in the next round
   (the outcome), household covariates, and objective shocks (3-month SPI from
   CHIRPS, MODIS NDVI z-scores, distance to and timing of ACLED conflict events).
2. **Models** (`scripts/model/train_models.py`, `src/nigeria_resilience/models/`).
   Six distributional models of log consumption at t+1, all log-normal: the
   two-stage OLS+GLM benchmark (Cissé and Barrett 2018), a two-stage Elastic Net,
   XGBoostLSS with one Gaussian (XGB-1D) or a three-component mixture (XGB-Mix),
   a single-Gaussian neural network (NN-1D) and a four-component mixture density
   network (MDN) trained with quantile recalibration. Hyperparameters are
   selected on the validation split (Optuna, 50 trials); each model is then
   refitted on train + validation. Two designs: a temporal split (train
   2010-2012, validate 2013, test 2015) and an LGA-temporal split in which the
   test households live in LGAs unseen in training and validation.
3. **Analysis** (`scripts/analysis/`, 16 numbered steps). Calibration of the
   predicted distributions (PIT histograms, Q-Q and inverse-quantile plots),
   forecast scores (MSE, NLL, CRPS, EMD accuracy), resilience scores
   P(y_{t+1} > poverty line) at the $1.90, $3.20, $5.50 and national lines,
   Brier skill and Murphy decomposition, correlates of the scores, targeting
   errors, agreement and disagreement with the MDN, and the attrition checks.

## Requirements

**Software.** Python 3.9 and [uv](https://docs.astral.sh/uv/) (installs the
exact Python version and every package from `uv.lock`). Main packages:
TensorFlow 2.8.0, TensorFlow Probability 0.15.0, XGBoost 2.1.4, PyTorch 2.2.2
(CPU, used by XGBoostLSS), Optuna 4.3.0, NumPy 1.22.4, pandas 1.5.3,
SciPy 1.10.1, statsmodels 0.13.5, scikit-learn 1.3.2, matplotlib 3.5.1. These
are the versions that produced the published results. XGBoostLSS is vendored
in `src/xgboostlss/` (Apache 2.0, see its `NOTICE`).

Alternatively, use the Docker image (no local Python needed; this is also the
route for Apple-silicon Macs, for which TensorFlow 2.8 has no wheels):

```bash
docker build -t nigeria-resilience .
docker run --rm -v "$PWD/outputs:/app/outputs" nigeria-resilience
```

**Hardware.** A standard laptop; no GPU (the published results are CPU results
and the pipeline forces CPU execution). About 4 GB of RAM and 3 GB of disk
(mostly TensorFlow and PyTorch).

**Operating systems.** Tested on Windows 11 (AMD Ryzen 5 6600H laptop) and,
by the CI, on GitHub-hosted Ubuntu 24.04 and Windows Server 2022. On Windows, keep the repository at a short
path (for example `C:\work\nigeria-resilience-mdn`) or enable
[long path support](https://learn.microsoft.com/windows/win32/fileio/maximum-file-path-limitation).

## Reproducing the results

```bash
git clone https://github.com/noureini/nigeria-resilience-mdn.git
cd nigeria-resilience-mdn
uv sync
uv run python reproduce.py
```

`reproduce.py` runs four stages; each step runs in its own process, with the
settings listed under [Reproducibility notes](#reproducibility-notes), and writes
a log to `outputs/logs/`.

| Stage | What it does | Time |
|---|---|---|
| `data` | builds the household extract from the GHS-Panel and EPAR files (`00`), then the analysis panel (`01`), into `outputs/data/` | 6 min |
| `predict` | loads the six fitted models (`models/<scenario>/model/`) and predicts the test households of both designs into `outputs/predictions/` | 1 min |
| `analysis` | `scripts/analysis/01`–`16`: every figure and table, from `outputs/predictions/` | 8 min |
| `verify` | compares the predictions with those saved when the models were fitted, and the exhibits with `reference/` | seconds |

Useful options:

```bash
uv run python reproduce.py --retrain                 # refit the six models first (see below)
uv run python reproduce.py --stage analysis --from 9 # resume at a step
uv run python reproduce.py --stage verify            # only re-check outputs
uv run python scripts/analysis/05_fig6_fig7_brier.py # run one step on its own
```

Every script's header lists the exact files it reads and writes.

## Exhibits and the scripts that produce them

Every exhibit file is named after its number in the paper. Figures are written
as `.pdf` and `.png` to `outputs/figures/`; the numbers behind them, and the
tables, to `outputs/tables/`. The single source of truth for this mapping is
[`src/nigeria_resilience/paper.py`](src/nigeria_resilience/paper.py).

### Main text

| Exhibit | Content | Script (`scripts/analysis/`) | Output |
|---|---|---|---|
| Fig. 1 | Decision tree of the six specifications (diagram) | — | `docs/Fig1_model_decision_tree.dot` (Graphviz) |
| Fig. 2 | PIT histograms | `01_fig2_pit_histograms.py` | `Fig2_pit_histograms` |
| Fig. 3 | PIT Q-Q plots: (a) all, (b) rural, (c) urban | `02_fig3_pit_qq.py` | `Fig3{a_all,b_rural,c_urban}_pit_qq` |
| Fig. 4 | Inverse-quantile plots by household type | `03_fig4_inverse_quantiles.py` | `Fig4_inverse_quantiles` |
| Fig. 5 | MSE, NLL, CRPS, EMD accuracy relative to the MDN | `04_fig5_figA1_forecast_metrics.py` | `Fig5_relative_performance_vs_mdn` |
| Fig. 6 | Brier skill score ratios to the MDN | `05_fig6_fig7_brier.py` | `Fig6_brier_skill_ratio_vs_mdn` |
| Fig. 7 | Murphy decomposition | `05_fig6_fig7_brier.py` | `Fig7_murphy_decomposition` |
| Fig. 8 | Distribution of predicted resilience scores | `06_fig8_resilience_distributions.py` | `Fig8_resilience_score_distributions` |
| Fig. 9 | Mean resilience score ratios to the MDN | `07_fig9_mean_resilience_ratios.py` | `Fig9_mean_resilience_ratio_vs_mdn` |
| Fig. 10 | Correlates of the resilience scores, $1.90/day | `08_fig10_resilience_correlates.py` | `Fig10_resilience_correlates` |
| Fig. 11 | Targeting performance at τ = 0.5 | `09_fig11_figB1_B5_targeting.py` | `Fig11_targeting_tau05` |
| Fig. 12 | PIT Q-Q plots, new LGAs at t+1 | `10_fig12_fig13_new_lgas.py` | `Fig12_new_lgas_pit_qq` |
| Fig. 13 | Forecast metrics relative to the MDN, new LGAs at t+1 | `10_fig12_fig13_new_lgas.py` | `Fig13_new_lgas_relative_performance` |

### Appendix

| Exhibit | Content | Script (`scripts/analysis/`) | Output |
|---|---|---|---|
| Table A.1 | Selected neural-network hyperparameters | `16_tablesA1_A3_specification.py` | `TableA1_nn_hyperparameters.tex` |
| Table A.2 | Selected XGBoost hyperparameters | `16_tablesA1_A3_specification.py` | `TableA2_xgb_hyperparameters.tex` |
| Table A.3 | List of household covariates | `16_tablesA1_A3_specification.py` | `TableA3_covariates.tex` |
| Table A.4 | Attrition: training set | `15_tablesA4_A5_attrition.py` | `TableA4_attrition_train.tex` |
| Table A.5 | Attrition: test set | `15_tablesA4_A5_attrition.py` | `TableA5_attrition_test.tex` |
| Fig. A.1 | Forecast metrics (levels) | `04_fig5_figA1_forecast_metrics.py` | `FigA1_forecast_metrics` |
| Fig. A.2 | Agreement with the MDN resilience classes | `11_figA2_agreement.py` | `FigA2_agreement_with_mdn` |
| Fig. A.3 | Type 2 disagreement (SMD) | `12_figA3_figB7_disagreement.py` | `FigA3_disagreement_type2` |
| Fig. A.4 | Correlates of the resilience scores, $3.20/day | `08_fig10_resilience_correlates.py` | `FigA4_resilience_correlates_320` |
| Fig. A.5 | Correlates of the resilience scores, $5.50/day | `08_fig10_resilience_correlates.py` | `FigA5_resilience_correlates_550` |
| Fig. A.6 | Correlates of the resilience scores, national line | `08_fig10_resilience_correlates.py` | `FigA6_resilience_correlates_national` |
| Fig. B.1 | Coverage gap by eligibility threshold | `09_fig11_figB1_B5_targeting.py` | `FigB1_coverage_gap` |
| Fig. B.2 | Exclusion error | `09_fig11_figB1_B5_targeting.py` | `FigB2_exclusion_error` |
| Fig. B.3 | Inclusion error | `09_fig11_figB1_B5_targeting.py` | `FigB3_inclusion_error` |
| Fig. B.4 | Precision-recall trade-offs | `09_fig11_figB1_B5_targeting.py` | `FigB4_precision_recall` |
| Fig. B.5 | Targeting under a fixed budget | `09_fig11_figB1_B5_targeting.py` | `FigB5_fixed_budget_targeting` |
| Fig. B.6 | Households across resilience categories | `13_figB6_category_shares.py` | `FigB6_resilience_categories` |
| Fig. B.7 | Type 1 disagreement (SMD) | `12_figA3_figB7_disagreement.py` | `FigB7_disagreement_type1` |
| Fig. B.8 | Effort to reach the resilience thresholds | `14_figB8_effort.py` | `FigB8_effort_vs_mdn` |

All bootstrap intervals use 400 paired replications with fixed seeds, as in the paper.

## Verification against the published results

`reference/` holds the published version of every table and of the data
behind every figure. The `verify` stage compares them with the regenerated
outputs and writes `outputs/verification_report.md`:

* **Tables and figure data (CSV)**: every number is compared; *identical*
  means equal values, *close* means the largest difference in each column is
  below 10⁻⁴ of the column's magnitude.
* **LaTeX tables**: identical text, or every number within one unit of its last
  printed digit (*within rounding*).
* **Data stage**: every input against `data/checksums.csv`; the panel is then checked through the predictions of the shipped models (and, on the authors' machine, against the panel the models were fitted on).
* **Inference**: the predictions recomputed from the model files against the
  predictions saved when the models were fitted (`models/<scenario>/test_scored.parquet`).
* **Refitted models** (with `--retrain`): the predictions of the refitted models
  against those of the shipped models.

Images are not compared byte for byte (PDF and PNG files embed timestamps).
[`paper_exhibits/`](paper_exhibits/) holds the outputs of the reference run, named
by their number in the paper.

**Result.** The default run passes on Windows 11, Ubuntu 24.04 and Windows Server
2022: 15–16 of the 24 tables are identical to `reference/`, the others agree to
better than 10⁻⁶ of their magnitude (and PIT values rounded to four decimals
within one unit of the last digit); the predictions recomputed from the model
files agree with the saved ones to floating-point precision; the panel built from
the World Bank files equals the one the models were fitted on (to 2·10⁻¹¹). A refit
from that panel on a GitHub Linux runner (`--retrain`, AMD EPYC 7763; three
independent runs, identical to each other) reproduces the shipped predictions to
4·10⁻¹³; from the authors' copy of the panel it reproduces them exactly.

## Repository layout

```
.
├── reproduce.py              one-command driver (stages: data, model, analysis, verify)
├── pyproject.toml, uv.lock   pinned environment (Linux x86-64 and Windows x86-64)
├── Dockerfile                container alternative to uv
├── config/hparams/           selected hyperparameters (Optuna search), per scenario
├── data/                     see data/README.md
│   ├── sample/               household identifiers of the estimation and attrition samples
│   ├── derived/              conflict measures (ACLED), climate values (CHIRPS, MODIS)
│   ├── raw/deflators/        CPI, PPP
│   ├── raw/ghs/, raw/epar/   survey files you download / build (not in the repository)
│   └── checksums.csv         SHA-256 of every input
├── models/                   per scenario: the six fitted models (model/), their
│                             predictions, hyperparameters and provenance
├── scripts/
│   ├── data/                 00  household extract from the GHS-Panel files,
│                             01  analysis panel; optional_gee_climate.py (Earth Engine)
│   ├── model/                train_models.py (refit, or full Optuna search with --hpo),
│                             predict.py (inference with the fitted models)
│   └── analysis/             01–16  one script per exhibit (or group of exhibits)
├── src/nigeria_resilience/   library code: data, models (models/store.py: model files),
│                             scoring.py (inference), evaluation, resilience scores,
│                             targeting, plotting; paper.py (exhibit map), verify.py
├── src/xgboostlss/           vendored XGBoostLSS (Apache 2.0)
├── third_party/epar/         EPAR's Stata code for the GHS-Panel indicators (BSD-3)
├── paper_exhibits/           every figure and table, named by its number in the paper
│                             (INDEX.md: title and page; All_figures_in_paper_order.pdf)
├── reference/                the same exhibits under their script names (for verification)
├── docs/                     Fig. 1 source, Table A.3, audit of the numbers in the text
├── tests/                    unit tests (uv run pytest)
└── outputs/                  everything the pipeline writes (generated)
```

## Data availability

The household data are the Nigeria General Household Survey-Panel, waves 1-3
(National Bureau of Statistics and World Bank, LSMS-ISA). They are public use files
of the World Bank Microdata Library: free to download after registration, but the
terms of use do not allow redistribution. The package therefore contains the code
that builds every analysis file from the original downloads, and only data that may
be shared:

| In the repository | Not in the repository (obtain as described in [`data/README.md`](data/README.md)) |
|---|---|
| code from the World Bank files to every exhibit; EPAR's Stata code for the farm and income indicators | GHS-Panel waves 1-3, Stata files (World Bank Microdata Library, free registration) |
| conflict measures by household and round, computed from ACLED (ACLED's terms allow derived data) | ACLED event file (only needed to recompute the conflict measures) |
| monthly CHIRPS rainfall and MODIS NDVI by household, without coordinates | |
| CPI and PPP deflators; household identifiers of the samples; checksums of every input | |
| the six fitted models and their predictions (household identifiers and model outputs only) | |

To reproduce the paper:

1. Register at the World Bank Microdata Library and download the Stata files of
   waves 1-3 (catalog entries 1002, 1952, 2734; data releases
   NGA_2010_GHSP-W1_v03_M, NGA_2012_GHSP-W2_v02_M, NGA_2015_GHSP-W3_v02_M); unzip them
   into `data/raw/ghs/wave1`, `wave2` and `wave3`.
2. In Stata, run `third_party/epar/run_epar.do` (set the package folder on its first
   line); it writes the EPAR indicators to `data/raw/epar/`.
3. `uv run python reproduce.py`.

The data stage checks every input against `data/checksums.csv`, stops with a
message if a file is missing and names any file that differs from the release used
(the authors keep the exact files and will help compare a later release; see
`data/README.md`). Without the survey files, the stages that need them
cannot run; the shipped model files, predictions and `reference/` exhibits can still
be inspected.

## Refitting the models and rerunning the hyperparameter search

**Shipped models.** `models/<scenario>/` contains, for each of the two designs:

| File | Content |
|---|---|
| `model/ols_glm.json`, `model/elastic_net.json` | coefficients of the two-stage models (and the feature scaler of the Elastic Net) |
| `model/xgb_1d.json`, `model/xgb_mix.json`, `model/xgb_meta.json` | XGBoostLSS boosters (XGBoost JSON format) and their start values |
| `model/mdn.weights.h5`, `model/nn_1d.weights.h5`, `model/networks.json` | network weights (Keras HDF5), architecture and feature scaler |
| `test_scored.parquet`, `mixture_params.npz` | the predictions made when the models were fitted |
| `hparams.json`, `training_info.json` | hyperparameters used; provenance (seed, CPU, settings, SHA-256 of the panel, git commit, package versions) |

`predict` loads these files and predicts the test households again; `verify`
checks that the result equals the saved predictions.

**Reference environment.** The shipped models were fitted by the
[*Reference refit*](.github/workflows/linux-refit.yml) workflow on a GitHub-hosted
`ubuntu-24.04` runner (4 vCPUs, AMD EPYC 7763), with the pinned lock file and the
settings of `reproduce.py`. Repeated runs of that workflow on runners with an
AMD EPYC 7763 or 9V74 (the processors GitHub assigns most often) produce
byte-identical model files. Built from the World Bank files, the panel differs from
the authors' copy in the last binary digit of a few values (the extract stores
decimals with 8 significant digits), and a refit in this environment then matches
the shipped models to about 10⁻¹³. The figures and tables of the article are
produced from the shipped models. Refitting the neural networks gives identical
models on the reference processor and slightly different weights on other
processors, without changing any conclusion.

**Refit.** `--retrain` refits the six models of both designs with the selected
hyperparameters (`config/hparams/`) into `models_retrained/`, predicts with
them and reruns the analysis; `verify` then reports how far the predictions
move from those of the shipped models.

```bash
uv run python reproduce.py --retrain
```

To refit in the reference environment, fork the repository and run the
*Reference refit* workflow (Actions tab, *Run workflow*); it uploads the fitted
models as an artifact, and its job summary shows the CPU and the stopping
losses. Runs on an AMD EPYC 7763 or 9V74 reproduce `models/` byte for byte;
GitHub sometimes assigns another processor (an Intel Xeon or an AMD EPYC 9V45),
in which case rerun the job. On another
CPU the refit is close but not identical: see *Cross-platform* below.

**Full search.** The hyperparameters were selected by 50-trial Optuna searches
(TPE sampler, seed 42) on the validation split; the MDN and NN-1D refits stop
when the training loss reaches the value of the selected trial at its best
validation epoch (`target_nll` in `config/hparams/`). To rerun the searches
(several hours per design):

```bash
uv run python scripts/model/train_models.py --scenario nigeria_temporal --hpo --out-dir models_hpo
NIGERIA_RESILIENCE_MODELS_DIR=models_hpo uv run python reproduce.py --stage predict
uv run python reproduce.py --stage analysis
```

The XGBoostLSS search has a 10-minute time budget per model, so on a slower
machine it may complete fewer than 50 trials and select different
hyperparameters; the shipped selection is the reference.

## Reproducibility notes

* **Determinism.** Every random step is seeded: Python, NumPy and TensorFlow
  (42), the Optuna searches (42), and every bootstrap (42; 1 and 123 where the
  paper used them). `reproduce.py` starts each step in a new process with
  `PYTHONHASHSEED=42`, TensorFlow deterministic ops, CPU only, single-threaded
  BLAS, and the CPU kernels of oneDNN, OpenBLAS, NumPy and PyTorch limited to
  AVX2 so that CPUs with and without AVX-512 run the same code. TensorFlow keeps
  its default thread pools, as in the reference refit.
* **Same environment, same result.** Two runs in the same environment (same CPU
  model, operating system and lock file) give byte-identical models, predictions
  and exhibits.
* **Cross-platform.** The default run does not fit any model, so every exhibit
  is the same on every computer: inference with the shipped models differs from
  the saved predictions by floating-point noise (about 10⁻⁷; a PIT value, rounded
  to four decimals, can move by 10⁻⁴), and no printed number changes (see the
  verification report, run on Linux and Windows by the CI). A *refit*, however,
  depends on the CPU: the neural networks are trained by millions of
  floating-point operations, whose rounding differs between processor models,
  and gradient descent amplifies those differences. TensorFlow documents the
  same limit: determinism holds on the same hardware and software. Across the
  GitHub runners and laptops we tested, a refit moves the MDN's EMD accuracy by
  up to about ±0.03 and the ratios to the MDN accordingly, without changing the
  ranking of the two-stage and distributional models; the verification report
  shows the size of the change for any refit.
* **Reproducibility checks.** On every push the CI runs the unit tests. Run by
  hand (*Actions → CI → Run workflow*) where the survey files are available (the
  authors run it on their private copy; in a fork, store your own copy of
  `data/raw/ghs` and `data/raw/epar` in a private repository and set the variable
  `DATA_REPO` and the secret `DATA_REPO_SSH_KEY`), it builds the data from the World
  Bank files, reproduces the paper from the shipped models on Linux and Windows and
  compares every exhibit with `reference/`. It also refits the six models on Linux, only on
  the reference CPUs (AMD EPYC 7763/9V74; on another CPU the refit is skipped with a
  warning), and compares them with the shipped ones. The *Reference refit*
  workflow checks that repeated refits are byte-identical.
* **Tests.** `uv run pytest` runs fast unit tests (exhibit map, splits,
  hyperparameter files, scoring helpers, shipped model outputs).

## Citation and license

Please cite the article (see [`CITATION.cff`](CITATION.cff)):

```bibtex
@article{sayouti2026resilience,
  author  = {Sayouti, Nour{\'e}ini Souleymane and Mensah, Edouard Romeo},
  title   = {Predicting Poverty-Resilient Households using Machine Learning in Sub-Saharan Africa},
  journal = {Oxford Bulletin of Economics and Statistics},
  year    = {2026}
}
```

and, for the code and data, the replication package:

```bibtex
@software{sayouti2026replication,
  author  = {Sayouti, Nour{\'e}ini Souleymane and Mensah, Edouard Romeo},
  title   = {Replication package for: Predicting Poverty-Resilient Households using Machine Learning in Sub-Saharan Africa},
  version = {v1.0.2},
  year    = {2026},
  doi     = {10.5281/zenodo.24200611},
  url     = {https://github.com/noureini/nigeria-resilience-mdn}
}
```

Code: MIT License ([`LICENSE`](LICENSE)). Data: terms of the original
providers ([`data/README.md`](data/README.md)).

Corresponding author: Edouard Romeo Mensah, mensahedouard@gmail.com.
