"""Helpers shared by the analysis scripts (scripts/analysis/*.py).

Every analysis script starts from the test-set predictions recomputed from the
fitted models by scripts/model/predict.py (outputs/predictions/<scenario>/
test_scored.parquet and mixture_params.npz), writes its figure(s) under the paper's exhibit number
in outputs/figures/ and the numbers behind them in outputs/tables/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nigeria_resilience.config import (
    PATHS,
    list_exog_shocks,
    list_hh_characteristcs,
    list_shocks_comm,
)
from nigeria_resilience.models.xgboost_mixture import compute_national_poverty_line
from nigeria_resilience.visualization.style import set_defaults

# Covariates used for the SMD / meta-regression exhibits (Figs. 10, A.3, B.7)
COVARS = list_hh_characteristcs + list_shocks_comm + list_exog_shocks

PIT_COLS = ["cdf_ols_glm", "cdf_elastic_net", "cdf_xgb_1d", "cdf_nn", "cdf_xgb_md", "cdf_mdn"]
PIT_TITLES = ["OLS + GLM", "Elastic Net", "XGB -- 1D", "NN -- 1D", "XGB -- Mix", "MDN"]
SUFFIXES_6 = ("ols_glm", "elastic_net", "xgb_1d", "nn", "xgb_md", "mdn")

BOOT_B = 400      # bootstrap replications used throughout the paper
SEED = 42

FIG_DIR = PATHS["figures"]
TAB_DIR = PATHS["tables"]


def setup() -> None:
    """Create output folders and apply the plotting defaults used in the paper."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    set_defaults()


def national_line() -> float:
    """National poverty line in 2016 PPP $/day (2.54)."""
    return compute_national_poverty_line()


def load_scored(scenario: str = "nigeria_temporal") -> pd.DataFrame:
    """Scored 2015 test households of a scenario, with the household-type variable."""
    df = pd.read_parquet(PATHS["predictions"] / scenario / "test_scored.parquet")
    ag = "ag_hh" if "ag_hh" in df.columns else "agactivities_hh"
    conds = [
        (df["rural"] == 1) & (df[ag] == 1),
        (df["rural"] == 1) & (df[ag] == 0),
        (df["rural"] == 0) & (df[ag] == 0),
        (df["rural"] == 0) & (df[ag] == 1),
    ]
    labels = ["rural_ag", "rural_non_ag", "urban_non_ag", "urban_ag"]
    df["ag_rural_hh"] = pd.Categorical(np.select(conds, labels, default=np.nan),
                                       categories=labels, ordered=True)
    return df


def load_mixtures(scenario: str = "nigeria_temporal") -> dict:
    """Predicted (alpha, mu, sigma) arrays of the MDN, NN-1D and XGB-Mix."""
    z = np.load(PATHS["predictions"] / scenario / "mixture_params.npz")
    return {k: z[k] for k in z.files}


def save_figure(fig, name: str) -> None:
    """Write outputs/figures/<name>.pdf and .png (300 dpi)."""
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.{{pdf,png}}")


def save_current_figure(name: str) -> None:
    save_figure(plt.gcf(), name)


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> None:
    """Write outputs/tables/<name>.csv (full precision)."""
    df.to_csv(TAB_DIR / f"{name}.csv", index=index, float_format="%.10g")
    print(f"  wrote tables/{name}.csv")


def flatten(obj, keys=("level",)) -> pd.DataFrame:
    """Turn nested dicts of summary statistics into a long table.

    Bootstrap draws (arrays) are dropped; scalar leaves are kept.  Used to save
    the numbers behind each figure for verification.
    """
    rows: list[dict] = []

    def walk(o, path):
        if isinstance(o, dict):
            leaves = {k: v for k, v in o.items() if np.isscalar(v) or v is None}
            if leaves and len(leaves) == len([v for v in o.values()
                                              if not isinstance(v, (np.ndarray, list))]):
                rows.append({**{f"key{i}": p for i, p in enumerate(path)}, **leaves})
            for k, v in o.items():
                if isinstance(v, (dict, pd.DataFrame)):
                    walk(v, path + [k])
        elif isinstance(o, pd.DataFrame):
            d = o.reset_index() if not isinstance(o.index, pd.RangeIndex) else o.copy()
            for i, p in enumerate(path):
                d.insert(i, f"key{i}", p)
            rows.extend(d.to_dict("records"))
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, path + [i])

    walk(obj, [])
    return pd.DataFrame(rows)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
