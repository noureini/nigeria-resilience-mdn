"""Tables A.1-A.3 -- Model specification.

Table A.1  selected neural-network hyperparameters (MDN, NN-1D)
Table A.2  selected XGBoostLSS hyperparameters (XGB-1D, XGB-Mix)
Table A.3  list of household covariates (descriptive; checked against the model features)

Reads   models/nigeria_temporal/hparams.json, docs/TableA3_covariates.tex
Writes  outputs/tables/TableA1_nn_hyperparameters.{tex,csv}
        outputs/tables/TableA2_xgb_hyperparameters.{tex,csv}
        outputs/tables/TableA3_covariates.tex
"""
import shutil

import pandas as pd

from nigeria_resilience import exhibits as ex
from nigeria_resilience.config import PATHS, list_all


def sci(x: float) -> str:
    """1.5e-05 -> '1.5 \\times 10^{-5}' as printed in the paper."""
    m, e = f"{x:.1e}".split("e")
    return f"${m} \\times 10^{{{int(e)}}}$"


def to_tex(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(df.columns)
    lines = ["\\begin{table}[H]", "\\centering", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", "\\begin{tabular}{l" + "c" * (len(cols) - 1) + "}",
             "\\toprule", " & ".join(cols) + " \\\\", "\\midrule"]
    lines += [" & ".join(str(v) for v in row) + " \\\\" for row in df.itertuples(index=False)]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


ex.setup()
H = ex.read_json(PATHS["models"] / "nigeria_temporal" / "hparams.json")

# ---- Table A.1 -------------------------------------------------------------
m, n = H["mdn"], H["nn_1d"]
a1 = pd.DataFrame([
    ("Hidden layers", m["params"]["n_hidden"], n["params"]["n_hidden"]),
    ("Neurons per layer", m["params"]["neurons"], n["params"]["neurons"]),
    ("Dropout rate", f"{m['params']['dropout']:.2f}", f"{n['params']['dropout']:.2f}"),
    ("Mixture components", m["params"]["n_components"], n["params"]["n_components"]),
    ("L2 regularization (hidden)", sci(m["params"]["l2_reg_v1"]), sci(n["params"]["l2_reg_v1"])),
    ("L2 regularization (output)", sci(m["params"]["l2_reg_v2"]), sci(n["params"]["l2_reg_v2"])),
    ("Learning rate", sci(m["params"]["learning_rate"]), sci(n["params"]["learning_rate"])),
    ("Batch size", m["params"]["batch_size"], n["params"]["batch_size"]),
    ("QRT bandwidth", f"{m['kde_b']:.2f}", f"{n['kde_b']:.2f}"),
], columns=["Hyperparameter", "MDN", "NN-1D"])
ex.save_table(a1, "TableA1_nn_hyperparameters")
(ex.TAB_DIR / "TableA1_nn_hyperparameters.tex").write_text(
    to_tex(a1, "Selected Neural Network Hyperparameters", "tab:nn_hparams"), encoding="utf-8")

# ---- Table A.2 -------------------------------------------------------------
x1, xm = H["xgb_1d"], H["xgb_mix"]
a2 = pd.DataFrame([
    ("Mixture components", 1, xm["M"]),
    ("Learning rate ($\\eta$)", f"{x1['eta']:.2f}", f"{xm['eta']:.3f}"),
    ("Maximum tree depth", x1["max_depth"], xm["max_depth"]),
    ("Regularization ($\\gamma$)", sci(x1["gamma"]), f"{xm['gamma']:.2f}"),
    ("Subsample ratio", f"{x1['subsample']:.2f}", f"{xm['subsample']:.2f}"),
    ("Column subsample ratio", f"{x1['colsample_bytree']:.2f}", f"{xm['colsample_bytree']:.2f}"),
    ("Minimum child weight", f"{x1['min_child_weight']:.1f}", f"{xm['min_child_weight']:.1f}"),
    ("Optimal boosting rounds", x1["opt_rounds"], xm["opt_rounds"]),
], columns=["Hyperparameter", "XGB-1D", "XGB-Mix"])
ex.save_table(a2, "TableA2_xgb_hyperparameters")
(ex.TAB_DIR / "TableA2_xgb_hyperparameters.tex").write_text(
    to_tex(a2, "Selected XGBoost Hyperparameters", "tab:xgb_hparams"), encoding="utf-8")

# ---- Table A.3 (descriptive list; model features are config.list_all) -------
shutil.copyfile(PATHS["data"].parent / "docs" / "TableA3_covariates.tex",
                ex.TAB_DIR / "TableA3_covariates.tex")
print(f"  wrote tables/TableA3_covariates.tex  ({len(list_all)} model features in config.list_all)")
