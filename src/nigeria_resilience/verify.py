"""Compare a reproduction with the published results in reference/.

Checks
------
1. Exhibit files: every figure (.png and .pdf) and table listed in paper.py exists.
2. Tables (.csv): every numeric value is compared with reference/tables/.
   identical    all values equal (to 1e-12 relative)
   close        largest difference in each column <= 1e-4 x the column's magnitude
3. Tables (.tex): identical text, or every number within one unit of its last
   printed digit ("within rounding").
4. Model outputs (only when NIGERIA_RESILIENCE_MODELS_DIR points to a retrained
   set): predicted PIT values and resilience scores vs the published ones in models/.
5. Data stage: the analysis panel (outputs/data/analysis_panel.xlsx) vs the authors'
   analysis file data/processed/final_data_with_shocks_vf_NGA_for_ML.xlsx, when present.

Writes outputs/verification_report.md and returns True when nothing FAILs.
Figures are compared through the numbers behind them (the CSVs), not as images:
PDF and PNG files embed timestamps and font-rendering details.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from nigeria_resilience.paper import figure_files, table_files

REL_TOL = 1e-4
NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _compare_csv(new: Path, ref: Path) -> tuple[str, str]:
    a, b = pd.read_csv(new), pd.read_csv(ref)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return "FAIL", f"shape/columns differ ({a.shape} vs {b.shape})"
    worst, status = 0.0, "identical"
    for c in a.columns:
        x = pd.to_numeric(a[c], errors="coerce")
        y = pd.to_numeric(b[c], errors="coerce")
        if x.notna().sum() == 0 and y.notna().sum() == 0:
            if not a[c].astype(str).equals(b[c].astype(str)):
                return "FAIL", f"text column '{c}' differs"
            continue
        if not x.isna().equals(y.isna()):
            return "FAIL", f"missing values differ in '{c}'"
        d = (x - y).abs().max()
        scale = max(y.abs().max(), 1e-12)
        rel = float(d / scale) if pd.notna(d) else 0.0
        if rel > REL_TOL and _rounded_4dp(y) and d <= ROUND_4DP:
            # PIT values are stored rounded to 4 decimals: floating-point noise in
            # the prediction can move one of them by one unit of the last digit.
            status = "within rounding"
            continue
        worst = max(worst, rel)
        if rel > REL_TOL:
            return "FAIL", f"column '{c}': max |diff| = {d:.3g} ({rel:.2g} of magnitude)"
        if rel > 1e-12 and status == "identical":
            status = "close"
    if status == "within rounding":
        return status, ("values rounded to 4 decimals differ by at most one unit; "
                        f"other columns: max relative diff {worst:.2g}")
    return status, ("" if status == "identical" else f"max relative diff {worst:.2g}")


ROUND_4DP = 1.001e-4  # one unit of the 4th decimal, plus float32 representation error


def _rounded_4dp(y: pd.Series) -> bool:
    v = y.dropna().to_numpy(dtype=float)
    return len(v) > 0 and bool(np.all(np.abs(v * 1e4 - np.round(v * 1e4)) < 1e-3))


def _compare_tex(new: Path, ref: Path) -> tuple[str, str]:
    t_new, t_ref = new.read_text(encoding="utf-8"), ref.read_text(encoding="utf-8")
    if t_new == t_ref:
        return "identical", ""
    n_new, n_ref = NUM.findall(t_new), NUM.findall(t_ref)
    if NUM.sub("#", t_new) != NUM.sub("#", t_ref) or len(n_new) != len(n_ref):
        return "FAIL", "text differs"
    n_diff = 0
    for u, v in zip(n_new, n_ref):
        if u == v:
            continue
        n_diff += 1
        dec = len(v.split(".")[1]) if "." in v and "e" not in v.lower() else 0
        if abs(float(u) - float(v)) > 10 ** (-dec) + 1e-12:
            return "FAIL", f"{u} vs {v}"
    return "within rounding", f"{n_diff} number(s) differ in the last digit"


def _compare_predictions(new_dir: Path, ref_dir: Path, label: str,
                         tol: float) -> list[tuple[str, str, str]]:
    """Largest difference in the predicted means, sds, PIT values and resilience scores."""
    rows = []
    for sc in ("nigeria_temporal", "nigeria_lga_temporal"):
        f_new, f_ref = new_dir / sc / "test_scored.parquet", ref_dir / sc / "test_scored.parquet"
        if not f_new.exists() or not f_ref.exists():
            rows.append((f"{label} {sc}", "FAIL", "missing predictions"))
            continue
        a, b = pd.read_parquet(f_new), pd.read_parquet(f_ref)
        cols = [c for c in b.columns if c.startswith(
            ("mean_predicted_", "std_predicted_", "cdf_", "resilience_score_"))]
        d = max(float((a[c].astype(float) - b[c].astype(float)).abs().max()) for c in cols)
        rows.append((f"{label} {sc}",
                     "identical" if d == 0 else ("close" if d <= tol else "differs"),
                     f"max |diff| = {d:.3g} over {len(cols)} columns"))
    return rows


def _compare_panel(root: Path) -> tuple[str, str, str] | None:
    new = root / "outputs" / "data" / "analysis_panel.xlsx"
    ref = root / "data" / "processed" / "final_data_with_shocks_vf_NGA_for_ML.xlsx"
    if not new.exists():
        return None
    if not ref.exists():   # the authors' panel is not redistributed
        return ("analysis panel (built from the GHS-Panel files)", "built",
                "authors' copy not in the package: checked through the predictions of the shipped models")
    a, b = pd.read_excel(new), pd.read_excel(ref)
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return ("data/processed panel", "FAIL", f"shape {a.shape} vs {b.shape}")
    a = a.sort_values(["hhid", "surveyprd"]).reset_index(drop=True)
    b = b.sort_values(["hhid", "surveyprd"]).reset_index(drop=True)
    # Every cell: same missing-value pattern, same text, same numbers bit for bit.
    if not a.isna().equals(b.isna()):
        n = int((a.isna() != b.isna()).to_numpy().sum())
        return ("data/processed panel", "FAIL", f"{n} cells differ in missingness")
    num = [c for c in b.columns if pd.api.types.is_numeric_dtype(b[c])
           and pd.api.types.is_numeric_dtype(a[c])]
    txt = [c for c in b.columns if c not in num]
    n_txt = int((a[txt].astype(str) != b[txt].astype(str)).to_numpy().sum()) if txt else 0
    if n_txt:
        return ("data/processed panel", "FAIL", f"{n_txt} text cells differ")
    x, y = a[num].to_numpy(float), b[num].to_numpy(float)
    both = ~np.isnan(y)
    n_diff = int((x[both] != y[both]).sum())
    d = float(np.abs(x[both] - y[both]).max()) if both.any() else 0.0
    detail = (f"all {a.size:,} cells equal" if n_diff == 0 else
              f"{n_diff:,} of {a.size:,} cells differ, max |diff| = {d:.3g}")
    return ("data/processed panel (built from data/raw)",
            "identical" if n_diff == 0 else ("close" if d < 1e-8 else "FAIL"), detail)


def verify(root: Path) -> bool:
    out, ref = root / "outputs", root / "reference"
    lines = ["# Verification report", "",
             "Regenerated exhibits in `outputs/` compared with the published versions in "
             "`reference/` (see `src/nigeria_resilience/verify.py` for the rules).", ""]
    ok = True

    missing = [f"figures/{f}.{e}" for f in figure_files() for e in ("png", "pdf")
               if not (out / "figures" / f"{f}.{e}").exists()]
    missing += [f"tables/{t}" for t in table_files() if not (out / "tables" / t).exists()]
    lines += ["## Exhibit files", "",
              f"{len(figure_files())} figures and {len(table_files())} tables expected; "
              + ("all present." if not missing else f"**missing: {', '.join(missing)}**"), ""]
    ok &= not missing

    lines += ["## Tables and figure data", "", "| File | Result | Note |", "|---|---|---|"]
    for t in table_files():
        new, old = out / "tables" / t, ref / "tables" / t
        if not new.exists() or not old.exists():
            res, note = "FAIL" if not new.exists() else "no reference", ""
        elif t.endswith(".csv"):
            res, note = _compare_csv(new, old)
        else:
            res, note = _compare_tex(new, old)
        ok &= res != "FAIL"
        lines.append(f"| `{t}` | {res} | {note} |")
    lines.append("")

    models_dir = Path(os.environ.get("NIGERIA_RESILIENCE_MODELS_DIR", root / "models"))
    pred_dir = Path(os.environ.get("NIGERIA_RESILIENCE_PREDICTIONS_DIR",
                                   root / "outputs" / "predictions"))
    # Inference: predictions recomputed from the model files vs those saved at fitting
    # time. Only floating-point noise is allowed (another CPU may differ near 1e-7;
    # PIT values are rounded to 4 decimals, so one may move by 1e-4).
    rows = _compare_predictions(pred_dir, models_dir, "predictions from", tol=ROUND_4DP)
    lines += ["## Inference from the fitted models", "",
              f"Predictions in `outputs/predictions/` recomputed from the model files in "
              f"`{models_dir.name}/` vs the predictions saved when they were fitted.", "",
              "| Output | Result | Note |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    lines.append("")
    ok &= all(r[1] != "FAIL" and r[1] != "differs" for r in rows)

    if models_dir.resolve() != (root / "models").resolve():
        lines += ["## Retrained models vs shipped models", "",
                  "In the reference environment (AMD EPYC 7763/9V74) a refit matches the shipped "
                  "models to about 1e-13 from the panel built from the World Bank files, and "
                  "exactly from the authors' panel; on another CPU it differs (see the README).", "",
                  "| Output | Result | Note |", "|---|---|---|"]
        lines += [f"| {a} | {b} | {c} |" for a, b, c in
                  _compare_predictions(models_dir, root / "models", "refit", tol=1e-5)]
        lines.append("")

    panel = _compare_panel(root)
    if panel:
        lines += ["## Data stage", "", "| Output | Result | Note |", "|---|---|---|",
                  f"| {panel[0]} | {panel[1]} | {panel[2]} |", ""]
        ok &= panel[1] != "FAIL"

    lines += ["## Overall", "", "**PASS**" if ok else "**FAIL**", ""]
    report = out / "verification_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-2:]))
    print(f"Report: {report.relative_to(root)}")
    return ok
