"""Build the GHS-Panel household extract and the attrition sample from the World Bank files.

Reads the GHS-Panel waves 1-3 (data/raw/ghs/) and the EPAR household indicators
(data/raw/epar/), checks them against data/checksums.csv, and writes

    outputs/data/ghs_extract.xlsx     (input of 01_build_panel.py)
    outputs/data/attrition_sample.xlsx (input of Tables A.4-A.5)

If the authors' copies are present in data/raw/lsms/, each column
is compared with them and the result is written to outputs/data/ghs_extract_check.csv.
See src/nigeria_resilience/data/ghs_extract.py for every construction rule.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nigeria_resilience.config import PATHS  # noqa: E402
from nigeria_resilience.data import ghs_extract as gx  # noqa: E402

OUT = PATHS["results"] / "data"
CHECKSUMS = PATHS["data"] / "checksums.csv"
AUTHORS_COPY = {"extract": PATHS["raw"] / "lsms" / "dev_resilience_cst_outiliers_prepared.xlsx",
            "attrition": PATHS["raw"] / "lsms" / "Attrition_sample_for_testing.xlsx"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_inputs() -> None:
    """Compare the World Bank and EPAR files with the checksums of the authors' copies."""
    ref = pd.read_csv(CHECKSUMS)
    ref = ref[ref.source.isin(["ghs", "epar"])]
    bad = []
    for r in ref.itertuples():
        if r.source == "ghs":
            same = sha256(gx.module(r.file)) == r.sha256
        else:   # EPAR output: compare the variables used (Stata writes the save date into the file)
            same = gx.epar_content_hash(int(r.file[13])) == r.content_sha256
        if not same:
            bad.append(r.file)
    if bad:
        print(f"WARNING: {len(bad)} input file(s) differ from the authors' copies "
              f"(another release of the survey, or of EPAR's code?): {', '.join(bad)}")
    else:
        print(f"All {len(ref)} input files match data/checksums.csv.")


def compare(new: pd.DataFrame, ref: pd.DataFrame, rtol: float = 1e-7) -> pd.DataFrame:
    m = ref.merge(new, on=gx.KEY, how="left", suffixes=("", "__new"), indicator=True)
    rows = []
    for c in [c for c in ref.columns if c in new.columns and c not in gx.KEY]:
        a, b = m[c], m[c + "__new"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            x, y = a.astype(float).to_numpy(), b.astype(float).to_numpy()
            both = ~np.isnan(x) & ~np.isnan(y)
            same_na = np.isnan(x) == np.isnan(y)
            rel = np.zeros_like(x)
            rel[both] = np.abs(x[both] - y[both]) / np.maximum(np.abs(x[both]), 1e-12)
            exact = np.where(both, x == y, same_na).mean()
            close = np.where(both, rel <= rtol, same_na).mean()
            rows.append((c, "number", exact, close, rel[both].max() if both.any() else 0.0))
        else:
            eq = ((a.astype(str) == b.astype(str)) | (a.isna() & b.isna())).mean()
            rows.append((c, "text", eq, eq, np.nan))
    r = pd.DataFrame(rows, columns=["column", "type", "share_identical", "share_within_1e-7", "max_rel_diff"])
    r.attrs["rows_found"] = (m._merge == "both").mean()
    return r


def missing_inputs() -> list[str]:
    """Names of the World Bank and EPAR files that are not in place."""
    out = [f"data/raw/ghs/.../{n}.dta" for names in gx.GHS_FILES.values() for n in names
           if not any(gx.GHS_DIR.rglob(f"{n}.dta"))]
    out += [f"data/raw/epar/{n}.dta" for n in gx.EPAR_FILES if not (gx.EPAR_DIR / f"{n}.dta").exists()]
    return out


def main() -> None:
    missing = missing_inputs()
    if missing:
        sys.exit(f"{len(missing)} survey file(s) missing, e.g. {', '.join(missing[:4])}.\n"
                 "Download the GHS-Panel waves 1-3 (World Bank Microdata Library) into "
                 "data/raw/ghs/ and build the EPAR indicators with third_party/epar/run_epar.do: "
                 "see data/README.md.")
    OUT.mkdir(parents=True, exist_ok=True)
    check_inputs()
    print("Building every household-round from the GHS-Panel files...")
    panel = gx.all_household_rounds()
    print("Selecting the estimation sample (see data/README.md, 'The estimation sample'):")
    extract, attrition = gx.build(panel)
    extract.to_excel(OUT / "ghs_extract.xlsx", index=False)
    attrition.to_excel(OUT / "attrition_sample.xlsx", index=False)
    print(f"  extract: {extract.shape}, attrition sample: {attrition.shape}")

    reports = []
    for name, new in (("extract", extract), ("attrition", attrition)):
        if not AUTHORS_COPY[name].exists():
            print(f"  ({AUTHORS_COPY[name].name} not present: no comparison with the authors' copy)")
            continue
        r = compare(new, pd.read_excel(AUTHORS_COPY[name]))
        r.insert(0, "file", name)
        reports.append(r)
        worst = r.sort_values("share_within_1e-7").head(3)
        print(f"  {name}: rows found {r.attrs['rows_found']:.4f}; columns all within 1e-7: "
              f"{int((r['share_within_1e-7'] == 1).sum())} of {len(r)}; lowest:\n"
              + worst.to_string(index=False))
    if reports:
        pd.concat(reports).to_csv(OUT / "ghs_extract_check.csv", index=False)


if __name__ == "__main__":
    main()
