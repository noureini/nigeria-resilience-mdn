"""Copy the exhibits of a run into paper_exhibits/, named by their number in the paper.

    python scripts/collect_paper_exhibits.py                      # from outputs/
    python scripts/collect_paper_exhibits.py --source <run dir>   # e.g. a CI artifact

Fig2_pit_histograms.pdf -> Figure_02_pit_histograms.pdf (and .jpg, 300 dpi), FigA4_... -> Figure_A4_...,
TableA4_attrition_train.tex -> Table_A4_attrition_train.tex; the data behind the
figures go to paper_exhibits/data/. INDEX.md lists every exhibit with its page in
the final manuscript (corrected proofs, 12 November 2026).
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nigeria_resilience.paper import EXHIBITS  # noqa: E402
from PIL import Image  # noqa: E402

# Page of each exhibit in the final manuscript (corrected proofs, 12 November 2026).
MANUSCRIPT_PAGE = {
    "Fig. 1": 14, "Fig. 2": 16, "Fig. 3": 17, "Fig. 4": 19, "Fig. 5": 20, "Fig. 6": 22, "Fig. 7": 23,
    "Fig. 8": 24, "Fig. 9": 25, "Fig. 10": 27, "Fig. 11": 30, "Fig. 12": 31, "Fig. 13": 33,
    "Table A.1": 44, "Table A.2": 45, "Table A.3": 49, "Table A.4": 50, "Table A.5": 50,
    "Fig. A.1": 51, "Fig. A.2": 52, "Fig. A.3": 54, "Fig. A.4": 55, "Fig. A.5": 56, "Fig. A.6": 57,
    "Fig. B.1": 64, "Fig. B.2": 65, "Fig. B.3": 66, "Fig. B.4": 67, "Fig. B.5": 68, "Fig. B.6": 69,
    "Fig. B.7": 71, "Fig. B.8": 72,
}


def paper_name(stem: str) -> str:
    """Fig2_x -> Figure_02_x, Fig3a_x -> Figure_03a_x, FigA4_x -> Figure_A4_x, TableA1_x -> Table_A1_x."""
    m = re.match(r"(Fig|Table)(A|B)?(\d+)([a-c]?)_(.*)", stem)
    kind, app, num, sub, rest = m.groups()
    label = f"{app}{num}" if app else f"{int(num):02d}"
    return f"{'Figure' if kind == 'Fig' else 'Table'}_{label}{sub}_{rest}"


def to_jpg(png: Path, jpg: Path) -> None:
    """A figure's 300-dpi PNG as a high-quality JPG on a white background."""
    im = Image.open(png)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, "white"); bg.paste(im, mask=im.split()[-1]); im = bg
    im.convert("RGB").save(jpg, "JPEG", quality=95, dpi=(300, 300), subsampling=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=ROOT / "outputs",
                    help="run folder with figures/ and tables/")
    ap.add_argument("--dest", type=Path, default=ROOT / "paper_exhibits")
    args = ap.parse_args()
    src_fig, src_tab, dest = args.source / "figures", args.source / "tables", args.dest
    (dest / "data").mkdir(parents=True, exist_ok=True)
    for f in dest.rglob("*"):         # empty the folder (file by file: synced folders
        if f.is_file():               # such as OneDrive may lock directory removal)
            f.unlink()

    rows, missing = [], []
    for number, title, script, figures, tables in EXHIBITS:
        files = []
        if number == "Fig. 1":                       # diagram, not computed
            src = ROOT / "reference" / "figures" / "Fig1_model_decision_tree.pdf"
            new = "Figure_01_model_decision_tree.pdf"
            shutil.copy2(src, dest / new); files.append(new)
            to_jpg(src.with_suffix(".png"), dest / "Figure_01_model_decision_tree.jpg")
            files.append("Figure_01_model_decision_tree.jpg")
            rows.append((number, title, MANUSCRIPT_PAGE.get(number, ""), files, []))
            continue
        for stem in figures:
            src = src_fig / f"{stem}.pdf"
            if not src.exists():
                missing.append(str(src)); continue
            new = f"{paper_name(stem)}.pdf"
            shutil.copy2(src, dest / new); files.append(new)
            to_jpg(src.with_suffix(".png"), dest / f"{paper_name(stem)}.jpg")
            files.append(f"{paper_name(stem)}.jpg")
        data = []
        for t in tables:
            src = src_tab / t
            if not src.exists():
                missing.append(str(src)); continue
            if t.startswith("Table"):                # the table itself
                new = paper_name(Path(t).stem) + Path(t).suffix
                shutil.copy2(src, dest / new); files.append(new)
            else:                                    # numbers behind a figure
                shutil.copy2(src, dest / "data" / t); data.append(f"data/{t}")
        rows.append((number, title, MANUSCRIPT_PAGE.get(number, ""), files, data))

    lines = ["# Exhibits of the paper, as produced by the replication package", "",
             "Each file is named by its number in the paper (figures as PDF and as 300-dpi JPG); the page is that of the final manuscript",
             "(corrected proofs, 12 November 2026). `All_figures_in_paper_order.pdf` has every figure in one file.",
             "", "| Exhibit | Title | Page | File(s) | Data behind it |", "|---|---|---|---|---|"]
    for number, title, page, files, data in rows:
        lines.append(f"| {number} | {title} | {page} | {', '.join(f'`{f}`' for f in files)} | "
                     f"{', '.join(f'`{d}`' for d in data)} |")
    (dest / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{sum(len(r[3]) for r in rows)} exhibit files and "
          f"{len(list((dest / 'data').iterdir()))} data files written to {dest}")
    if missing:
        sys.exit("missing outputs:\n  " + "\n  ".join(missing))


if __name__ == "__main__":
    main()
