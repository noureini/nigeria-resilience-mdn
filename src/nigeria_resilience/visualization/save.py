from pathlib import Path

import matplotlib.pyplot as plt

from nigeria_resilience.config import PATHS


def save_and_show(fig, filename: str, subfolder: str = "figures", dpi: int = 300, close: bool = True):
    """Save ``fig`` as PNG (``dpi``) and as vector PDF next to it.

    ``filename`` may be absolute or relative to outputs/<subfolder>/.
    """
    outdir = PATHS.get(subfolder, PATHS["results"])
    filepath = outdir / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    if filepath.suffix.lower() == ".png":
        fig.savefig(filepath.with_suffix(".pdf"), bbox_inches="tight")
    print(f"  wrote {Path(filepath).name}")
    plt.show()
    if close:
        plt.close(fig)
