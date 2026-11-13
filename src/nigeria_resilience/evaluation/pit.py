"""PIT (Probability Integral Transform) histograms and computation helpers."""

import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from nigeria_resilience.visualization.save import save_and_show

# ---------------------------------------------------------------------------
# PIT histogram panel
# ---------------------------------------------------------------------------

def plot_pit_panel(
    df, cols, titles, *,
    bins=20,
    stat="probability",
    same_ylim=True,
    show_baseline=True,
    out_path=None,
    return_fig=False,
    ncols=None,
    nrows=None,
):
    """PIT histogram panel with optional common y-limit and flexible *stat*.

    Parameters
    ----------
    stat : str
        ``"probability"`` (bar heights sum to 1, baseline = 1/bins),
        ``"count"`` (raw counts, baseline = n/bins),
        ``"percent"`` (0 .. 100 %, baseline = 100/bins),
        ``"density"`` (density on [0,1], baseline = 1.0).
    out_path : str, optional
        If given the figure is saved **and** displayed via ``save_and_show``.
    """
    if len(cols) != len(titles):
        raise ValueError("`cols` and `titles` must have the same length.")

    nplots = len(cols)
    if nplots == 0:
        raise ValueError("`cols` is empty.")

    edges = np.linspace(0.0, 1.0, bins + 1)
    width = edges[1] - edges[0]

    # --- compute global y-limit so all panels are comparable ---------------
    y_max = None
    if same_ylim:
        global_max = 0.0
        global_base = 0.0
        for c in cols:
            u = np.clip(df[c].to_numpy(dtype=float), 1e-9, 1 - 1e-9)
            n = max(len(u), 1)
            counts, _ = np.histogram(u, bins=edges)
            if stat in ("probability", "proportion"):
                heights = counts / n
                base = 1.0 / bins
            elif stat in ("count", "frequency"):
                heights = counts
                base = n / bins
            elif stat == "percent":
                heights = 100.0 * counts / n
                base = 100.0 / bins
            elif stat == "density":
                heights = counts / (n * width)
                base = 1.0
            else:
                raise ValueError("Unsupported stat.")
            if heights.size:
                global_max = max(global_max, float(heights.max()))
            global_base = max(global_base, float(base))

        y_max = max(global_max * 1.15, global_base * 1.6)

    # --- layout ------------------------------------------------------------
    if ncols is None and nrows is None:
        ncols = 3
        nrows = math.ceil(nplots / ncols)
    elif ncols is not None and nrows is None:
        nrows = math.ceil(nplots / ncols)
    elif ncols is None and nrows is not None:
        ncols = math.ceil(nplots / nrows)

    if ncols * nrows < nplots:
        raise ValueError(
            f"Grid too small: ncols*nrows={ncols * nrows} < {nplots} plots."
        )

    fig, ax = plt.subplots(
        nrows, ncols, sharex=True, sharey=same_ylim, figsize=(14, 10)
    )
    ax = np.atleast_1d(ax).ravel()

    for i, (c, title) in enumerate(zip(cols, titles)):
        u = np.clip(df[c].to_numpy(dtype=float), 1e-9, 1 - 1e-9)
        n = max(len(u), 1)

        sns.histplot(
            u,
            bins=bins,
            binrange=(0, 1),
            stat=stat,
            edgecolor="black",
            alpha=0.7,
            ax=ax[i],
        )

        if show_baseline:
            if stat in ("probability", "proportion"):
                base = 1.0 / bins
                label = f"Uniform (1/{bins})"
            elif stat in ("count", "frequency"):
                base = n / bins
                label = f"Uniform (n/{bins})"
            elif stat == "percent":
                base = 100.0 / bins
                label = f"Uniform (100/{bins}%)"
            elif stat == "density":
                base = 1.0
                label = "Uniform density (=1)"
            else:
                raise ValueError("Unsupported stat.")
            ax[i].axhline(base, ls="--", lw=1.2, color="k", label=label)

        if same_ylim and y_max is not None:
            ax[i].set_ylim(0, y_max)
        ax[i].set_xlim(0, 1)
        ax[i].set_title(title)
        ax[i].set_xlabel("PIT")
        ax[i].set_ylabel(
            "Count"
            if stat in ("count", "frequency")
            else (
                "Percent"
                if stat == "percent"
                else (
                    "Proportion"
                    if stat in ("probability", "proportion")
                    else "Density"
                )
            )
        )
        ax[i].grid(True, axis="y", alpha=0.3)
        if show_baseline:
            ax[i].legend(loc="upper right")

    for j in range(nplots, len(ax)):
        fig.delaxes(ax[j])

    fig.tight_layout()

    if out_path:
        save_and_show(fig, out_path)
    elif not return_fig:
        plt.show()

    if return_fig:
        return fig, ax
