"""Plot the median joint iteration count against the condition ratio."""
import csv
import math
from pathlib import Path

from plot_convergence import NAMES, PLOT_STYLES
from problems import KAPPAS
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


def plot_iterations_by_kappa():
    """Plot five-draw median joint crossing indices and interquartile ranges."""
    # Group the joint first-hit counts for the 1e-15 target.
    counts = {}
    with (ROOT / "results/runs.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["family"], row["method"], int(row["kappa"]))
            raw = row["kjoint_15"]
            counts.setdefault(key, []).append(int(raw) if raw else math.inf)

    panel_labels = ("(a) Ridge regression", "(b) Asymmetric squared loss",
                    "(c) Binary logistic regression", "(d) Log-sum-exp")
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.2), layout="constrained")
    for ax, family, panel_label in zip(axes.flat, NAMES, panel_labels):
        for method, (label, color, style) in PLOT_STYLES.items():
            crossings = [counts[family, method, kappa] for kappa in KAPPAS]
            # Plot the median and the interquartile range over the five runs.
            lower, medians, upper = np.quantile(crossings, [.25, .5, .75], axis=1)
            ax.fill_between(KAPPAS, lower, upper, color=color, alpha=.10, linewidth=0)
            ax.loglog(KAPPAS, medians, color=color, ls=style, lw=1.05, marker="o",
                      markersize=3.3, label=label,
                      zorder=4 if method in ("fg_original", "widecap") else 2)
        ax.set(xlabel=r"Prescribed condition ratio $L/\mu$" + "\n" + panel_label,
               ylabel="Accepted iterations")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False)
    out = ROOT / "figures" / "iterations_by_kappa"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"median_iterations_1e-15.{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    plot_iterations_by_kappa()
