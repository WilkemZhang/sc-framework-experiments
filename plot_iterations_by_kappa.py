"""Median stopping iterations and interquartile ranges by condition ratio."""

import numpy as np
import matplotlib.pyplot as plt

from problems import KAPPAS, NUMBER_OF_DRAWS
from solvers import METHODS


FAMILY_NAMES = {
    "ridge": "Ridge regression",
    "asymmetric": "Asymmetric squared loss",
    "logistic": "Binary logistic regression",
    "logsumexp": "Log-sum-exp",
}


def plot_iterations_by_kappa(rows, exponent, figures):
    """Plot median joint stopping iterations for one target over the five draws."""
    counts = {(row["family"], row["replicate"], row["kappa"], row["method"]):
              row[f"kjoint_{exponent}"] for row in rows}
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "mathtext.fontset": "stix", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.7,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.1))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.14, top=0.885,
                        wspace=0.23, hspace=0.34)
    for ax, (family, title) in zip(axes.flat, FAMILY_NAMES.items()):
        for method, (label, color, linestyle) in METHODS.items():
            # Rows are conditioning levels; columns are the five paired draws.
            values = np.array([
                [counts[family, replicate, kappa, method]
                 for replicate in range(NUMBER_OF_DRAWS)]
                for kappa in KAPPAS
            ])
            lower, center, upper = np.percentile(
                values, [25, 50, 75], axis=1, method="linear"
            )
            ax.fill_between(
                KAPPAS, lower, upper, color=color, alpha=0.075, linewidth=0,
            )
            ax.plot(
                KAPPAS, center, color=color, linestyle=linestyle,
                linewidth=1.55, marker="o", markersize=3, label=label,
            )
        ax.set(
            xscale="log", yscale="log", title=title,
            xlabel=r"Prescribed condition ratio $L/\mu$",
            ylabel="Accepted iterations",
        )
        ax.set_xticks([2, 100, 10000, 1000000])
        ax.set_xticklabels(["2", r"$10^2$", r"$10^4$", r"$10^6$"])
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(axis="y", which="major", color="#e3e3e3", linewidth=0.65)
        ax.set_xlim(1.6, 1.25e6)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=4, frameon=False,
        fontsize=9, bbox_to_anchor=(0.53, 0.015), columnspacing=2.0,
    )
    for extension in ("png", "pdf"):
        fig.savefig(figures / f"median_iterations_1e-{exponent}.{extension}", dpi=160)
    plt.close(fig)
