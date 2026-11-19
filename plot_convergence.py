"""Convergence curves for the four objective families."""

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator

from problems import EXPONENTS
from solvers import METHODS


NAMES = {"ridge": "Ridge", "asymmetric": "Asymmetric squared loss",
         "logistic": "Binary logistic", "logsumexp": "Log-sum-exp"}


def plot_convergence(trajectories, figures):
    """Plot both relative errors for the first draw at kappa = 1,000,000."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "mathtext.fontset": "stix", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": .65,
                         "pdf.fonttype": 42})
    ratios = [r"\frac{\Delta_k}{\Delta_0}",
              r"\frac{\Vert x_k-x^\star\Vert^2}{\Vert x_0-x^\star\Vert^2}"]
    grid, grid_axes = plt.subplots(4, 2, figsize=(7.8, 8.4))
    for row, family in enumerate(NAMES):
        traces = {method: trajectories[f"{family}__{method}"] for method in METHODS}
        # Clip a single long run at 105% of the second-longest run.
        ends = sorted(int(trace[-1, 0]) for trace in traces.values())
        cropped = ends[-1] > 2 * ends[-2]
        xmax = max(1, math.ceil(1.05 * ends[-2])) if cropped else ends[-1]
        visible = {method: trace[:xmax + 1] for method, trace in traces.items()}

        for col, ax in enumerate(grid_axes[row], start=1):
            positives = [t[:, col][t[:, col] > 0] for t in visible.values()]
            minimum = min(1e-15, *(q.min() for q in positives))
            maximum = max(q.max() for q in positives)
            ymin, ymax = minimum * .36, maximum * 2.3
            ticks = sorted({0, -4, -8, -12, -15} |
                           set(range(4, math.floor(math.log10(ymax)) + 1, 4)) |
                           set(range(-20, math.floor(math.log10(ymin)) - 1, -5)))
            for m, (label, color, style) in METHODS.items():
                t = visible[m]
                y = np.ma.masked_less_equal(t[:, col], 0)
                highlighted = m in ("fg_original", "widecap")
                ax.semilogy(t[:, 0], y, color=color, ls=style, lw=1.05,
                            zorder=4 if highlighted else 2, label=label)
            for p in EXPONENTS:
                ax.axhline(10.0**-p, color="#747474", lw=.7, ls=(0, (3, 4)), zorder=0)
            ax.set_ylabel("$" + ratios[col - 1] + "$", fontsize=12, labelpad=5)
            ax.set_xlabel(r"$k$", fontsize=11, labelpad=1)
            ax.set_xlim(0, max(1., xmax) * (1 if cropped else 1.018))
            ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
            ax.ticklabel_format(style="plain", axis="x", useOffset=False)
            ax.set_yticks([10.0**e for e in ticks])
            ax.set_ylim(ymin, ymax)
            ax.yaxis.set_minor_locator(NullLocator())
            ax.tick_params(length=3, width=.65, labelsize=8.5)
        grid_axes[row, 0].set_title(NAMES[family] + r"     $\kappa_f=10^6$",
                                    loc="left", fontsize=10.5, pad=6)
    handles, labels = grid_axes[0, 0].get_legend_handles_labels()
    grid.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
                fontsize=8.5, bbox_to_anchor=(.52, .005), columnspacing=1.8,
                handlelength=2.8)
    grid.subplots_adjust(left=.105, right=.987, bottom=.11, top=.956,
                         wspace=.36, hspace=.65)
    for ext in ("pdf", "png"):
        grid.savefig(figures / f"all_families_grid.{ext}", dpi=180)
    plt.close(grid)
