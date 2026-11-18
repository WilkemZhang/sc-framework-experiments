"""Draw convergence trajectories from the saved error history."""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Plot labels and styles.
NAMES = {"ridge": "Ridge", "asymmetric": "Asymmetric squared loss",
         "logistic": "Binary logistic", "logsumexp": "Log-sum-exp"}
PLOT_STYLES = {
    "adanag_g12": (r"AdaNAG-G$_{12}$", "#6a3d9a", "--"),
    "adanag_g_half": (r"AdaNAG-G$_{1/2}$", "#b8860b", "--"),
    "ac_fgm_a01": ("AC-FGM", "#008b45", "--"),
    "adaagm55": (r"AdaAGM$_{5.5}$", "#c51b7d", "--"),
    "fg_original": ("VC-FG", "#0066ff", "-"),
    "widecap": ("CC-FG", "#e31a1c", "-"),
    "nag_sc_oracle": ("NAG-SC (oracle)", "#444444", "--"),
}


def plot_convergence():
    """Plot both relative errors for the four problem families."""
    from main import TARGETS

    root = Path(__file__).resolve().parent
    fig, axes = plt.subplots(4, 2, figsize=(8, 9), layout="constrained")
    ylabel = (r"$\Delta_k/\Delta_0$",
              r"$\Vert x_k-x^\star\Vert^2/\Vert x_0-x^\star\Vert^2$")
    with np.load(root / "results" / "curves.npz") as history:
        for row, (family, name) in enumerate(NAMES.items()):
            traces = {method: history[f"{family}__{method}"] for method in PLOT_STYLES}
            # Crop the view if a single long run would compress the other curves.
            ends = sorted(len(t) - 1 for t in traces.values())
            if ends[-1] > 2 * ends[-2]:
                xmax = math.ceil(1.05 * ends[-2])
                traces = {method: t[:xmax+1] for method, t in traces.items()}
            else:
                xmax = ends[-1]

            # The columns of each trace are iteration, objective error, and distance error.
            for col, ax in enumerate(axes[row], start=1):
                for method, (label, color, style) in PLOT_STYLES.items():
                    t = traces[method]
                    ax.semilogy(t[:, 0], t[:, col], nonpositive="mask",
                                label=label, color=color, ls=style, lw=1.05,
                                zorder=4 if method in ("fg_original", "widecap") else 2)
                for target in TARGETS:
                    ax.axhline(target, color="gray", ls=":", lw=.7, zorder=0)
                ax.set(xlabel="$k$", ylabel=ylabel[col-1], xlim=(0, xmax))
            axes[row, 0].set_title(name + r"    $\kappa_f=10^6$", loc="left")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False)
    out = root / "figures" / "convergence"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"all_families_grid.{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    plot_convergence()
