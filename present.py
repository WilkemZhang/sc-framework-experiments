"""Rebuild iteration tables, convergence plots, and conditioning comparisons."""
import csv
import math
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator
import numpy as np

from problems import FAMILIES, KAPPAS


ROOT = Path(__file__).resolve().parent
NAMES = {"ridge": "Ridge", "asymmetric": "Asymmetric squared loss",
         "logistic": "Binary logistic", "logsumexp": "Log-sum-exp"}
EXPONENTS = (8, 12, 15)
METHODS = {
    "adanag_g12": (r"AdaNAG-G$_{12}$", "#6a3d9a", "--"),
    "adanag_g_half": (r"AdaNAG-G$_{1/2}$", "#b8860b", "--"),
    "ac_fgm_a01": ("AC-FGM", "#008b45", "--"),
    "adaagm55": (r"AdaAGM$_{5.5}$", "#c51b7d", "--"),
    "fg_original": ("VC-FG", "#0066ff", "-"),
    "widecap": ("CC-FG", "#e31a1c", "-"),
    "nag_sc_oracle": ("NAG-SC (oracle)", "#444444", "--"),
}
TABLE_LABELS = {
    "adanag_g12": r"G$_{12}$", "adanag_g_half": r"G$_{1/2}$",
    "ac_fgm_a01": "AC-FGM", "adaagm55": r"AGM$_{5.5}$",
    "fg_original": "VC-FG", "widecap": "CC-FG", "nag_sc_oracle": "NAG-SC",
}
CSV_LABELS = {
    "adanag_g12": "AdaNAG-G12", "adanag_g_half": "AdaNAG-G1/2",
    "adaagm55": "AdaAGM5.5",
}


def read_iterations():
    """A missing first crossing contributes +infinity to the five-run median."""
    with (ROOT / "results/runs.csv").open(newline="") as stream:
        runs = list(csv.DictReader(stream))
    groups = {}
    for row in runs:
        family, kappa, method = row["family"], int(row["kappa"]), row["method"]
        for exponent in EXPONENTS:
            key = (family, kappa, method, exponent)
            group = groups.setdefault(key, {})
            replicate = int(row["replicate"])
            if replicate in group:
                raise ValueError(f"Duplicate observation: {key}, {replicate}")
            crossings = []
            for metric in ("kf", "kd", "kjoint"):
                raw = row[f"{metric}_{exponent}"]
                crossings.append(int(raw) if raw else math.inf)
            if max(crossings[:2]) > crossings[2]:
                raise ValueError(f"Joint crossing precedes an individual crossing: {key}")
            group[replicate] = crossings
    expected = {(f, k, m, p) for f in FAMILIES for k in KAPPAS
                for m in METHODS for p in EXPONENTS}
    if set(groups) != expected or any(set(g) != set(range(5)) for g in groups.values()):
        raise ValueError("The input does not contain five runs per grid cell and method")
    return groups


def iteration_text(k):
    return f"{k:,.0f}" if math.isfinite(k) else r"$\infty$"


def write_table(path, caption, label, header, rows, note, columns):
    tex = [r"\begin{table}[H]", r"\centering\small", r"\caption{" + caption + "}",
           r"\label{" + label + "}", r"\footnotesize", r"\setlength{\tabcolsep}{3.2pt}",
           r"\begin{tabular}{@{}" + columns + r"@{}}",
           r"\toprule", header, r"\midrule", *rows, r"\bottomrule", r"\end{tabular}",
           r"\par\smallskip", r"\begin{minipage}{.79\linewidth}\footnotesize " +
           note + r"\end{minipage}", r"\end{table}", ""]
    path.write_text("\n".join(tex), encoding="utf-8")


def write_tables(groups):
    out = ROOT / "tables"
    out.mkdir(exist_ok=True)
    joint = {}
    with (out / "median_iterations.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["family", "kappa", "method", "tolerance", "median_k_f",
                         "median_k_d", "median_k_joint", "n_hit_f", "n_hit_d",
                         "n_hit_joint", "n_instances"])
        for f in FAMILIES:
            for k in KAPPAS:
                for m in METHODS:
                    for p in EXPONENTS:
                        key = (f, k, m, p)
                        measures = list(zip(*groups[key].values()))
                        medians = [median(v) for v in measures]
                        reached = [sum(math.isfinite(i) for i in v) for v in measures]
                        joint[key] = medians[2]
                        writer.writerow([f, k, m, f"1e-{p}", *medians, *reached, 5])
    # Each method's median is computed before the minimum over methods is taken.
    columns = " & ".join(NAMES.values()) + r" \\"
    note = (r"Parentheses give the median first iteration $k$ at which both "
            r"relative errors in~\eqref{eq:num-relative-errors} meet $\varepsilon$ "
            r"in the same iterate, over five runs. "
            r"All methods tied for the smallest finite median are listed. "
            r"NAG-SC is the oracle baseline supplied with the global bounds.")
    rows = []
    with (out / "winners.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["kappa", "tolerance", *NAMES.values()])
        for k in KAPPAS:
            if rows:
                rows.append(r"\addlinespace[2pt]")
            k_text = str(k) if k < 100 else rf"$10^{{{int(math.log10(k))}}}$"
            for i, p in enumerate(EXPONENTS):
                cells = [k_text if i == 0 else "", rf"$10^{{-{p}}}$"]
                csv_cells = [k, f"1e-{p}"]
                for f in NAMES:
                    best = min(joint[f, k, m, p] for m in METHODS)
                    winners = [m for m in METHODS if joint[f, k, m, p] == best]
                    if math.isfinite(best):
                        labels = "/".join(TABLE_LABELS[m] for m in winners)
                        cells.append(f"{labels} ({iteration_text(best)})")
                        names = "/".join(CSV_LABELS.get(m, METHODS[m][0]) for m in winners)
                        csv_cells.append(f"{names} ({int(best)})")
                    else:
                        cells.append("---")
                        csv_cells.append("")
                rows.append(" & ".join(cells) + r" \\")
                writer.writerow(csv_cells)
    write_table(out / "winners.tex", "Fastest method (median iteration count).",
                "tab:winners", r"$\kappa_f$ & $\varepsilon$ & " + columns,
                rows, note, "llcccc")


def draw_panel(ax, visible, col, xmax, cropped, compact=False):
    ratios = [r"\frac{\Delta_k}{\Delta_0}",
              r"\frac{\Vert x_k-x^\star\Vert^2}{\Vert x_0-x^\star\Vert^2}"]
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
    ax.set_ylabel("$" + ratios[col - 1] + "$", fontsize=12 if compact else 14, labelpad=5)
    ax.set_xlabel(r"$k$", fontsize=11 if compact else 13, labelpad=1)
    ax.set_xlim(0, max(1., xmax) * (1 if cropped else 1.018))
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.ticklabel_format(style="plain", axis="x", useOffset=False)
    ax.set_yticks([10.0**e for e in ticks])
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(length=3, width=.65, labelsize=8.5 if compact else 9)


def write_figures(history, groups):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "mathtext.fontset": "stix", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": .65,
                         "pdf.fonttype": 42})
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    singles = out / "single"
    singles.mkdir(exist_ok=True)
    grid, grid_axes = plt.subplots(4, 2, figsize=(7.8, 8.4))
    for row, f in enumerate(NAMES):
        traces = {m: history[f"{f}__{m}"] for m in METHODS}
        for m, t in traces.items():
            if (t.ndim != 2 or t.shape[1] != 3 or not np.isfinite(t).all()
                    or (t < 0).any() or not np.array_equal(t[:, 0], np.arange(len(t)))):
                raise ValueError(f"Invalid iteration trace for {f}, {m}")
            for p in EXPONENTS:
                mask = t[:, 1:] <= 10.0**-p
                hits = [np.flatnonzero(v) for v in (mask[:, 0], mask[:, 1], mask.all(axis=1))]
                crossed = [int(v[0]) if v.size else math.inf for v in hits]
                if crossed != groups[f, 1_000_000, m, p][0]:
                    raise ValueError(f"Trace and tabulated first-hit iterations disagree: {f}, {m}")
        # One unusually long run should not compress the six other trajectories.
        ends = sorted(len(t)-1 for t in traces.values())
        cropped = ends[-1] > 2 * ends[-2]
        xmax = math.ceil(1.05 * ends[-2]) if cropped else ends[-1]
        visible = {m: t[:xmax+1] for m, t in traces.items()}
        layouts = [
            ((1, 2), "pair", out),
            ((1,), "objective", singles),
            ((2,), "distance", singles),
        ]
        for columns, suffix, directory in layouts:
            paired = len(columns) == 2
            fig, axes = plt.subplots(
                1, len(columns), squeeze=False,
                figsize=(7.8, 3.55) if paired else (6.6, 4.6),
            )
            for ax, col in zip(axes[0], columns):
                draw_panel(ax, visible, col, xmax, cropped)
            ratio_kind = r"$\kappa_f=10^6$"
            fig.suptitle(NAMES[f] + "     " + ratio_kind,
                         x=.095 if paired else .14, y=.97, ha="left", fontsize=12)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=4 if paired else 3, frameon=False, fontsize=8.5,
                       bbox_to_anchor=(.52, .002), columnspacing=1.8, handlelength=2.8)
            fig.subplots_adjust(left=.105 if paired else .16, right=.987,
                                bottom=.285 if paired else .30, top=.855, wspace=.36)
            for ext in ("pdf", "png"):
                fig.savefig(directory / f"{f}_{suffix}.{ext}", dpi=180)
            plt.close(fig)

        for col, ax in enumerate(grid_axes[row], start=1):
            draw_panel(ax, visible, col, xmax, cropped, compact=True)
        grid_axes[row, 0].set_title(NAMES[f] + r"     $\kappa_f=10^6$",
                                    loc="left", fontsize=10.5, pad=6)
    handles, labels = grid_axes[0, 0].get_legend_handles_labels()
    grid.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
                fontsize=8.5, bbox_to_anchor=(.52, .005), columnspacing=1.8,
                handlelength=2.8)
    grid.subplots_adjust(left=.105, right=.987, bottom=.11, top=.956,
                         wspace=.36, hspace=.65)
    for ext in ("pdf", "png"):
        grid.savefig(out / f"all_families_grid.{ext}", dpi=180)
    plt.close(grid)


def write_median_iterations(groups):
    """Plot five-draw median joint crossing indices and interquartile ranges."""
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    kappas = np.asarray(KAPPAS)
    titles = ("Ridge regression", "Asymmetric squared loss",
              "Binary logistic regression", "Log-sum-exp")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "mathtext.fontset": "stix", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": .65,
                         "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.2))
        for ax, family, title in zip(axes.flat, NAMES, titles):
            for method, (label, color, style) in METHODS.items():
                crossings = np.asarray([
                    [groups[family, kappa, method, 15][replicate][2]
                     for replicate in range(5)] for kappa in KAPPAS
                ])
                lower, medians, upper = np.quantile(
                    crossings, [.25, .5, .75], axis=1, method="linear"
                )
                ax.fill_between(kappas, lower, upper, color=color,
                                alpha=.10, linewidth=0, zorder=1)
                ax.loglog(kappas, medians, color=color, ls=style,
                          lw=1.05, marker="o", markersize=3.3, label=label,
                          zorder=4 if method in ("fg_original", "widecap") else 2)
            ax.set_title(title, fontsize=11.5, pad=8)
            ax.set_xlabel(r"Prescribed condition ratio $L/\mu$", fontsize=10.5)
            ax.set_ylabel("Accepted iterations", fontsize=10.5)
            ax.set_xticks([2, 100, 10000, 1000000],
                          ["2", r"$10^2$", r"$10^4$", r"$10^6$"])
            ax.grid(axis="y", which="major", color="#dddddd", lw=.6)
            ax.tick_params(labelsize=9.5, length=3, width=.65)
            ax.margins(x=.02, y=.06)
            ax.set_ylim(bottom=10.0**math.floor(math.log10(ax.dataLim.ymin)))
        fig.suptitle("Median iterations to reach both relative errors "
                     r"$\leq 10^{-15}$", y=.983, fontsize=12, fontweight="bold")
        fig.text(.5, .938,
                 "Lines: medians. Shading: interquartile range across five matched "
                 "draws; not confidence intervals.",
                 ha="center", fontsize=9.5, color="#444444")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
                   fontsize=9.5, bbox_to_anchor=(.52, .012), columnspacing=1.5,
                   handlelength=2.3)
        fig.subplots_adjust(left=.105, right=.985, bottom=.185, top=.85,
                            wspace=.35, hspace=.60)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"median_iterations_1e-15.{ext}", dpi=180)
        plt.close(fig)



if __name__ == "__main__":
    groups = read_iterations()
    write_tables(groups)
    with np.load(ROOT / "results/curves.npz") as history:
        write_figures(history, groups)
    write_median_iterations(groups)
