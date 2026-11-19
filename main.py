"""Run the SC Framework experiments or redraw figures from saved results."""

import csv
import json
import math
import os
import platform
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

# Set numerical-library thread limits before importing NumPy and SciPy.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")
import numpy as np
import scipy

from problems import EXPONENTS, FAMILIES, KAPPAS, NUMBER_OF_DRAWS, Problem
from solvers import METHODS, run_solver
from plot_convergence import plot_convergence
from plot_iterations_by_kappa import plot_iterations_by_kappa


MODE = "full"                         # "full" or "reference"
OUTPUT_FOLDER = Path("output") / MODE
SAVED_COUNTS = Path("reference_runs.csv")  # Used only in reference mode
MAX_STEPS = 150_000


def measure_method(problem, method, reference, keep_trajectory):
    """Record the first accepted iterate meeting each error target."""
    xref, _, gap0 = reference
    distance0 = np.linalg.norm(problem.x0 - xref)
    record = {"method": method, "rf": 1.0, "rd": 1.0}
    for exponent in EXPONENTS:
        for name in ("kf", "kd", "kjoint"):
            record[f"{name}_{exponent}"] = math.inf
    trajectory = []

    def record_errors(k, x):
        if k == 0:
            objective_error = distance_error = 1.0
        else:
            gap = problem.bregman(x, xref)
            objective_error = gap / gap0
            distance_error = (np.linalg.norm(x - xref) / distance0 )**2
        record["rf"], record["rd"] = objective_error, distance_error
        if keep_trajectory:
            trajectory.append((k, objective_error, distance_error))
        for exponent in EXPONENTS:
            target = 10.0**-exponent
            errors = (objective_error, distance_error,
                      max(objective_error, distance_error))
            for name, error in zip(("kf", "kd", "kjoint"), errors):
                key = f"{name}_{exponent}"
                if error <= target and record[key] == math.inf:
                    record[key] = k
        return max(objective_error, distance_error) <= 10.0**-EXPONENTS[-1]

    result = run_solver(method, problem.oracle, problem.x0, problem.mu, problem.L,
                        MAX_STEPS, bregman=problem.bregman, monitor=record_errors)
    record.update(status=result["status"], iterations=result["iterations"])
    return record, np.asarray(trajectory)


def save_csv(path, rows):
    """An empty crossing cell means the target was not reached."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: "" if value == math.inf else value
                         for key, value in row.items()} for row in rows)


def run_experiments():
    """Run the same data and starting point for every method and condition ratio."""
    rows, instances, references, trajectories = [], [], {}, {}
    for replicate in range(NUMBER_OF_DRAWS):
        for family in FAMILIES:
            problem = Problem(family, replicate)
            for kappa in KAPPAS:
                problem.set_kappa(kappa)
                reference = problem.reference()
                xref, _, gap0 = reference
                distance0 = np.linalg.norm(problem.x0 - xref)
                instances.append(dict(
                    family=family, replicate=replicate, kappa=kappa,
                    n=problem.n, features=problem.p, dimension=problem.dim,
                    C=problem.C, lam=getattr(problem, "lam", None),
                    mu=problem.mu, L=problem.L,
                    value_at_ref=problem.oracle(xref)[0],
                    initial_gap_to_ref=gap0, initial_distance_to_ref=distance0))
                references[f"{family}__{replicate}__{kappa}"] = xref
                keep_trajectory = replicate == 0 and kappa == 1_000_000
                for method in METHODS:
                    record, trajectory = measure_method(problem, method, reference,
                                                        keep_trajectory)
                    rows.append(dict(family=family, replicate=replicate, kappa=kappa,
                                     **record))
                    if keep_trajectory:
                        trajectories[f"{family}__{method}"] = trajectory
            print(f"Finished {family}, draw {replicate + 1}/{NUMBER_OF_DRAWS}", flush=True)

    results = OUTPUT_FOLDER / "results"
    results.mkdir(parents=True, exist_ok=True)
    save_csv(results / "runs.csv", rows)
    save_csv(results / "instances.csv", instances)
    np.savez_compressed(results / "references.npz", **references)
    np.savez_compressed(results / "curves.npz", **trajectories)
    return rows, trajectories


def read_saved_counts(path):
    """Read saved iteration counts for the median iteration figures."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for name in ("replicate", "kappa", "iterations"):
            row[name] = int(row[name])
        for exponent in EXPONENTS:
            for name in ("kf", "kd", "kjoint"):
                key = f"{name}_{exponent}"
                row[key] = int(row[key]) if row[key] else math.inf
    return rows


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    figures = OUTPUT_FOLDER / "figures"
    figures.mkdir(exist_ok=True)

    figure_names = []
    if MODE == "full":
        rows, trajectories = run_experiments()
        plot_convergence(trajectories, figures)
        figure_names.append("all_families_grid")
    else:
        rows = read_saved_counts(SAVED_COUNTS)
        (OUTPUT_FOLDER / "results").mkdir(exist_ok=True)
        save_csv(OUTPUT_FOLDER / "results" / "runs.csv", rows)

        curves_file = SAVED_COUNTS.with_name("curves.npz")
        if curves_file.is_file():
            with np.load(curves_file) as trajectories:
                plot_convergence(trajectories, figures)
            figure_names.append("all_families_grid")

    for exponent in EXPONENTS:
        plot_iterations_by_kappa(rows, exponent, figures)
        figure_names.append(f"median_iterations_1e-{exponent}")

    # Save a short record of the environment used for this execution.
    environment = {"mode": MODE, "python": platform.python_version(),
                   "platform": platform.platform(), "numpy": np.__version__,
                   "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
                   "workers": 1, "draws": NUMBER_OF_DRAWS, "max_steps": MAX_STEPS,
                   "condition_ratios": KAPPAS, "targets": [10.0**-p for p in EXPONENTS],
                   "saved_counts": str(SAVED_COUNTS) if MODE == "reference" else None}
    (OUTPUT_FOLDER / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    print(f"Finished. Results and figures are in {OUTPUT_FOLDER.resolve()}")

    archive = Path("sc_framework_outputs.zip")
    data_files = ["runs.csv"]
    if MODE == "full":
        data_files.extend(["instances.csv", "references.npz", "curves.npz"])

    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        bundle.write(OUTPUT_FOLDER / "environment.json", "environment.json")
        for name in data_files:
            bundle.write(OUTPUT_FOLDER / "results" / name, f"results/{name}")
        for name in figure_names:
            for ext in ("pdf", "png"):
                bundle.write(figures / f"{name}.{ext}", f"figures/{name}.{ext}")
    print(f"Results archive: {archive.resolve()}")


if __name__ == "__main__":
    main()
