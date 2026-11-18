"""Compute the experiments, save the raw results, and draw the two figures."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from problems import FAMILIES, KAPPAS, Problem
from solvers import METHODS, run_solver


# Experiment settings.
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
REPLICATES = range(5)
EXPONENTS = (8, 12, 15)
TARGETS = tuple(10.0 ** -power for power in EXPONENTS)
MAX_STEPS = 150_000


# 1. Compute the experiment results.
def run_base(job):
    """Reuse one sampled instance across all condition numbers and methods."""
    family, replicate = job
    rows, instances = [], []
    references, curves = {}, {}
    with threadpool_limits(1):
        problem = Problem(family, replicate)
        for kappa in KAPPAS:
            problem.set_kappa(kappa)
            xref, gref, gap0, d0 = problem.reference()

            instance = dict(family=family, replicate=replicate, kappa=kappa)
            instances.append(dict(
                **instance, n=problem.n, features=problem.p, dimension=problem.dim,
                C=problem.C, lam=getattr(problem, "lam", None), mu=problem.mu, L=problem.L,
            ))
            references[f"{family}__{replicate}__{kappa}"] = xref
            display = replicate == 0 and kappa == 1_000_000
            for method in METHODS:
                trace = []
                first_k = {metric: [] for metric in ("f", "d", "joint")}
                rf, rd = 1.0, 1.0

                def monitor(k, x):
                    """Record errors at an accepted iterate; stop on the joint target."""
                    nonlocal rf, rd
                    if k:
                        diff = x - xref
                        # Compute f(x)-f(xref) without direct subtraction.
                        delta = problem.bregman(x, xref) + gref @ diff
                        rf = delta/gap0
                        rd = (np.linalg.norm(diff)/d0)**2
                    if display:
                        trace.append((k, rf, rd))
                    for metric, value in (("f", rf), ("d", rd), ("joint", max(rf, rd))):
                        hits = first_k[metric]
                        while len(hits) < len(TARGETS) and value <= TARGETS[len(hits)]:
                            hits.append(k)
                    return len(first_k["joint"]) == len(TARGETS)

                result = run_solver(method, problem.oracle, problem.x0, problem.mu,
                                    problem.L, MAX_STEPS, bregman=problem.bregman,
                                    monitor=monitor)
                row = dict(**instance, method=method, status=result["status"],
                           iterations=result["iterations"], rf=rf, rd=rd)
                for i, exponent in enumerate(EXPONENTS):
                    for metric, hits in first_k.items():
                        row[f"k{metric}_{exponent}"] = hits[i] if i < len(hits) else None
                rows.append(row)
                if display:
                    curves[f"{family}__{method}"] = np.asarray(trace)
        print(f"Completed {family}, instance {replicate+1}/{len(REPLICATES)}", flush=True)
    return rows, instances, references, curves


def run_experiments(workers):
    """Collect the results for every problem family and replicate."""
    jobs = [(family, replicate) for replicate in REPLICATES for family in FAMILIES]
    if workers == 1:
        return list(map(run_base, jobs))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_base, jobs))


# 2. Save the raw results.
def save_csv(path, rows):
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def save_results(records):
    """Save runs, instance parameters, reference points, and convergence histories."""
    rows, instances = [], []
    references, curves = {}, {}
    for run_rows, instance_rows, reference_points, histories in records:
        rows.extend(run_rows)
        instances.extend(instance_rows)
        references.update(reference_points)
        curves.update(histories)

    OUT.mkdir(exist_ok=True)
    save_csv(OUT / "runs.csv", rows)
    save_csv(OUT / "instances.csv", instances)
    np.savez_compressed(OUT / "references.npz", **references)
    np.savez_compressed(OUT / "curves.npz", **curves)


# 3. Draw the two figures from the saved data.
def make_figures():
    from plot_convergence import plot_convergence
    from plot_iterations_by_kappa import plot_iterations_by_kappa

    plot_convergence()
    plot_iterations_by_kappa()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "run", "plot"), default="all",
                        help="Run experiments, plot saved results, or do both (default).")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of independent experiment workers; use 1 for serial execution.")
    args = parser.parse_args()
    if args.mode in ("all", "run"):
        records = run_experiments(args.workers)
        save_results(records)
    if args.mode in ("all", "plot"):
        make_figures()


if __name__ == "__main__":
    main()
