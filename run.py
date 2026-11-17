"""Record the accepted iteration indices for three prescribed error thresholds."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from problems import FAMILIES, KAPPAS, Problem
from solvers import METHODS, run_solver


OUT = Path(__file__).resolve().parent / "results"
REPLICATES = range(5)
EXPONENTS = (8, 12, 15)
TARGETS = tuple(10.0 ** -power for power in EXPONENTS)
MAX_STEPS = 150_000


def run_base(job):
    family, replicate = job
    rows, instances = [], []
    references, curves = {}, {}
    with threadpool_limits(1):
        problem = Problem(family, replicate)
        for kappa in KAPPAS:
            problem.set_kappa(kappa)
            xref, gref, pref, gap0, cert = problem.reference()
            gnorm = np.linalg.norm(gref)
            radius = gnorm / problem.mu
            d0 = np.linalg.norm(problem.x0 - xref)
            if gap0 <= 0 or d0 <= radius or (radius / (d0 - radius))**2 >= TARGETS[-1]:
                raise ArithmeticError("Reference accuracy is insufficient for the distance target")

            instance = dict(family=family, replicate=replicate, kappa=kappa)
            instances.append(dict(
                **instance, n=problem.n, features=problem.p, dimension=problem.dim,
                C=problem.C, lam=getattr(problem, "lam", None), mu=problem.mu, L=problem.L,
                value_at_ref=problem.oracle(xref)[0], reference_grad_norm=gnorm,
                initial_gap_to_ref=gap0, initial_distance_to_ref=d0,
                reference_gap_bound=cert, reference_radius=radius,
                certificate_relative=cert/gap0, radius_relative=radius/d0,
            ))
            references[f"{family}__{replicate}__{kappa}"] = xref
            display = replicate == 0 and kappa == 1_000_000
            for method in METHODS:
                trace = []
                first_k = {metric: [] for metric in ("f", "d", "joint")}
                rf, rd = 1.0, 1.0

                def monitor(k, x):
                    nonlocal rf, rd
                    if k:
                        diff = x - xref
                        delta = problem.bregman(x, xref, pref) + gref @ diff
                        rf = max(delta/gap0, (delta + cert)/(gap0 + cert))
                        rd = ((np.linalg.norm(diff) + radius)/(d0 - radius))**2
                    if not (np.isfinite(rf) and np.isfinite(rd) and min(rf, rd) >= 0):
                        raise ArithmeticError((family, replicate, kappa, method, k, rf, rd))
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


def save_csv(path, rows):
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    jobs = [(family, replicate) for replicate in REPLICATES for family in FAMILIES]
    if args.workers == 1:
        records = list(map(run_base, jobs))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            records = list(pool.map(run_base, jobs))
    OUT.mkdir(exist_ok=True)
    save_csv(OUT/"runs.csv", [r for rows, _, _, _ in records for r in rows])
    save_csv(OUT/"instances.csv", [r for _, rows, _, _ in records for r in rows])
    references = {k: v for _, _, refs, _ in records for k, v in refs.items()}
    curves = {k: v for _, _, _, traces in records for k, v in traces.items()}
    np.savez_compressed(OUT/"references.npz", **references)
    np.savez_compressed(OUT/"curves.npz", **curves)


if __name__ == "__main__":
    main()
