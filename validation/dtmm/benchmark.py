"""
Runtime of DirichletTreeMixture versus DirichletMixture.

Single process, single thread. Data: Dirichlet-tree mixtures with clades of 6
taxa (theta_root = 30, depth 5,000). Grid over N (S = 60), over S (N = 300)
and over K (N = 300, S = 60). Median of 3 runs (1 run for the largest sizes).

Fits:
  dmm         DirichletMixture
  dtmm_star   DirichletTreeMixture, star tree, no node selection (the DMM
              through the tree code)
  dtmm_clade  clade tree (taxonomy-like: S/6 + 1 internal nodes), selection on
  dtmm_data   default data-built binary tree (S - 1 nodes), selection on
  dtmm_data_all  same, selection off

Run (nothing else running):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 venv/bin/python validation/dtmm/benchmark.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from pydmm import DirichletMixture, DirichletTreeMixture, Tree

OUT = Path(__file__).parent / "results"
CLADE_SIZE = 6


def data(n, s, k, seed=0, theta_root=30.0, depth=5000, delta=0.8):
    rng = np.random.default_rng(seed)
    n_clades = s // CLADE_SIZE
    p0 = np.exp(rng.normal(0, 0.7, n_clades))
    roots = []
    for g in range(k):
        shift = np.zeros(n_clades)
        if g:
            shift[rng.choice(n_clades, 4, replace=False)] = [delta, -delta, delta, -delta]
        r = np.exp(np.log(p0) + shift)
        roots.append(r / r.sum() * theta_root)
    within = [np.exp(rng.normal(0, 1.0, CLADE_SIZE)) for _ in range(n_clades)]
    within = [w / w.sum() * 50.0 for w in within]
    labels = rng.integers(0, k, n)
    X = np.zeros((n, s), dtype=np.int32)
    for i in range(n):
        top = rng.dirichlet(roots[labels[i]])
        p = np.concatenate([top[c] * rng.dirichlet(within[c]) for c in range(n_clades)])
        X[i] = rng.multinomial(depth, p / p.sum())
    parents = np.full(s + n_clades + 1, s + n_clades)
    parents[-1] = -1
    for c in range(n_clades):
        parents[c * CLADE_SIZE:(c + 1) * CLADE_SIZE] = s + c
    return X, Tree.from_parents(parents, s), labels


def timed(make, X, reps):
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        make().fit(X)
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def run(n, s, k):
    X, clade_tree, _ = data(n, s, k)
    data_tree = Tree.from_correlation(X)
    reps = 1 if n * s >= 300 * 240 * 2 else 3
    fits = {
        "dmm": lambda: DirichletMixture(n_components=k),
        "dtmm_star": lambda: DirichletTreeMixture(n_components=k, tree=Tree.star(s), node_selection=False),
        "dtmm_clade": lambda: DirichletTreeMixture(n_components=k, tree=clade_tree),
        "dtmm_data": lambda: DirichletTreeMixture(n_components=k, tree=data_tree),
        "dtmm_data_all": lambda: DirichletTreeMixture(n_components=k, tree=data_tree, node_selection=False),
    }
    row = {"N": n, "S": s, "K": k}
    for name, make in fits.items():
        row[name] = timed(make, X, reps)
    start = time.perf_counter()
    Tree.from_correlation(X)
    row["build_data_tree"] = time.perf_counter() - start
    return row


def main():
    OUT.mkdir(exist_ok=True)
    grid = ([(n, 60, 2) for n in (100, 300, 1000, 3000)]
            + [(300, s, 2) for s in (30, 120, 240)]
            + [(300, 60, k) for k in (3, 4)])
    rows = []
    for n, s, k in grid:
        row = run(n, s, k)
        rows.append(row)
        print("  ".join(f"{key}={val:.2f}" if isinstance(val, float) else f"{key}={val}"
                        for key, val in row.items()))
        sys.stdout.flush()
    with open(OUT / "benchmark.json", "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
