"""
Bootstrap-averaged assignment probabilities (`n_bootstrap`): gains and costs on
synthetic data with known clusters.

A single fit's assignment probabilities treat the fitted parameters as exact.
`n_bootstrap=B` refits on B bootstrap resamples and averages the probabilities
of the fit and the refits (components matched by Hungarian assignment), so the
probabilities also carry the uncertainty of the fit. This script measures what
that buys and what it costs.

Scenarios (10 replicates each, K = 2 unless stated):
  dmm         well-specified DMM data, overlapping clusters, N = 300, S = 30
  dmm_easy    as dmm, better separated (larger component shifts)
  dmm_small   as dmm with N = 100: the fit itself is noisier
  dmm_k3      as dmm with three clusters
  lognorm0    logistic-normal data, independent features (N = 300, S = 60);
              misspecified for the DMM, which is then over-confident
  lognorm6    as lognorm0 with within-module correlation 0.6 (strongly
              over-confident single fits)

Fits:
  dmm, dmm_onestep               DirichletMixture, untempered / temperature="one-step"
  dmm_bag4, dmm_bag8, dmm_bag16  DirichletMixture(n_bootstrap=4/8/16)
  dmm_bag8_onestep               n_bootstrap=8 and temperature="one-step"
  dtmm, dtmm_bag8                DirichletTreeMixture(n_clades=S // 3), without / with n_bootstrap=8

Metrics against the true labels (clusters matched by permutation): accuracy,
mean maximum probability (confidence), expected calibration error (ECE, 10
bins), log-loss, Brier score, and "wrong splits" (accuracy below
1/K + 0.4 (1 - 1/K)). Lower ECE, log-loss and Brier are better.

Runtime is measured separately in one process, single-threaded and
sequentially (no other fits running), over N, S and B.

Run: OMP_NUM_THREADS=1 venv/bin/python validation/bagging/compare.py [quality|runtime]
"""

import json
import sys
import time
from itertools import permutations
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "correlation"))
from experiment import lognormal_modules  # noqa: E402

from pydmm import DirichletMixture, DirichletTreeMixture  # noqa: E402

OUT = HERE / "results"
N_REPS = 10
SCENARIOS = ["dmm", "dmm_easy", "dmm_small", "dmm_k3", "lognorm0", "lognorm6"]
FITS = {
    "dmm": ("dmm", {}),
    "dmm_onestep": ("dmm", {"temperature": "one-step"}),
    "dmm_bag4": ("dmm", {"n_bootstrap": 4}),
    "dmm_bag8": ("dmm", {"n_bootstrap": 8}),
    "dmm_bag16": ("dmm", {"n_bootstrap": 16}),
    "dmm_bag8_onestep": ("dmm", {"n_bootstrap": 8, "temperature": "one-step"}),
    "dtmm": ("dtmm", {}),
    "dtmm_bag8": ("dtmm", {"n_bootstrap": 8}),
}


def dm_mixture(rng, n=300, s=30, k=2, depth=3000, spread=0.45, theta=40.0):
    """Well-specified DMM data (as validation/correlation/experiment.py:dmm_data,
    with the component spread and N as parameters)."""
    base = rng.gamma(1.0, 1.0, size=s) + 0.2
    alphas = []
    for _ in range(k):
        a = base * np.exp(rng.normal(0, spread, size=s))
        alphas.append(a / a.sum() * theta)
    labels = rng.integers(0, k, size=n)
    X = np.empty((n, s), dtype=np.int64)
    for i in range(n):
        X[i] = rng.multinomial(depth, rng.dirichlet(alphas[labels[i]]))
    return X, labels


def make(kind, rep, n=None, s=None):
    rng = np.random.default_rng(9000 + 100 * SCENARIOS.index(kind) + rep
                                if kind in SCENARIOS else rep)
    if kind == "dmm":
        return dm_mixture(rng, spread=0.3)
    if kind == "dmm_easy":
        return dm_mixture(rng, spread=0.6)
    if kind == "dmm_small":
        return dm_mixture(rng, n=100, spread=0.3)
    if kind == "dmm_k3":
        return dm_mixture(rng, k=3, spread=0.4)
    if kind == "lognorm0":
        return lognormal_modules(rng, 0.0, delta=1.0)
    if kind == "lognorm6":
        return lognormal_modules(rng, 0.6, delta=0.8)
    raise ValueError(kind)


def align(z, labels):
    k = z.shape[1]
    best, best_acc = None, -1.0
    for perm in permutations(range(k)):
        acc = np.mean(z[:, list(perm)].argmax(1) == labels)
        if acc > best_acc:
            best, best_acc = list(perm), acc
    return z[:, best]


def metrics(z, labels, n_bins=10):
    z = align(z, labels)
    k = z.shape[1]
    conf = z.max(1)
    correct = z.argmax(1) == labels
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = sum(abs(correct[bins == b].mean() - conf[bins == b].mean()) * np.mean(bins == b)
              for b in range(n_bins) if np.any(bins == b))
    p_true = np.clip(z[np.arange(len(labels)), labels], 1e-12, 1)
    onehot = np.eye(k)[labels]
    return {"accuracy": float(correct.mean()), "confidence": float(conf.mean()),
            "ece": float(ece), "log_loss": float(-np.log(p_true).mean()),
            "brier": float(((z - onehot) ** 2).sum(1).mean())}


def model(family, k, s, rep, **kw):
    if family == "dmm":
        return DirichletMixture(n_components=k, random_state=rep, **kw)
    return DirichletTreeMixture(n_components=k, random_state=rep, n_clades=s // 3, **kw)


def job(args):
    kind, rep = args
    X, labels = make(kind, rep)
    X = np.ascontiguousarray(X.astype(np.int32))
    k = int(labels.max()) + 1
    out = {"kind": kind, "rep": rep, "k": k}
    for name, (family, kw) in FITS.items():
        t = time.process_time()
        m = model(family, k, X.shape[1], rep, **kw).fit(X)
        out[name] = metrics(m.result_.group_assignments, labels)
        out[name]["cpu_s"] = time.process_time() - t
    return out


def quality():
    jobs = [(kind, rep) for kind in SCENARIOS for rep in range(N_REPS)]
    with Pool() as pool:
        results = pool.map(job, jobs)
    summary = {"n_reps": N_REPS, "scenarios": {}}
    for kind in SCENARIOS:
        runs = [r for r in results if r["kind"] == kind]
        k = runs[0]["k"]
        threshold = 1 / k + (1 - 1 / k) * 0.4
        s = {}
        print(kind)
        for f in FITS:
            s[f] = {m: float(np.mean([r[f][m] for r in runs])) for m in runs[0][f]}
            s[f]["wrong_splits"] = int(sum(r[f]["accuracy"] < threshold for r in runs))
            print(f"  {f:17s} acc={s[f]['accuracy']:.3f} conf={s[f]['confidence']:.3f} "
                  f"ECE={s[f]['ece']:.3f} LL={s[f]['log_loss']:.3f} "
                  f"Brier={s[f]['brier']:.3f} wrong={s[f]['wrong_splits']} "
                  f"cpu={s[f]['cpu_s']:.2f}s")
        summary["scenarios"][kind] = s
        sys.stdout.flush()
    with open(OUT / "compare.json", "w") as fh:
        json.dump(summary, fh, indent=1)


def runtime(n_repeat=3):
    """Median wall time of single fits, sequential and single-threaded."""
    cases = [("dmm", 300, 60, 2), ("dmm", 1000, 60, 2), ("dmm", 3000, 60, 2),
             ("dmm", 1000, 200, 4), ("dtmm", 300, 60, 2), ("dtmm", 1000, 60, 2),
             ("dtmm", 1000, 200, 4)]
    rows = []
    for family, n, s, k in cases:
        rng = np.random.default_rng(n + s + k)
        X, _ = dm_mixture(rng, n=n, s=s, k=k, spread=0.3)
        X = np.ascontiguousarray(X.astype(np.int32))
        base = None
        for B in (0, 4, 8, 16):
            times = []
            for r in range(n_repeat if B <= 8 else 1):
                t = time.perf_counter()
                model(family, k, s, 0, n_bootstrap=B).fit(X)
                times.append(time.perf_counter() - t)
            sec = float(np.median(times))
            base = sec if B == 0 else base
            rows.append({"model": family, "N": n, "S": s, "K": k, "B": B,
                         "seconds": sec, "ratio": sec / base})
            print(f"{family:5s} N={n:5d} S={s:4d} K={k} B={B:2d} {sec:7.2f}s  x{sec / base:5.1f}")
            sys.stdout.flush()
    with open(OUT / "runtime.json", "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    what = sys.argv[1:] or ["quality", "runtime"]
    if "quality" in what:
        quality()
    if "runtime" in what:
        runtime()
