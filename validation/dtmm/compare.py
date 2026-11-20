"""
DirichletMixture (DMM) versus DirichletTreeMixture (DTMM) on synthetic data.

Scenarios (N = 300, S = 60 unless stated, 10 replicates each):
  tree      Dirichlet-tree data, 10 clades of 6 taxa. The groups differ in 4
            clade totals; within-clade composition is shared. theta_root sets
            how strongly whole clades bloom (lower = stronger). K = 2.
  tree3     As tree (theta_root = 30) with three groups. K = 3.
  wide      As tree (theta_root = 30) with 40 clades (S = 240): the same 4
            differential clades are now a small part of the data.
  dmm       Plain DMM data (no tree structure), to check the DTMM does not
            lose accuracy when the tree adds nothing.
  lognorm   Logistic-normal data, 10 modules of 6 features with within-module
            correlation rho (validation/correlation/experiment.py).

Fits:
  dmm             DirichletMixture
  dmm_auto        DirichletMixture(temperature="auto")
  dtmm_true       DirichletTreeMixture with the generating clade tree
  dtmm_true_all   same, node_selection=False
  dtmm_data       DirichletTreeMixture with the default data-built tree
  oracle          (tree scenarios) assignments from the true parameters

Reports accuracy, confidence, ECE, log-loss, wrong splits (accuracy below
1/K + (1 - 1/K) * 0.4, i.e. < 0.7 for K = 2), how often BIC prefers each
DTMM over the DMM, and the number of component-specific nodes.

Runtime is measured separately, single-threaded, by benchmark.py.

Run: OMP_NUM_THREADS=1 venv/bin/python validation/dtmm/compare.py
"""

import json
import sys
from itertools import permutations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "correlation"))
from experiment import dmm_data, lognormal_modules  # noqa: E402

from pydmm import DirichletMixture, DirichletTreeMixture, Tree  # noqa: E402
from pydmm.dtmm import tree_component_log_likelihood  # noqa: E402

OUT = HERE / "results"
N_REPS = 10
N_CLADES, CLADE_SIZE = 10, 6
S = N_CLADES * CLADE_SIZE
CLADES = [list(range(c * CLADE_SIZE, (c + 1) * CLADE_SIZE)) for c in range(N_CLADES)]
SCENARIOS = [("tree", 100.0), ("tree", 30.0), ("tree", 10.0), ("tree3", 30.0),
             ("wide", 30.0), ("dmm", 0.0), ("lognorm", 0.0), ("lognorm", 0.6)]
WIDE_CLADES = 40
FITS = ["dmm", "dmm_auto", "dtmm_true", "dtmm_true_all", "dtmm_data"]


def clade_tree(n_clades=N_CLADES) -> Tree:
    s = n_clades * CLADE_SIZE
    parents = np.full(s + n_clades + 1, s + n_clades)
    parents[-1] = -1
    for c in range(n_clades):
        parents[c * CLADE_SIZE:(c + 1) * CLADE_SIZE] = s + c
    return Tree.from_parents(parents, s)


def tree_data(rng, theta_root, k=2, n=300, depth=5000, delta=0.8, theta_in=50.0,
              n_clades=N_CLADES):
    """
    Dirichlet-tree mixture data. Returns counts, labels and the generating
    parameters per component as an (n_edges, K) alpha matrix for clade_tree().
    """
    p0 = np.exp(rng.normal(0, 0.7, n_clades))
    roots = [p0 / p0.sum()]
    for g in range(1, k):
        shift = np.zeros(n_clades)
        idx = rng.choice(n_clades, 4, replace=False) if g > 1 else np.arange(4)
        shift[idx] = [delta, -delta, delta, -delta]
        r = np.exp(np.log(p0) + shift)
        roots.append(r / r.sum())
    within = [np.exp(rng.normal(0, 1.0, CLADE_SIZE)) for _ in range(n_clades)]
    within = [w / w.sum() * theta_in for w in within]
    labels = rng.integers(0, k, n)
    X = np.zeros((n, n_clades * CLADE_SIZE), dtype=np.int64)
    for i in range(n):
        top = rng.dirichlet(roots[labels[i]] * theta_root)
        p = np.concatenate([top[c] * rng.dirichlet(within[c]) for c in range(n_clades)])
        X[i] = rng.multinomial(depth, p / p.sum())
    # alpha per edge in clade_tree() order: clade nodes first, then the root
    alpha = np.column_stack([np.concatenate(within + [roots[g] * theta_root]) for g in range(k)])
    return X, labels, alpha


def make(kind, value, rep):
    rng = np.random.default_rng({"tree": 3000, "tree3": 6000, "wide": 7000, "dmm": 4000,
                                 "lognorm": 5000}[kind] + rep)
    if kind in ("tree", "tree3"):
        return tree_data(rng, value, k=3 if kind == "tree3" else 2)
    if kind == "wide":
        return tree_data(rng, value, n_clades=WIDE_CLADES)
    if kind == "dmm":
        X, labels = dmm_data(rng, s=S)
        return X, labels, None
    X, labels = lognormal_modules(rng, value)
    return X, labels, None


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
    conf = z.max(1)
    correct = z.argmax(1) == labels
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = sum(abs(correct[bins == b].mean() - conf[bins == b].mean()) * np.mean(bins == b)
              for b in range(n_bins) if np.any(bins == b))
    p_true = np.clip(z[np.arange(len(labels)), labels], 1e-12, 1)
    return {"accuracy": float(correct.mean()), "confidence": float(conf.mean()),
            "ece": float(ece), "log_loss": float(-np.log(p_true).mean())}


def job(args):
    kind, value, rep = args
    X, labels, alpha = make(kind, value, rep)
    X = np.ascontiguousarray(X.astype(np.int32))
    k = int(labels.max()) + 1
    out = {"kind": kind, "value": value, "rep": rep, "k": k}

    dmm = DirichletMixture(n_components=k, random_state=rep, temperature="auto").fit(X)
    out["dmm"] = metrics(dmm.result_.group_assignments_untempered, labels)
    out["dmm_auto"] = metrics(dmm.result_.group_assignments, labels)
    out["bic_dmm"] = dmm.result_.goodness_of_fit["BIC"]

    n_clades = WIDE_CLADES if kind == "wide" else N_CLADES
    true_tree = clade_tree(n_clades) if kind != "dmm" else Tree.star(S)
    fits = {"dtmm_true": dict(tree=true_tree),
            "dtmm_true_all": dict(tree=true_tree, node_selection=False),
            "dtmm_data": dict()}
    for name, kw in fits.items():
        m = DirichletTreeMixture(n_components=k, random_state=rep, **kw).fit(X)
        out[name] = metrics(m.result_.group_assignments, labels)
        out[f"bic_{name}"] = m.result_.goodness_of_fit["BIC"]
        out[f"specific_{name}"] = int((~m.result_.node_shared).sum())
        out[f"nodes_{name}"] = int(m.tree_.n_nodes)

    if alpha is not None:
        ll = tree_component_log_likelihood(X, clade_tree(n_clades), alpha) + np.log(np.bincount(labels, minlength=k) / len(labels))
        out["oracle"] = metrics(np.exp(ll - logsumexp(ll, axis=1, keepdims=True)), labels)
    return out


def summarise(runs):
    k = runs[0]["k"]
    threshold = 1 / k + (1 - 1 / k) * 0.4
    fits = FITS + (["oracle"] if "oracle" in runs[0] else [])
    s = {"k": k}
    for f in fits:
        s[f] = {m: float(np.mean([r[f][m] for r in runs])) for m in runs[0][f]}
        s[f]["wrong_splits"] = int(sum(r[f]["accuracy"] < threshold for r in runs))
    for name in FITS[2:]:
        s[name]["bic_preferred_over_dmm"] = int(sum(r[f"bic_{name}"] < r["bic_dmm"] for r in runs))
        s[name]["specific_nodes"] = float(np.mean([r[f"specific_{name}"] for r in runs]))
        s[name]["nodes"] = float(np.mean([r[f"nodes_{name}"] for r in runs]))
    return s


def main():
    OUT.mkdir(exist_ok=True)
    jobs = [(kind, value, rep) for kind, value in SCENARIOS for rep in range(N_REPS)]
    with Pool() as pool:
        results = pool.map(job, jobs)
    summary = {"n_reps": N_REPS, "scenarios": {}}
    for kind, value in SCENARIOS:
        runs = [r for r in results if r["kind"] == kind and r["value"] == value]
        s = summarise(runs)
        summary["scenarios"][f"{kind}:{value}"] = s
        print(f"{kind}:{value}")
        for f in FITS + (["oracle"] if "oracle" in s else []):
            extra = ""
            if f in FITS[2:]:
                extra = (f" | BIC<DMM {s[f]['bic_preferred_over_dmm']}/{N_REPS}"
                         f" | specific {s[f]['specific_nodes']:.1f}/{s[f]['nodes']:.0f}")
            print(f"  {f:14s} acc={s[f]['accuracy']:.3f} conf={s[f]['confidence']:.3f} "
                  f"ECE={s[f]['ece']:.3f} LL={s[f]['log_loss']:.3f} wrong={s[f]['wrong_splits']}{extra}")
        sys.stdout.flush()
    with open(OUT / "compare.json", "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
