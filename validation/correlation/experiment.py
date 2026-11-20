"""
Are pyDMM's cluster assignment probabilities over-confident when features are
correlated?

The Dirichlet-multinomial only has the (weak, negative) correlation implied by
the sum constraint. Groups of co-varying features (taxa from one guild,
ASVs of one strain, ...) therefore each add their own term to the per-sample
log-likelihood, and the log-odds between components counts the same signal
several times.

Experiments
-----------
A  Exact duplication (well-specified DMM data): every column copied m times.
   Same information, so the assignment probabilities should not change.
B  Correlated modules (logistic-normal mixture): S features in modules of b,
   within-module correlation rho. Higher rho = less independent information,
   so accuracy drops; a calibrated model should lose confidence too.

For each fit we report accuracy, mean confidence (max assignment probability),
expected calibration error (ECE), log-loss of the true label, and the effect of
two candidate fixes:
  * tempering z_ik ∝ (w_k p(x_i | alpha_k))^tau with tau = S_eff / S, S_eff the
    participation ratio of the pooled within-cluster CLR correlation matrix
    (unsupervised), and with the oracle tau that minimises log-loss;
  * tempering with tau from Good's weight-of-evidence condition (var = 2*mean),
    also unsupervised (pydmm.calibration.estimate_temperature, i.e.
    DirichletMixture(temperature="auto"));
  * aggregating features into correlation modules (hierarchical clustering on
    the within-cluster CLR correlation) before fitting.

Run: venv/bin/python validation/correlation/experiment.py
"""

import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.special import gammaln, logsumexp

from pydmm import DirichletMixture
from pydmm.calibration import estimate_temperature

OUT = Path(__file__).parent / "results"
TAU_GRID = np.round(np.linspace(0.02, 1.0, 50), 3)


# --------------------------------------------------------------------------- data

def dmm_data(rng, n=300, s=30, depth=3000, k=2):
    """Well-specified DMM data with moderately overlapping components."""
    base = rng.gamma(1.0, 1.0, size=s) + 0.2
    alphas = []
    for _ in range(k):
        a = base * np.exp(rng.normal(0, 0.45, size=s))
        alphas.append(a / a.sum() * 40.0)  # precision theta = 40
    labels = rng.integers(0, k, size=n)
    X = np.empty((n, s), dtype=np.int64)
    for i in range(n):
        p = rng.dirichlet(alphas[labels[i]])
        X[i] = rng.multinomial(depth, p)
    return X, labels


def lognormal_modules(rng, rho, n=300, n_modules=10, b=6, depth=5000,
                      n_diff=4, delta=1.2, sigma=1.0):
    """Two-cluster logistic-normal mixture with block-correlated features."""
    s = n_modules * b
    mu0 = rng.normal(0, 1.0, size=s)
    shift = np.zeros(s)
    for m in range(n_diff):
        shift[m * b:(m + 1) * b] = delta * (1 if m % 2 == 0 else -1)
    block = np.full((b, b), rho) + (1 - rho) * np.eye(b)
    cov = sigma ** 2 * np.kron(np.eye(n_modules), block)
    chol = np.linalg.cholesky(cov + 1e-12 * np.eye(s))
    labels = rng.integers(0, 2, size=n)
    eta = mu0 + labels[:, None] * shift + rng.standard_normal((n, s)) @ chol.T
    p = np.exp(eta - logsumexp(eta, axis=1, keepdims=True))
    X = np.vstack([rng.multinomial(depth, pi) for pi in p])
    return X, labels


# --------------------------------------------------------------------- helpers

def dm_loglik(X, alpha):
    """Per-sample DM log-likelihood (without multinomial coefficient). alpha (S,)."""
    a0 = alpha.sum()
    n = X.sum(axis=1)
    return (gammaln(a0) - gammaln(a0 + n)
            + (gammaln(X + alpha) - gammaln(alpha)).sum(axis=1))


def component_loglik(model, X):
    alpha = model.result_.parameter_estimates["Estimate"]  # (S, K)
    return np.column_stack([dm_loglik(X, alpha[:, k]) for k in range(alpha.shape[1])])


def tempered(ll, w, tau):
    """Same as pydmm.calibration.tempered_responsibilities (weights tempered too)."""
    z = tau * (ll + np.log(w)[None, :])
    return np.exp(z - logsumexp(z, axis=1, keepdims=True))


def align(z, labels):
    """Permute columns of z to best match the true labels."""
    k = z.shape[1]
    best, best_acc = None, -1
    for perm in permutations(range(k)):
        acc = np.mean(z[:, list(perm)].argmax(1) == labels)
        if acc > best_acc:
            best, best_acc = list(perm), acc
    return z[:, best]


def metrics(z, labels, n_bins=10):
    conf = z.max(1)
    pred = z.argmax(1)
    correct = pred == labels
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = sum(abs(correct[bins == b].mean() - conf[bins == b].mean()) * np.mean(bins == b)
              for b in range(n_bins) if np.any(bins == b))
    p_true = np.clip(z[np.arange(len(labels)), labels], 1e-12, 1)
    return {
        "accuracy": float(correct.mean()),
        "confidence": float(conf.mean()),
        "ece": float(ece),
        "log_loss": float(-np.log(p_true).mean()),
        "frac_gt_0.99": float(np.mean(conf > 0.99)),
    }


def clr(X):
    L = np.log(X + 0.5)
    return L - L.mean(axis=1, keepdims=True)


def within_cluster_corr(X, hard):
    """Pooled within-cluster correlation of CLR-transformed counts."""
    C = clr(X)
    resid = np.vstack([C[hard == k] - C[hard == k].mean(0) for k in np.unique(hard)
                       if np.sum(hard == k) > 2])
    R = np.corrcoef(resid, rowvar=False)
    return np.nan_to_num(R)


def effective_features(R):
    """Participation ratio (sum lambda)^2 / sum lambda^2 of a correlation matrix."""
    lam = np.clip(np.linalg.eigvalsh(R), 0, None)
    return float(lam.sum() ** 2 / np.sum(lam ** 2))


def tau_weight_of_evidence(ll, w):
    """Unsupervised tau (Good's weight-of-evidence condition); see pydmm.calibration."""
    return estimate_temperature(ll, w)


def aggregate_modules(X, R, cut=0.5):
    """Sum features whose within-cluster correlation exceeds roughly `cut`."""
    D = np.clip(1 - R, 0, 2)
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method="average")
    groups = fcluster(Z, t=1 - cut, criterion="distance")
    return np.column_stack([X[:, groups == g].sum(1) for g in np.unique(groups)])


def evaluate(X, labels, k=2, seed=0):
    m = DirichletMixture(n_components=k, random_state=seed).fit(X)
    w = m.result_.mixture_weights
    ll = component_loglik(m, X)
    z = align(tempered(ll, w, 1.0), labels)
    out = {"plain": metrics(z, labels)}

    hard = z.argmax(1)
    R = within_cluster_corr(X, hard)
    s_eff = effective_features(R)
    tau = s_eff / X.shape[1]
    out["S"] = X.shape[1]
    out["S_eff"] = s_eff
    out["tau_seff"] = tau
    out["tempered_seff"] = metrics(align(tempered(ll, w, tau), labels), labels)

    t_woe = tau_weight_of_evidence(ll, w)
    out["tau_woe"] = t_woe
    out["tempered_woe"] = metrics(align(tempered(ll, w, t_woe), labels), labels)

    losses = [metrics(align(tempered(ll, w, t), labels), labels)["log_loss"] for t in TAU_GRID]
    t_best = float(TAU_GRID[int(np.argmin(losses))])
    out["tau_oracle"] = t_best
    out["tempered_oracle"] = metrics(align(tempered(ll, w, t_best), labels), labels)

    Xa = aggregate_modules(X, R)
    ma = DirichletMixture(n_components=k, random_state=seed).fit(Xa)
    za = align(ma.result_.group_assignments, labels)
    out["aggregated"] = metrics(za, labels)
    out["S_aggregated"] = Xa.shape[1]
    out["wrong_partition"] = float(out["plain"]["accuracy"] < 0.7)
    return out


def mean_of(runs):
    """Average nested metric dicts over replicates."""
    keys = runs[0].keys()
    res = {}
    for key in keys:
        if isinstance(runs[0][key], dict):
            res[key] = {m: float(np.mean([r[key][m] for r in runs])) for m in runs[0][key]}
        else:
            res[key] = float(np.mean([r[key] for r in runs]))
    return res


# ------------------------------------------------------------------------ main

def main(n_reps=10):
    OUT.mkdir(exist_ok=True)
    results = {"duplication": {}, "modules": {}}

    print("A. exact duplication of every feature (well-specified DMM data)")
    for mult in [1, 2, 4, 8]:
        runs = []
        for rep in range(n_reps):
            rng = np.random.default_rng(100 + rep)
            X, labels = dmm_data(rng)
            runs.append(evaluate(np.repeat(X, mult, axis=1), labels, seed=rep))
        r = mean_of(runs)
        results["duplication"][mult] = r
        print(f"  m={mult}: acc={r['plain']['accuracy']:.3f} conf={r['plain']['confidence']:.3f} "
              f"ECE={r['plain']['ece']:.3f} logloss={r['plain']['log_loss']:.3f} "
              f"| S_eff/S={r['tau_seff']:.2f} -> ECE={r['tempered_seff']['ece']:.3f} "
              f"| woe tau={r['tau_woe']:.2f} -> logloss={r['tempered_woe']['log_loss']:.3f} "
              f"| oracle tau={r['tau_oracle']:.2f} | aggregated ECE={r['aggregated']['ece']:.3f}")
        sys.stdout.flush()

    print("B. block-correlated features (logistic-normal mixture, 10 modules x 6)")
    for rho in [0.0, 0.3, 0.6, 0.9]:
        runs = []
        for rep in range(n_reps):
            rng = np.random.default_rng(200 + rep)
            X, labels = lognormal_modules(rng, rho)
            runs.append(evaluate(X, labels, seed=rep))
        r = mean_of(runs)
        results["modules"][rho] = r
        print(f"  rho={rho}: acc={r['plain']['accuracy']:.3f} conf={r['plain']['confidence']:.3f} "
              f"ECE={r['plain']['ece']:.3f} logloss={r['plain']['log_loss']:.3f} "
              f"| S_eff/S={r['tau_seff']:.2f} -> ECE={r['tempered_seff']['ece']:.3f} "
              f"logloss={r['tempered_seff']['log_loss']:.3f} "
              f"| woe tau={r['tau_woe']:.2f} -> ECE={r['tempered_woe']['ece']:.3f} "
              f"logloss={r['tempered_woe']['log_loss']:.3f} "
              f"| oracle tau={r['tau_oracle']:.2f} logloss={r['tempered_oracle']['log_loss']:.3f} "
              f"| wrong partitions={r['wrong_partition']:.1f} "
              f"| aggregated S={r['S_aggregated']:.1f} acc={r['aggregated']['accuracy']:.3f} "
              f"ECE={r['aggregated']['ece']:.3f}")
        sys.stdout.flush()

    with open(OUT / "experiment.json", "w") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
