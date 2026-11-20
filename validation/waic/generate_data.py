"""
Synthetic count data for validating pyDMM's WAIC.

Two kinds of data:

* ``comparison_dataset()``: small, clean 3-cluster Dirichlet-multinomial mixture
  that both pyDMM and prodiphy can fit (equal read depth for every sample,
  because prodiphy uses a single depth; concentrations sum(alpha) in 10-30).
* ``scaling_dataset(kind, n_samples, seed)``: larger data for the N-scaling
  experiment (pyDMM only):
    - "clean":    exact 3-cluster DMM (control; every criterion should settle on 3)
    - "gradient": continuum between 3 enterotype prototypes; each sample's mean
                  composition is a random blend of the prototypes, then DM noise.
                  No finite K is "true".

Run as a script to write ``data/comparison_counts.csv`` (and its labels).
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

N_FEATURES = 10
DEPTH = 2000

# Three enterotype-like prototypes over 10 taxa (rows sum to 1)
_PROTOTYPES = np.array(
    [
        [30, 20, 10, 8, 6, 6, 6, 6, 4, 4],   # dominated by taxa 0-1
        [4, 6, 6, 30, 22, 10, 8, 6, 4, 4],   # dominated by taxa 3-4
        [4, 4, 6, 6, 6, 8, 10, 14, 20, 22],  # dominated by taxa 8-9
    ],
    dtype=float,
)
PROTOTYPES = _PROTOTYPES / _PROTOTYPES.sum(axis=1, keepdims=True)
CONCENTRATIONS = np.array([15.0, 20.0, 25.0])  # sum(alpha) per cluster
WEIGHTS = np.array([0.40, 0.35, 0.25])
TRUE_ALPHA = PROTOTYPES * CONCENTRATIONS[:, None]  # (3, S)


def _dm_counts(rng, alpha_rows, depth):
    p = np.vstack([rng.dirichlet(a) for a in alpha_rows])
    return np.vstack([rng.multinomial(depth, pi) for pi in p])


def clean_mixture(n_samples, seed, depth=DEPTH):
    """Exact 3-component DMM. Returns (counts, labels)."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(3, size=n_samples, p=WEIGHTS)
    X = _dm_counts(rng, TRUE_ALPHA[labels], depth)
    return X.astype(np.int32), labels


def gradient_mixture(n_samples, seed, depth=DEPTH, blend_concentration=0.7):
    """
    Continuum between the 3 prototypes.

    blend ~ Dirichlet(blend_concentration * 1); mean composition = blend @ prototypes;
    alpha_i = 20 * mean composition; counts ~ DM(depth, alpha_i).
    Returns (counts, nearest-prototype labels).
    """
    rng = np.random.default_rng(seed)
    blend = rng.dirichlet(np.full(3, blend_concentration), size=n_samples)
    mean = blend @ PROTOTYPES
    X = _dm_counts(rng, 20.0 * mean, depth)
    return X.astype(np.int32), blend.argmax(axis=1)


def comparison_dataset(seed=20261120):
    """N=300, S=10, K_true=3, depth 2000. Returns (DataFrame, labels)."""
    X, labels = clean_mixture(300, seed)
    df = pd.DataFrame(
        X,
        index=[f"S{i:03d}" for i in range(X.shape[0])],
        columns=[f"taxon_{j}" for j in range(X.shape[1])],
    )
    return df, labels


def scaling_dataset(kind, n_samples, seed):
    if kind == "clean":
        return clean_mixture(n_samples, seed)
    if kind == "gradient":
        return gradient_mixture(n_samples, seed)
    raise ValueError(f"unknown kind {kind!r}")


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    df, labels = comparison_dataset()
    df.to_csv(DATA_DIR / "comparison_counts.csv")
    pd.Series(labels, index=df.index, name="true_cluster").to_csv(
        DATA_DIR / "comparison_labels.csv"
    )
    print(f"wrote {DATA_DIR / 'comparison_counts.csv'}: {df.shape}, "
          f"row sums {df.sum(1).min()}-{df.sum(1).max()}, "
          f"cluster sizes {np.bincount(labels)}")
