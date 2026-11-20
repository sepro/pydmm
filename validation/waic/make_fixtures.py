"""
Reference values for pyDMM's WAIC unit tests, computed with PyMC and arviz.

Run in the prodiphy environment after run_prodiphy.py (needs the K=3 trace):

    validation/waic/.venv-prodiphy/bin/python validation/waic/make_fixtures.py

Writes tests/data/waic/:
    logp_reference.npz   PyMC log-density of a DM mixture (equal and variable depth)
    arith_reference.npz  arviz.waic on a random log-likelihood matrix
    swap_reference.npz   prodiphy posterior draws (K=3, selected chain, 200 draws)
                         and arviz.waic of prodiphy's own log-likelihood for them
"""

import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from generate_data import DATA_DIR, TRUE_ALPHA, WEIGHTS  # noqa: E402

OUT = HERE.parent.parent / "tests" / "data" / "waic"
TRACES = HERE / "results" / "traces"


def mixture_logp(X, n, w, alpha):
    comps = [pm.DirichletMultinomial.dist(n=n, a=a, shape=X.shape) for a in alpha]
    return pm.logp(pm.Mixture.dist(w=w, comp_dists=comps), X).eval()


def logp_reference(rng):
    df = pd.read_csv(DATA_DIR / "comparison_counts.csv", index_col=0)
    X_eq = df.values[:30].astype(np.int64)
    # perturbed parameters, so the test is not tied to the generating values
    alpha = TRUE_ALPHA * rng.uniform(0.5, 1.5, size=TRUE_ALPHA.shape)
    w = rng.dirichlet(WEIGHTS * 20)
    logp_eq = mixture_logp(X_eq, X_eq.sum(1), w, alpha)

    # variable depth, including zeros and a very shallow sample
    depths = np.array([1, 5, 37, 150, 999, 2500, 10000, 40, 3, 777])
    X_var = np.vstack([rng.multinomial(d, rng.dirichlet(alpha[i % 3]))
                       for i, d in enumerate(depths)]).astype(np.int64)
    logp_var = mixture_logp(X_var, X_var.sum(1), w, alpha)
    return dict(X_eq=X_eq, X_var=X_var, w=w, alpha=alpha,
                logp_eq=logp_eq, logp_var=logp_var)


def arith_reference(rng):
    # heterogeneous variances so p_waic_i differs across observations
    log_lik = rng.normal(-5, 1, size=(1, 400, 40)) * rng.uniform(0.1, 1.5, size=40)
    idata = az.from_dict(log_likelihood={"obs": log_lik})
    res = az.waic(idata, pointwise=True)
    return dict(log_lik=log_lik[0], elpd_waic=res.elpd_waic, p_waic=res.p_waic,
                se=res.se, elpd_waic_i=res.waic_i.values,
                p_waic_i=res.p_waic_i.values if hasattr(res, "p_waic_i") else np.nan)


def swap_reference():
    """First 200 draws of prodiphy's selected K=3 chain."""
    draws = np.load(HERE / "results" / "prodiphy_draws_K3.npz")
    ll = np.load(TRACES / "prodiphy_loglik_K3.npz")["log_lik"][:200]  # (200, N)
    res = az.waic(az.from_dict(log_likelihood={"obs": ll[None]}))
    X = pd.read_csv(DATA_DIR / "comparison_counts.csv", index_col=0).values
    return dict(X=X.astype(np.int64), w=draws["w"][:200], alpha=draws["alpha"][:200],
                elpd_waic=res.elpd_waic, p_waic=res.p_waic, se=res.se,
                log_lik_first10=ll[:, :10])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    np.savez_compressed(OUT / "logp_reference.npz", **logp_reference(rng))
    np.savez_compressed(OUT / "arith_reference.npz", **arith_reference(rng))
    np.savez_compressed(OUT / "swap_reference.npz", **swap_reference())
    for f in sorted(OUT.glob("*.npz")):
        print(f"{f.relative_to(OUT.parent.parent.parent)}: {f.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
