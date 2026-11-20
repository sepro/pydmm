"""
Compare pyDMM's Laplace-approximation WAIC with prodiphy's NUTS WAIC.

Uses only the cached prodiphy results (results/prodiphy_*.csv/npz), so it runs
in the normal pyDMM environment:

    venv/bin/python validation/waic/compare.py

For each K it reports, on the arviz elpd scale (higher is better):
    elpd_prodiphy   arviz WAIC of prodiphy's NUTS run
    elpd_swap       prodiphy's posterior draws through pyDMM's log-likelihood
                    code (must equal elpd_prodiphy up to float error)
    elpd_pydmm      pyDMM fit with compute_waic=True (Laplace draws)
    d_prodiphy, d_pydmm   elpd difference from prodiphy's best model
    dse             SE of that difference (arviz.compare, prodiphy)

Selected K uses the rule from scaling.py: the smallest K such that no larger K
is better by more than 2 SEs of the pointwise difference.

Acceptance (agreed plan): same selected K, and |d_pydmm - d_prodiphy| <= 2 * dse
for every K. prodiphy's reference is a single chain per K (see run_prodiphy.py).
Writes results/comparison.csv.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from generate_data import DATA_DIR  # noqa: E402
from scaling import confident_k  # noqa: E402

from pydmm import DirichletMixture  # noqa: E402
from pydmm.waic import (  # noqa: E402
    dirichlet_mixture_log_likelihood,
    waic_from_log_likelihood,
)

RESULTS = HERE / "results"


def main():
    X = pd.read_csv(DATA_DIR / "comparison_counts.csv", index_col=0)
    ref = pd.read_csv(RESULTS / "prodiphy_waic.csv").set_index("K")
    comp = pd.read_csv(RESULTS / "prodiphy_compare.csv", index_col=0)
    comp.index = [int(i.lstrip("K")) for i in comp.index]

    rows, pw_ref, pw_pydmm = [], {}, {}
    for K in ref.index:
        draws = np.load(RESULTS / f"prodiphy_draws_K{K}.npz")
        swap = waic_from_log_likelihood(
            dirichlet_mixture_log_likelihood(X.values, draws["w"], draws["alpha"]))
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = DirichletMixture(K, compute_waic=True, n_waic_draws=1000).fit(X)
        g = fit.result.goodness_of_fit
        pw_ref[K] = swap["elpd_waic_i"]
        pw_pydmm[K] = waic_from_log_likelihood(
            fit.result.pointwise_log_likelihood)["elpd_waic_i"]
        rows.append({
            "K": K,
            "elpd_prodiphy": ref.loc[K, "elpd_waic"],
            "elpd_swap": swap["elpd_waic"],
            "elpd_pydmm": g["elpd_waic"],
            "p_waic_prodiphy": ref.loc[K, "p_waic"],
            "p_waic_pydmm": g["p_waic"],
            "se_prodiphy": ref.loc[K, "se"],
            "se_pydmm": g["WAIC_se"],
            "dse": comp.loc[K, "dse"],
            "runtime_prodiphy_s": ref.loc[K, "runtime_s"],
            "runtime_pydmm_s": time.time() - t0,
        })
    df = pd.DataFrame(rows).set_index("K")
    best_ref = df["elpd_prodiphy"].idxmax()
    best_pydmm = df["elpd_pydmm"].idxmax()
    sel_ref, sel_pydmm = confident_k(pw_ref), confident_k(pw_pydmm)
    df["d_prodiphy"] = df.loc[best_ref, "elpd_prodiphy"] - df["elpd_prodiphy"]
    df["d_pydmm"] = df.loc[best_ref, "elpd_pydmm"] - df["elpd_pydmm"]
    df["within_2dse"] = (df["d_pydmm"] - df["d_prodiphy"]).abs() <= 2 * df["dse"].clip(lower=1e-12)
    df.loc[best_ref, "within_2dse"] = True  # reference model, difference is 0 by definition

    df.to_csv(RESULTS / "comparison.csv")
    pd.set_option("display.width", 200)
    print(df.round(2).to_string())
    print()
    print(f"swap check: max |elpd_swap - elpd_prodiphy| = "
          f"{(df['elpd_swap'] - df['elpd_prodiphy']).abs().max():.2e}")
    print(f"argmin K: prodiphy {best_ref}, pyDMM {best_pydmm}")
    print(f"selected K (no larger K confidently better): prodiphy {sel_ref}, pyDMM {sel_pydmm}")
    ok = (sel_ref == sel_pydmm) and bool(df["within_2dse"].all())
    print("ACCEPTANCE:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
