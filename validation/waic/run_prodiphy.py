"""
Reference WAIC from prodiphy (PyMC NUTS) on the comparison dataset.

Run in the separate prodiphy environment (see requirements-prodiphy.txt):

    validation/waic/.venv-prodiphy/bin/python validation/waic/run_prodiphy.py

Settings: DMM.fit per K (so sample_kwargs reach PyMC), nutpie sampler,
4 chains x 500 draws, 500 tuning steps, concentration bounds 1-500
(DMM.fit default).

WAIC is evaluated per chain. prodiphy's chains can land in different modes
(clusters at different indices, or a collapsed mode where one component takes
all the weight), so pooling chains is not meaningful. The reference for each K
is the chain with the highest lppd (best in-sample fit); all chains are
reported in prodiphy_waic_chains.csv.

Writes to results/:
    prodiphy_waic_chains.csv   elpd_waic, p_waic, lppd, se, weights per chain
    prodiphy_waic.csv          selected chain per K
    prodiphy_compare.csv       arviz.compare of the selected chains (dse)
    prodiphy_draws_K{K}.npz    selected chain's draws: w (D, K), alpha (D, K, S)
    traces/prodiphy_loglik_K{K}.npz  selected chain's log-likelihood (D, N)
                               (gitignored; used by make_fixtures.py)

K=1 is not run: with a single component PyMC's Mixture treats the last
(feature) axis as the mixture dimension, so prodiphy's DMM(clusters=1) is not a
one-component DMM. nutpie ignores idata_kwargs, so the log-likelihood is
computed after sampling with pm.compute_log_likelihood.
"""

import argparse
import sys
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from prodiphy import DMM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from generate_data import DATA_DIR  # noqa: E402

RESULTS = HERE / "results"
TRACES = RESULTS / "traces"
N_CHAINS = 4


def chain_draws(trace, K, chain):
    post = trace.posterior.sel(chain=chain)
    w = post["w"].values  # (D, K)
    alpha = np.stack([post[f"a[{c}]"].values for c in range(K)], axis=1)
    return w, alpha


def chain_loglik(trace, chain):
    return trace.log_likelihood["obs"].sel(chain=chain).values  # (D, N)


def single_chain_idata(log_lik):
    return az.from_dict(log_likelihood={"obs": log_lik[None]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--sampler", default="nutpie")
    ap.add_argument("--seed", type=int, default=20261120)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    TRACES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_DIR / "comparison_counts.csv", index_col=0)
    assert df.sum(axis=1).nunique() == 1, "prodiphy needs equal read depth"

    chain_rows, selected_rows, selected = [], [], {}
    for K in args.ks:
        print(f"=== prodiphy K={K} ===", flush=True)
        model = DMM(clusters=K, chains=N_CHAINS, cores=N_CHAINS, samples=500, tune=500)
        t0 = time.time()
        model.fit(
            df, lower=1, upper=500,
            sample_kwargs={"nuts_sampler": args.sampler, "random_seed": args.seed + K},
        )
        runtime = time.time() - t0
        trace = model.trace
        if "log_likelihood" not in trace.groups():
            with model.model:
                pm.compute_log_likelihood(trace, progressbar=False)

        per_chain = []
        for c in trace.posterior.chain.values:
            ll = chain_loglik(trace, c)
            res = az.waic(single_chain_idata(ll))
            w, _ = chain_draws(trace, K, c)
            row = {
                "K": K, "chain": int(c), "elpd_waic": res.elpd_waic, "p_waic": res.p_waic,
                "lppd": res.elpd_waic + res.p_waic, "se": res.se, "warning": bool(res.warning),
                "weights_sorted": " ".join(f"{x:.2f}" for x in np.sort(w.mean(0))[::-1]),
                "runtime_s_all_chains": runtime,
            }
            per_chain.append(row)
            print("CHAIN", row, flush=True)
        chain_rows += per_chain
        best = max(per_chain, key=lambda r: r["lppd"])
        c = best["chain"]
        w, alpha = chain_draws(trace, K, c)
        ll = chain_loglik(trace, c)
        np.savez_compressed(RESULTS / f"prodiphy_draws_K{K}.npz", w=w, alpha=alpha, chain=c)
        np.savez_compressed(TRACES / f"prodiphy_loglik_K{K}.npz", log_lik=ll, chain=c)
        selected[f"K{K}"] = single_chain_idata(ll)
        selected_rows.append({**{k: best[k] for k in ("K", "chain", "elpd_waic", "p_waic",
                                                       "lppd", "se", "warning", "weights_sorted")},
                              "runtime_s": runtime})
        pd.DataFrame(chain_rows).to_csv(RESULTS / "prodiphy_waic_chains.csv", index=False)
        pd.DataFrame(selected_rows).to_csv(RESULTS / "prodiphy_waic.csv", index=False)
        print(f"SELECTED chain {c} for K={K}", flush=True)

    if len(selected) > 1:
        comp = az.compare(selected, ic="waic")
        comp.to_csv(RESULTS / "prodiphy_compare.csv")
        print(comp)


if __name__ == "__main__":
    main()
