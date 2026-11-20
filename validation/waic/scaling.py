"""
N-scaling experiment: which K do BIC, AIC and WAIC pick as the sample size grows?

    venv/bin/python validation/waic/scaling.py [--reps 2] [--kmax 8] [--jobs 16]

For each data kind ("clean" 3-cluster DMM, "gradient" continuum), each N and
each replicate, fits K = 1..kmax once (C core, with Hessian) and computes WAIC
with both Laplace variants from that same fit:

    full    joint observed information (pyDMM default)
    block   per-component Hessians, assignments fixed (first implementation)

Selected K per criterion:

    BIC, AIC              argmin
    WAIC_<m>_argmin       argmin
    WAIC_<m>              smallest K such that no larger K is confidently
                          better: elpd(K') - elpd(K) <= Z * SE(difference) for
                          every K' > K (SE from pointwise elpd differences)

Writes results/scaling.csv (one row per fit, appended as tasks finish) and
results/scaling_selected.csv.
"""

import argparse
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from generate_data import scaling_dataset  # noqa: E402

from pydmm import pydmm_core  # noqa: E402
from pydmm.waic import compute_waic  # noqa: E402

RESULTS = HERE / "results"
SIZES = [250, 500, 1000, 2000, 4000]
KINDS = ["clean", "gradient"]
METHODS = ["full", "block"]
Z = 2.0


def confident_k(elpd_i):
    """Smallest K with no larger K better by more than Z SEs of the difference."""
    ks = sorted(elpd_i)
    for k in ks:
        ok = True
        for k2 in ks:
            if k2 <= k:
                continue
            diff = elpd_i[k2] - elpd_i[k]
            if diff.sum() > Z * np.sqrt(diff.size * np.var(diff)):
                ok = False
                break
        if ok:
            return k
    return ks[-1]


def run_task(task):
    kind, n, rep, kmax = task
    warnings.simplefilter("ignore")
    X, _ = scaling_dataset(kind, n, seed=1000 * rep + n + (0 if kind == "clean" else 7))
    rows, elpd_i = [], {m: {} for m in METHODS}
    for K in range(1, kmax + 1):
        t0 = time.time()
        res = pydmm_core.dirichlet_fit(X, n_components=K, seed=rep, return_hessian=True)
        t_fit = time.time() - t0
        g = res["GoodnessOfFit"]
        row = {"kind": kind, "N": n, "rep": rep, "K": K, "BIC": g["BIC"], "AIC": g["AIC"],
               "fit_s": t_fit}
        for m in METHODS:
            t0 = time.time()
            w = compute_waic(X, res["Fit"]["Estimate"], res["Mixture"]["Weight"],
                             res["Hessian"], n_draws=1000, random_state=rep, method=m)
            row[f"WAIC_{m}"] = w["WAIC"]
            row[f"p_waic_{m}"] = w["p_waic"]
            row[f"se_{m}"] = w["se"]
            row[f"waic_s_{m}"] = time.time() - t0
            if m == "full":
                row["n_regularised"] = w["n_regularised"]
            elpd_i[m][K] = w["elpd_waic_i"]
        rows.append(row)
    sel = {"kind": kind, "N": n, "rep": rep}
    df = pd.DataFrame(rows).set_index("K")
    sel["BIC"] = int(df["BIC"].idxmin())
    sel["AIC"] = int(df["AIC"].idxmin())
    for m in METHODS:
        sel[f"WAIC_{m}"] = confident_k(elpd_i[m])
        sel[f"WAIC_{m}_argmin"] = int(df[f"WAIC_{m}"].idxmin())
    return pd.DataFrame(rows), sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    tasks = [(k, n, r, args.kmax) for k in KINDS for n in args.sizes
             for r in range(args.reps)]
    tasks.sort(key=lambda t: -t[1])  # biggest first for load balance
    fits, sels = [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futures = [ex.submit(run_task, t) for t in tasks]
        for f in as_completed(futures):
            df, sel = f.result()
            fits.append(df)
            sels.append(sel)
            pd.concat(fits).sort_values(["kind", "N", "rep", "K"]).to_csv(
                RESULTS / "scaling.csv", index=False)
            pd.DataFrame(sels).sort_values(["kind", "N", "rep"]).to_csv(
                RESULTS / "scaling_selected.csv", index=False)
            print(sel, flush=True)

    sel = pd.DataFrame(sels).sort_values(["kind", "N", "rep"])
    print(sel.to_string(index=False))
    df = pd.concat(fits)
    t = df.groupby("N")[["fit_s", "waic_s_full", "waic_s_block"]].mean()
    print("\nmean seconds per K:\n", t.round(2).to_string())


if __name__ == "__main__":
    main()
