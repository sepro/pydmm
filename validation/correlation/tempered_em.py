"""
Solution 1 (tempered assignment probabilities, DirichletMixture(temperature="auto"))
versus solution 2 (tempered EM: the same tau also used in the E-step, so alpha and
the partition can change).

Uses the simulation settings of experiment.py:
  A  well-specified DMM data with every feature copied m times
  B  logistic-normal data with block-correlated features (rho)

Per setting and replicate:
  plain       standard EM, standard responsibilities
  sol1        same fit, temperature="auto"
  sol2        tempered EM with tau fixed to sol1's estimate (two-stage)
  sol2 + tau  the sol2 fit with its tempered E-step responsibilities
  sweep       tempered EM with fixed tau in TAU_SWEEP (best case for sol2)

Also stores histograms of the per-sample maximum assignment probability for
plain and sol1, split by correct / incorrect assignment.

Run: venv/bin/python validation/correlation/tempered_em.py
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from experiment import align, dmm_data, lognormal_modules, metrics  # noqa: E402

from pydmm import DirichletMixture, pydmm_core  # noqa: E402

OUT = Path(__file__).parent / "results"
N_REPS = 10
TAU_SWEEP = [0.5, 0.25, 0.1, 0.05]
BINS = np.linspace(0.5, 1.0, 11)  # K = 2: max probability lies in [0.5, 1]


def settings():
    for mult in [1, 2, 4, 8]:
        yield ("duplication", mult)
    for rho in [0.0, 0.3, 0.6, 0.9]:
        yield ("modules", rho)


def make_data(kind, value, rep):
    if kind == "duplication":
        X, labels = dmm_data(np.random.default_rng(100 + rep))
        X = np.repeat(X, value, axis=1)
    else:
        X, labels = lognormal_modules(np.random.default_rng(200 + rep), value)
    return np.ascontiguousarray(X.astype(np.int32)), labels


def tempered_em(X, tau, seed):
    res = pydmm_core.dirichlet_fit(X, n_components=2, seed=seed, temperature=float(tau))
    return res["Group"]


def hist(z, labels):
    conf = z.max(1)
    correct = z.argmax(1) == labels
    h_ok, _ = np.histogram(np.clip(conf[correct], 0.5, 1.0), bins=BINS)
    h_bad, _ = np.histogram(np.clip(conf[~correct], 0.5, 1.0), bins=BINS)
    return h_ok.tolist(), h_bad.tolist()


def job(args):
    kind, value, rep = args
    X, labels = make_data(kind, value, rep)
    m = DirichletMixture(n_components=2, random_state=rep, temperature="auto").fit(X)
    z_plain = align(m.result_.group_assignments_untempered, labels)
    z_sol1 = align(m.result_.group_assignments, labels)
    tau = m.temperature_
    z_sol2 = align(tempered_em(X, tau, rep), labels)
    out = {
        "kind": kind, "value": value, "rep": rep, "tau": tau,
        "plain": metrics(z_plain, labels),
        "sol1": metrics(z_sol1, labels),
        "sol2": metrics(z_sol2, labels),
        "sweep": {str(t): metrics(align(tempered_em(X, t, rep), labels), labels)
                  for t in TAU_SWEEP},
        "hist_plain": hist(z_plain, labels),
        "hist_sol1": hist(z_sol1, labels),
        "maxprob_plain": np.round(z_plain.max(1), 5).tolist(),
        "maxprob_sol1": np.round(z_sol1.max(1), 5).tolist(),
    }
    return out


def summarise(runs):
    def avg(key, metric, sub=None):
        vals = [(r[key][sub] if sub else r[key])[metric] for r in runs]
        return float(np.mean(vals))

    def wrong(key, sub=None):
        return int(sum((r[key][sub] if sub else r[key])["accuracy"] < 0.7 for r in runs))

    s = {"tau": float(np.mean([r["tau"] for r in runs]))}
    for key in ["plain", "sol1", "sol2"]:
        s[key] = {m: avg(key, m) for m in runs[0][key]}
        s[key]["wrong_splits"] = wrong(key)
    s["sweep"] = {}
    for t in TAU_SWEEP:
        s["sweep"][str(t)] = {m: avg("sweep", m, str(t)) for m in runs[0]["sweep"][str(t)]}
        s["sweep"][str(t)]["wrong_splits"] = wrong("sweep", str(t))
    for key in ["hist_plain", "hist_sol1"]:
        s[key] = {"correct": np.sum([r[key][0] for r in runs], axis=0).tolist(),
                  "incorrect": np.sum([r[key][1] for r in runs], axis=0).tolist()}
    for key in ["maxprob_plain", "maxprob_sol1"]:
        v = np.concatenate([r[key] for r in runs])
        s[key + "_quantiles"] = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95]).round(4).tolist()
        s[key + "_frac_gt_0.99"] = float(np.mean(v > 0.99))
    return s


def main():
    OUT.mkdir(exist_ok=True)
    jobs = [(kind, value, rep) for kind, value in settings() for rep in range(N_REPS)]
    with Pool() as pool:
        results = pool.map(job, jobs)
    summary = {"bins": BINS.tolist(), "n_reps": N_REPS, "duplication": {}, "modules": {}}
    for kind, value in settings():
        runs = [r for r in results if r["kind"] == kind and r["value"] == value]
        s = summarise(runs)
        summary[kind][str(value)] = s
        line = f"{kind:11s} {value:>4}: tau={s['tau']:.2f} | "
        for key in ["plain", "sol1", "sol2"]:
            line += (f"{key} acc={s[key]['accuracy']:.3f} conf={s[key]['confidence']:.3f} "
                     f"LL={s[key]['log_loss']:.3f} wrong={s[key]['wrong_splits']} | ")
        line += "sweep acc " + " ".join(
            f"{t}:{s['sweep'][str(t)]['accuracy']:.3f}/w{s['sweep'][str(t)]['wrong_splits']}"
            for t in TAU_SWEEP)
        print(line)
    with open(OUT / "tempered_em.json", "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
