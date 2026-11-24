# Bootstrap-averaged assignment probabilities

Gains and costs of `n_bootstrap` (bagging) for `DirichletMixture` and
`DirichletTreeMixture` on synthetic data with known clusters. The summary is in
the main README ("Bootstrap-Averaged Assignment Probabilities"). The real-data
evidence is in `docs/dtmm_comparison.md` ("Stability on real data").

| File | Purpose |
|---|---|
| `compare.py quality` | 6 scenarios × 10 replicates: well-specified DMM data (overlapping, well separated, N = 100, K = 3) and misspecified logistic-normal data (independent features, correlated modules). Compares plain, tempered and bagged fits (B = 4, 8, 16) on accuracy, confidence, ECE, log-loss, Brier and wrong splits. Results in `results/compare.json` |
| `compare.py runtime` | Single-threaded, sequential wall time of single and bagged fits over N, S, K and B. Results in `results/runtime.json` |

```bash
OMP_NUM_THREADS=1 venv/bin/python validation/bagging/compare.py quality
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 venv/bin/python validation/bagging/compare.py runtime   # run alone
```

`compare.py` imports `lognormal_modules` from `validation/correlation/experiment.py`.

## Results (24 Nov 2026)

* **Gains** come where single fits are over-confident. On misspecified data, the
  DMM's log-loss dropped from 2.00 to 0.56 (0.39 with one-step tempering on top),
  and the DTMM's from 1.55 to 0.39, with accuracy up from 0.80 to 0.875. With
  N = 100, ECE dropped from 0.073 to 0.051.
* **No gain** on well-separated clusters. With three clusters, results were
  slightly worse (log-loss 0.150 vs 0.157).
* **Wrong partitions** are not repaired. With strongly correlated modules,
  bagging alone (log-loss 2.12) did less than one-step tempering alone (1.24).
* **B = 8** is a good default: B = 4 was noisier and B = 16 added little.
* **Runtime** is about B + 1 single fits for the DMM and up to twice that for
  the DTMM: ×16.4 for B = 8 at N = 300, and ×17.6 (36 s → 10.5 min) at
  N = 1000, S = 200, K = 4. The refits run sequentially.
