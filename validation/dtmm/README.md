# Dirichlet-tree mixture validation

Checks and comparisons for `DirichletTreeMixture`. The write-up with
conclusions is [`docs/dtmm_comparison.md`](../../docs/dtmm_comparison.md).

| File | Purpose |
|---|---|
| `reference.py` | NumPy/SciPy reference DTMM (EM, node selection). Prototype used to choose the design; reproduces pyDMM's DMM with a star tree and the C DTMM's likelihood |
| `compare.py` | DMM vs DTMM on synthetic data (7 scenarios × 10 replicates): accuracy, calibration, wrong splits, BIC, selected nodes. Results in `results/compare.json` |
| `benchmark.py` | Single-threaded runtime over N, S, K and tree shape. Results in `results/benchmark.json` |

```bash
OMP_NUM_THREADS=1 venv/bin/python validation/dtmm/compare.py     # ~5 min on 16 cores
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 venv/bin/python validation/dtmm/benchmark.py   # run alone
```

`compare.py` imports the data generators of `validation/correlation/experiment.py`.
