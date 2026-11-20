# WAIC validation

Checks of pyDMM's WAIC (`DirichletMixture(compute_waic=True)`) against PyMC/arviz
and prodiphy, plus an experiment on how BIC, AIC and WAIC choose K as N grows.
Results below are from the cached files in `results/` (20 Nov 2026).

## Files

| File | Environment | Purpose |
|---|---|---|
| `generate_data.py` | pyDMM venv | Comparison dataset (N=300, S=10, 3 clusters, depth 2,000) and scaling generators |
| `run_prodiphy.py` | `.venv-prodiphy` | prodiphy DMM per K (nutpie, 4 chains × 500 draws, 500 tune, bounds 1–500) |
| `make_fixtures.py` | `.venv-prodiphy` | Reference values for `tests/test_waic.py` (`tests/data/waic/`) |
| `compare.py` | pyDMM venv | pyDMM vs cached prodiphy results |
| `scaling.py` | pyDMM venv | BIC / AIC / WAIC selection for N = 250–4,000 |
| `requirements-prodiphy.txt` | | Separate environment for prodiphy, PyMC, arviz, nutpie |

## How pyDMM computes WAIC

EM gives the posterior mode only. pyDMM draws 1,000 parameter sets from a joint
normal (Laplace) approximation over log α and the weight logits. The precision
is the observed information of the mixture, from Louis' identity: the C core's
per-component Hessians minus the covariance of the complete-data score over the
cluster assignments. Directions where the precision is below 1, including
negative curvature at saddle points of overfitted fits, get a posterior SD of 1.

A first version used independent per-component Hessians with assignments held
fixed ("block", still available as `compute_waic(..., method="block")`). It
underestimates p_waic for redundant components; see the scaling results.

## Exact checks (unit tests)

| Check | Result |
|---|---|
| Pointwise log-likelihood vs `pm.logp` of the PyMC DM mixture, equal and variable depth | match to 1e-10 |
| WAIC arithmetic vs `az.waic` on the same matrix | match to 1e-12 |
| prodiphy's posterior draws through pyDMM's code vs prodiphy's own WAIC | match to 4e-11 |
| Observed information vs finite-difference Hessian of the negative log posterior | match to ~1e-5 |
| C Hessian vs finite differences (regression test for the precedence bug fixed in `hessian()`) | match to ~1e-5 |

## pyDMM vs prodiphy (`compare.py`)

prodiphy's chains land in different modes, so its WAIC is computed per chain and
the chain with the highest lppd is used.

| K | prodiphy elpd | pyDMM elpd | Note |
|---|---|---|---|
| 2 | −16615.2 | −16373.7 | All 4 prodiphy chains collapsed to one occupied cluster |
| 3 | −16001.5 (p_waic 31.8) | −16002.0 (p_waic 32.5) | Same model, agree within 0.6 (SE 32) |
| 4 | −16001.7 | −16025.6 | prodiphy empties the extra component; pyDMM splits a cluster |
| 5 | −16001.0 | −16022.5 | as K=4 |

Selected K (smallest K with no larger K better by > 2 SE of the difference): **3 for both**.

The pre-agreed check (same selected K, and every elpd difference within 2 × dse)
reports FAIL. The causes are structural, not numerical:

* K=2: prodiphy's sampler never leaves the collapsed one-cluster mode.
* K≥4: prodiphy's Dir(1/K) weight prior lets redundant components go empty at no
  cost, so its K=3, 4 and 5 models are practically identical (dse ≈ 0.3).
  pyDMM's EM has no weight prior and splits real clusters, which the full Laplace
  approximation charges about 20 elpd for. The two packages then describe
  different posteriors, and a 0.3-wide tolerance cannot hold.

Runtime per K: prodiphy 58–101 s (nutpie, 4 chains); pyDMM 1–7 s including WAIC.

Other prodiphy notes: `DMM(clusters=1)` is not a one-component model (PyMC's
`Mixture` treats the feature axis as components when given one component), and
nutpie ignores `idata_kwargs`, so the log-likelihood must be computed after sampling.

## Scaling experiment (`scaling.py`)

K = 1..8, 2 replicates per cell. WAIC columns use the "no larger K confidently
better" rule (z = 2); argmin is shown for contrast.

| Data | N | BIC | AIC | WAIC full | WAIC full argmin | WAIC block | WAIC block argmin |
|---|---|---|---|---|---|---|---|
| clean | 250 | 3, 3 | 3, 3 | 3, 3 | 3, 3 | 3, 3 | 8, 8 |
| clean | 500 | 3, 3 | 3, 3 | 3, 3 | 3, 3 | 5, 4 | 8, 8 |
| clean | 1000 | 3, 3 | 3, 3 | 3, 3 | 3, 4 | 4, 4 | 8, 8 |
| clean | 2000 | 3, 3 | 3, 3 | 3, 3 | 3, 3 | 5, 5 | 8, 8 |
| clean | 4000 | 3, 3 | 3, 3 | 3, 3 | 3, 3 | 4, 6 | 8, 8 |
| gradient | 250 | 2, 3 | 3, 3 | 3, 3 | 6, 7 | 4, 5 | 7, 8 |
| gradient | 500 | 3, 3 | 3, 3 | 4, 3 | 4, 6 | 5, 6 | 8, 8 |
| gradient | 1000 | 3, 3 | 3, 5 | 5, 5 | 6, 5 | 6, 6 | 8, 8 |
| gradient | 2000 | 3, 3 | 5, 6 | 6, 6 | 8, 6 | 7, 7 | 8, 8 |
| gradient | 4000 | 5, 5 | 6, 5 | 6, 5 | 6, 8 | 7, 8 | 8, 8 |

* On data that really are 3 clusters, full WAIC with the selection rule picks
  3 every time. The block approximation and plain argmin over-select.
* On the continuum ("gradient") data, BIC drifts from 3 to 5 as N grows, which
  matches what is seen on real data. **WAIC drifts at least as much** (3 → 5–6),
  in line with AIC. WAIC estimates predictive accuracy, and with more samples
  extra components genuinely predict a continuum better. WAIC does not stop
  cluster numbers from growing with N.

Mean seconds per K (fit / WAIC full): N=250 2.5 / 2.1, N=1000 13.1 / 6.2,
N=4000 50.4 / 13.7.

## Re-running

```bash
python validation/waic/generate_data.py
validation/waic/.venv-prodiphy/bin/python validation/waic/run_prodiphy.py   # ~6 min
validation/waic/.venv-prodiphy/bin/python validation/waic/make_fixtures.py
venv/bin/python validation/waic/compare.py
venv/bin/python validation/waic/scaling.py --jobs 16                       # ~25 min on 16 cores
```
