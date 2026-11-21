# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pyDMM is a Python package providing a Python interface to a C implementation of Dirichlet Mixture Model fitting (ported from the R DirichletMultinomial package) for compositional data analysis. It's designed for analyzing microbiome data and other count-based compositional datasets. The C core uses GSL for optimization, special functions and linear algebra.

A Rust port exists on the `rust` branch but was slower than the C core; `main` stays on C.

## Architecture

- **Python Interface** (`src/pydmm/core.py`):
  - `DirichletMixture`: Main scikit-learn compatible class
  - `DirichletMixtureResult`: Results container with pandas integration
- **WAIC** (`src/pydmm/waic.py`): joint Laplace posterior draws (observed information
  via Louis' identity from the C per-component Hessians + per-sample scores; flat/saddle
  directions regularised to SD <= 1), pointwise Dirichlet-multinomial mixture
  log-likelihood, WAIC arithmetic (matches `arviz.waic`). `method="block"` (assignments
  held fixed) is kept only for comparison: it underestimates p_waic for overfitted K.
- **Calibration** (`src/pydmm/calibration.py`): tempered assignment probabilities
  z_ik ∝ (w_k DM(x_i|α_k))^τ, applied after the fit (α, w, NLE/BIC/AIC/WAIC and the
  argmax are unchanged). `estimate_temperature` (`temperature="auto"`) uses Good's
  weight-of-evidence condition (calibrated normal LLR: var = 2·mean), pooled over
  component pairs, iterated to a fixed point (weights = tempered assignments).
  `temperature="one-step"` is the first iterate only (`max_iter=1`, weights =
  untempered assignments); it cannot drift, which "auto" does on heavy-tailed DTMM
  ratios (BUGS.md bug 5). Both go through `resolve_temperature`. Corrects
  over-confidence from correlated features and from the Dirichlet's fixed
  per-feature variance; cannot repair a wrong partition.
- **Dirichlet-tree mixture** (`src/pydmm/dtmm.py`, `src/pydmm/tree.py`):
  `DirichletTreeMixture` subclasses `DirichletMixture` (reuses input validation,
  `predict`, `fit_predict`, `score`). Each internal node of a `Tree` over the features
  splits its count with its own DM; star tree + `node_selection=False` == DMM (tested).
  Node selection (default on, it is a new model): a node stays component-specific
  only if its expected log-likelihood gain beats 0.5·log N·(K−1)·n_children, otherwise
  one α is shared and the node drops out of the E-step. `tree=None` builds
  `Tree.from_correlation` (average linkage on 1 − corr(CLR)). `Tree` stores CSR child
  lists in post-order (ids < S leaves, S+u internal node u, root last); unary nodes
  are collapsed. No WAIC for the DTMM yet.
- **C Extension**:
  - `src/pydmm/wrapper.c`: Python C API wrapper exposing `pydmm_core.dirichlet_fit`
    and `pydmm_core.dirichlet_tree_fit` (validates the tree before fitting)
  - `src/pydmm/dirichlet_fit_standalone.c/.h`: DMM EM, likelihood, BIC/AIC/Laplace
  - `src/pydmm/dirichlet_node.c/.h`: shared by both models: soft k-means, weighted
    DM objective/gradient, BFGS (`optimise_lambda_k`), `neg_log_evidence_i`, `hessian`,
    and the collapse restart (`collapsed`, `hard_kmeans`). Moved unchanged from the
    DMM/DTMM files; DMM output is bit-identical to before.
  - Collapse restart: soft k-means can start all components equal (very similar
    compositions); EM then stays at that saddle (all z = 1/K). The DTMM always restarts
    from hard k-means; the DMM only with `restart_collapsed=True` (off by default).
  - `src/pydmm/dirichlet_tree_fit.c/.h`: DTMM EM (per-node `data_t` views), node
    selection rounds, block-diagonal Hessian, criteria. `Fit/*` outputs are `(n_edges, K)`.
  - `src/pydmm/dirichlet_fit_main.c/.h`: original R-package version, not compiled
- **Build System**: `setup.py` defines the C extension (links `gsl`, `gslcblas`, `m`); metadata in `pyproject.toml`

### Critical Data Layout Issues

The C code expects data in **column-major** format (features × samples) while NumPy uses row-major (samples × features). `wrapper.c` transposes the input into a contiguous copy, and transposes outputs back: `Group` → `(N, K)`, `Fit/*` → `(S, K)`. The optional `Hessian` output (`return_hessian=True`) is `(K, S, S)`, w.r.t. λ = log α.

### Model details worth knowing

- NLE (and therefore BIC, AIC, Laplace) includes a Gamma(0.1, 0.1) prior on every α (`GAMMA_ITA`, `GAMMA_NU`) and the multinomial coefficient. WAIC uses the likelihood only, so its absolute value is not directly comparable to BIC/AIC; compare each criterion across K.
- All criteria in `goodness_of_fit` are on the negative-log-likelihood scale (lower is better), not the textbook −2 scale. `WAIC = -elpd_waic`.
- `hessian()` had an operator-precedence bug inherited from the R package (`adPi[i] * n ? a : b`); it is fixed and covered by `tests/test_waic.py::TestHessianExport::test_matches_numerical_hessian`.

## Development Commands

### Building and Installation
```bash
# System requirements
sudo apt-get install libgsl-dev

python3 -m venv venv
source venv/bin/activate
pip install -e .            # builds the C extension

# After modifying C code, rebuild in place
python setup.py build_ext --inplace
# The compiled extension appears as src/pydmm/pydmm_core.*.so
```

### Running Tests
```bash
pytest                                   # all tests
pytest -v
pytest tests/test_core.py
pytest --cov=pydmm --cov-report=term-missing
pytest tests/test_core.py::TestDirichletMixture::test_clustering_performance
```

### Assignment calibration (`validation/correlation/`)
```bash
venv/bin/python validation/correlation/experiment.py    # over-confidence vs correlation, candidate fixes
venv/bin/python validation/correlation/tempered_em.py   # tempered assignments vs tempered EM (needs the patch in that dir)
```

### Dirichlet-tree mixture (`validation/dtmm/`)
```bash
OMP_NUM_THREADS=1 venv/bin/python validation/dtmm/compare.py     # DMM vs DTMM on synthetic data
OMP_NUM_THREADS=1 venv/bin/python validation/dtmm/benchmark.py   # runtime (run alone)
```
`validation/dtmm/reference.py` is a NumPy reference implementation (prototype and oracle).
Write-up: `docs/dtmm_comparison.md`.

### WAIC validation (`validation/waic/`)
```bash
python validation/waic/generate_data.py            # comparison dataset
venv/bin/python validation/waic/compare.py         # pyDMM vs cached prodiphy results
venv/bin/python validation/waic/scaling.py         # BIC/AIC/WAIC vs N
# prodiphy reference runs use a separate env (PyMC, arviz, nutpie):
python -m venv validation/waic/.venv-prodiphy
validation/waic/.venv-prodiphy/bin/pip install -r validation/waic/requirements-prodiphy.txt
validation/waic/.venv-prodiphy/bin/python validation/waic/run_prodiphy.py
validation/waic/.venv-prodiphy/bin/python validation/waic/make_fixtures.py   # tests/data/waic/*.npz
```

## Dependencies and Build Requirements

- **System**: GSL (`libgsl-dev`), C compiler, `python3-dev`
- **Build**: setuptools, numpy (headers)
- **Runtime**: NumPy >= 1.19.0, pandas >= 1.0.0, scikit-learn >= 1.0.0 (SciPy comes with scikit-learn)
- **Testing**: pytest >= 6.0, pytest-cov >= 2.10
- PyMC/arviz/prodiphy are only used in the separate validation environment, never by the package or its tests.

## Test Suite

- **`test_core.py`**: DirichletMixture functionality, input validation, pandas integration
- **`test_c_extension.py`**: C extension integration, memory management, data layout handling
- **`test_model_selection.py`**: BIC/AIC model selection, overclustering detection
- **`test_sklearn_compatibility.py`**: scikit-learn API conformance
- **`test_dtmm.py`**: `Tree` builders, C tree fit (star == DMM, NLE vs Python likelihood, invalid trees), node selection, `DirichletTreeMixture` API and sklearn compatibility
- **`test_calibration.py`**: tempered assignments, `temperature="auto"` estimator, default path unchanged
- **`test_bug_fixes.py`**: regression tests for bugs fixed from `BUGS.md`; two tests
  compile C harnesses in `tests/c/` with the DMM sources (singular-Hessian fault injection,
  heap poisoning for the k-means start); skipped without a C compiler/GSL
- **`test_waic.py`**: Hessian export, pointwise log-likelihood vs PyMC, WAIC vs arviz, prodiphy draws through pyDMM code (reference values in `tests/data/waic/`)
- **`conftest.py`**: Shared test fixtures

## Key Design Patterns

1. **Scikit-learn API**: `fit()`, `predict_proba()`, `fit_predict()`; constructor params are stored unchanged (`n_components`, `verbose`, `random_state`, `compute_waic`, `n_waic_draws`, `temperature`, `restart_collapsed`)
2. **Pandas Integration**: DataFrames preserve sample/feature names through the pipeline
3. **Opt-in extras**: WAIC (and the Hessian export it needs) only run when `compute_waic=True`, and tempering only when `temperature` is set, so the default fit path stays as fast as before and its output bit-identical. New output-changing methods must follow the same pattern (off by default)
4. **Result Object Pattern**: `DirichletMixtureResult` gives structured access to all outputs
5. **Clustering Evaluation**: label-invariant metrics (purity, optimal assignment)

## Data Flow and Memory Management

1. **Input**: DataFrame or NumPy array → validation → int32 array
2. **C Interface**: transpose to column-major, allocate outputs, call `dirichlet_fit_main`, convert outputs to NumPy arrays with the expected shapes
3. **Output**: wrapped in `DirichletMixtureResult`; with `compute_waic=True`, WAIC metrics are added to `goodness_of_fit` and draws × samples log-likelihoods to `pointwise_log_likelihood`
4. **Memory**: output arrays are created with `NPY_ARRAY_OWNDATA` so NumPy frees them

## Common Issues

- **GSL not found**: install `libgsl-dev`
- **Data types**: input must be integer counts (int32)
- **Label assignment**: cluster labels are arbitrary; use label-invariant metrics
