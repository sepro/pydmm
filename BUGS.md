# Known bugs

This file lists open bugs. Each entry gives the symptom, the cause and the steps to reproduce it. The text uses ASD-STE100 Simplified Technical English.

## 1. DirichletMixture stops with all components equal

**Status:** Open. Found on 20 Nov 2026 (branch `dtmm`).

**Symptom:** The fit gives the same assignment probabilities to all samples. With two components, each probability is 0.5. The mixture weights are equal. The clusters are not found.

**Cause:** The soft k-means start in `kmeans()` (`src/pydmm/dirichlet_node.c`) uses `SOFT_BETA = 50`. When the samples have almost the same composition, the distances are very small. Then all samples get almost equal assignments, and all components start almost equal. EM then stays at this symmetric saddle point.

**Where it occurs:**

- Many features (for example 240) and many samples (for example 300).
- Low-noise counts with small differences between the groups.
- One replicate in `validation/correlation/experiment.py` (duplication m = 4, replicate 5).

**To reproduce:**

1. Build the package with `pip install -e .`.
2. Run this script:

```python
import numpy as np
from pydmm import DirichletMixture

rng = np.random.default_rng(0)
n, s = 150, 100
p = rng.dirichlet(np.ones(s) * 5)
labels = rng.integers(0, 2, n)
X = np.zeros((n, s), np.int32)
for i in range(n):
    q = p.copy()
    if labels[i]:
        q[:5] *= 1.5
    X[i] = rng.multinomial(20000, rng.dirichlet(q / q.sum() * 2000))

m = DirichletMixture(n_components=2).fit(X)
print(m.result_.mixture_weights)              # [0.5 0.5]
print(m.result_.group_assignments[:3])        # all rows [0.5 0.5]
```

3. Make sure that all assignment probabilities are 0.5.

**Expected result:** The fit finds the two groups. `DirichletTreeMixture` finds them on the same data (accuracy 0.97).

**Possible fix:** Use the same fix as in `src/pydmm/dirichlet_tree_fit.c`. After EM, examine if all samples have the same assignments (`collapsed()`). If yes, start again from hard k-means with k-means++ seeding (`hard_kmeans()`). Note: this fix changes the output of `DirichletMixture` for these data. Make it an option, or approve the change first.

## 2. DirichletMixture stops the Python process when the Hessian is singular

**Status:** Open. Found by code review on 20 Nov 2026. Not seen in a test run.

**Symptom:** The Python process stops. No Python exception occurs.

**Cause:** In `dirichlet_fit_main()` (`src/pydmm/dirichlet_fit_standalone.c`), the code calls `exit(1)` when `gsl_linalg_LU_decomp()` fails on the Hessian.

**To reproduce:**

No tested data set is available yet.

1. Find data where a component Hessian is singular. Possible examples (not tested): a component that gets no samples, or very large `n_components` for few samples.
2. Fit `DirichletMixture` on these data.
3. Make sure that the Python process stops.

**Expected result:** The fit continues. The confidence intervals are NaN, or the code raises a Python exception.

**Possible fix:** Do not call `exit(1)`. Set the errors and `LogDet` to NaN. `dirichlet_tree_fit.c` uses this procedure.

## 3. The soft k-means start reads memory that is not initialised

**Status:** Open. Found by code review on 20 Nov 2026.

**Symptom:** None seen. The result can change between runs or between computers.

**Cause:** `dirichlet_fit_main()` allocates `aadLambda` with `calloc`. The file maps `calloc` to `malloc`, so the memory is not set to zero. The code then gives `aadLambda` to `kmeans()` as `aadMu`. In the first iteration, `kmeans()` reads `aadMu` to calculate `dMaxChange`.

**To reproduce:**

1. Build the extension with `-fsanitize=memory` (clang), or run the tests with `valgrind --track-origins=yes`.
2. Fit `DirichletMixture` on any data.
3. Make sure that the tool reports a read of uninitialised memory in `kmeans()`.

**Possible fix:** Set `aadLambda` to zero before the call to `kmeans()`. This fix can change the output of `DirichletMixture`. Compare the results with the saved baseline first.

## 4. The random number generator is not released

**Status:** Open. Found by code review on 20 Nov 2026.

**Symptom:** Each `DirichletMixture` fit keeps a small quantity of memory.

**Cause:** `dirichlet_fit_main()` calls `gsl_rng_alloc()` but does not call `gsl_rng_free()`.

**To reproduce:**

1. Fit `DirichletMixture` 10,000 times in one Python process.
2. Monitor the memory of the process.
3. Make sure that the memory increases.

**Possible fix:** Call `gsl_rng_free(ptGSLRNG)` at the end of `dirichlet_fit_main()`. This fix does not change the results.
