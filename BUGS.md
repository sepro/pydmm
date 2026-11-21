# Known bugs

This file lists open bugs. Each entry gives the symptom, the cause and the steps to reproduce it. The text uses ASD-STE100 Simplified Technical English.

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

## 5. temperature="auto" makes a DirichletTreeMixture almost uniform

**Status:** Open. Found on 20 Nov 2026 in the American Gut case study (`validation/case_study/`, local only, not in git). Workaround: use `temperature="one-step"`. This option stops after the first step of the iteration.

**Symptom:** On real data, `estimate_temperature()` can give a very small τ for a `DirichletTreeMixture`. For the case-study fit (N = 1000, 236 features, K = 4), τ is 0.013. The mean maximum assignment probability then is 0.35, and 0 of 1000 samples have a probability above 0.8. This is almost the uniform value 0.25. The bootstrap stability of the same partition is 0.66, so the tempered probabilities are too low. For `DirichletMixture` fits on the same data, the estimate is stable (τ ≈ 0.08–0.10).

**Cause:** `estimate_temperature()` (`src/pydmm/calibration.py`) is a fixed-point iteration. Each step weights the log-likelihood ratios with the tempered assignments of the previous step. When τ decreases, the weights of component k include more samples of the other components. Then the weighted mean of the ratio decreases and its variance increases, so the next τ = 2 · mean / var is smaller again. For this fit, the first step gives τ = 0.16. The iteration then decreases to 0.115, 0.093, 0.078, … and stops at 0.013. The DTMM ratios are heavy-tailed (shared nodes drop out, and the remaining nodes give large ratios for a small number of samples), and this makes the effect stronger.

**To reproduce:**

1. Build the package with `pip install -e .`.
2. Run this script from the repository root. The fit takes approximately 10 minutes.

```python
import sys
sys.path.insert(0, "validation/case_study")
from prepare_data import load_subset
from pydmm import DirichletTreeMixture

X = load_subset(1000)
m = DirichletTreeMixture(n_components=4, random_state=4, temperature="auto").fit(X)
print(m.temperature_)                                      # 0.013
print(m.result_.group_assignments.max(axis=1).mean())      # 0.35
```

3. Make sure that τ is near 0.013 and that the mean maximum probability is near 0.35.

**Expected result:** τ near the first-step value (0.16) or another value that agrees with the bootstrap stability.

**Possible fix:** Use the first step only (weights from the untempered assignments), or use hard assignments as weights in each step. Then the weights do not depend on τ, and the feedback stops. Use a robust estimate of the variance for heavy-tailed ratios. These fixes change the output only when `temperature="auto"` is set. Do the simulations in `validation/correlation/` again after the change.
