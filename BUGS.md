# Known bugs

This file lists open bugs. Each entry gives the symptom, the cause and the steps to reproduce it. The text uses ASD-STE100 Simplified Technical English.

## 5. temperature="auto" makes a DirichletTreeMixture almost uniform

**Status:** Open. Found on 20 Nov 2026 in the American Gut case study (`validation/case_study/`, local only, not in git). Examined again on 21 Nov 2026 (branch `bug-fix`). Two fixes were tried. Neither fix is satisfactory, so the code is not changed (see "Tried fixes"). Workaround: use `temperature="one-step"`.

**Symptom:** `estimate_temperature()` can give a very small τ for a `DirichletTreeMixture`. Then all assignment probabilities are almost equal to the mixture weights.

- Case study (N = 1000, 236 features, K = 4): τ is 0.013. The mean maximum assignment probability is 0.35 (uniform value: 0.25). The bootstrap stability of the same partition is 0.66.
- Synthetic data (procedure below, K = 2): τ is 0.031. The mean maximum probability is 0.51 (uniform value: 0.50). The τ with the lowest log-loss for the true labels (oracle τ) is 0.61.
- In the simulations below, τ from "auto" is less than one third of the oracle τ for 11 % of the DTMM fits (0 % of the DMM fits).

**Cause:** `estimate_temperature()` (`src/pydmm/calibration.py`) is a fixed-point iteration τ ← f(τ). Each step weights the log-likelihood ratios (LLR) with the tempered assignments of the previous step. Two effects make f(τ) smaller than τ for all τ above a value near zero. Then the iteration has no fixed point except the degenerate one near zero, where the tempered assignments are almost the mixture weights.

1. Heavy tails. The DTMM ratios are skewed and heavy-tailed (case study: excess kurtosis up to 37; DMM fits on the same data: near 0). A small number of large ratios make the variance large, so τ = 2 · mean / var is small. This effect occurs for all K, also for K = 2.
2. Samples of other components (only for K > 2). When τ decreases, the weights of component k include more samples of a third component m. The ratio between k and l of these samples gives no evidence about k, but it decreases the mean and increases the variance.

For the case study, f(τ) is less than τ for all τ above 0.013 (for example, f(0.3) = 0.144, f(0.16) = 0.114, f(0.04) = 0.030). For the DMM fit on the same data, f(τ) is almost constant (0.08 to 0.10), and the iteration stops at 0.082.

**To reproduce:**

1. Build the package with `pip install -e .`.
2. Run this script from the repository root. It takes approximately 1 minute.

```python
import sys
sys.path.insert(0, "validation/dtmm")
import numpy as np
from compare import make
from pydmm import DirichletTreeMixture

X, labels, _ = make("tree", 10.0, 2)
X = X.astype(np.int32)
for temp in ("auto", "one-step"):
    m = DirichletTreeMixture(n_components=2, random_state=2, temperature=temp).fit(X)
    print(temp, m.temperature_, m.result_.group_assignments.max(axis=1).mean())
# auto      0.031  0.513
# one-step  0.761  0.738
```

3. Make sure that τ from "auto" is near 0.031 and that the mean maximum probability is near 0.51.

**Expected result:** τ near the oracle τ (0.61) or near the one-step value (0.76).

**Simulations used to examine the fixes:** 18 scenarios × 6 replicates, DMM and DTMM, true K. The data come from the generators in `validation/dtmm/compare.py` (tree, tree3, wide, lognorm, dmm) and `validation/correlation/experiment.py` (feature duplication, correlated modules). Fits with accuracy below 0.7 (wrong partition) are not used: 100 DMM fits and 74 DTMM fits remain. "Excess log-loss" is the log-loss minus the log-loss at the oracle τ. "Collapse" is the fraction of fits with τ below one third of the oracle τ.

| Estimator | DMM excess log-loss | DMM ECE | DTMM excess log-loss | DTMM ECE | DTMM collapse | Case study DTMM τ (max. p) |
|---|---|---|---|---|---|---|
| "auto" (current) | 0.023 | 0.042 | 0.104 | 0.054 | 11 % | 0.013 (0.35) |
| "one-step" (current) | 0.072 | 0.058 | 0.108 | 0.044 | 0 % | 0.162 (0.79) |
| Attempt 1: robust | 0.028 | 0.043 | 0.101 | 0.040 | 0 % | 0.201 (0.83) |
| Attempt 2: pairwise | 0.023 | 0.042 | 0.103 | 0.053 | 11 % | 0.093 (0.68) |
| Pairwise and robust | 0.029 | 0.043 | 0.101 | 0.040 | 0 % | 0.235 (0.85) |
| Maximum likelihood (see below) | 0.023 | 0.042 | 0.093 | 0.043 | 4 % | 0.029 (0.45) |

**Tried fixes:**

1. Attempt 1: robust estimate in each step. Use the weighted median instead of the weighted mean, and (1.4826 · weighted MAD)² instead of the weighted variance. For normal ratios this gives the same τ.
   - Result: the collapse stops (0 % of the DTMM fits). For the scenario of the reproduction above (tree, θ = 10), the mean excess log-loss decreases from 0.080 to 0.007.
   - Problem: the DMM results become worse (excess log-loss 0.023 → 0.028; for correlated modules with ρ = 0.9, 0.026 → 0.042). For the DTMM "lognorm, ρ = 0.6" scenario, the excess log-loss increases from 0.012 to 0.133. The case-study probabilities (mean maximum 0.83) are more confident than the bootstrap stability (0.66). The DMM case-study τ changes from 0.082 to 0.136.
2. Attempt 2: pairwise weights. For the pair (k, l), use only the samples whose most likely component is k or l. Weight them with the two-component probability z_ik / (z_ik + z_il). This removes effect 2. The most likely component does not change with τ. For K = 2 the estimator does not change (bit-identical).
   - Result: the case study is correct. τ is 0.093, and the mean maximum probability (0.68) agrees with the bootstrap stability (0.66). The DMM results do not change (K = 2), and the DMM case-study τ changes only a little (K = 4: 0.082 → 0.087; K = 7: 0.096 → 0.111).
   - Problem: it does not remove effect 1. The synthetic reproduction above (K = 2) does not change. Effect 2 did not occur in synthetic data: on 4-group Dirichlet-tree data (N = 600, K = 4, 24 DMM and 24 DTMM fits, 13 of them with accuracy 0.7 or more), the current estimator did not collapse, and attempt 2 gave the same τ. Thus attempt 2 is tested only on the case-study ratios.

**Also examined (not implemented):**

- Hard assignments as weights. The weights do not depend on τ, so the estimate cannot drift. But the hard assignments cut the ratio distribution at zero, and τ is too large (DMM excess log-loss 0.074, like "one-step").
- Maximum likelihood of Good's model. For each pair, fit LLR ~ π N(a, 2a/τ) + (1 − π) N(−b, 2b/τ) to the samples of k and l. The DTMM log-loss is the best of all estimators, but τ still collapses (4 % of the DTMM fits, 40 % on the "tree, θ = 10" scenario). On the case study τ is 0.029.

**Possible fix:** These changes change the output of `temperature="auto"`. Get approval first. Do the simulations in `validation/correlation/` again after the change.

- Add attempt 2. It is correct on the case study and has no effect on K = 2. It does not repair effect 1.
- For effect 1, use "one-step" for `DirichletTreeMixture(temperature="auto")` only, or tell users to use "one-step" for the DTMM (the README already does). In the simulations above, "one-step" never collapsed on the DTMM, and its DTMM ECE is lower than the ECE of "auto" (0.044 against 0.054).
