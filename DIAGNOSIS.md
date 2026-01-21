# Diagnosis: Class Assignment Discrepancy

## Problem Statement
There is a discrepancy between class assignments obtained from the C fitting code (via `.result.get_best_component()` and `.result.get_group_assignments_df()`) and the Python `.predict()` and `.predict_proba()` methods when applied to the same training data.

## Root Cause

### C Code (Fitting)
The C code in `dirichlet_fit_standalone.c` uses the **Dirichlet-multinomial distribution** to compute group assignments during fitting. Specifically:

1. In `calc_z()` (lines 267-300), it computes responsibilities using `neg_log_evidence_i()`
2. `neg_log_evidence_i()` (lines 241-265) computes the Dirichlet-multinomial log likelihood:
   ```
   log P(x | alpha) = sum(log Gamma(alpha_i + x_i)) - log Gamma(sum(alpha_i + x_i))
                      + sum(log Gamma(alpha_i)) - log Gamma(sum(alpha_i))
   ```

This is the **correct** formula for count data, as it accounts for the discrete nature of the observations.

### Python Code (Prediction)
The Python code in `core.py` uses the **Dirichlet distribution** (not Dirichlet-multinomial) to compute predictions. Specifically:

1. `predict_proba()` (lines 295-332) calls `_compute_dirichlet_log_likelihood()`
2. `_compute_dirichlet_log_likelihood()` (lines 212-237) computes:
   - Converts counts to proportions with pseudocount
   - Computes Dirichlet PDF: `log(B(alpha)) + sum((alpha_i - 1) * log(x_i))`

This is **incorrect** for count data because:
- It treats the data as continuous proportions on the simplex
- It doesn't account for the discrete counting process
- The multinomial coefficient is missing

## Mathematical Difference

**Dirichlet-multinomial** (used by C code):
```
P(x | alpha, n) = [Gamma(n+1) / prod(Gamma(x_i+1))] *
                  [Gamma(sum(alpha)) / prod(Gamma(alpha_i))] *
                  [prod(Gamma(alpha_i + x_i)) / Gamma(sum(alpha_i) + n)]
```

**Dirichlet** (incorrectly used by Python code):
```
P(p | alpha) = [Gamma(sum(alpha)) / prod(Gamma(alpha_i))] * prod(p_i^(alpha_i - 1))
```

The Dirichlet-multinomial is the compound distribution that arises from:
1. Drawing probabilities from a Dirichlet distribution
2. Drawing counts from a Multinomial distribution with those probabilities

## Solution
The `_compute_dirichlet_log_likelihood()` method in `core.py` must be replaced with a method that computes the **Dirichlet-multinomial** log likelihood, matching the C code's calculation.

## Impact
This bug causes:
- Incorrect predictions on new data
- Inconsistent results when calling `.predict()` on training data vs. using `.result.get_best_component()`
- Potentially incorrect classification boundaries
