# Fix Summary: Prediction Class Mismatch

## Issue
There was a discrepancy between class assignments from the C portion of the code (accessed via `.result.get_best_component()` and `.result.get_group_assignments_df()`) and the Python `.predict()` and `.predict_proba()` methods when applied to training data.

## Root Cause
The Python `predict_proba()` method was using the **Dirichlet distribution** likelihood, while the C fitting code uses the **Dirichlet-multinomial distribution** likelihood. These are fundamentally different:

- **Dirichlet**: For continuous compositional data on the simplex
- **Dirichlet-multinomial**: For discrete count data (the correct choice for this use case)

## Changes Made

### 1. Fixed `core.py` (src/pydmm/core.py)
- **Renamed and rewrote** `_compute_dirichlet_log_likelihood()` to `_compute_dirichlet_multinomial_log_likelihood()`
- **New implementation** matches the C code's `neg_log_evidence_i()` function:
  ```python
  log_likelihood = log B(alpha + counts) - log B(alpha)
  where B(alpha) = sum(log Gamma(alpha_i)) - log Gamma(sum(alpha_i))
  ```
- **Updated** `predict_proba()` to call the new method

### 2. Added Regression Test (tests/test_core.py)
- **New test**: `test_predict_matches_c_code_assignments_on_training_data()`
- **Validates**:
  - Class assignments from C code match `predict()` on training data
  - Probabilities from C code closely match `predict_proba()` on training data
  - Uses `rtol=1e-5, atol=1e-8` to allow for minor numerical differences

### 3. Documentation (DIAGNOSIS.md)
- **Detailed analysis** of the mathematical difference between the two distributions
- **Explanation** of why Dirichlet-multinomial is correct for count data
- **Impact assessment** of the bug

## Mathematical Details

### Before (Incorrect)
```
P(p | alpha) = [Gamma(sum(alpha)) / prod(Gamma(alpha_i))] * prod(p_i^(alpha_i - 1))
```
This treated data as continuous proportions.

### After (Correct)
```
P(x | alpha) = [prod(Gamma(alpha_i + x_i)) / Gamma(sum(alpha_i + x_i))] *
               [Gamma(sum(alpha_i)) / prod(Gamma(alpha_i))]
```
This properly handles discrete count data.

## Testing

### What Was Verified
✓ Python syntax is correct (compiled successfully)
✓ Logic matches C code implementation
✓ New test added to prevent regression

### What Needs To Be Verified
⚠️ **Full test suite** needs to run with:
```bash
pip install -e .[dev]
pytest tests/ -v --cov=pydmm --cov-report=term-missing
```

⚠️ **Expected behavior after fix**:
1. All existing tests should still pass
2. New test `test_predict_matches_c_code_assignments_on_training_data` should pass
3. Predictions on training data should now match C code assignments

⚠️ **Note**: Cannot run tests locally due to GSL installation requirement and network issues

## Impact

### Benefits
✓ Predictions are now mathematically correct for count data
✓ Consistency between fitting and prediction
✓ `.predict()` on training data matches `.result.get_best_component()`
✓ Proper probability calculations for classification

### Backward Compatibility
⚠️ **Breaking change**: Predictions from this version will differ from previous versions
- Previous predictions were mathematically incorrect
- Users who saved predictions should re-run with the fixed version
- Model parameters (alpha estimates) from old fits are still valid

## Files Changed
- `src/pydmm/core.py`: Core fix
- `tests/test_core.py`: Regression test
- `DIAGNOSIS.md`: Technical documentation (new)
- `FIX_SUMMARY.md`: This summary (new)
