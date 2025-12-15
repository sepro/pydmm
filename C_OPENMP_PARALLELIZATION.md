# Parallelization with OpenMP (C Code - No Porting Required)

**Project:** pyDMM (Python Dirichlet Mixture Model)
**Analysis Date:** 2025-12-15
**Current Implementation:** C with GSL + Python C API
**Proposed Enhancement:** Add OpenMP parallelization to existing C code

---

## Executive Summary

This document provides an **alternative approach** to the Rust migration analyzed in `PARALLELIZATION_ANALYSIS.md`. Instead of porting to Rust, we can add parallelization to the existing C code using **OpenMP**, achieving similar performance gains (**4-8x speedup**) with **minimal code changes**.

**Key Advantages over Rust Migration:**
- ✅ **Much simpler**: Add compiler pragmas, no language change
- ✅ **Faster implementation**: Days instead of months
- ✅ **Lower risk**: Smaller code changes, easier to validate
- ✅ **No new dependencies**: OpenMP widely available with GCC/Clang
- ✅ **Incremental adoption**: Can parallelize one loop at a time

**Trade-offs vs. Rust:**
- ❌ Still has C's memory safety issues
- ❌ Less modern tooling
- ⚠️ Slightly higher overhead than Rayon (but negligible for these workloads)
- ⚠️ Manual thread safety management

---

## 1. What is OpenMP?

**OpenMP** (Open Multi-Processing) is an API for parallel programming in C/C++/Fortran that uses:
- **Compiler directives** (`#pragma omp`) to mark parallel regions
- **Runtime library** for thread management
- **Environment variables** for configuration

**Example:**
```c
// Sequential loop
for (int i = 0; i < n; i++) {
    result[i] = expensive_computation(data[i]);
}

// Parallel loop (just add one line!)
#pragma omp parallel for
for (int i = 0; i < n; i++) {
    result[i] = expensive_computation(data[i]);
}
```

**Compiler Support:**
- GCC 4.2+ (all modern versions)
- Clang 3.7+ (all modern versions)
- MSVC 2008+ (Windows)
- ICC (Intel Compiler)

---

## 2. Parallelization Implementation

### 2.1 **CRITICAL: Parallel M-Step (Component Optimization)**

**Current Code** (`dirichlet_fit_standalone.c:518-519`):
```c
for (k = 0; k < K; k++) {
    optimise_lambda_k(aadLambda[k], data, aadZ[k]);
}
```

**Parallelized Version:**
```c
#pragma omp parallel for schedule(dynamic)
for (k = 0; k < K; k++) {
    optimise_lambda_k(aadLambda[k], data, aadZ[k]);
}
```

**Explanation:**
- `parallel for`: Distribute loop iterations across threads
- `schedule(dynamic)`: Dynamic load balancing (some components may converge faster)
- Each iteration is completely independent (no data races)

**Expected Speedup:**
- K=5, 8 cores: **~4x**
- K=10, 16 cores: **~8x**

---

### 2.2 **CRITICAL: Parallel E-Step (Posterior Probability Computation)**

**Current Code** (`dirichlet_fit_standalone.c:282-299`):
```c
for (i = 0; i < N; i++) {
    double dSum = 0.0;
    double dOffset = BIG_DBL;
    for (k = 0; k < K; k++) {
        double dNegLogEviI = neg_log_evidence_i(data, data->aanX + i,
                                                 aadLambda[k],
                                                 aadLngammaLambda0 + k*S);
        if (dNegLogEviI < dOffset)
            dOffset = dNegLogEviI;
        adStore[k] = dNegLogEviI;
    }
    for (k = 0; k < K; k++) {
        aadZ[k][i] = adW[k] * exp(-(adStore[k] - dOffset));
        dSum += aadZ[k][i];
    }
    for (k = 0; k < K; k++)
        aadZ[k][i] /= dSum;
}
```

**Parallelized Version:**
```c
#pragma omp parallel for schedule(static) private(k, dSum, dOffset)
for (i = 0; i < N; i++) {
    double dSum = 0.0;
    double dOffset = BIG_DBL;
    double adStore[K];  // Thread-private array

    for (k = 0; k < K; k++) {
        double dNegLogEviI = neg_log_evidence_i(data, data->aanX + i,
                                                 aadLambda[k],
                                                 aadLngammaLambda0 + k*S);
        if (dNegLogEviI < dOffset)
            dOffset = dNegLogEviI;
        adStore[k] = dNegLogEviI;
    }
    for (k = 0; k < K; k++) {
        aadZ[k][i] = adW[k] * exp(-(adStore[k] - dOffset));
        dSum += aadZ[k][i];
    }
    for (k = 0; k < K; k++)
        aadZ[k][i] /= dSum;
}
```

**Explanation:**
- `schedule(static)`: Static partitioning (each iteration has similar cost)
- `private(...)`: Each thread gets its own copy of these variables
- `adStore` array moved inside loop (was stack-allocated outside)

**Expected Speedup:**
- N=1000, 8 cores: **~5-6x**
- N=10000, 16 cores: **~8-10x**

---

### 2.3 **HIGH PRIORITY: Parallel K-means Distance Computation**

**Current Code** (`dirichlet_fit_standalone.c:82-95`):
```c
for (i = 0; i < N; i++) {
    double dNorm = 0.0, adDist[K];
    for (k = 0; k < K; k++) {
        adDist[k] = 0.0;
        for (j = 0; j < S; j++) {
            const double dDiff = aadMu[k][j] - aadY[j * N + i];
            adDist[k] += dDiff * dDiff;
        }
        adDist[k] = sqrt(adDist[k]);
        dNorm += exp(-SOFT_BETA * adDist[k]);
    }
    for (k = 0; k < K; k++)
        aadZ[k][i] = exp(-SOFT_BETA * adDist[k]) / dNorm;
}
```

**Parallelized Version:**
```c
#pragma omp parallel for schedule(static) private(k, j, dNorm)
for (i = 0; i < N; i++) {
    double dNorm = 0.0;
    double adDist[K];  // Thread-private array

    for (k = 0; k < K; k++) {
        adDist[k] = 0.0;
        for (j = 0; j < S; j++) {
            const double dDiff = aadMu[k][j] - aadY[j * N + i];
            adDist[k] += dDiff * dDiff;
        }
        adDist[k] = sqrt(adDist[k]);
        dNorm += exp(-SOFT_BETA * adDist[k]);
    }
    for (k = 0; k < K; k++)
        aadZ[k][i] = exp(-SOFT_BETA * adDist[k]) / dNorm;
}
```

**Expected Speedup:** **~4-6x** on 8+ cores

---

### 2.4 **MEDIUM PRIORITY: Parallel Negative Log-Likelihood**

**Current Code** (`dirichlet_fit_standalone.c:329-357`):
```c
for (i = 0; i < N; i++) {
    double dProb = 0.0, dFactor = 0.0, dSum = 0.0, adLogStore[K],
        dOffset = -BIG_DBL;

    for (j = 0; j < S; j++) {
        dSum += aanX[j * N + i];
        dFactor += gsl_sf_lngamma(aanX[j * N + i] + 1.0);
    }
    dFactor -= gsl_sf_lngamma(dSum + 1.0);

    for (k = 0; k < K; k++) {
        double dSumAlphaKN = 0.0, dLogBAlphaN = 0.0;
        for (j = 0; j < S; j++) {
            int countN = aanX[j * N + i];
            double dAlphaN = exp(aadLambda[k][j]) + countN;
            dSumAlphaKN += dAlphaN;
            dLogBAlphaN += countN ? gsl_sf_lngamma(dAlphaN) :
                aadLngammaLambda0[k * S + j];
        }
        dLogBAlphaN -= gsl_sf_lngamma(dSumAlphaKN);
        adLogStore[k] = dLogBAlphaN - adLogBAlpha[k] - dFactor;
        if (adLogStore[k] > dOffset)
            dOffset = adLogStore[k];
    }

    for (k = 0; k < K; k++)
        dProb += adPi[k]*exp(-dOffset + adLogStore[k]);
    dRet += log(dProb)+dOffset;
}
```

**Parallelized Version:**
```c
double dRet = 0.0;
#pragma omp parallel for schedule(static) reduction(+:dRet) \
    private(j, k, dProb, dFactor, dSum, dOffset)
for (i = 0; i < N; i++) {
    double dProb = 0.0, dFactor = 0.0, dSum = 0.0,
        dOffset = -BIG_DBL;
    double adLogStore[K];  // Thread-private

    for (j = 0; j < S; j++) {
        dSum += aanX[j * N + i];
        dFactor += gsl_sf_lngamma(aanX[j * N + i] + 1.0);
    }
    dFactor -= gsl_sf_lngamma(dSum + 1.0);

    for (k = 0; k < K; k++) {
        double dSumAlphaKN = 0.0, dLogBAlphaN = 0.0;
        for (j = 0; j < S; j++) {
            int countN = aanX[j * N + i];
            double dAlphaN = exp(aadLambda[k][j]) + countN;
            dSumAlphaKN += dAlphaN;
            dLogBAlphaN += countN ? gsl_sf_lngamma(dAlphaN) :
                aadLngammaLambda0[k * S + j];
        }
        dLogBAlphaN -= gsl_sf_lngamma(dSumAlphaKN);
        adLogStore[k] = dLogBAlphaN - adLogBAlpha[k] - dFactor;
        if (adLogStore[k] > dOffset)
            dOffset = adLogStore[k];
    }

    for (k = 0; k < K; k++)
        dProb += adPi[k]*exp(-dOffset + adLogStore[k]);

    // This gets reduced across threads
    dRet += log(dProb)+dOffset;
}
```

**Explanation:**
- `reduction(+:dRet)`: Automatically sum `dRet` across all threads
- Each thread accumulates its partial sum, then OpenMP combines them

**Expected Speedup:** **~3-5x** on 8+ cores

---

### 2.5 **LOW PRIORITY: Parallel Hessian Computation**

**Current Code** (`dirichlet_fit_standalone.c:544-569`):
```c
for (k = 0; k < K; k++) {
    data->adPi = aadZ[k];
    if (k > 0)
        dLogDet += 2.0 * log(N) - log(adW[k]);
    hessian(ptHessian, aadLambda[k], data);

    status = gsl_linalg_LU_decomp(ptHessian, p, &signum);
    // ... error handling ...
    gsl_linalg_LU_invert(ptHessian, p, ptInverseHessian);
    for (j = 0; j < S; j++) {
        aadErr[k][j] = gsl_matrix_get(ptInverseHessian, j, j);
        dTemp = gsl_matrix_get(ptHessian, j, j);
        dLogDet += log(fabs(dTemp));
    }
}
```

**Parallelized Version:**
```c
// Note: Need separate matrices per thread due to GSL operations
#pragma omp parallel for schedule(dynamic) \
    private(j, status, signum, dTemp) \
    reduction(+:dLogDet)
for (k = 0; k < K; k++) {
    // Allocate thread-local matrices
    gsl_matrix *ptHessian_local = gsl_matrix_alloc(S, S);
    gsl_matrix *ptInverseHessian_local = gsl_matrix_alloc(S, S);
    gsl_permutation *p_local = gsl_permutation_alloc(S);

    struct data_t data_local = *data;  // Copy data structure
    data_local.adPi = aadZ[k];

    double dLogDetContrib = 0.0;
    if (k > 0)
        dLogDetContrib = 2.0 * log(N) - log(adW[k]);

    hessian(ptHessian_local, aadLambda[k], &data_local);

    int status_local, signum_local;
    status_local = gsl_linalg_LU_decomp(ptHessian_local, p_local, &signum_local);

    if (status_local == GSL_SUCCESS) {
        gsl_linalg_LU_invert(ptHessian_local, p_local, ptInverseHessian_local);
        for (j = 0; j < S; j++) {
            aadErr[k][j] = gsl_matrix_get(ptInverseHessian_local, j, j);
            double dTemp_local = gsl_matrix_get(ptHessian_local, j, j);
            dLogDetContrib += log(fabs(dTemp_local));
        }
    }

    dLogDet += dLogDetContrib;

    // Clean up thread-local resources
    gsl_matrix_free(ptHessian_local);
    gsl_matrix_free(ptInverseHessian_local);
    gsl_permutation_free(p_local);
}
```

**Note:** This is more complex due to GSL's requirement for separate matrices per thread.

**Expected Speedup:** **~2-3x** (but low impact since Hessian is <5% of runtime)

---

## 3. Complete Modified File

Here are the key changes needed to `dirichlet_fit_standalone.c`:

### 3.1 Add OpenMP Header

```c
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_permutation.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>  // Add this
#endif

#include "dirichlet_fit_standalone.h"
```

### 3.2 Modified `kmeans()` Function

```c
static void kmeans(struct data_t *data, gsl_rng *ptGSLRNG,
                   double* adW, double **aadZ, double **aadMu)
{
    const int S = data->S, N = data->N, K = data->K,
        *aanX = data->aanX;
    int i, j, k, iter = 0;

    double *aadY, *adMu;
    double dMaxChange = BIG_DBL;

    if (data->verbose)
        printf("  Soft kmeans\n");

    aadY = (double *) calloc(N * S, sizeof(double));
    adMu = (double *) calloc(S, sizeof(double));

    // Parallel normalization
    #pragma omp parallel for private(j) schedule(static)
    for (i = 0; i < N; i++) {
        double dTotal = 0.0;
        for (j = 0; j < S; j++)
            dTotal += aanX[j * N + i];
        for (j = 0; j < S; j++)
            aadY[j * N + i] = (aanX[j * N + i]) / dTotal;
    }

    /* initialise (not parallelized due to RNG) */
    for (i = 0; i < N; i++) {
        k = gsl_rng_uniform_int (ptGSLRNG, K);
        for (j = 0; j < K; j++)
            aadZ[j][i] = 0.0;
        aadZ[k][i] = 1.0;
    }

    while (dMaxChange > K_MEANS_THRESH && iter < MAX_ITER) {
        /* update mu (not easily parallelizable due to dependencies) */
        dMaxChange = 0.0;
        for (i = 0; i < K; i++){
            double dNormChange = 0.0;
            adW[i] = 0.0;
            for (j = 0; j < N; j++)
                adW[i] += aadZ[i][j];
            for (j = 0; j < S; j++) {
                adMu[j] = 0.0;
                for (k = 0; k < N; k++)
                    adMu[j] += aadZ[i][k] * aadY[j * N + k];
            }

            for (j = 0; j < S; j++) {
                double dDiff = 0.0;
                adMu[j] /= adW[i];
                dDiff = (adMu[j] - aadMu[i][j]);
                dNormChange += dDiff * dDiff;
                aadMu[i][j] = adMu[j];
            }
            dNormChange = sqrt(dNormChange);
            if (dNormChange > dMaxChange)
                dMaxChange = dNormChange;
        }

        /* calc distances and update Z - PARALLELIZED */
        #pragma omp parallel for private(k, j) schedule(static)
        for (i = 0; i < N; i++) {
            double dNorm = 0.0;
            double adDist[K];

            for (k = 0; k < K; k++) {
                adDist[k] = 0.0;
                for (j = 0; j < S; j++) {
                    const double dDiff = aadMu[k][j] - aadY[j * N + i];
                    adDist[k] += dDiff * dDiff;
                }
                adDist[k] = sqrt(adDist[k]);
                dNorm += exp(-SOFT_BETA * adDist[k]);
            }
            for (k = 0; k < K; k++)
                aadZ[k][i] = exp(-SOFT_BETA * adDist[k]) / dNorm;
        }
        iter++;
        if (data->verbose && (iter % 10 == 0))
            printf("    iteration %d change %f\n", iter, dMaxChange);
    }

    free(aadY);
    free(adMu);
}
```

### 3.3 Modified `calc_z()` Function

```c
static void calc_z(double **aadZ, const struct data_t *data,
                   const double *adW, double **aadLambda)
{
    int i, k, j;
    const int N = data->N, K = data->K, S = data->S;
    double *aadLngammaLambda0 = (double*)calloc(S*K,sizeof(double));

    // Pre-compute log-gamma values (not worth parallelizing, small loop)
    for(k = 0; k < K; k++) {
        for(j = 0; j < S; j++) {
            const double dAlpha = exp(aadLambda[k][j]);
            aadLngammaLambda0[k*S +j] = gsl_sf_lngamma(dAlpha);
        }
    }

    // PARALLELIZED E-step
    #pragma omp parallel for private(k) schedule(static)
    for (i = 0; i < N; i++) {
        double dSum = 0.0;
        double dOffset = BIG_DBL;
        double adStore[K];  // Thread-private

        for (k = 0; k < K; k++) {
            double dNegLogEviI =
                neg_log_evidence_i(data, data->aanX + i, aadLambda[k],
                                   aadLngammaLambda0 + k*S);
            if (dNegLogEviI < dOffset)
                dOffset = dNegLogEviI;
            adStore[k] = dNegLogEviI;
        }
        for (k = 0; k < K; k++) {
            aadZ[k][i] = adW[k] * exp(-(adStore[k] - dOffset));
            dSum += aadZ[k][i];
        }
        for (k = 0; k < K; k++)
            aadZ[k][i] /= dSum;
    }

    free(aadLngammaLambda0);
}
```

### 3.4 Modified EM Algorithm in `dirichlet_fit_main()`

```c
/* simple EM algorithm */
int iter = 0;
double dNLL = 0.0, dNew, dChange = BIG_DBL;

if (data->verbose)
    printf("  Expectation Maximization\n");

while (dChange > 1.0e-6 && iter < 100) {
    calc_z(aadZ, data, adW, aadLambda); /* latent var expectation (parallelized inside) */

    // PARALLELIZED M-step
    #pragma omp parallel for schedule(dynamic)
    for (k = 0; k < K; k++) {
        optimise_lambda_k(aadLambda[k], data, aadZ[k]);
    }

    for (k = 0; k < K; k++) { /* current likelihood & weights */
        adW[k] = 0.0;
        for(i = 0; i < N; i++)
            adW[k] += aadZ[k][i];
    }

    dNew = neg_log_likelihood(adW, aadLambda, data);
    dChange = fabs(dNLL - dNew);
    dNLL = dNew;
    iter++;

    if (data->verbose && (iter % 10) == 0)
        printf("    iteration %d change %f\n", iter, dChange);
}
```

---

## 4. Build Configuration Changes

### 4.1 Modified `setup.py`

```python
from setuptools import setup, Extension, find_packages
import numpy as np
import os

# Check if OpenMP is available
def has_openmp():
    import subprocess
    import tempfile

    test_code = """
    #include <omp.h>
    int main() {
        #pragma omp parallel
        { }
        return 0;
    }
    """

    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
        f.write(test_code)
        temp_file = f.name

    try:
        result = subprocess.run(
            ['gcc', '-fopenmp', temp_file, '-o', temp_file + '.out'],
            capture_output=True
        )
        success = result.returncode == 0
        os.unlink(temp_file)
        if success:
            os.unlink(temp_file + '.out')
        return success
    except:
        return False

# Configure compiler flags
extra_compile_args = ['-O3', '-Wall', '-std=c99']
extra_link_args = []

if has_openmp():
    print("OpenMP support detected - enabling parallelization")
    extra_compile_args.append('-fopenmp')
    extra_link_args.append('-fopenmp')
else:
    print("OpenMP not available - building sequential version")

# Define the C extension module
ext_modules = [
    Extension(
        'pydmm.pydmm_core',
        sources=[
            'src/pydmm/wrapper.c',
            'src/pydmm/dirichlet_fit_standalone.c'
        ],
        include_dirs=[
            np.get_include(),
            'src/pydmm',
            '/usr/include/gsl',
        ],
        libraries=['gsl', 'gslcblas', 'm'],
        library_dirs=['/usr/lib/x86_64-linux-gnu'],
        define_macros=[
            ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name='pydmm',
    version='0.1.1',  # Bump version
    description='Python wrapper for Dirichlet Mixture Model fitting (with OpenMP parallelization)',
    # ... rest of setup.py unchanged ...
    ext_modules=ext_modules,
    # ... rest unchanged ...
)
```

### 4.2 Environment Variables

Users can control OpenMP behavior:

```bash
# Set number of threads
export OMP_NUM_THREADS=8

# Run Python script
python my_analysis.py

# Or set inline
OMP_NUM_THREADS=16 python my_analysis.py
```

---

## 5. Thread Safety Considerations

### 5.1 GSL Random Number Generator

**Issue:** GSL's RNG is not thread-safe by default.

**Solution:** Only used in initialization (k-means), which we don't parallelize for the RNG-dependent parts.

### 5.2 GSL Special Functions

**Thread Safety:** GSL special functions (`gsl_sf_lngamma`, `gsl_sf_psi`, etc.) are **thread-safe** as long as error handling is disabled:

```c
gsl_set_error_handler_off();  // Already in code at line 469
```

### 5.3 GSL Optimization

**Thread Safety:** `gsl_multimin_fdfminimizer` creates independent state per call, so parallel calls to `optimise_lambda_k()` are safe (each thread gets its own minimizer instance).

### 5.4 Memory Access Patterns

**E-step:** Each thread writes to different columns of `aadZ`, no conflicts.

**M-step:** Each thread modifies different rows of `aadLambda`, no conflicts.

---

## 6. Performance Analysis

### 6.1 Expected Speedups (OpenMP vs Sequential C)

| Workload | Sequential | OpenMP (4 cores) | OpenMP (8 cores) | OpenMP (16 cores) |
|----------|------------|------------------|------------------|-------------------|
| Small (N=100, K=3) | 0.5s | 0.25s (2x) | 0.18s (2.8x) | 0.15s (3.3x) |
| Medium (N=1K, K=5) | 5s | 1.5s (3.3x) | 0.9s (5.6x) | 0.7s (7.1x) |
| Large (N=10K, K=5) | 50s | 15s (3.3x) | 8s (6.3x) | 5s (10x) |
| Many comp (N=1K, K=20) | 20s | 6s (3.3x) | 3.5s (5.7x) | 2.5s (8x) |

### 6.2 Comparison: OpenMP vs Rust/Rayon

| Metric | OpenMP (C) | Rayon (Rust) |
|--------|------------|--------------|
| **Implementation Effort** | **1-2 days** | 2-3 months |
| **Code Changes** | **~20 lines** | Full rewrite |
| **Risk** | **Very Low** | Medium |
| **Peak Speedup (16 cores)** | **8-10x** | 8-12x |
| **Memory Safety** | No (still C) | Yes |
| **Error Handling** | C-style | Rust Result types |
| **Build Complexity** | **Low** | Medium |
| **Dependencies** | GCC/Clang (OpenMP) | Rust toolchain |
| **Distribution** | **Easy** (binary wheels) | Medium (Rust wheels) |
| **Debugging** | Familiar C tools | Rust tools |

**Bottom Line:** OpenMP gives **~80% of the performance gains** with **<5% of the implementation effort**.

---

## 7. Testing & Validation

### 7.1 Correctness Tests

```python
import pytest
import numpy as np
from pydmm import DirichletMixture

def test_openmp_sequential_equivalence():
    """Ensure OpenMP and sequential versions give identical results."""
    np.random.seed(42)
    X = np.random.randint(0, 100, (100, 20), dtype=np.int32)

    # Test with 1 thread (should be sequential)
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    result_seq = DirichletMixture(n_components=3, random_state=42).fit(X)

    # Test with 8 threads
    os.environ['OMP_NUM_THREADS'] = '8'
    result_par = DirichletMixture(n_components=3, random_state=42).fit(X)

    # Compare results (should be identical with same random seed)
    np.testing.assert_allclose(
        result_seq.result_.mixture_weights,
        result_par.result_.mixture_weights,
        rtol=1e-10,
        atol=1e-12
    )

    np.testing.assert_allclose(
        result_seq.result_.group_assignments,
        result_par.result_.group_assignments,
        rtol=1e-10,
        atol=1e-12
    )

def test_openmp_performance():
    """Verify speedup with multiple threads."""
    import time
    import os

    np.random.seed(42)
    X = np.random.randint(0, 100, (1000, 50), dtype=np.int32)

    # Time sequential
    os.environ['OMP_NUM_THREADS'] = '1'
    start = time.time()
    DirichletMixture(n_components=5, random_state=42, verbose=False).fit(X)
    time_seq = time.time() - start

    # Time parallel
    os.environ['OMP_NUM_THREADS'] = '8'
    start = time.time()
    DirichletMixture(n_components=5, random_state=42, verbose=False).fit(X)
    time_par = time.time() - start

    speedup = time_seq / time_par
    print(f"Speedup: {speedup:.2f}x")

    # Assert we get at least 2x speedup on 8 cores
    assert speedup >= 2.0, f"Expected speedup >= 2x, got {speedup:.2f}x"
```

### 7.2 Benchmark Script

```python
#!/usr/bin/env python3
import numpy as np
import time
import os
from pydmm import DirichletMixture

def benchmark(n_samples, n_features, n_components, n_threads):
    """Benchmark with given parameters."""
    np.random.seed(42)
    X = np.random.randint(0, 100, (n_samples, n_features), dtype=np.int32)

    os.environ['OMP_NUM_THREADS'] = str(n_threads)

    start = time.time()
    model = DirichletMixture(n_components=n_components, random_state=42, verbose=False)
    result = model.fit(X)
    elapsed = time.time() - start

    return elapsed

if __name__ == '__main__':
    scenarios = [
        (100, 20, 3, "Small"),
        (1000, 50, 5, "Medium"),
        (5000, 100, 5, "Large"),
        (1000, 50, 10, "Many Components"),
    ]

    thread_counts = [1, 2, 4, 8]

    print("OpenMP Parallelization Benchmark")
    print("=" * 80)

    for n, s, k, name in scenarios:
        print(f"\n{name} (N={n}, S={s}, K={k}):")
        print(f"{'Threads':<10} {'Time (s)':<12} {'Speedup':<10}")
        print("-" * 40)

        baseline_time = None
        for threads in thread_counts:
            elapsed = benchmark(n, s, k, threads)

            if baseline_time is None:
                baseline_time = elapsed
                speedup = 1.0
            else:
                speedup = baseline_time / elapsed

            print(f"{threads:<10} {elapsed:<12.3f} {speedup:<10.2f}x")
```

---

## 8. Incremental Implementation Strategy

### Phase 1: M-Step Only (Highest Impact, Lowest Risk)

**Estimated Time:** 2-3 hours

1. Add `#include <omp.h>`
2. Add one `#pragma` to M-step loop
3. Update `setup.py` with `-fopenmp`
4. Test and benchmark

**Expected Result:** 3-5x speedup already

### Phase 2: Add E-Step

**Estimated Time:** 2-3 hours

1. Move `adStore` array inside loop
2. Add `#pragma` to E-step loop
3. Test for correctness

**Expected Result:** 5-8x total speedup

### Phase 3: Add K-means

**Estimated Time:** 1-2 hours

1. Add `#pragma` to distance computation
2. Move `adDist` inside loop
3. Test

**Expected Result:** 6-10x total speedup

### Phase 4: Optimize & Tune

**Estimated Time:** 4-8 hours

1. Experiment with scheduling strategies
2. Profile to find remaining bottlenecks
3. Add parallelization to likelihood computation
4. Fine-tune thread counts

**Total Implementation Time: 1-2 days**

---

## 9. Potential Issues & Solutions

### 9.1 Issue: False Sharing

**Problem:** Multiple threads writing to nearby memory locations causes cache invalidation.

**Example:** In E-step, if `aadZ[k][i]` arrays are small, different threads might write to the same cache line.

**Solution:**
```c
// Add padding between thread-accessed data
#pragma omp parallel for schedule(static) private(k)
for (i = 0; i < N; i++) {
    // Access pattern is fine: each thread writes to different i values
    // and aadZ[k][i] writes are separated by stride N
}
```

**Reality:** Unlikely to be an issue since stride is N (number of samples), which is typically large enough.

### 9.2 Issue: Load Imbalance in M-Step

**Problem:** Some components might converge faster than others in BFGS2 optimization.

**Solution:** Use `schedule(dynamic)` instead of `schedule(static)`:
```c
#pragma omp parallel for schedule(dynamic)  // Dynamic scheduling
for (k = 0; k < K; k++) {
    optimise_lambda_k(aadLambda[k], data, aadZ[k]);
}
```

### 9.3 Issue: Nested Parallelism

**Problem:** If GSL internally uses threads (unlikely), nested parallelism could cause issues.

**Solution:** Disable nested parallelism:
```c
#ifdef _OPENMP
omp_set_nested(0);  // Disable nested parallelism
#endif
```

Or in Python:
```python
import os
os.environ['OMP_NESTED'] = 'FALSE'
```

---

## 10. Documentation for Users

Add to README:

```markdown
## Parallel Execution

pyDMM now supports multi-threaded execution using OpenMP. By default, it will use all available CPU cores.

### Controlling Thread Count

Set the `OMP_NUM_THREADS` environment variable:

```bash
# Use 8 threads
export OMP_NUM_THREADS=8
python your_script.py

# Or inline
OMP_NUM_THREADS=16 python your_script.py
```

### Performance Tips

- For small datasets (N < 100), parallelization overhead may outweigh benefits. Use `OMP_NUM_THREADS=1`.
- For large datasets (N > 1000) or many components (K > 5), use all available cores.
- Optimal thread count is typically equal to the number of physical cores (not hyperthreads).

### Checking OpenMP Status

```python
import os
print(f"Using {os.environ.get('OMP_NUM_THREADS', 'all')} threads")
```
```

---

## 11. Recommendation

**For pyDMM, I strongly recommend starting with OpenMP over Rust migration:**

### Reasons:

1. **Minimal Effort:** ~1-2 days vs 2-3 months
2. **Low Risk:** Small, localized changes
3. **High Impact:** 80% of the performance gains
4. **Easy Maintenance:** Team already knows C
5. **Fast Validation:** Can A/B test old vs new quickly
6. **No API Changes:** Zero user-facing changes
7. **Easy Distribution:** No new build dependencies

### Migration Path:

1. **Now:** Implement OpenMP (this document)
2. **Later:** Consider Rust if additional benefits (memory safety, ecosystem) become important
3. **Hybrid:** Could even keep C+OpenMP for core algorithms and use Rust for new features

### Success Criteria:

- ✅ 4x+ speedup on 8-core machines for typical workloads
- ✅ Numerical equivalence to sequential version
- ✅ No breaking changes
- ✅ Implementation in <1 week

---

## Appendix: Complete Diff

Here's a complete diff showing all changes needed:

```diff
diff --git a/src/pydmm/dirichlet_fit_standalone.c b/src/pydmm/dirichlet_fit_standalone.c
index 1234567..abcdefg 100644
--- a/src/pydmm/dirichlet_fit_standalone.c
+++ b/src/pydmm/dirichlet_fit_standalone.c
@@ -7,6 +7,10 @@
 #include <gsl/gsl_permutation.h>
 #include <math.h>

+#ifdef _OPENMP
+#include <omp.h>
+#endif
+
 #include "dirichlet_fit_standalone.h"
 /* re-map to R transient memory allocation */
 #define calloc(_nelm, _elsize) malloc((_nelm) * (_elsize))
@@ -80,6 +84,7 @@ static void kmeans(struct data_t *data, gsl_rng *ptGSLRNG,
         }

         /* calc distances and update Z */
+        #pragma omp parallel for schedule(static) private(k, j)
         for (i = 0; i < N; i++) {
             double dNorm = 0.0, adDist[K];
             for (k = 0; k < K; k++) {
@@ -274,6 +279,7 @@ static void calc_z(double **aadZ, const struct data_t *data,
         }
     }

+    #pragma omp parallel for schedule(static) private(k)
     for (i = 0; i < N; i ++) {
         double dSum = 0.0;
         double dOffset = BIG_DBL;
@@ -515,6 +521,7 @@ void dirichlet_fit_main(struct data_t *data, int rseed)
     while (dChange > 1.0e-6 && iter < 100) {
         calc_z(aadZ, data, adW, aadLambda); /* latent var expectation */

+        #pragma omp parallel for schedule(dynamic)
         for (k = 0; k < K; k++) /* mixture components, given pi */
             optimise_lambda_k(aadLambda[k], data, aadZ[k]);

diff --git a/setup.py b/setup.py
index 1234567..abcdefg 100644
--- a/setup.py
+++ b/setup.py
@@ -1,8 +1,33 @@
 from setuptools import setup, Extension, find_packages
 import numpy as np
+import os
+
+def has_openmp():
+    """Check if OpenMP is available."""
+    import subprocess
+    import tempfile
+
+    test_code = '#include <omp.h>\nint main() { return 0; }\n'
+
+    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
+        f.write(test_code)
+        temp_file = f.name
+
+    try:
+        result = subprocess.run(['gcc', '-fopenmp', temp_file], capture_output=True)
+        os.unlink(temp_file)
+        return result.returncode == 0
+    except:
+        return False

 # Define the C extension module
+extra_compile_args = ['-O3', '-Wall', '-std=c99']
+extra_link_args = []
+if has_openmp():
+    extra_compile_args.append('-fopenmp')
+    extra_link_args.append('-fopenmp')
+
 ext_modules = [
     Extension(
         'pydmm.pydmm_core',
@@ -20,7 +45,8 @@ ext_modules = [
         define_macros=[
             ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
         ],
-        extra_compile_args=['-O3', '-Wall', '-std=c99'],
+        extra_compile_args=extra_compile_args,
+        extra_link_args=extra_link_args,
     )
 ]
```

---

**Conclusion:** OpenMP provides an excellent path to parallelization with minimal risk and effort. Implement this first, measure the gains, then decide if Rust migration is worthwhile.
