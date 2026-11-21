/* Fault injection for BUGS.md bug 2 (tests/test_bug_fixes.py).
   The real hessian() is renamed to hessian_orig at compile time; this
   hessian() replaces the matrix of component 1 with a singular one.
   dirichlet_fit_main() must return (not exit) and report NaN. */
#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include "dirichlet_fit_standalone.h"

void hessian_orig(gsl_matrix *, const double *, const struct data_t *);
static int nCalls = 0;
void hessian(gsl_matrix *H, const double *adLambda, const struct data_t *data)
{
    hessian_orig(H, adLambda, data);
    if (nCalls++ == 1)
        gsl_matrix_set_all(H, 1.0); /* rank 1 */
}

enum { N = 30, S = 4, K = 2 };

int main(void)
{
    int i, k, aanX[S * N];
    double adGroup[K * N], adW[K], adLo[K * S], adMpe[K * S], adUp[K * S];
    struct data_t data = {0};
    gsl_rng *r = gsl_rng_alloc(gsl_rng_default);

    for (i = 0; i < S * N; i++)
        aanX[i] = 10 + (int)(gsl_rng_uniform(r) * 40) +
            ((i / N) == (i % N) % 2 ? 100 : 0);
    gsl_rng_free(r);
    data.N = N; data.S = S; data.K = K; data.aanX = aanX;
    data.group = adGroup; data.mixture_wt = adW;
    data.fit_lower = adLo; data.fit_mpe = adMpe; data.fit_upper = adUp;
    dirichlet_fit_main(&data, 1);

    printf("NLE %g\nLogDet %g\nLaplace %g\nBIC %g\n",
           data.NLE, data.LogDet, data.fit_laplace, data.fit_bic);
    for (k = 0; k < K; k++)
        printf("Lower%d %g\nUpper%d %g\n", k, adLo[k * S], k, adUp[k * S]);
    return 0;
}
