/* Regression harness for former BUGS.md bug 3 (tests/test_bug_fixes.py).
   dirichlet_fit_main() must not depend on what malloc returns. Before each
   fit, chunks of the size of aadLambda (K * S doubles) are filled with zero
   or with 0xFF bytes (NaN) and freed; glibc gives them back to the next
   malloc of that size. Prints the NLE and the assignments of both fits. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_rng.h>
#include "dirichlet_fit_standalone.h"

enum { N = 40, S = 12, K = 3, N_CHUNKS = 7 };

int main(void)
{
    int i, j, t, rep, aanX[S * N];
    double aadGroup[2][K * N], adW[K], adLo[K * S], adMpe[K * S], adUp[K * S];
    struct data_t data = {0};
    gsl_rng *r = gsl_rng_alloc(gsl_rng_default);

    gsl_rng_set(r, 1);
    for (i = 0; i < N; i++)     /* three groups, column-major [j * N + i] */
        for (j = 0; j < S; j++)
            aanX[j * N + i] = (int)(gsl_rng_uniform(r) * 50) +
                (j % 3 == i % 3 ? 200 : 0);
    gsl_rng_free(r);

    data.N = N; data.S = S; data.K = K; data.aanX = aanX;
    data.mixture_wt = adW;
    data.fit_lower = adLo; data.fit_mpe = adMpe; data.fit_upper = adUp;
    for (rep = 0; rep < 2; rep++) {
        void *apChunk[N_CHUNKS];
        for (t = 0; t < N_CHUNKS; t++) {
            apChunk[t] = malloc(K * S * sizeof(double));
            memset(apChunk[t], rep ? 0xFF : 0x00, K * S * sizeof(double));
        }
        for (t = 0; t < N_CHUNKS; t++)
            free(apChunk[t]);
        data.group = aadGroup[rep];
        dirichlet_fit_main(&data, 7);
        printf("NLE%d %.17g\n", rep, data.NLE);
    }
    printf("SameGroup %d\n",
           memcmp(aadGroup[0], aadGroup[1], sizeof(aadGroup[0])) == 0);
    return 0;
}
