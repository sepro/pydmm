#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_multimin.h>
#include <math.h>

#include "dirichlet_node.h"
/* re-map to R transient memory allocation */
#define calloc(_nelm, _elsize) malloc((_nelm) * (_elsize))
#define free(_ptr) free(_ptr)

/* Routines shared by the Dirichlet mixture (dirichlet_fit_standalone.c) and
   the Dirichlet-tree mixture (dirichlet_tree_fit.c): soft k-means
   initialisation and the weighted Dirichlet-multinomial fit of one parameter
   vector. The tree model calls them once per internal node, with a data_t
   whose S columns are the node's children. */

const double BIG_DBL = 1.0e9;
const double K_MEANS_THRESH = 1.0e-6;
const double SOFT_BETA = 50.0;
const double GAMMA_ITA = 0.1;
const double GAMMA_NU = 0.1;
const int MAX_ITER = 1000;
const size_t MAX_GRAD_ITER = 1000;
/* largest |z_ik - w_k / N| below which all components are taken to be equal */
const double COLLAPSE_THRESH = 1.0e-3;
const int N_KMEANS_STARTS = 5;

void kmeans(struct data_t *data, gsl_rng *ptGSLRNG,
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

    for (i = 0; i < N; i++) {
        double dTotal = 0.0;
        for (j = 0; j < S; j++)
            dTotal += aanX[j * N + i];
        for (j = 0; j < S; j++)
            aadY[j * N + i] = (aanX[j * N + i]) / dTotal;
    }

    /* initialise */
    for (i = 0; i < N; i++) {
        k = gsl_rng_uniform_int (ptGSLRNG, K);
        for (j = 0; j < K; j++)
            aadZ[j][i] = 0.0;
        aadZ[k][i] = 1.0;
    }

    while (dMaxChange > K_MEANS_THRESH && iter < MAX_ITER) {
        /* update mu */
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

        /* calc distances and update Z */
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
        iter++;
        if (data->verbose && (iter % 10 == 0))
            printf("    iteration %d change %f\n", iter, dMaxChange);
    }

    free(aadY);
    free(adMu);
}

/* 1 if every sample has (almost) the same assignments, i.e. EM stopped at
   the symmetric saddle point where all components are equal */
int collapsed(const struct data_t *data, double **aadZ,
              const double *adW)
{
    int i, k;
    for (k = 0; k < data->K; k++)
        for (i = 0; i < data->N; i++)
            if (fabs(aadZ[k][i] - adW[k] / data->N) > COLLAPSE_THRESH)
                return 0;
    return 1;
}

/* Hard k-means (Lloyd) on relative abundances with k-means++ seeding, best
   of N_KMEANS_STARTS by within-cluster sum of squares. Used when the soft
   k-means start collapses, which happens when many features make all
   samples' compositions close. Fills aadZ with 0/1 assignments. */
void hard_kmeans(const struct data_t *data, double **aadZ,
                 gsl_rng *ptGSLRNG)
{
    const int N = data->N, S = data->S, K = data->K, *aanX = data->aanX;
    int i, j, k, start, iter;
    double dBest = BIG_DBL;
    double *aadY = (double *) calloc(N * S, sizeof(double));
    double *aadC = (double *) calloc(K * S, sizeof(double));
    double *adD = (double *) calloc(N, sizeof(double));
    int *anLabel = (int *) calloc(N, sizeof(int));
    int *anCount = (int *) calloc(K, sizeof(int));

    for (i = 0; i < N; i++) {
        double dTotal = 0.0;
        for (j = 0; j < S; j++)
            dTotal += aanX[j * N + i];
        for (j = 0; j < S; j++)
            aadY[i * S + j] = dTotal > 0.0 ? aanX[j * N + i] / dTotal : 0.0;
    }

    for (start = 0; start < N_KMEANS_STARTS; start++) {
        /* k-means++ seeding */
        int c = gsl_rng_uniform_int(ptGSLRNG, N);
        for (j = 0; j < S; j++)
            aadC[j] = aadY[c * S + j];
        for (k = 1; k < K; k++) {
            double dSum = 0.0, dU;
            for (i = 0; i < N; i++) {
                int m;
                adD[i] = BIG_DBL;
                for (m = 0; m < k; m++) {
                    double dDist = 0.0;
                    for (j = 0; j < S; j++) {
                        const double dDiff = aadY[i * S + j] - aadC[m * S + j];
                        dDist += dDiff * dDiff;
                    }
                    if (dDist < adD[i])
                        adD[i] = dDist;
                }
                dSum += adD[i];
            }
            dU = gsl_rng_uniform(ptGSLRNG) * dSum;
            for (c = 0; c < N - 1; c++) {
                dU -= adD[c];
                if (dU <= 0.0)
                    break;
            }
            for (j = 0; j < S; j++)
                aadC[k * S + j] = aadY[c * S + j];
        }

        /* Lloyd iterations */
        double dInertia = 0.0;
        for (iter = 0; iter < MAX_ITER; iter++) {
            int nChanged = 0;
            dInertia = 0.0;
            for (i = 0; i < N; i++) {
                int kBest = 0;
                double dMin = BIG_DBL;
                for (k = 0; k < K; k++) {
                    double dDist = 0.0;
                    for (j = 0; j < S; j++) {
                        const double dDiff = aadY[i * S + j] - aadC[k * S + j];
                        dDist += dDiff * dDiff;
                    }
                    if (dDist < dMin) {
                        dMin = dDist;
                        kBest = k;
                    }
                }
                if (iter == 0 || anLabel[i] != kBest)
                    nChanged++;
                anLabel[i] = kBest;
                dInertia += dMin;
            }
            if (iter > 0 && nChanged == 0)
                break;
            for (k = 0; k < K; k++) {
                anCount[k] = 0;
                for (j = 0; j < S; j++)
                    aadC[k * S + j] = 0.0;
            }
            for (i = 0; i < N; i++) {
                anCount[anLabel[i]]++;
                for (j = 0; j < S; j++)
                    aadC[anLabel[i] * S + j] += aadY[i * S + j];
            }
            for (k = 0; k < K; k++)
                if (anCount[k] > 0)
                    for (j = 0; j < S; j++)
                        aadC[k * S + j] /= anCount[k];
        }

        if (dInertia < dBest) {
            dBest = dInertia;
            for (i = 0; i < N; i++)
                for (k = 0; k < K; k++)
                    aadZ[k][i] = anLabel[i] == k ? 1.0 : 0.0;
        }
    }

    free(anCount);
    free(anLabel);
    free(adD);
    free(aadC);
    free(aadY);
}

double neg_log_evidence_lambda_pi(const gsl_vector *lambda,
                                         void *params)
{
    int i, j;
    const struct data_t *data = (const struct data_t *) params;
    const int S = data->S, N = data->N, *aanX = data->aanX;
    const double *adPi = data->adPi;
    double dLogE = 0.0, dLogEAlpha = 0.0, dSumAlpha = 0.0, dSumLambda = 0.0;
    double adSumAlphaN[N], dWeight = 0.0;

    for (i = 0; i < N; i++) {
        adSumAlphaN[i] = 0.0;
        dWeight += adPi[i];
    }

    for (j = 0; j < S; j++) {
        const double dLambda = gsl_vector_get(lambda, j);
        const double dAlpha = exp(dLambda);
        dLogEAlpha += gsl_sf_lngamma(dAlpha);
        dSumLambda += dLambda;
        dSumAlpha += dAlpha;
        const double lngammaAlpha0 = gsl_sf_lngamma(dAlpha);
        for (i = 0; i < N; i++) {
            const double dN = aanX[j * N + i];
            const double dAlphaN = dAlpha + dN;
            const double lngammaAlphaN =
                dN ? gsl_sf_lngamma(dAlphaN) : lngammaAlpha0;
            adSumAlphaN[i] += dAlphaN; /*weight by pi*/
            dLogE -= adPi[i] * lngammaAlphaN; /*weight by pi*/
        }
    }
    dLogEAlpha -= gsl_sf_lngamma(dSumAlpha);

    for(i = 0; i < N; i++)
        dLogE += adPi[i] * gsl_sf_lngamma(adSumAlphaN[i]);

    return dLogE + dWeight*dLogEAlpha + GAMMA_NU*dSumAlpha -
        GAMMA_ITA * dSumLambda;
}

void neg_log_derive_evidence_lambda_pi(const gsl_vector *ptLambda,
                                              void *params, gsl_vector* g)
{
    const struct data_t *data = (const struct data_t *) params;
    const int S = data->S, N = data->N, *aanX = data->aanX;
    const double *adPi = data->adPi;

    int i, j;
    double adDeriv[S], adStore[N], adAlpha[S];
    double dSumStore = 0.0, dStore = 0.0;
    double dWeight = 0;

    for (i = 0; i < N; i++) {
        adStore[i] = 0.0;
        dWeight += adPi[i];
    }

    for (j = 0; j < S; j++) {
        adAlpha[j] = exp(gsl_vector_get(ptLambda, j));
        dStore += adAlpha[j];
        adDeriv[j] = dWeight* gsl_sf_psi(adAlpha[j]);
        double alphaS0 = gsl_sf_psi(adAlpha[j]);
        for (i = 0; i < N; i++) {
            int dN = aanX[j * N + i];
            double dAlphaN = adAlpha[j] + dN;

            double psiAlphaN = dN ? gsl_sf_psi(dAlphaN) : alphaS0;
            adDeriv[j] -= adPi[i] * psiAlphaN;
            adStore[i] += dAlphaN;
        }
    }

    for (i = 0; i < N; i++)
        dSumStore += adPi[i] * gsl_sf_psi(adStore[i]);
    dStore = dWeight * gsl_sf_psi(dStore);

    for (j = 0; j < S; j++) {
        double value = adAlpha[j] *
            (GAMMA_NU + adDeriv[j] - dStore + dSumStore) - GAMMA_ITA;
        gsl_vector_set(g, j, value);
    }
}

void neg_log_FDF_lamba_pi(const gsl_vector *x, void *params,
                                 double *f, gsl_vector *g)
{
    *f = neg_log_evidence_lambda_pi(x, params);
    neg_log_derive_evidence_lambda_pi(x, params, g);
}

void optimise_lambda_k(double *adLambdaK, struct data_t *data,
                              double *adZ)
{
    const int S = data->S;

    int i, status;
    size_t iter = 0;

    const gsl_multimin_fdfminimizer_type *T;
    gsl_multimin_fdfminimizer *s;
    gsl_multimin_function_fdf fdf;
    gsl_vector *ptLambda;

    /*initialise vector*/
    ptLambda = gsl_vector_alloc(S);
    for (i = 0; i < S; i++)
        gsl_vector_set(ptLambda, i, adLambdaK[i]);

    /*initialise function to be solved*/
    data->adPi = adZ;
    fdf.n = S;
    fdf.f = neg_log_evidence_lambda_pi;
    fdf.df = neg_log_derive_evidence_lambda_pi;
    fdf.fdf = neg_log_FDF_lamba_pi;
    fdf.params = data;

    T = gsl_multimin_fdfminimizer_vector_bfgs2;
    s = gsl_multimin_fdfminimizer_alloc(T, S);

    gsl_multimin_fdfminimizer_set(s, &fdf, ptLambda, 1.0e-6, 0.1);

    do {
        iter++;
        status = gsl_multimin_fdfminimizer_iterate(s);
        if (status)
            break;
        status = gsl_multimin_test_gradient(s->gradient, 1e-3);
   } while (status == GSL_CONTINUE && iter < MAX_GRAD_ITER);

    for (i = 0; i < S; i++)
        adLambdaK[i] = gsl_vector_get(s->x, i);

    gsl_vector_free(ptLambda);
    gsl_multimin_fdfminimizer_free(s);
}

double neg_log_evidence_i(const struct data_t *data,
                                 const int *anX, const double* adLambda,
                                 const double* aadLnGammaLambda0)
{
    int j;
    const int S = data->S, N = data->N;
    double dLogE = 0.0, dLogEAlpha = 0.0, dSumAlpha = 0.0,
        dSumAlphaN = 0.0;

    for (j = 0; j < S; j++) {
        const double n = anX[j * N];
        const double dAlpha = exp(adLambda[j]);
        const double dAlphaN = n + dAlpha;

        dLogEAlpha += aadLnGammaLambda0[j];
        dSumAlpha += dAlpha;
        dSumAlphaN += dAlphaN;
        dLogE -= n ? gsl_sf_lngamma(dAlphaN) : aadLnGammaLambda0[j] ;
    }

    dLogEAlpha -= gsl_sf_lngamma(dSumAlpha);
    dLogE += gsl_sf_lngamma(dSumAlphaN);

    return dLogE + dLogEAlpha;
}

void hessian(gsl_matrix* ptHessian, const double* adLambda,
                    const struct data_t *data)
{
    const int S = data->S, N = data->N, *aanX = data->aanX;
    const double *adPi = data->adPi;

    int i = 0, j = 0;
    double adAlpha[S], adAJK[S], adCJK[S], adAJK0[S], adCJK0[S];
    double dCK0 = 0.0, dAK0;
    double dCSum, dAlphaSum = 0.0, dW = 0.0, dCK = 0.0, dAK;

    for (j = 0; j < S; j++) {
        adAlpha[j] = exp(adLambda[j]);
        dAlphaSum += adAlpha[j];
        adAJK0[j] = adAJK[j] = adCJK0[j] = adCJK[j] = 0.0;
        const double dPsiAlpha = gsl_sf_psi(adAlpha[j]);
        const double dPsi1Alpha = gsl_sf_psi_1(adAlpha[j]);
        for (i = 0; i < N; i++) {
            const int n = aanX[j * N + i];
            adCJK0[j] += adPi[i] * (n ? gsl_sf_psi(adAlpha[j] + n) : dPsiAlpha);
            adAJK0[j] += adPi[i] * dPsiAlpha;
            adCJK[j] += adPi[i] * (n ? gsl_sf_psi_1(adAlpha[j] + n) : dPsi1Alpha);
            adAJK[j] += adPi[i] * dPsi1Alpha;
        }
    }

    for (i = 0; i < N; i++) {
        dW += adPi[i];
        dCSum = 0.0;
        for (j = 0; j < S; j++)
            dCSum += adAlpha[j] + aanX[j * N + i];
        dCK  += adPi[i]*gsl_sf_psi_1(dCSum);
        dCK0 += adPi[i]*gsl_sf_psi(dCSum);
    }

    dAK = dW * gsl_sf_psi_1(dAlphaSum);
    dAK0 = dW * gsl_sf_psi(dAlphaSum);
    for (i = 0; i < S; i++)
        for (j = 0; j < S; j++) {
            double dVal = 0.0;
            if (i == j) {
                double dG1 = -adAlpha[i] *
                    (dAK0 - dCK0 + adCJK0[i] - adAJK0[i]);
                double dG2 = -adAlpha[i] *
                    adAlpha[i]*(dAK - dCK + adCJK[i] - adAJK[i]);
                double dG3 = adAlpha[i]*GAMMA_NU;
                dVal = dG1 + dG2 + dG3;
            } else
                dVal = -adAlpha[i] * adAlpha[j] * (dAK - dCK);
            gsl_matrix_set(ptHessian, i, j, dVal);
        }
}
