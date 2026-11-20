#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_permutation.h>
#include <math.h>

#include "dirichlet_fit_standalone.h"
#include "dirichlet_node.h"
/* re-map to R transient memory allocation */
#define calloc(_nelm, _elsize) malloc((_nelm) * (_elsize))
#define free(_ptr) free(_ptr)

static void calc_z(double **aadZ, const struct data_t *data,
                   const double *adW, double **aadLambda)
{
    int i, k, j;
    const int N = data->N, K = data->K, S = data->S;
    double adStore[K];
    double *aadLngammaLambda0 = (double*)calloc(S*K,sizeof(double));

    for(k = 0; k < K; k++) {
        for(j = 0; j < S; j++) {
            const double dAlpha = exp(aadLambda[k][j]);
            aadLngammaLambda0[k*S +j] = gsl_sf_lngamma(dAlpha);
        }
    }

    for (i = 0; i < N; i ++) {
        double dSum = 0.0;
        double dOffset = BIG_DBL;
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
}

static double neg_log_likelihood(double *adW, double** aadLambda,
                                 const struct data_t *data)
{
    const int S = data->S, N = data->N, K = data->K,
        *aanX = data->aanX;
    int i, j, k;
    double adPi[K], adLogBAlpha[K];
    double dRet = 0.0, dL5 = 0.0, dL6 = 0.0, dL7 = 0.0, dL8 = 0.0;
    double dK = K, dN = N, dS = S;

    double *aadLngammaLambda0 = (double*)calloc(S*K,sizeof(double));

    for (k = 0; k < K; k++){
        double dSumAlphaK = 0.0;
        adLogBAlpha[k] = 0.0;
        adPi[k] = adW[k]/dN;
        for (j = 0; j < S; j++){
            double dAlpha = exp(aadLambda[k][j]);
            double lngammaAlpha = gsl_sf_lngamma(dAlpha);
            aadLngammaLambda0[k * S + j] = lngammaAlpha;

            dSumAlphaK += dAlpha;
            adLogBAlpha[k] += lngammaAlpha;
        }
        adLogBAlpha[k] -= gsl_sf_lngamma(dSumAlphaK);
    }

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
    dL5 = -dS * dK * gsl_sf_lngamma(GAMMA_ITA);
    dL6 = GAMMA_ITA * dK * dS * log(GAMMA_NU);

    for (i = 0; i < K; i++)
        for (j = 0; j < S; j++) {
            dL7 += exp(aadLambda[i][j]);
            dL8 += aadLambda[i][j];
        }
    dL7 *= -GAMMA_NU;
    dL8 *= GAMMA_ITA;

    return -dRet -dL5 - dL6 -dL7 -dL8;
}

static void group_output(struct data_t *data, double** aadZ)
{
    const int N = data->N, K = data->K;
    int i, k;
    for(k = 0; k < K; k++)
        for (i = 0; i < N; i++)
            data->group[k * N + i] = aadZ[k][i];
}

static void mixture_output(struct data_t *data, double *adW,
                           double** aadLambda, double **aadErr)
{
    const int N = data->N, S = data->S;
    int i, k;
    for (k = 0; k < data->K; k++)
        data->mixture_wt[k] = adW[k] / N;
    for (i = 0; i < data->S; i++) {
        for (k = 0; k < data->K; k++) {
            double dErr = aadErr[k][i], dL = 0.0, dU = 0.0;
            int bIll = FALSE;
            if (dErr >= 0.0) {
                dErr = sqrt(dErr);
                if (dErr < 100.0) {
                    dL =  exp(aadLambda[k][i] - 2.0*dErr);
                    dU =  exp(aadLambda[k][i] + 2.0*dErr);
                } else bIll = TRUE;
            } else bIll = TRUE;

            if (bIll)
                dL = dU = NAN;
            data->fit_lower[k * S + i] = dL;
            data->fit_mpe[k * S + i] = exp(aadLambda[k][i]);
            data->fit_upper[k * S + i] = dU;
        }
    }
}

void dirichlet_fit_main(struct data_t *data, int rseed)
{
    const int N = data->N, S = data->S, K = data->K;
    int i, j, k;

    gsl_rng *ptGSLRNG;
    gsl_rng_env_setup();
    gsl_set_error_handler_off();
    ptGSLRNG = gsl_rng_alloc(gsl_rng_default);
    gsl_set_error_handler_off();
    gsl_rng_set(ptGSLRNG, rseed);

    /* allocate matrices */
    double **aadZ, **aadLambda, **aadErr, *adW;
    adW = (double *) calloc(K, sizeof(double));

    aadZ = (double **) calloc(K, sizeof(double *));
    aadLambda = (double **) calloc(K, sizeof(double *));
    aadErr = (double **) calloc(K, sizeof(double*));

    aadZ[0] = (double *) calloc(K * N, sizeof(double));
    aadLambda[0] = (double *) calloc(K * S, sizeof(double));
    aadErr[0] = (double *) calloc(K * S, sizeof(double));

    for (k = 1; k < K; k++) {
        aadZ[k] = aadZ[0] + k * N;
        aadLambda[k] = aadLambda[0] + k * S;
        aadErr[k] = aadErr[0] + k * S;
    }

    /* soft k means initialiser */
    kmeans(data, ptGSLRNG, adW, aadZ, aadLambda);
    for (k = 0; k < K; k++) {
        adW[k] = 0.0;
        for (i = 0; i < N; i++)
            adW[k] += aadZ[k][i];
    }

    if (data->verbose)
        printf("  Expectation Maximization setup\n");
    for (k = 0; k < K; k++) {
        for (j = 0; j < S; j++) {
            const double x = aadLambda[k][j];
            aadLambda[k][j] = (x > 0.0) ? log(x) : -10;
        }
        optimise_lambda_k(aadLambda[k], data, aadZ[k]);
    }

    /* simple EM algorithm */
    int iter = 0;
    double dNLL = 0.0, dNew, dChange = BIG_DBL;

    if (data->verbose)
        printf("  Expectation Maximization\n");
    while (dChange > 1.0e-6 && iter < 100) {
        calc_z(aadZ, data, adW, aadLambda); /* latent var expectation */
        for (k = 0; k < K; k++) /* mixture components, given pi */
            optimise_lambda_k(aadLambda[k], data, aadZ[k]);
        for (k = 0; k < K; k++) { /* current likelihood & weights */
            adW[k] = 0.0;
            for(i = 0; i < N; i++)
                adW[k] += aadZ[k][i];
        }

        dNew = neg_log_likelihood(adW, aadLambda, data);
        dChange = fabs(dNLL - dNew);
        dNLL = dNew;
        iter++;
        /* User interrupt check removed */
        if (data->verbose && (iter % 10) == 0)
            printf("    iteration %d change %f\n", iter, dChange);
    }

    /* hessian */
    if (data->verbose)
        printf("  Hessian\n");
    gsl_matrix *ptHessian = gsl_matrix_alloc(S, S),
        *ptInverseHessian = gsl_matrix_alloc(S, S);
    gsl_permutation *p = gsl_permutation_alloc(S);
    double dLogDet = 0., dTemp;
    int signum, status;

    for (k = 0; k < K; k++) {
        data->adPi = aadZ[k];
        if (k > 0)
            dLogDet += 2.0 * log(N) - log(adW[k]);
        hessian(ptHessian, aadLambda[k], data);
        if (data->hessian_out)
            for (i = 0; i < S; i++)
                for (j = 0; j < S; j++)
                    data->hessian_out[(size_t)k * S * S + i * S + j] =
                        gsl_matrix_get(ptHessian, i, j);

        status = gsl_linalg_LU_decomp(ptHessian, p, &signum);
        if (status != GSL_SUCCESS) {
            gsl_matrix_free(ptHessian);
            gsl_matrix_free(ptInverseHessian);
            gsl_permutation_free(p);

            free(aadErr[0]); free(aadErr);
            free(aadLambda[0]); free(aadLambda);
            free(aadZ[0]); free(aadZ);
            free(adW);

            fprintf(stderr, "Error: LU decompistion of a singular matrix during Hessian\n"); exit(1);
        }
        gsl_linalg_LU_invert(ptHessian, p, ptInverseHessian);
        for (j = 0; j < S; j++) {
            aadErr[k][j] = gsl_matrix_get(ptInverseHessian, j, j);
            dTemp = gsl_matrix_get(ptHessian, j, j);
            dLogDet += log(fabs(dTemp));
        }
    }

    gsl_matrix_free(ptHessian);
    gsl_matrix_free(ptInverseHessian);
    gsl_permutation_free(p);

    /* results */
    double dP = K * S + K - 1;
    data->NLE = dNLL; data->LogDet = dLogDet;
    data->fit_laplace = dNLL + 0.5 * dLogDet - 0.5 * dP * log(2. * M_PI);
    data->fit_bic = dNLL + 0.5 * log(N) * dP;
    data->fit_aic = dNLL + dP;

    group_output(data, aadZ);
    mixture_output(data, adW, aadLambda, aadErr);

    free(aadErr[0]); free(aadErr);
    free(aadLambda[0]); free(aadLambda);
    free(aadZ[0]); free(aadZ);
    free(adW);
}
