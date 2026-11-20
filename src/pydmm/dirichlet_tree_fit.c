#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_permutation.h>
#include <math.h>
#include <string.h>

#include "dirichlet_fit_standalone.h"
#include "dirichlet_node.h"
#include "dirichlet_tree_fit.h"
/* re-map to R transient memory allocation */
#define calloc(_nelm, _elsize) malloc((_nelm) * (_elsize))
#define free(_ptr) free(_ptr)

const int MAX_SELECT_ROUNDS = 5;
const int MAX_EM_ITER = 100;
const double EM_THRESH = 1.0e-6;
/* largest |z_ik - w_k / N| below which all components are taken to be equal */
const double COLLAPSE_THRESH = 1.0e-3;
const int N_KMEANS_STARTS = 5;

/* One internal node: the counts of its children and a data_t view of them,
   so the shared Dirichlet-multinomial routines can fit its parameters. */
struct node_t {
    int C;              /* number of children */
    int nOffset;        /* index of its first edge */
    int *aanX;          /* child counts, [c * N + i] */
    struct data_t dm;
};

static void node_setup(struct tree_data_t *data, struct node_t *aNode,
                       int *aanTotal)
{
    const int N = data->N, S = data->S;
    int u, c, i;

    for (u = 0; u < data->nNodes; u++) {
        struct node_t *node = aNode + u;
        node->nOffset = data->anChildStart[u];
        node->C = data->anChildStart[u + 1] - node->nOffset;
        node->aanX = (int *) calloc(node->C * N, sizeof(int));
        for (i = 0; i < N; i++)
            aanTotal[u * N + i] = 0;
        for (c = 0; c < node->C; c++) {
            const int child = data->anChild[node->nOffset + c];
            const int *anSrc = child < S ? data->aanX + child * N
                : aanTotal + (child - S) * N;
            for (i = 0; i < N; i++) {
                node->aanX[c * N + i] = anSrc[i];
                aanTotal[u * N + i] += anSrc[i];
            }
        }
        node->dm.verbose = false;
        node->dm.N = N;
        node->dm.S = node->C;
        node->dm.K = data->K;
        node->dm.aanX = node->aanX;
        node->dm.adPi = NULL;
        node->dm.hessian_out = NULL;
    }
}

/* lambda of every edge from the k-means means: log of each child's share of
   its parent's mass (for a star tree this is log(mu), as in the DMM) */
static void init_lambda(struct tree_data_t *data, double **aadMu,
                        double **aadLambda)
{
    const int S = data->S, K = data->K, nNodes = data->nNodes;
    int k, u, c;
    double *adMass = (double *) calloc(S + nNodes, sizeof(double));

    for (k = 0; k < K; k++) {
        for (c = 0; c < S; c++)
            adMass[c] = aadMu[k][c];
        for (u = 0; u < nNodes; u++) {
            adMass[S + u] = 0.0;
            for (c = data->anChildStart[u]; c < data->anChildStart[u + 1]; c++)
                adMass[S + u] += adMass[data->anChild[c]];
        }
        for (u = 0; u < nNodes; u++) {
            const double dParent = adMass[S + u];
            for (c = data->anChildStart[u]; c < data->anChildStart[u + 1]; c++) {
                const double x = dParent > 0.0 ?
                    adMass[data->anChild[c]] / dParent : 0.0;
                aadLambda[k][c] = (x > 0.0) ? log(x) : -10;
            }
        }
    }
    free(adMass);
}

/* fit one node's parameters: per component, or once on all samples if shared */
static void optimise_node(struct node_t *node, int K, bool bShared,
                          double **aadZ, double *adOnes, double **aadLambda)
{
    int k;
    if (bShared) {
        optimise_lambda_k(aadLambda[0] + node->nOffset, &node->dm, adOnes);
        for (k = 1; k < K; k++)
            memcpy(aadLambda[k] + node->nOffset, aadLambda[0] + node->nOffset,
                   node->C * sizeof(double));
    } else
        for (k = 0; k < K; k++)
            optimise_lambda_k(aadLambda[k] + node->nOffset, &node->dm, aadZ[k]);
}

static double *node_lngamma(const struct node_t *node, const double *adLambda)
{
    int c;
    double *adLnGamma = (double *) calloc(node->C, sizeof(double));
    for (c = 0; c < node->C; c++)
        adLnGamma[c] = gsl_sf_lngamma(exp(adLambda[node->nOffset + c]));
    return adLnGamma;
}

/* per-sample, per-component negative log evidence summed over nodes,
   [k * N + i]; shared nodes are skipped when bSkipShared (they add the same
   term to every component) */
static void tree_neg_log_evidence(const struct tree_data_t *data,
                                  const struct node_t *aNode,
                                  const int *anShared, double **aadLambda,
                                  bool bSkipShared, double *adOut)
{
    const int N = data->N, K = data->K;
    int u, k, i;

    for (i = 0; i < K * N; i++)
        adOut[i] = 0.0;
    for (u = 0; u < data->nNodes; u++) {
        const struct node_t *node = aNode + u;
        if (bSkipShared && anShared[u])
            continue;
        for (k = 0; k < K; k++) {
            const double *adLambda = aadLambda[k] + node->nOffset;
            double *adLnGamma = node_lngamma(node, aadLambda[k]);
            for (i = 0; i < N; i++)
                adOut[k * N + i] += neg_log_evidence_i(&node->dm,
                                                       node->aanX + i,
                                                       adLambda, adLnGamma);
            free(adLnGamma);
        }
    }
}

static void calc_z_tree(double **aadZ, const struct tree_data_t *data,
                        const struct node_t *aNode, const int *anShared,
                        const double *adW, double **aadLambda,
                        double *adStore)
{
    const int N = data->N, K = data->K;
    int i, k;

    tree_neg_log_evidence(data, aNode, anShared, aadLambda, true, adStore);
    for (i = 0; i < N; i++) {
        double dSum = 0.0, dOffset = BIG_DBL;
        for (k = 0; k < K; k++)
            if (adStore[k * N + i] < dOffset)
                dOffset = adStore[k * N + i];
        for (k = 0; k < K; k++) {
            aadZ[k][i] = adW[k] * exp(-(adStore[k * N + i] - dOffset));
            dSum += aadZ[k][i];
        }
        for (k = 0; k < K; k++)
            aadZ[k][i] /= dSum;
    }
}

/* number of distinct alpha parameters (shared nodes counted once) */
static int n_alpha(const struct tree_data_t *data, const struct node_t *aNode,
                   const int *anShared)
{
    int u, n = 0;
    for (u = 0; u < data->nNodes; u++)
        n += anShared[u] ? aNode[u].C : data->K * aNode[u].C;
    return n;
}

static double neg_log_likelihood_tree(const struct tree_data_t *data,
                                      const struct node_t *aNode,
                                      const int *anShared, const double *adW,
                                      double **aadLambda, double *adStore)
{
    const int S = data->S, N = data->N, K = data->K, *aanX = data->aanX;
    int i, j, k, u, c;
    double dRet = 0.0, dL5, dL6, dL7 = 0.0, dL8 = 0.0;
    const double dP = n_alpha(data, aNode, anShared);

    tree_neg_log_evidence(data, aNode, anShared, aadLambda, false, adStore);
    for (i = 0; i < N; i++) {
        double dProb = 0.0, dFactor = 0.0, dSum = 0.0, dOffset = -BIG_DBL;
        for (j = 0; j < S; j++) {
            dSum += aanX[j * N + i];
            dFactor += gsl_sf_lngamma(aanX[j * N + i] + 1.0);
        }
        dFactor -= gsl_sf_lngamma(dSum + 1.0);
        for (k = 0; k < K; k++)
            if (-adStore[k * N + i] > dOffset)
                dOffset = -adStore[k * N + i];
        for (k = 0; k < K; k++)
            dProb += adW[k] / N * exp(-adStore[k * N + i] - dOffset);
        dRet += log(dProb) + dOffset - dFactor;
    }
    dL5 = -dP * gsl_sf_lngamma(GAMMA_ITA);
    dL6 = GAMMA_ITA * dP * log(GAMMA_NU);
    for (u = 0; u < data->nNodes; u++)
        for (k = 0; k < (anShared[u] ? 1 : K); k++)
            for (c = 0; c < aNode[u].C; c++) {
                const double dLambda = aadLambda[k][aNode[u].nOffset + c];
                dL7 += exp(dLambda);
                dL8 += dLambda;
            }
    dL7 *= -GAMMA_NU;
    dL8 *= GAMMA_ITA;

    return -dRet - dL5 - dL6 - dL7 - dL8;
}

static double em(struct tree_data_t *data, struct node_t *aNode,
                 const int *anShared, double **aadZ, double *adW,
                 double **aadLambda, double *adStore)
{
    const int N = data->N, K = data->K;
    int i, k, u, iter = 0;
    double dNLL = 0.0, dNew, dChange = BIG_DBL;

    while (dChange > EM_THRESH && iter < MAX_EM_ITER) {
        calc_z_tree(aadZ, data, aNode, anShared, adW, aadLambda, adStore);
        for (u = 0; u < data->nNodes; u++)
            if (!anShared[u])
                for (k = 0; k < K; k++)
                    optimise_lambda_k(aadLambda[k] + aNode[u].nOffset,
                                      &aNode[u].dm, aadZ[k]);
        for (k = 0; k < K; k++) {
            adW[k] = 0.0;
            for (i = 0; i < N; i++)
                adW[k] += aadZ[k][i];
        }
        dNew = neg_log_likelihood_tree(data, aNode, anShared, adW, aadLambda,
                                       adStore);
        dChange = fabs(dNLL - dNew);
        dNLL = dNew;
        iter++;
        if (data->verbose && (iter % 10) == 0)
            printf("    iteration %d change %f\n", iter, dChange);
    }
    return dNLL;
}

/* 1 if every sample has (almost) the same assignments, i.e. EM stopped at
   the symmetric saddle point where all components are equal */
static int collapsed(const struct tree_data_t *data, double **aadZ,
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
static void hard_kmeans(const struct tree_data_t *data, double **aadZ,
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

/* expected complete-data log-likelihood of one node's counts */
static double node_expected_loglik(const struct node_t *node,
                                   const double *adLambda, const double *adZ)
{
    int i;
    double dRet = 0.0;
    double *adLnGamma = node_lngamma(node, adLambda);
    for (i = 0; i < node->dm.N; i++)
        dRet -= adZ[i] * neg_log_evidence_i(&node->dm, node->aanX + i,
                                            adLambda + node->nOffset,
                                            adLnGamma);
    free(adLnGamma);
    return dRet;
}

/* Node selection: a node keeps component-specific parameters only if their
   expected log-likelihood gain over one shared parameter vector exceeds the
   BIC penalty 0.5 * log(N) * (K - 1) * C. Returns 1 if any node changed. */
static int select_nodes(struct tree_data_t *data, struct node_t *aNode,
                        int *anShared, double **aadZ, double *adOnes,
                        double **aadLambda)
{
    const int N = data->N, K = data->K, E = data->anChildStart[data->nNodes];
    int u, k, nChanged = 0, nSpecific = 0, uBest = 0;
    double dBest = -BIG_DBL;
    int *anNew = (int *) calloc(data->nNodes, sizeof(int));
    double *adShared = (double *) calloc(E, sizeof(double));
    double **aadSpecific = (double **) calloc(K, sizeof(double *));

    aadSpecific[0] = (double *) calloc(K * E, sizeof(double));
    for (k = 1; k < K; k++)
        aadSpecific[k] = aadSpecific[0] + k * E;
    for (k = 0; k < K; k++)
        memcpy(aadSpecific[k], aadLambda[k], E * sizeof(double));
    memcpy(adShared, aadLambda[0], E * sizeof(double));

    for (u = 0; u < data->nNodes; u++) {
        struct node_t *node = aNode + u;
        double dGain = 0.0, dPenalty = 0.5 * log(N) * (K - 1) * node->C;
        optimise_lambda_k(adShared + node->nOffset, &node->dm, adOnes);
        dGain -= node_expected_loglik(node, adShared, adOnes);
        for (k = 0; k < K; k++) {
            optimise_lambda_k(aadSpecific[k] + node->nOffset, &node->dm, aadZ[k]);
            dGain += node_expected_loglik(node, aadSpecific[k], aadZ[k]);
        }
        anNew[u] = dGain < dPenalty;
        if (!anNew[u])
            nSpecific++;
        if (dGain - dPenalty > dBest) {
            dBest = dGain - dPenalty;
            uBest = u;
        }
    }
    /* keep the most informative node, or the components cannot differ */
    if (nSpecific == 0) {
        anNew[uBest] = 0;
        nSpecific = 1;
    }

    for (u = 0; u < data->nNodes; u++) {
        const struct node_t *node = aNode + u;
        if (anNew[u] != anShared[u])
            nChanged = 1;
        anShared[u] = anNew[u];
        for (k = 0; k < K; k++)
            memcpy(aadLambda[k] + node->nOffset,
                   (anShared[u] ? adShared : aadSpecific[k]) + node->nOffset,
                   node->C * sizeof(double));
    }
    if (data->verbose)
        printf("  Node selection: %d of %d nodes component-specific\n",
               nSpecific, data->nNodes);

    free(aadSpecific[0]); free(aadSpecific);
    free(adShared);
    free(anNew);
    return nChanged;
}

void dirichlet_tree_fit_main(struct tree_data_t *data, int rseed)
{
    const int N = data->N, S = data->S, K = data->K, nNodes = data->nNodes,
        E = data->anChildStart[nNodes];
    int i, k, u, c;

    gsl_rng *ptGSLRNG;
    gsl_rng_env_setup();
    gsl_set_error_handler_off();
    ptGSLRNG = gsl_rng_alloc(gsl_rng_default);
    gsl_set_error_handler_off();
    gsl_rng_set(ptGSLRNG, rseed);

    /* allocate matrices */
    double **aadZ, **aadLambda, **aadErr, **aadMu, *adW, *adOnes, *adStore;
    int *anShared, *aanTotal;
    struct node_t *aNode;

    adW = (double *) calloc(K, sizeof(double));
    adOnes = (double *) calloc(N, sizeof(double));
    adStore = (double *) calloc(K * N, sizeof(double));
    anShared = (int *) calloc(nNodes, sizeof(int));
    aanTotal = (int *) calloc(nNodes * N, sizeof(int));
    aNode = (struct node_t *) calloc(nNodes, sizeof(struct node_t));

    aadZ = (double **) calloc(K, sizeof(double *));
    aadLambda = (double **) calloc(K, sizeof(double *));
    aadErr = (double **) calloc(K, sizeof(double *));
    aadMu = (double **) calloc(K, sizeof(double *));

    aadZ[0] = (double *) calloc(K * N, sizeof(double));
    aadLambda[0] = (double *) calloc(K * E, sizeof(double));
    aadErr[0] = (double *) calloc(K * E, sizeof(double));
    aadMu[0] = (double *) calloc(K * S, sizeof(double));

    for (k = 1; k < K; k++) {
        aadZ[k] = aadZ[0] + k * N;
        aadLambda[k] = aadLambda[0] + k * E;
        aadErr[k] = aadErr[0] + k * E;
        aadMu[k] = aadMu[0] + k * S;
    }
    for (i = 0; i < N; i++)
        adOnes[i] = 1.0;
    for (u = 0; u < nNodes; u++)
        anShared[u] = 0;

    node_setup(data, aNode, aanTotal);

    /* soft k means initialiser on the leaf counts, as in the DMM */
    struct data_t leaves;
    leaves.verbose = data->verbose;
    leaves.N = N; leaves.S = S; leaves.K = K;
    leaves.aanX = data->aanX;
    leaves.adPi = NULL;
    leaves.hessian_out = NULL;
    for (k = 0; k < K; k++)
        for (c = 0; c < S; c++)
            aadMu[k][c] = 0.0;
    kmeans(&leaves, ptGSLRNG, adW, aadZ, aadMu);
    for (k = 0; k < K; k++) {
        adW[k] = 0.0;
        for (i = 0; i < N; i++)
            adW[k] += aadZ[k][i];
    }

    if (data->verbose)
        printf("  Expectation Maximization setup\n");
    init_lambda(data, aadMu, aadLambda);
    for (u = 0; u < nNodes; u++)
        optimise_node(aNode + u, K, false, aadZ, adOnes, aadLambda);

    if (data->verbose)
        printf("  Expectation Maximization\n");
    double dNLL = em(data, aNode, anShared, aadZ, adW, aadLambda, adStore);

    /* soft k-means can start all components (almost) equal when the samples'
       compositions are close; EM then stays at that saddle point */
    if (K > 1 && collapsed(data, aadZ, adW)) {
        if (data->verbose)
            printf("  Components identical; restarting from hard k-means\n");
        hard_kmeans(data, aadZ, ptGSLRNG);
        for (k = 0; k < K; k++) {
            adW[k] = 0.0;
            for (i = 0; i < N; i++)
                adW[k] += aadZ[k][i];
        }
        init_lambda(data, aadMu, aadLambda);
        for (u = 0; u < nNodes; u++)
            optimise_node(aNode + u, K, false, aadZ, adOnes, aadLambda);
        dNLL = em(data, aNode, anShared, aadZ, adW, aadLambda, adStore);
    }

    data->nSelectRounds = 0;
    if (data->bSelect && K > 1) {
        while (data->nSelectRounds < MAX_SELECT_ROUNDS) {
            data->nSelectRounds++;
            if (!select_nodes(data, aNode, anShared, aadZ, adOnes, aadLambda))
                break;
            if (data->verbose)
                printf("  Expectation Maximization (selection round %d)\n",
                       data->nSelectRounds);
            dNLL = em(data, aNode, anShared, aadZ, adW, aadLambda, adStore);
        }
        /* assignments and likelihood for the final parameters */
        calc_z_tree(aadZ, data, aNode, anShared, adW, aadLambda, adStore);
        for (k = 0; k < K; k++) {
            adW[k] = 0.0;
            for (i = 0; i < N; i++)
                adW[k] += aadZ[k][i];
        }
        dNLL = neg_log_likelihood_tree(data, aNode, anShared, adW, aadLambda,
                                       adStore);
    }

    /* hessian, per node (block diagonal) */
    if (data->verbose)
        printf("  Hessian\n");
    double dLogDet = 0.0;
    for (k = 1; k < K; k++)
        dLogDet += 2.0 * log(N) - log(adW[k]);
    for (u = 0; u < nNodes; u++) {
        struct node_t *node = aNode + u;
        const int C = node->C;
        gsl_matrix *ptHessian = gsl_matrix_alloc(C, C),
            *ptInverseHessian = gsl_matrix_alloc(C, C);
        gsl_permutation *p = gsl_permutation_alloc(C);
        int signum, status, j;

        for (k = 0; k < (anShared[u] ? 1 : K); k++) {
            node->dm.adPi = anShared[u] ? adOnes : aadZ[k];
            hessian(ptHessian, aadLambda[k] + node->nOffset, &node->dm);
            status = gsl_linalg_LU_decomp(ptHessian, p, &signum);
            if (status == GSL_SUCCESS)
                status = gsl_linalg_LU_invert(ptHessian, p, ptInverseHessian);
            for (j = 0; j < C; j++) {
                const double dErr = status == GSL_SUCCESS ?
                    gsl_matrix_get(ptInverseHessian, j, j) : NAN;
                aadErr[k][node->nOffset + j] = dErr;
                dLogDet += status == GSL_SUCCESS ?
                    log(fabs(gsl_matrix_get(ptHessian, j, j))) : NAN;
            }
        }
        if (anShared[u])
            for (k = 1; k < K; k++)
                memcpy(aadErr[k] + node->nOffset, aadErr[0] + node->nOffset,
                       C * sizeof(double));
        gsl_matrix_free(ptHessian);
        gsl_matrix_free(ptInverseHessian);
        gsl_permutation_free(p);
    }

    /* results */
    double dP = n_alpha(data, aNode, anShared) + K - 1;
    data->NLE = dNLL; data->LogDet = dLogDet;
    data->fit_laplace = dNLL + 0.5 * dLogDet - 0.5 * dP * log(2. * M_PI);
    data->fit_bic = dNLL + 0.5 * log(N) * dP;
    data->fit_aic = dNLL + dP;

    for (k = 0; k < K; k++) {
        data->mixture_wt[k] = adW[k] / N;
        for (i = 0; i < N; i++)
            data->group[k * N + i] = aadZ[k][i];
        for (c = 0; c < E; c++) {
            double dErr = aadErr[k][c], dL = NAN, dU = NAN;
            if (dErr >= 0.0) {
                dErr = sqrt(dErr);
                if (dErr < 100.0) {
                    dL = exp(aadLambda[k][c] - 2.0 * dErr);
                    dU = exp(aadLambda[k][c] + 2.0 * dErr);
                }
            }
            data->fit_lower[k * E + c] = dL;
            data->fit_mpe[k * E + c] = exp(aadLambda[k][c]);
            data->fit_upper[k * E + c] = dU;
        }
    }
    for (u = 0; u < nNodes; u++)
        data->node_shared[u] = anShared[u];

    for (u = 0; u < nNodes; u++)
        free(aNode[u].aanX);
    free(aNode);
    free(aanTotal);
    free(anShared);
    free(adStore);
    free(adOnes);
    free(adW);
    free(aadMu[0]); free(aadMu);
    free(aadErr[0]); free(aadErr);
    free(aadLambda[0]); free(aadLambda);
    free(aadZ[0]); free(aadZ);
    gsl_rng_free(ptGSLRNG);
    if (data->verbose)
        fflush(stdout);
}
