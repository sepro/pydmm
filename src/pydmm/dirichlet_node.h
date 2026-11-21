#ifndef _DIRICHLET_NODE_H_
#define _DIRICHLET_NODE_H_

#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

#include "dirichlet_fit_standalone.h"

extern const double BIG_DBL;
extern const double K_MEANS_THRESH;
extern const double SOFT_BETA;
extern const double GAMMA_ITA;
extern const double GAMMA_NU;
extern const int MAX_ITER;
extern const size_t MAX_GRAD_ITER;
extern const double COLLAPSE_THRESH;
extern const int N_KMEANS_STARTS;

/* soft k-means on relative abundances; fills weights, assignments and means */
void kmeans(struct data_t *data, gsl_rng *ptGSLRNG,
            double* adW, double **aadZ, double **aadMu);

/* 1 if every sample has (almost) the same assignments, i.e. EM stopped at the
   symmetric saddle point where all components are equal */
int collapsed(const struct data_t *data, double **aadZ, const double *adW);

/* hard k-means with k-means++ seeding (best of N_KMEANS_STARTS); fills aadZ
   with 0/1 assignments. Restart for a collapsed soft k-means start. */
void hard_kmeans(const struct data_t *data, double **aadZ, gsl_rng *ptGSLRNG);

/* weighted negative log evidence (+ Gamma prior) of one Dirichlet-multinomial,
   its gradient w.r.t. lambda = log(alpha), and both together (for GSL) */
double neg_log_evidence_lambda_pi(const gsl_vector *lambda, void *params);
void neg_log_derive_evidence_lambda_pi(const gsl_vector *ptLambda,
                                       void *params, gsl_vector* g);
void neg_log_FDF_lamba_pi(const gsl_vector *x, void *params,
                          double *f, gsl_vector *g);

/* BFGS fit of lambda for weights adZ (sets data->adPi = adZ) */
void optimise_lambda_k(double *adLambdaK, struct data_t *data, double *adZ);

/* negative log evidence of one sample (anX points at its first count, stride N) */
double neg_log_evidence_i(const struct data_t *data,
                          const int *anX, const double* adLambda,
                          const double* aadLnGammaLambda0);

/* Hessian of the weighted negative log posterior w.r.t. lambda (weights data->adPi) */
void hessian(gsl_matrix* ptHessian, const double* adLambda,
             const struct data_t *data);

#endif
