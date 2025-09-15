#ifndef _DIRICHLET_FIT_STANDALONE_H_
#define _DIRICHLET_FIT_STANDALONE_H_

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

// Define types that were from R
typedef bool Rboolean;
#define TRUE true
#define FALSE false

struct data_t {
    Rboolean verbose;
    int N, S, K;
    const int *aanX;
    double* adPi;
    /* result */
    double NLE, LogDet;
    double *group;
    double *mixture_wt;
    double fit_laplace, fit_bic, fit_aic,
        *fit_lower, *fit_mpe, *fit_upper;
};

void dirichlet_fit_main(struct data_t *data, int rseed);

// Utility macros to replace R functions
#define R_alloc(nelm, elsize) malloc((nelm) * (elsize))
#define Rprintf printf
#define Rf_error(msg) do { fprintf(stderr, "Error: %s\n", msg); exit(1); } while(0)
#define R_CheckUserInterrupt() do {} while(0)
#define R_NaN (0.0/0.0)

#endif