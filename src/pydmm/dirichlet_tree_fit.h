#ifndef _DIRICHLET_TREE_FIT_H_
#define _DIRICHLET_TREE_FIT_H_

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

/* Dirichlet-tree multinomial mixture.

   The S features are the leaves of a rooted tree. Every internal node splits
   the count arriving at it over its children with a Dirichlet-multinomial,
   with its own parameters per component, or one set of parameters shared by
   all components (node selection).

   Tree layout: internal node u (0 <= u < nNodes) has children
   anChild[anChildStart[u]] .. anChild[anChildStart[u + 1] - 1]. A child id
   j < S is leaf j, a child id S + u is internal node u. Nodes are in
   post-order (children before parents), so the root is node nNodes - 1.
   Parameters are stored per edge, in anChild order (E = anChildStart[nNodes]). */
struct tree_data_t {
    bool verbose;
    int N, S, K;
    const int *aanX;          /* leaf counts, [j * N + i] */
    int nNodes;
    const int *anChildStart;  /* nNodes + 1 */
    const int *anChild;       /* E */
    bool bSelect;             /* choose shared / component-specific per node */
    /* result */
    double NLE, LogDet;
    double *group;            /* [k * N + i] */
    double *mixture_wt;       /* K */
    int *node_shared;         /* nNodes: 1 if the node's alpha is shared */
    int nSelectRounds;
    double fit_laplace, fit_bic, fit_aic,
        *fit_lower, *fit_mpe, *fit_upper;  /* [k * E + e] */
};

void dirichlet_tree_fit_main(struct tree_data_t *data, int rseed);

#endif
