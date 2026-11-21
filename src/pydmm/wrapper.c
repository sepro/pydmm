#include <Python.h>
#include <numpy/arrayobject.h>
#include <string.h>
#include "dirichlet_fit_standalone.h"
#include "dirichlet_tree_fit.h"

static PyObject *dirichlet_fit_py(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *counts_array = NULL;
    int n_components = 2;
    int verbose = 0;
    int seed = 42;
    int return_hessian = 0;
    int restart_collapsed = 0;

    static char *kwlist[] = {"counts", "n_components", "verbose", "seed",
                             "return_hessian", "restart_collapsed", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|iiipp", kwlist,
                                     &PyArray_Type, &counts_array,
                                     &n_components, &verbose, &seed,
                                     &return_hessian, &restart_collapsed)) {
        return NULL;
    }

    // Ensure array is 2D integer array
    if (PyArray_NDIM(counts_array) != 2) {
        PyErr_SetString(PyExc_ValueError, "Input array must be 2-dimensional");
        return NULL;
    }

    if (PyArray_TYPE(counts_array) != NPY_INT32) {
        PyErr_SetString(PyExc_ValueError, "Input array must be int32 dtype");
        return NULL;
    }

    // Get array dimensions
    npy_intp *dims = PyArray_DIMS(counts_array);
    int N = dims[0];  // number of samples/communities
    int S = dims[1];  // number of features/taxa
    int K = n_components;

    // The C code expects data in column-major format (features x samples)
    // but NumPy provides row-major format (samples x features)
    // So we need to create a transposed, contiguous copy
    PyArrayObject *counts_transposed = (PyArrayObject *)PyArray_Transpose(counts_array, NULL);
    if (!counts_transposed) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to transpose input array");
        return NULL;
    }

    // Ensure the transposed array is contiguous
    PyArrayObject *counts_contiguous = (PyArrayObject *)PyArray_ContiguousFromAny(
        (PyObject *)counts_transposed, NPY_INT32, 2, 2
    );
    Py_DECREF(counts_transposed);

    if (!counts_contiguous) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create contiguous transposed array");
        return NULL;
    }

    if (N <= 0 || S <= 0 || K <= 0) {
        Py_DECREF(counts_contiguous);
        PyErr_SetString(PyExc_ValueError, "Array dimensions and n_components must be positive");
        return NULL;
    }

    // Prepare data structure
    struct data_t *data = (struct data_t *)malloc(sizeof(struct data_t));
    if (!data) {
        Py_DECREF(counts_contiguous);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for data structure");
        return NULL;
    }

    data->verbose = verbose;
    data->restart_collapsed = restart_collapsed;
    data->N = N;  // C code expects N samples
    data->S = S;  // C code expects S features
    data->K = K;
    data->aanX = (int *)PyArray_DATA(counts_contiguous);
    data->hessian_out = NULL;

    // Optional Hessian output (K x S x S), owned by a NumPy array
    PyArrayObject *hessian_array = NULL;
    if (return_hessian) {
        npy_intp hess_dims[3] = {K, S, S};
        hessian_array = (PyArrayObject *)PyArray_SimpleNew(3, hess_dims, NPY_DOUBLE);
        if (!hessian_array) {
            free(data);
            Py_DECREF(counts_contiguous);
            return NULL;
        }
        data->hessian_out = (double *)PyArray_DATA(hessian_array);
    }

    // Allocate output arrays
    data->group = (double *)malloc(N * K * sizeof(double));
    data->mixture_wt = (double *)malloc(K * sizeof(double));
    data->fit_lower = (double *)malloc(S * K * sizeof(double));
    data->fit_mpe = (double *)malloc(S * K * sizeof(double));
    data->fit_upper = (double *)malloc(S * K * sizeof(double));

    if (!data->group || !data->mixture_wt || !data->fit_lower ||
        !data->fit_mpe || !data->fit_upper) {
        // Clean up allocated memory
        free(data->group);
        free(data->mixture_wt);
        free(data->fit_lower);
        free(data->fit_mpe);
        free(data->fit_upper);
        free(data);
        Py_XDECREF(hessian_array);
        Py_DECREF(counts_contiguous);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate output arrays");
        return NULL;
    }

    // Call the main fitting function
    dirichlet_fit_main(data, seed);

    // Create Python return objects
    // C code stores group as [k * N + i], so dimensions are (K, N)
    npy_intp group_dims[2] = {K, N};
    npy_intp mixture_dims[1] = {K};
    npy_intp fit_dims[2] = {S, K};

    PyArrayObject *group_array_raw = (PyArrayObject *)PyArray_SimpleNewFromData(
        2, group_dims, NPY_DOUBLE, data->group);
    // Transpose to get (N, K) layout expected by Python code and make a copy
    PyArrayObject *group_array_transposed = (PyArrayObject *)PyArray_Transpose(group_array_raw, NULL);
    PyArrayObject *group_array = (PyArrayObject *)PyArray_Copy(group_array_transposed);
    Py_DECREF(group_array_raw);
    Py_DECREF(group_array_transposed);

    PyArrayObject *mixture_array = (PyArrayObject *)PyArray_SimpleNewFromData(
        1, mixture_dims, NPY_DOUBLE, data->mixture_wt);

    // C code stores fit arrays as [k * S + i], so dimensions are (K, S)
    npy_intp fit_c_dims[2] = {K, S};
    PyArrayObject *fit_lower_raw = (PyArrayObject *)PyArray_SimpleNewFromData(
        2, fit_c_dims, NPY_DOUBLE, data->fit_lower);
    PyArrayObject *fit_mpe_raw = (PyArrayObject *)PyArray_SimpleNewFromData(
        2, fit_c_dims, NPY_DOUBLE, data->fit_mpe);
    PyArrayObject *fit_upper_raw = (PyArrayObject *)PyArray_SimpleNewFromData(
        2, fit_c_dims, NPY_DOUBLE, data->fit_upper);

    // Transpose to get (S, K) layout expected by Python code and make copies
    PyArrayObject *fit_lower_transposed = (PyArrayObject *)PyArray_Transpose(fit_lower_raw, NULL);
    PyArrayObject *fit_mpe_transposed = (PyArrayObject *)PyArray_Transpose(fit_mpe_raw, NULL);
    PyArrayObject *fit_upper_transposed = (PyArrayObject *)PyArray_Transpose(fit_upper_raw, NULL);

    PyArrayObject *fit_lower_array = (PyArrayObject *)PyArray_Copy(fit_lower_transposed);
    PyArrayObject *fit_mpe_array = (PyArrayObject *)PyArray_Copy(fit_mpe_transposed);
    PyArrayObject *fit_upper_array = (PyArrayObject *)PyArray_Copy(fit_upper_transposed);

    // Clean up intermediate arrays
    Py_DECREF(fit_lower_raw);
    Py_DECREF(fit_mpe_raw);
    Py_DECREF(fit_upper_raw);
    Py_DECREF(fit_lower_transposed);
    Py_DECREF(fit_mpe_transposed);
    Py_DECREF(fit_upper_transposed);

    if (!group_array || !mixture_array || !fit_lower_array ||
        !fit_mpe_array || !fit_upper_array) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create output arrays");
        free(data->group);
        free(data->mixture_wt);
        free(data->fit_lower);
        free(data->fit_mpe);
        free(data->fit_upper);
        free(data);
        Py_XDECREF(hessian_array);
        Py_DECREF(counts_contiguous);
        return NULL;
    }

    // Set arrays to own their data
    // group_array is a copy, so it already owns its data
    PyArray_ENABLEFLAGS(mixture_array, NPY_ARRAY_OWNDATA);
    PyArray_ENABLEFLAGS(fit_lower_array, NPY_ARRAY_OWNDATA);
    PyArray_ENABLEFLAGS(fit_mpe_array, NPY_ARRAY_OWNDATA);
    PyArray_ENABLEFLAGS(fit_upper_array, NPY_ARRAY_OWNDATA);

    // Create return dictionary
    PyObject *result = PyDict_New();
    if (!result) {
        Py_DECREF(group_array);
        Py_DECREF(mixture_array);
        Py_DECREF(fit_lower_array);
        Py_DECREF(fit_mpe_array);
        Py_DECREF(fit_upper_array);
        Py_XDECREF(hessian_array);
        free(data);
        Py_DECREF(counts_contiguous);
        return NULL;
    }

    // Add goodness of fit metrics
    PyObject *goodness_dict = PyDict_New();
    PyDict_SetItemString(goodness_dict, "NLE", PyFloat_FromDouble(data->NLE));
    PyDict_SetItemString(goodness_dict, "LogDet", PyFloat_FromDouble(data->LogDet));
    PyDict_SetItemString(goodness_dict, "Laplace", PyFloat_FromDouble(data->fit_laplace));
    PyDict_SetItemString(goodness_dict, "BIC", PyFloat_FromDouble(data->fit_bic));
    PyDict_SetItemString(goodness_dict, "AIC", PyFloat_FromDouble(data->fit_aic));
    PyDict_SetItemString(result, "GoodnessOfFit", goodness_dict);

    // Add group assignments
    PyDict_SetItemString(result, "Group", (PyObject *)group_array);

    // Add mixture weights
    PyObject *mixture_dict = PyDict_New();
    PyDict_SetItemString(mixture_dict, "Weight", (PyObject *)mixture_array);
    PyDict_SetItemString(result, "Mixture", mixture_dict);

    // Add fit results
    PyObject *fit_dict = PyDict_New();
    PyDict_SetItemString(fit_dict, "Lower", (PyObject *)fit_lower_array);
    PyDict_SetItemString(fit_dict, "Estimate", (PyObject *)fit_mpe_array);
    PyDict_SetItemString(fit_dict, "Upper", (PyObject *)fit_upper_array);
    PyDict_SetItemString(result, "Fit", fit_dict);

    if (hessian_array) {
        PyDict_SetItemString(result, "Hessian", (PyObject *)hessian_array);
        Py_DECREF(hessian_array);
    }

    // Clean up
    Py_DECREF(counts_contiguous);  // Clean up transposed array
    free(data);

    return result;
}

/* Wrap a C array of doubles laid out [k * M + m] as a NumPy (M, K) array */
static PyObject *km_to_mk_array(const double *adSrc, int K, int M)
{
    npy_intp dims[2] = {M, K};
    PyArrayObject *array = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!array)
        return NULL;
    double *adDst = (double *)PyArray_DATA(array);
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++)
            adDst[m * K + k] = adSrc[k * M + m];
    return (PyObject *)array;
}

static int check_tree(const int *anChildStart, int nNodes, const int *anChild,
                      int nEdges, int S)
{
    if (nNodes < 1 || anChildStart[0] != 0 || anChildStart[nNodes] != nEdges) {
        PyErr_SetString(PyExc_ValueError, "child_start must start at 0 and end at len(child)");
        return 0;
    }
    int *anSeen = (int *)calloc(S + nNodes, sizeof(int));
    if (!anSeen) {
        PyErr_NoMemory();
        return 0;
    }
    int ok = 1;
    for (int u = 0; u < nNodes && ok; u++) {
        if (anChildStart[u + 1] - anChildStart[u] < 2) {
            PyErr_SetString(PyExc_ValueError, "every internal node needs at least 2 children");
            ok = 0;
            break;
        }
        for (int e = anChildStart[u]; e < anChildStart[u + 1]; e++) {
            const int c = anChild[e];
            /* children must be leaves or earlier internal nodes (post-order) */
            if (c < 0 || c >= S + u || anSeen[c]) {
                PyErr_SetString(PyExc_ValueError,
                                "child ids must be leaves or earlier nodes, each used once");
                ok = 0;
                break;
            }
            anSeen[c] = 1;
        }
    }
    for (int j = 0; j < S + nNodes - 1 && ok; j++)
        if (!anSeen[j]) {
            PyErr_SetString(PyExc_ValueError,
                            "every leaf and non-root node must have a parent");
            ok = 0;
        }
    free(anSeen);
    return ok;
}

static PyObject *dirichlet_tree_fit_py(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *counts_array = NULL, *start_array = NULL, *child_array = NULL;
    int n_components = 2;
    int verbose = 0;
    int seed = 42;
    int node_selection = 1;

    static char *kwlist[] = {"counts", "child_start", "child", "n_components",
                             "verbose", "seed", "node_selection", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!O!|iiip", kwlist,
                                     &PyArray_Type, &counts_array,
                                     &PyArray_Type, &start_array,
                                     &PyArray_Type, &child_array,
                                     &n_components, &verbose, &seed,
                                     &node_selection)) {
        return NULL;
    }

    if (PyArray_NDIM(counts_array) != 2 || PyArray_TYPE(counts_array) != NPY_INT32) {
        PyErr_SetString(PyExc_ValueError, "counts must be a 2-dimensional int32 array");
        return NULL;
    }
    if (PyArray_NDIM(start_array) != 1 || PyArray_TYPE(start_array) != NPY_INT32 ||
        PyArray_NDIM(child_array) != 1 || PyArray_TYPE(child_array) != NPY_INT32) {
        PyErr_SetString(PyExc_ValueError, "child_start and child must be 1-dimensional int32 arrays");
        return NULL;
    }

    npy_intp *dims = PyArray_DIMS(counts_array);
    int N = dims[0];
    int S = dims[1];
    int K = n_components;
    int nNodes = (int)PyArray_DIM(start_array, 0) - 1;
    int E = (int)PyArray_DIM(child_array, 0);

    if (N <= 0 || S <= 0 || K <= 0) {
        PyErr_SetString(PyExc_ValueError, "Array dimensions and n_components must be positive");
        return NULL;
    }

    PyArrayObject *start_c = (PyArrayObject *)PyArray_ContiguousFromAny(
        (PyObject *)start_array, NPY_INT32, 1, 1);
    PyArrayObject *child_c = (PyArrayObject *)PyArray_ContiguousFromAny(
        (PyObject *)child_array, NPY_INT32, 1, 1);
    if (!start_c || !child_c) {
        Py_XDECREF(start_c);
        Py_XDECREF(child_c);
        return NULL;
    }
    if (!check_tree((int *)PyArray_DATA(start_c), nNodes,
                    (int *)PyArray_DATA(child_c), E, S)) {
        Py_DECREF(start_c);
        Py_DECREF(child_c);
        return NULL;
    }

    // Column-major copy of the counts (features x samples), as for the DMM
    PyArrayObject *counts_transposed = (PyArrayObject *)PyArray_Transpose(counts_array, NULL);
    PyArrayObject *counts_contiguous = counts_transposed ? (PyArrayObject *)PyArray_ContiguousFromAny(
        (PyObject *)counts_transposed, NPY_INT32, 2, 2) : NULL;
    Py_XDECREF(counts_transposed);
    if (!counts_contiguous) {
        Py_DECREF(start_c);
        Py_DECREF(child_c);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create contiguous transposed array");
        return NULL;
    }

    struct tree_data_t data;
    data.verbose = verbose;
    data.N = N;
    data.S = S;
    data.K = K;
    data.aanX = (int *)PyArray_DATA(counts_contiguous);
    data.nNodes = nNodes;
    data.anChildStart = (int *)PyArray_DATA(start_c);
    data.anChild = (int *)PyArray_DATA(child_c);
    data.bSelect = node_selection;

    data.group = (double *)malloc((size_t)N * K * sizeof(double));
    data.mixture_wt = (double *)malloc(K * sizeof(double));
    data.node_shared = (int *)malloc(nNodes * sizeof(int));
    data.fit_lower = (double *)malloc((size_t)E * K * sizeof(double));
    data.fit_mpe = (double *)malloc((size_t)E * K * sizeof(double));
    data.fit_upper = (double *)malloc((size_t)E * K * sizeof(double));

    PyObject *result = NULL;
    if (!data.group || !data.mixture_wt || !data.node_shared ||
        !data.fit_lower || !data.fit_mpe || !data.fit_upper) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate output arrays");
        goto cleanup;
    }

    dirichlet_tree_fit_main(&data, seed);

    // Outputs: Group (N, K), Weight (K,), Fit/* (E, K), NodeShared (nNodes,)
    PyObject *group_array = km_to_mk_array(data.group, K, N);
    PyObject *lower_array = km_to_mk_array(data.fit_lower, K, E);
    PyObject *mpe_array = km_to_mk_array(data.fit_mpe, K, E);
    PyObject *upper_array = km_to_mk_array(data.fit_upper, K, E);
    npy_intp k_dims[1] = {K}, node_dims[1] = {nNodes};
    PyArrayObject *weight_array = (PyArrayObject *)PyArray_SimpleNew(1, k_dims, NPY_DOUBLE);
    PyArrayObject *shared_array = (PyArrayObject *)PyArray_SimpleNew(1, node_dims, NPY_INT32);
    if (!group_array || !lower_array || !mpe_array || !upper_array ||
        !weight_array || !shared_array) {
        Py_XDECREF(group_array); Py_XDECREF(lower_array);
        Py_XDECREF(mpe_array); Py_XDECREF(upper_array);
        Py_XDECREF(weight_array); Py_XDECREF(shared_array);
        goto cleanup;
    }
    memcpy(PyArray_DATA(weight_array), data.mixture_wt, K * sizeof(double));
    memcpy(PyArray_DATA(shared_array), data.node_shared, nNodes * sizeof(int));

    result = PyDict_New();
    PyObject *goodness_dict = PyDict_New();
    PyObject *mixture_dict = PyDict_New();
    PyObject *fit_dict = PyDict_New();
    PyObject *values[] = {PyFloat_FromDouble(data.NLE), PyFloat_FromDouble(data.LogDet),
                          PyFloat_FromDouble(data.fit_laplace), PyFloat_FromDouble(data.fit_bic),
                          PyFloat_FromDouble(data.fit_aic), PyLong_FromLong(data.nSelectRounds)};
    const char *names[] = {"NLE", "LogDet", "Laplace", "BIC", "AIC"};
    for (int m = 0; m < 5; m++)
        PyDict_SetItemString(goodness_dict, names[m], values[m]);
    PyDict_SetItemString(result, "GoodnessOfFit", goodness_dict);
    PyDict_SetItemString(result, "Group", group_array);
    PyDict_SetItemString(mixture_dict, "Weight", (PyObject *)weight_array);
    PyDict_SetItemString(result, "Mixture", mixture_dict);
    PyDict_SetItemString(fit_dict, "Lower", lower_array);
    PyDict_SetItemString(fit_dict, "Estimate", mpe_array);
    PyDict_SetItemString(fit_dict, "Upper", upper_array);
    PyDict_SetItemString(result, "Fit", fit_dict);
    PyDict_SetItemString(result, "NodeShared", (PyObject *)shared_array);
    PyDict_SetItemString(result, "SelectRounds", values[5]);
    for (int m = 0; m < 6; m++)
        Py_DECREF(values[m]);
    Py_DECREF(goodness_dict); Py_DECREF(mixture_dict); Py_DECREF(fit_dict);
    Py_DECREF(group_array); Py_DECREF(lower_array); Py_DECREF(mpe_array);
    Py_DECREF(upper_array); Py_DECREF(weight_array); Py_DECREF(shared_array);

cleanup:
    free(data.group);
    free(data.mixture_wt);
    free(data.node_shared);
    free(data.fit_lower);
    free(data.fit_mpe);
    free(data.fit_upper);
    Py_DECREF(counts_contiguous);
    Py_DECREF(start_c);
    Py_DECREF(child_c);
    return result;
}

static PyMethodDef module_methods[] = {
    {"dirichlet_fit", (PyCFunction)dirichlet_fit_py,
     METH_VARARGS | METH_KEYWORDS,
     "Fit Dirichlet mixture model to count data\n\n"
     "Parameters:\n"
     "  counts : array_like, shape (N, S)\n"
     "      Count data with N samples and S features\n"
     "  n_components : int, optional (default=2)\n"
     "      Number of mixture components\n"
     "  verbose : int, optional (default=0)\n"
     "      Verbosity level\n"
     "  seed : int, optional (default=42)\n"
     "      Random seed\n"
     "  return_hessian : bool, optional (default=False)\n"
     "      Also return 'Hessian', shape (K, S, S): per-component Hessian of\n"
     "      the negative log posterior w.r.t. log(alpha)\n"
     "  restart_collapsed : bool, optional (default=False)\n"
     "      If EM ends with all components equal (every sample has the same\n"
     "      assignments), restart once from hard k-means (k-means++ seeding)\n\n"
     "Returns:\n"
     "  dict : Dictionary containing fit results"},
    {"dirichlet_tree_fit", (PyCFunction)dirichlet_tree_fit_py,
     METH_VARARGS | METH_KEYWORDS,
     "Fit Dirichlet-tree multinomial mixture model to count data\n\n"
     "Parameters:\n"
     "  counts : array_like, shape (N, S), int32\n"
     "      Count data with N samples and S features (the tree's leaves)\n"
     "  child_start : array_like, shape (n_nodes + 1,), int32\n"
     "  child : array_like, shape (n_edges,), int32\n"
     "      Internal node u has children child[child_start[u]:child_start[u+1]];\n"
     "      ids < S are leaves, S + u is internal node u. Children before\n"
     "      parents; the root is the last node\n"
     "  n_components : int, optional (default=2)\n"
     "  verbose : int, optional (default=0)\n"
     "  seed : int, optional (default=42)\n"
     "  node_selection : bool, optional (default=True)\n"
     "      Share a node's parameters across components unless that loses\n"
     "      more than a BIC penalty\n\n"
     "Returns:\n"
     "  dict : GoodnessOfFit, Group (N, K), Mixture/Weight (K,),\n"
     "      Fit/Lower|Estimate|Upper (n_edges, K), NodeShared (n_nodes,),\n"
     "      SelectRounds"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT,
    "pydmm_core",
    "Python wrapper for Dirichlet Mixture Model fitting",
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit_pydmm_core(void)
{
    PyObject *module = PyModule_Create(&module_definition);
    if (!module) return NULL;

    import_array();

    return module;
}