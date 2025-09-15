#include <Python.h>
#include <numpy/arrayobject.h>
#include "dirichlet_fit_standalone.h"

static PyObject *dirichlet_fit_py(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *counts_array = NULL;
    int n_components = 2;
    int verbose = 0;
    int seed = 42;

    static char *kwlist[] = {"counts", "n_components", "verbose", "seed", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|iii", kwlist,
                                     &PyArray_Type, &counts_array,
                                     &n_components, &verbose, &seed)) {
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
    data->N = N;  // C code expects N samples
    data->S = S;  // C code expects S features
    data->K = K;
    data->aanX = (int *)PyArray_DATA(counts_contiguous);

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

    // Clean up
    Py_DECREF(counts_contiguous);  // Clean up transposed array
    free(data);

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
     "      Random seed\n\n"
     "Returns:\n"
     "  dict : Dictionary containing fit results"},
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