"""
Minimal setup.py for C extension configuration.
All package metadata is in pyproject.toml.
"""
from setuptools import setup, Extension
import numpy as np

# Define the C extension module
ext_modules = [
    Extension(
        'pydmm.pydmm_core',
        sources=[
            'src/pydmm/wrapper.c',
            'src/pydmm/dirichlet_fit_standalone.c'
        ],
        include_dirs=[
            np.get_include(),
            'src/pydmm',
            '/usr/include/gsl',
        ],
        libraries=['gsl', 'gslcblas', 'm'],
        library_dirs=['/usr/lib/x86_64-linux-gnu'],
        define_macros=[
            ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
        ],
        extra_compile_args=['-O3', '-Wall', '-std=c99'],
    )
]

# All metadata is in pyproject.toml - this only defines the C extension
setup(ext_modules=ext_modules)
