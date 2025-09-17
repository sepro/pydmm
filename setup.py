from setuptools import setup, Extension, find_packages
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
            '/usr/include/gsl',  # GSL headers
        ],
        libraries=['gsl', 'gslcblas', 'm'],  # GSL libraries
        library_dirs=['/usr/lib/x86_64-linux-gnu'],  # Standard library path
        define_macros=[
            ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
        ],
        extra_compile_args=['-O3', '-Wall', '-std=c99'],
    )
]

setup(
    name='pydmm',
    version='0.1.0',
    description='Python wrapper for Dirichlet Mixture Model fitting',
    long_description="""
    pyDMM provides a Python interface to a C implementation of Dirichlet Mixture
    Model fitting for compositional data analysis. It's particularly useful for
    analyzing microbiome data and other count-based compositional datasets.

    Features:
    - Fast C implementation with GSL optimization
    - Pandas DataFrame integration
    - Scikit-learn compatible interface
    - Comprehensive model diagnostics and parameter estimates
    """,
    long_description_content_type='text/plain',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/your-username/pyDMM',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    ext_modules=ext_modules,
    install_requires=[
        'numpy>=1.19.0',
        'pandas>=1.0.0',
        'scipy>=1.5.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.10',
        ],
        'test': [
            'pytest>=6.0',
            'pytest-cov>=2.10',
        ]
    },
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    keywords='dirichlet mixture model compositional data microbiome',
    zip_safe=False,
)