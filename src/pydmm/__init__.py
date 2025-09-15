"""
pyDMM: Python wrapper for Dirichlet Mixture Model fitting

This package provides a Python interface to a C implementation of
Dirichlet Mixture Model fitting for compositional data.
"""

from .core import DirichletMixture

__version__ = "0.1.0"
__all__ = ["DirichletMixture"]