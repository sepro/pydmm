# pyDMM - Python Dirichlet Mixture Model

pyDMM provides a Python interface to a C implementation of Dirichlet Mixture Model fitting for compositional data analysis. It's particularly useful for analyzing microbiome data and other count-based compositional datasets.

## Features

- **High-performance C implementation** using GSL (GNU Scientific Library)
- **Scikit-learn compatible API** with `fit()`, `predict_proba()`, and `fit_predict()` methods
- **Pandas DataFrame integration** with preserved sample and feature names
- **Comprehensive model diagnostics** including BIC, AIC, and parameter estimates
- **Automatic model selection** using information criteria
- **Label-invariant clustering evaluation** for proper unsupervised assessment

## Installation

### Prerequisites

Install the GNU Scientific Library (GSL) development headers:

```bash
# Ubuntu/Debian
sudo apt-get install libgsl-dev

# macOS (with Homebrew)
brew install gsl

# CentOS/RHEL/Fedora
sudo yum install gsl-devel  # or dnf install gsl-devel
```

### Install pyDMM

```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

## Quick Start

```python
import numpy as np
import pandas as pd
from pydmm import DirichletMixture

# Create sample data (or load your own compositional count data)
data = np.array([
    [100, 50, 25, 5],   # Sample 1
    [95,  55, 30, 10],  # Sample 2
    [10,  15, 80, 45],  # Sample 3
    [5,   20, 85, 40],  # Sample 4
], dtype=np.int32)

# Fit Dirichlet mixture model
dmm = DirichletMixture(n_components=2, random_state=42)
dmm.fit(data)

# Get cluster assignments
predicted_labels = dmm.result_.get_best_component()
print("Cluster assignments:", predicted_labels)

# Get assignment probabilities
probabilities = dmm.result_.get_group_assignments_df()
print("Assignment probabilities:")
print(probabilities)

# Model summary
summary = dmm.result_.summary()
print(f"BIC: {summary['goodness_of_fit']['BIC']:.1f}")
print(f"AIC: {summary['goodness_of_fit']['AIC']:.1f}")
```

## Model Selection

pyDMM supports automatic model selection using information criteria:

```python
# Test different numbers of clusters
cluster_options = [1, 2, 3, 4]
bic_scores = []

for k in cluster_options:
    dmm = DirichletMixture(n_components=k, random_state=42)
    dmm.fit(data)
    bic = dmm.result_.goodness_of_fit['BIC']
    bic_scores.append(bic)
    print(f"{k} clusters: BIC = {bic:.1f}")

# Select optimal number of clusters
best_k = cluster_options[np.argmin(bic_scores)]
print(f"Optimal number of clusters: {best_k}")
```

## Working with Pandas DataFrames

```python
# Create DataFrame with sample and feature names
samples_df = pd.DataFrame(
    data,
    index=[f"Sample_{i}" for i in range(len(data))],
    columns=[f"Feature_{i}" for i in range(data.shape[1])]
)

# Fit model
dmm = DirichletMixture(n_components=2, random_state=42)
dmm.fit(samples_df)

# Results preserve sample/feature names
assignments_df = dmm.result_.get_group_assignments_df()
print(assignments_df)

# Parameter estimates with feature names
param_estimates = dmm.result_.get_parameter_estimates_df()
print(param_estimates['Estimate'])
```

## API Reference

### DirichletMixture

**Parameters:**
- `n_components` (int): Number of mixture components (default: 2)
- `verbose` (bool): Print fitting progress (default: False)
- `random_state` (int): Random seed for reproducibility (default: 42)

**Methods:**
- `fit(X)`: Fit the model to data X
- `predict_proba(X)`: Get cluster assignment probabilities
- `fit_predict(X)`: Fit model and return cluster assignments

### DirichletMixtureResult

**Attributes:**
- `goodness_of_fit`: Dictionary with NLE, BIC, AIC, etc.
- `group_assignments`: Cluster assignment probabilities
- `mixture_weights`: Weight of each mixture component
- `parameter_estimates`: Parameter estimates with confidence intervals

**Methods:**
- `get_best_component()`: Get most likely cluster for each sample
- `get_group_assignments_df()`: Get assignments as pandas DataFrame
- `get_parameter_estimates_df()`: Get parameter estimates as DataFrames
- `summary()`: Get comprehensive model summary

## Examples

See the included example files:

- `example_reference_counts.py`: Complete workflow with model selection and evaluation

Run an example:
```bash
python example_reference_counts.py
```

## For Developers

### Running Tests

The package includes a comprehensive test suite with 91% code coverage:

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=pydmm --cov-report=term-missing

# Run specific test file
pytest tests/test_core.py

# Run specific test
pytest tests/test_core.py::TestDirichletMixture::test_clustering_performance
```

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd pyDMM

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in development mode with test dependencies
pip install -e ".[dev]"

# Install GSL (if not already installed)
sudo apt-get install libgsl-dev  # Ubuntu/Debian

# Run tests to verify installation
pytest
```

### Modifying C Code

After making changes to the C extension:

```bash
# Reinstall to recompile the C extension
pip install -e .

# Run tests to ensure changes work correctly
pytest tests/test_c_extension.py
```

### Code Architecture

- **`src/pydmm/core.py`**: Main Python interface and result handling
- **`src/pydmm/wrapper.c`**: NumPy C API wrapper with data layout conversion
- **`src/pydmm/dirichlet_fit_standalone.c`**: Core C implementation with GSL
- **`setup.py`**: Build configuration for C extension
- **`tests/`**: Comprehensive test suite

## Requirements

- Python >= 3.7
- NumPy >= 1.19.0
- pandas >= 1.0.0
- GSL (GNU Scientific Library)

## License

MIT License

## Citation

If you use pyDMM in your research, please cite:

```
[Citation information to be added]
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `pytest`
6. Submit a pull request

## Support

For questions, issues, or contributions, please use the GitHub issue tracker.