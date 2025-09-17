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

# Get cluster assignments (two equivalent methods)
predicted_labels = dmm.result.get_best_component()  # From fitted model
# OR
predicted_labels = dmm.predict(data)  # Using predict method
print("Cluster assignments:", predicted_labels)

# Get assignment probabilities
probabilities = dmm.result.get_group_assignments_df()
print("Assignment probabilities:")
print(probabilities)

# Model summary
summary = dmm.result.summary()
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
    bic = dmm.result.goodness_of_fit['BIC']
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
assignments_df = dmm.result.get_group_assignments_df()
print(assignments_df)

# Parameter estimates with feature names
param_estimates = dmm.result.get_parameter_estimates_df()
print(param_estimates['Estimate'])
```

## Predicting New Data

After fitting a model, you can predict cluster assignments for new samples:

```python
# Fit model on training data
dmm = DirichletMixture(n_components=2, random_state=42)
dmm.fit(training_data)

# Predict cluster assignments for new data
new_data = np.array([
    [120, 60, 30, 15],  # New sample 1
    [8,   25, 90, 50],  # New sample 2
], dtype=np.int32)

# Get hard cluster assignments
new_labels = dmm.predict(new_data)
print("New sample cluster assignments:", new_labels)

# Get soft cluster assignments (probabilities)
new_probabilities = dmm.predict_proba(new_data)
print("New sample probabilities:")
print(new_probabilities)
```

## API Reference

### DirichletMixture

**Parameters:**
- `n_components` (int): Number of mixture components (default: 2)
- `verbose` (bool): Print fitting progress (default: False)
- `random_state` (int): Random seed for reproducibility (default: 42)

**Methods:**
- `fit(X)`: Fit the model to data X
- `predict(X)`: Predict the most likely cluster for each sample
- `predict_proba(X)`: Get cluster assignment probabilities for each sample
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

See the included example files in `docs/examples/`:

- `reference_counts.py`: Complete workflow with model selection and evaluation
- `probability_comparison.py`: Compare probability computations between C and Python implementations
- `classification_comparison.py`: Compare classification decisions between C and Python implementations

Run an example:
```bash
python docs/examples/reference_counts.py
```

## Developer Documentation

For developers who want to contribute to pyDMM, modify its functionality, or understand the codebase architecture, see the [Developer Documentation](docs/dev.md).

## Requirements

- Python >= 3.7
- NumPy >= 1.19.0
- pandas >= 1.0.0
- GSL (GNU Scientific Library)

## License

GNU Lesser General Public License v3.0 (LGPL v3)

This package is licensed under the LGPL v3 license to maintain compatibility with the underlying C code components that are also under LGPL license.

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