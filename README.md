[![Run Tests](https://github.com/sepro/pydmm/actions/workflows/test.yml/badge.svg)](https://github.com/sepro/pydmm/actions/workflows/test.yml)

# pyDMM - Python Dirichlet Mixture Model

pyDMM provides a Python interface to a C implementation of Dirichlet Mixture Model fitting for compositional data analysis. It's particularly useful for analyzing microbiome data and other count-based compositional datasets.

This includes a few experimental features like WAIC instead of the BIC, calibrating assignent probabilities and a Dirichlet Tree Mixtured model implementation which were not included in the original. Futhermore, this package is compatible with scikit-learn.

## Features

- **High-performance C implementation** using GSL (GNU Scientific Library)
- **Full scikit-learn compatibility** with BaseEstimator and ClassifierMixin
  - Works with GridSearchCV for hyperparameter tuning
  - Compatible with cross_val_score for cross-validation
  - Integrates with sklearn pipelines
  - Supports get_params() and set_params() for parameter management
- **Standard sklearn API** with `fit()`, `predict()`, `predict_proba()`, and `fit_predict()` methods
- **Pandas DataFrame integration** with preserved sample and feature names
- **Comprehensive model diagnostics** including BIC, AIC, optional WAIC, and parameter estimates
- **Dirichlet-tree mixtures** (`DirichletTreeMixture`) for taxa that vary together along a taxonomy or a tree built from the data
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

### Collapsed fits

EM starts from a soft k-means clustering. When the samples have very similar compositions (many features, small differences between groups), this start can make all components almost equal, and EM stays there: every assignment probability is 1/K and the weights are equal. Set `restart_collapsed=True` to detect this and restart once from hard k-means with k-means++ seeding. Fits that do not collapse are not changed. It is off by default so that existing results stay the same; `DirichletTreeMixture` always restarts.

```python
dmm = DirichletMixture(n_components=2, restart_collapsed=True).fit(data)
```

### WAIC

Set `compute_waic=True` to also compute the widely applicable information criterion (WAIC). EM only finds the posterior mode, so pyDMM draws `n_waic_draws` (default 1,000) parameter sets from a joint Laplace approximation around it. The covariance is the inverse observed information of the mixture over log α and the weight logits. It includes the uncertainty in which cluster each sample belongs to, which is what penalises redundant or split clusters. Directions the data barely constrain, which overfitted fits have, get their posterior SD capped at 1; `goodness_of_fit["WAIC_n_regularised"]` reports how many. WAIC is off by default, so ordinary fits are not slowed down.

```python
dmm = DirichletMixture(n_components=3, compute_waic=True, n_waic_draws=1000)
dmm.fit(data)
gof = dmm.result.goodness_of_fit
print(gof["WAIC"], gof["WAIC_se"], gof["p_waic"], gof["elpd_waic"])

# draws x samples log-likelihood matrix, for your own diagnostics
dmm.result.pointwise_log_likelihood.shape
```

`WAIC = -elpd_waic` and uses the same negative log-likelihood scale as BIC and AIC (lower is better). arviz and prodiphy report `elpd_waic` (higher is better); multiply by −2 for the deviance scale used in the literature. BIC, AIC and NLE include the Gamma(0.1, 0.1) prior on α while WAIC does not, so compare each criterion across K, not WAIC against BIC.

To choose K, don't take the minimum WAIC. Take the smallest K for which no larger K is better by more than about two standard errors of the difference. That SE comes from the pointwise values:

```python
from pydmm.waic import waic_from_log_likelihood

elpd_i = {}
for k in range(1, 9):
    m = DirichletMixture(n_components=k, compute_waic=True).fit(data)
    elpd_i[k] = waic_from_log_likelihood(m.result.pointwise_log_likelihood)["elpd_waic_i"]

def confidently_better(k_big, k_small, z=2.0):
    d = elpd_i[k_big] - elpd_i[k_small]
    return d.sum() > z * np.sqrt(d.size * d.var())

best_k = min(k for k in elpd_i
             if not any(confidently_better(k2, k) for k2 in elpd_i if k2 > k))
```

WAIC estimates out-of-sample predictive accuracy, so, like AIC, its penalty does not grow with the number of samples. See `validation/waic/` for a comparison with prodiphy's MCMC-based WAIC and an experiment on how BIC, AIC and WAIC behave as N grows.

## Calibrated Assignment Probabilities

The usual assignment probabilities are often too close to 0 or 1. The Dirichlet-multinomial treats every feature as independent evidence, so features that rise and fall together (taxa of one guild, ASVs of one strain) count the same signal several times. The Dirichlet also fixes how much each feature may vary, so fluctuations in abundant features are read as strong evidence. In simulations pyDMM reported about 98% confidence for samples it assigned correctly only 70–85% of the time.

Set `temperature` to temper the probabilities. With a value τ in (0, 1] they become z<sub>ik</sub> ∝ (w<sub>k</sub> · DM(x<sub>i</sub> | α<sub>k</sub>))<sup>τ</sup>. `"auto"` estimates τ from the fitted model without labels: a calibrated log-likelihood ratio between two components that is roughly normal has variance equal to twice its mean (Good's weight-of-evidence condition), and an inflated one does not. `"auto"` iterates this to a fixed point: each step weights the samples with the tempered probabilities of the previous step. `"one-step"` stops after the first step, so the weights are the untempered probabilities. On heavy-tailed data the iteration can drift to a τ that is much too small, and `"one-step"` avoids that; on real data it gave probabilities closer to bootstrap stability for the Dirichlet-tree mixture. Tempering is off by default.

```python
dmm = DirichletMixture(n_components=3, temperature="auto").fit(data)
dmm.temperature_                              # estimated tau, e.g. 0.2
# temperature="one-step" (first step only) or a fixed value, e.g. temperature=0.3
dmm.result.group_assignments                  # tempered probabilities
dmm.result.group_assignments_untempered       # the usual EM responsibilities
dmm.predict_proba(new_data)                   # tempered with the same tau
```

Tempering changes only the probabilities. The fitted α and weights, NLE, BIC, AIC and WAIC, and each sample's most likely component stay exactly as without it. It cannot repair a wrong partition, only report it with less confidence. A τ well below 1 on your data also tells you how strongly the features overlap in what they say. See `validation/correlation/` for the simulations.

## Dirichlet-Tree Mixtures

`DirichletTreeMixture` replaces each component's Dirichlet with a Dirichlet-tree: every internal node of a tree over the features splits its count over its children with its own Dirichlet-multinomial (Dennis, 1991; Wang & Zhao, 2017; Mao & Ma, 2022). Taxa in one clade can then rise and fall together, and every split has its own dispersion. In the DMM, features can only be negatively correlated and one precision fixes how much every feature varies, which makes its assignment probabilities over-confident when groups of taxa bloom together.

```python
import pandas as pd
from pydmm import DirichletTreeMixture, Tree

# taxonomy: one row per feature (same names as the count columns), ranks from top to bottom
taxonomy = pd.DataFrame({"Family": [...], "Genus": [...]}, index=counts.columns)
tree = Tree.from_taxonomy(taxonomy)

dtmm = DirichletTreeMixture(n_components=3, tree=tree, random_state=42).fit(counts)
dtmm.predict_proba(counts)
dtmm.result_.differential_nodes()          # clades whose composition separates the clusters
dtmm.result_.goodness_of_fit["BIC"]        # same scale as DirichletMixture's BIC

DirichletTreeMixture(n_components=3).fit(counts)   # no tree: built from feature correlations
```

Trees can also come from a SciPy linkage matrix (`Tree.from_linkage`), a parent array (`Tree.from_parents`) or `Tree.from_correlation(X)`. With `Tree.star(n_features)` and `node_selection=False` the model is exactly the DMM.

By default only nodes whose parameters differ enough between clusters to pay a BIC penalty are fitted per cluster (`node_selection=True`); the others are shared and do not influence the assignments. This matters: without it, nodes that do not differ add noise and the DTMM clusters less accurately than the DMM. `temperature` works as for `DirichletMixture`, but on real data `"auto"` can over-correct a DTMM to almost uniform probabilities (`BUGS.md`, bug 5). Use `temperature="one-step"` with the DTMM. WAIC is not available for the DTMM yet.

In simulations (`docs/dtmm_comparison.md`), with the true tree the DTMM's probabilities matched an oracle that knows the generating parameters (log-loss 0.41 vs 1.45 for the DMM) when whole clades bloomed together, and with the default data-built tree it removed the DMM's wrong splits on data where features differ in variability (accuracy 0.945 vs 0.896). It was worse than the DMM when strongly correlated modules dominated, even though BIC preferred it, so compare both models on your data. A fit takes about 1.5× as long as a DMM fit with a taxonomy-like tree and about 2.7× with the default data-built tree.

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

## Component Labeling

By default, clusters are identified by numeric indices (0, 1, 2, ...). For better interpretability, you can assign human-readable labels to components:

```python
# Fit model
dmm = DirichletMixture(n_components=3, random_state=42)
dmm.fit(data)

# Assign meaningful labels to components
dmm.result.set_component_labels({
    0: 'Healthy',
    1: 'Diseased',
    2: 'Control'
})

# Get cluster assignments with labels
labels = dmm.result.get_best_component()
print("Cluster assignments:", labels)  # Returns ['Healthy', 'Diseased', ...]

# Assignment probabilities also use labels
probabilities = dmm.result.get_group_assignments_df()
print(probabilities)  # Columns named 'Healthy', 'Diseased', 'Control'

# Predictions also return labeled clusters
new_labels = dmm.predict(new_data)
print("New sample labels:", new_labels)  # Returns labeled predictions
```

**Note:** Component labels must be set after fitting the model and before making predictions or retrieving results.

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

Inherits from `sklearn.base.BaseEstimator` and `sklearn.base.ClassifierMixin` for full sklearn compatibility.

**Parameters:**
- `n_components` (int): Number of mixture components (default: 2)
- `verbose` (bool): Print fitting progress (default: False)
- `random_state` (int): Random seed for reproducibility (default: 42)
- `restart_collapsed` (bool): Restart once from hard k-means when the fit ends with all components equal (default: False). See [Collapsed fits](#collapsed-fits)

**Methods:**
- `fit(X, y=None)`: Fit the model to data X (y is ignored, for sklearn compatibility)
- `predict(X)`: Predict the most likely cluster for each sample
- `predict_proba(X)`: Get cluster assignment probabilities for each sample
- `fit_predict(X)`: Fit model and return cluster assignments
- `score(X)`: Return the negative log-likelihood of the data
- `get_params(deep=True)`: Get parameters for this estimator
- `set_params(**params)`: Set parameters for this estimator

**Attributes (after fitting):**
- `result_`: DirichletMixtureResult object with detailed results
- `classes_`: Array of cluster labels [0, 1, ..., n_components-1]
- `is_fitted`: Boolean indicating if the model has been fitted

### DirichletTreeMixture

Subclass of `DirichletMixture`, with the same methods and attributes.

**Parameters:**
- `n_components` (int): Number of mixture components (default: 2)
- `tree` (`Tree` or None): Tree over the features; None builds one from the data with `Tree.from_correlation` (default: None). Leaves are matched to DataFrame columns by name when the tree has leaf names
- `node_selection` (bool): Share each node's parameters across components unless they differ enough to pay a BIC penalty (default: True)
- `verbose` (bool): Print fitting progress (default: False)
- `random_state` (int): Random seed for reproducibility (default: 42)
- `temperature` (None, float, "auto" or "one-step"): Temper the assignment probabilities (default: None, off)

**Attributes (after fitting):** `result_` (`DirichletTreeMixtureResult`), `tree_` (the tree used, leaves in the data's column order), `classes_`, `feature_names_in_`, `temperature_`

### DirichletTreeMixtureResult

As `DirichletMixtureResult`, with `parameter_estimates` per tree edge (shape `(n_edges, n_components)`), plus `tree`, `node_shared` (bool per internal node), `differential_nodes()` and `get_parameter_estimates_df()` indexed by (parent, child).

### Tree

Build with `Tree.from_taxonomy(df)`, `Tree.from_linkage(Z)`, `Tree.from_parents(parents, n_leaves)`, `Tree.from_correlation(X)` or `Tree.star(n_leaves)`. Internal nodes with one child are removed. Useful methods: `leaves(u)`, `children(u)`, `node_counts(X)`, `reorder(feature_names)`, `edge_index()`.

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
- `set_component_labels(labels)`: Assign human-readable labels to components (dict mapping component index to label string)

## Examples

See the included example files in `docs/examples/`:

- `sklearn_compatibility.py`: Comprehensive sklearn integration demonstration (GridSearchCV, cross-validation, parameter management)
- `reference_counts.py`: Complete workflow with model selection and evaluation
- `component_labeling.py`: Demonstration of human-readable component labeling for interpretable results
- `probability_comparison.py`: Compare probability computations between C and Python implementations
- `classification_comparison.py`: Compare classification decisions between C and Python implementations

Run an example:
```bash
python docs/examples/sklearn_compatibility.py
```

## Developer Documentation

For developers who want to contribute to pyDMM, modify its functionality, or understand the codebase architecture, see the [Developer Documentation](docs/dev.md).

## Requirements

- Python >= 3.10
- NumPy >= 1.19.0
- pandas >= 1.0.0
- scikit-learn >= 0.24.0
- scipy >= 1.5.0
- GSL (GNU Scientific Library)

## License

GNU Lesser General Public License v3.0 (LGPL v3)

This package is licensed under the LGPL v3 license to maintain compatibility with the underlying C code components that are also under LGPL license.

## Acknowledgments

This package is based on the R package [DirichletMultinomial](https://github.com/mtmorgan/DirichletMultinomial) by Martin Morgan, which in turn implements methods from:

> Holmes, I., Harris, K., & Quince, C. (2012). Dirichlet multinomial mixtures: generative models for microbial metagenomics. PLoS ONE, 7(2): e30126. https://doi.org/10.1371/journal.pone.0030126

These sources have been instrumental in the development of pyDMM, providing both the theoretical foundation and reference implementation.

The Dirichlet-tree mixture (`DirichletTreeMixture`) builds on:

> Dennis, S. Y. (1991). On the hyper-Dirichlet type 1 and hyper-Liouville distributions. Communications in Statistics – Theory and Methods, 20(12), 4069–4081. https://doi.org/10.1080/03610929108830757

> Wang, T., & Zhao, H. (2017). A Dirichlet-tree multinomial regression model for associating dietary nutrients with gut microorganisms. Biometrics, 73(3), 792–801. https://doi.org/10.1111/biom.12654

> Mao, J., & Ma, L. (2022). Dirichlet-tree multinomial mixtures for clustering microbiome compositions. The Annals of Applied Statistics, 16(3), 1476–1499. https://doi.org/10.1214/21-AOAS1552

## Citation

If you use pyDMM in your research, please cite:

```
[Citation information to be added]
```

Additionally, please consider citing the original publication:

```bibtex
@article{holmes2012dirichlet,
  title={Dirichlet multinomial mixtures: generative models for microbial metagenomics},
  author={Holmes, Ian and Harris, Keith and Quince, Christopher},
  journal={PLoS ONE},
  volume={7},
  number={2},
  pages={e30126},
  year={2012},
  publisher={Public Library of Science},
  doi={10.1371/journal.pone.0030126}
}
```

If you use `DirichletTreeMixture`, please also cite:

```bibtex
@article{dennis1991hyper,
  title={On the hyper-Dirichlet type 1 and hyper-Liouville distributions},
  author={Dennis, Samuel Y.},
  journal={Communications in Statistics - Theory and Methods},
  volume={20},
  number={12},
  pages={4069--4081},
  year={1991},
  doi={10.1080/03610929108830757}
}

@article{wang2017dirichlet,
  title={A Dirichlet-tree multinomial regression model for associating dietary nutrients with gut microorganisms},
  author={Wang, Tao and Zhao, Hongyu},
  journal={Biometrics},
  volume={73},
  number={3},
  pages={792--801},
  year={2017},
  doi={10.1111/biom.12654}
}

@article{mao2022dirichlet,
  title={Dirichlet-tree multinomial mixtures for clustering microbiome compositions},
  author={Mao, Jialiang and Ma, Li},
  journal={The Annals of Applied Statistics},
  volume={16},
  number={3},
  pages={1476--1499},
  year={2022},
  doi={10.1214/21-AOAS1552}
}
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

