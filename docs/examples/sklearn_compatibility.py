"""
Example demonstrating sklearn compatibility of DirichletMixture
"""

import numpy as np
from sklearn.base import is_classifier, clone
from sklearn.model_selection import cross_val_score, GridSearchCV
from pydmm import DirichletMixture

# Set random seed for reproducibility
np.random.seed(42)

# Create synthetic data with two distinct groups
print("=" * 60)
print("pyDMM sklearn Compatibility Demo")
print("=" * 60)

# Generate test data
group1_props = np.array([0.6, 0.3, 0.05, 0.05])
group2_props = np.array([0.05, 0.05, 0.6, 0.3])

samples = []
for i in range(50):
    total_count = np.random.poisson(1000) + 100
    counts = np.random.multinomial(total_count, group1_props)
    samples.append(counts)

for i in range(50):
    total_count = np.random.poisson(1000) + 100
    counts = np.random.multinomial(total_count, group2_props)
    samples.append(counts)

X = np.array(samples, dtype=np.int32)

print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")

# 1. Test classifier recognition
print("\n1. Classifier Recognition")
print("-" * 40)
dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)
print(f"   is_classifier(DirichletMixture): {is_classifier(dmm)}")
print(f"   _estimator_type: {getattr(dmm, '_estimator_type', 'via tags')}")

# 2. Test fit with y parameter (ignored)
print("\n2. Fit with y Parameter (sklearn compatibility)")
print("-" * 40)
dummy_y = np.random.randint(0, 2, size=100)
dmm.fit(X, y=dummy_y)  # y is ignored
print(f"   Model fitted successfully (y parameter ignored)")
print(f"   classes_: {dmm.classes_}")

# 3. Test get_params and set_params
print("\n3. Parameter Management")
print("-" * 40)
params = dmm.get_params()
print(f"   get_params(): {params}")
dmm_copy = DirichletMixture()
dmm_copy.set_params(**params)
print(f"   set_params() successful")

# 4. Test clone
print("\n4. Clone Estimator")
print("-" * 40)
dmm_cloned = clone(dmm)
print(f"   Original fitted: {dmm.is_fitted}")
print(f"   Clone fitted: {dmm_cloned.is_fitted}")
print(f"   Clone params match: {dmm.get_params() == dmm_cloned.get_params()}")

# 5. Test cross-validation
print("\n5. Cross-Validation")
print("-" * 40)
dmm_cv = DirichletMixture(n_components=2, verbose=False, random_state=42)
scores = cross_val_score(dmm_cv, X, cv=3, scoring=None)
print(f"   Cross-validation scores: {scores}")
print(f"   Mean score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# 6. Test GridSearchCV
print("\n6. Hyperparameter Tuning with GridSearchCV")
print("-" * 40)
param_grid = {'n_components': [2, 3, 4]}
grid_search = GridSearchCV(
    DirichletMixture(verbose=False, random_state=42),
    param_grid,
    cv=3,
    scoring=None
)
grid_search.fit(X)
print(f"   Best parameters: {grid_search.best_params_}")
print(f"   Best score: {grid_search.best_score_:.4f}")
print(f"   Best estimator fitted: {grid_search.best_estimator_.is_fitted}")

# 7. Test predictions
print("\n7. Predictions")
print("-" * 40)
best_model = grid_search.best_estimator_
predictions = best_model.predict(X[:10])
probabilities = best_model.predict_proba(X[:10])
print(f"   First 10 predictions: {predictions}")
print(f"   First 3 probability distributions:")
for i in range(3):
    print(f"      Sample {i}: {probabilities[i]}")

# 8. Backward compatibility
print("\n8. Backward Compatibility (fit without y)")
print("-" * 40)
dmm_old = DirichletMixture(n_components=2, verbose=False, random_state=42)
dmm_old.fit(X)  # Old-style call without y
print(f"   Old-style fit() still works")
print(f"   Model has result_: {dmm_old.result_ is not None}")
print(f"   Model has classes_: {hasattr(dmm_old, 'classes_')}")

print("\n" + "=" * 60)
print("All sklearn compatibility features working correctly!")
print("=" * 60)
