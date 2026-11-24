"""Bootstrap-averaged assignment probabilities (n_bootstrap) for DirichletMixture."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from pydmm import DirichletMixture


def _dm_data(seed, n=150, S=12, shift=0.8, theta=30.0, depth=2000):
    rng = np.random.default_rng(seed)
    base = np.exp(rng.normal(0, 0.7, S))
    alt = base * np.exp(np.r_[shift, -shift, np.zeros(S - 2)])
    labels = rng.integers(0, 2, n)
    X = np.zeros((n, S), dtype=np.int32)
    for i in range(n):
        a = (base, alt)[labels[i]]
        X[i] = rng.multinomial(depth, rng.dirichlet(a / a.sum() * theta))
    return X, labels


@pytest.fixture(scope="module")
def data():
    return _dm_data(0)


def test_default_unchanged(data):
    X, _ = data
    a = DirichletMixture(n_components=2).fit(X)
    b = DirichletMixture(n_components=2, n_bootstrap=0).fit(X)
    np.testing.assert_array_equal(a.result_.group_assignments, b.result_.group_assignments)
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))
    assert a.bootstrap_ == [] and a.result_.group_assignments_single is None


def test_bagged_probabilities(data):
    X, labels = data
    single = DirichletMixture(n_components=2).fit(X)
    m = DirichletMixture(n_components=2, n_bootstrap=4).fit(X)
    z = m.result_.group_assignments
    assert len(m.bootstrap_) == 4
    np.testing.assert_allclose(z.sum(1), 1.0)
    np.testing.assert_allclose(m.predict_proba(X), z)
    np.testing.assert_array_equal(m.result_.group_assignments_single,
                                  single.result_.group_assignments)
    assert m.result_.goodness_of_fit == single.result_.goodness_of_fit
    np.testing.assert_array_equal(m.result_.parameter_estimates["Estimate"],
                                  single.result_.parameter_estimates["Estimate"])
    # components stay aligned: same most likely component for (almost) all samples
    assert np.mean(z.argmax(1) == single.result_.group_assignments.argmax(1)) > 0.95
    # averaging can only lower the confidence on average (never above the max of its parts)
    assert z.max(1).mean() <= single.result_.group_assignments.max(1).mean() + 1e-9


def test_reproducible(data):
    X, _ = data
    a = DirichletMixture(n_components=2, n_bootstrap=2, random_state=3).fit(X)
    b = DirichletMixture(n_components=2, n_bootstrap=2, random_state=3).fit(X)
    np.testing.assert_array_equal(a.result_.group_assignments, b.result_.group_assignments)


def test_new_data_and_labels(data):
    X, _ = data
    m = DirichletMixture(n_components=2, n_bootstrap=2).fit(X[:100])
    p = m.predict_proba(X[100:])
    assert p.shape == (50, 2)
    np.testing.assert_allclose(p.sum(1), 1.0)
    m.result_.set_component_labels({0: "a", 1: "b"})
    df = m.predict_proba(X[100:])
    assert isinstance(df, pd.DataFrame)
    np.testing.assert_allclose(df.to_numpy(), p)


def test_with_temperature_and_waic(data):
    X, _ = data
    m = DirichletMixture(n_components=2, n_bootstrap=2, temperature="one-step",
                         compute_waic=True, n_waic_draws=50).fit(X)
    assert all(b["tau"] is not None for b in m.bootstrap_)
    assert "WAIC" in m.result_.goodness_of_fit
    np.testing.assert_allclose(m.predict_proba(X), m.result_.group_assignments)


@pytest.mark.parametrize("bad", [-1, 1.5, "3", True])
def test_invalid(data, bad):
    X, _ = data
    with pytest.raises(ValueError, match="n_bootstrap"):
        DirichletMixture(n_bootstrap=bad).fit(X)


def test_clone(data):
    m = DirichletMixture(n_components=3, n_bootstrap=5)
    assert clone(m).n_bootstrap == 5
