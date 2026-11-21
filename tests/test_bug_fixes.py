"""
Regression tests for bugs that were listed in BUGS.md.
"""

import numpy as np
import pytest
from sklearn.base import clone

from pydmm import DirichletMixture, pydmm_core


def _close_groups(seed, n=60, s=50):
    """Two groups with almost the same composition (5 of s features x1.5)."""
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(s) * 5)
    labels = rng.integers(0, 2, n)
    X = np.zeros((n, s), np.int32)
    for i in range(n):
        q = p.copy()
        if labels[i]:
            q[:5] *= 1.5
        X[i] = rng.multinomial(20000, rng.dirichlet(q / q.sum() * 2000))
    return X, labels


def _accuracy(z, labels):
    a = np.mean(z.argmax(axis=1) == labels)
    return max(a, 1.0 - a)


class TestRestartCollapsed:
    """Bug 1: soft k-means start collapses all components (restart_collapsed)."""

    @pytest.fixture(scope="class")
    def collapsing(self):
        # seed 1 collapses with the default soft k-means start
        return _close_groups(1)

    def test_default_is_off(self):
        m = DirichletMixture()
        assert m.restart_collapsed is False
        assert m.get_params()["restart_collapsed"] is False
        assert clone(DirichletMixture(restart_collapsed=True)).restart_collapsed is True

    def test_default_still_collapses(self, collapsing):
        X, _ = collapsing
        z = DirichletMixture(random_state=0).fit(X).result_.group_assignments
        np.testing.assert_allclose(z, 0.5, atol=1e-3)

    def test_restart_finds_groups(self, collapsing):
        X, labels = collapsing
        default = DirichletMixture(random_state=0).fit(X)
        m = DirichletMixture(random_state=0, restart_collapsed=True).fit(X)
        z = m.result_.group_assignments
        assert np.ptp(z[:, 0]) > 0.5
        assert _accuracy(z, labels) > 0.95
        assert m.result_.goodness_of_fit["BIC"] < default.result_.goodness_of_fit["BIC"]

    @pytest.mark.parametrize("seed", [0, 2])
    def test_no_change_without_collapse(self, seed):
        X, _ = _close_groups(seed)
        a = pydmm_core.dirichlet_fit(X, n_components=2, seed=3)
        b = pydmm_core.dirichlet_fit(X, n_components=2, seed=3, restart_collapsed=True)
        assert np.ptp(a["Group"][:, 0]) > 0.5
        np.testing.assert_array_equal(a["Group"], b["Group"])
        for key in a["Fit"]:
            np.testing.assert_array_equal(a["Fit"][key], b["Fit"][key])
        assert a["GoodnessOfFit"] == b["GoodnessOfFit"]

    def test_single_component(self, collapsing):
        X, _ = collapsing
        a = pydmm_core.dirichlet_fit(X, n_components=1, seed=0)
        b = pydmm_core.dirichlet_fit(X, n_components=1, seed=0, restart_collapsed=True)
        np.testing.assert_array_equal(a["Fit"]["Estimate"], b["Fit"]["Estimate"])
