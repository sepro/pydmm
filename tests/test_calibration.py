"""
Tests for tempered assignment probabilities (DirichletMixture(temperature=...)).
"""

import numpy as np
import pytest
from sklearn.base import clone

from pydmm import DirichletMixture
from pydmm.calibration import (MIN_TEMPERATURE, _pooled_weight_of_evidence,
                               component_log_likelihood, estimate_temperature,
                               tempered_responsibilities, validate_temperature)


def _dmm_data(seed, n=200, s=20, depth=2000, k=2):
    """Counts from a Dirichlet-multinomial mixture with overlapping components."""
    rng = np.random.default_rng(seed)
    base = rng.gamma(1.0, 1.0, size=s) + 0.2
    alphas = []
    for _ in range(k):
        a = base * np.exp(rng.normal(0, 0.45, size=s))
        alphas.append(a / a.sum() * 40.0)
    labels = rng.integers(0, k, size=n)
    X = np.vstack([rng.multinomial(depth, rng.dirichlet(alphas[l])) for l in labels])
    return X.astype(np.int32), labels


@pytest.fixture(scope="module")
def data():
    return _dmm_data(0)


class TestDefaultUnchanged:
    def test_default_is_off(self, data):
        X, _ = data
        m = DirichletMixture(n_components=2, random_state=1).fit(X)
        assert m.temperature is None
        assert m.temperature_ is None
        assert m.result_.temperature is None
        assert m.result_.group_assignments_untempered is None
        assert "temperature" not in m.result_.summary()

    def test_default_matches_plain_fit(self, data):
        X, _ = data
        a = DirichletMixture(n_components=2, random_state=1).fit(X)
        b = DirichletMixture(n_components=2, random_state=1, temperature=None).fit(X)
        np.testing.assert_array_equal(a.result_.group_assignments, b.result_.group_assignments)
        np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))

    def test_refit_without_temperature_resets(self, data):
        X, _ = data
        m = DirichletMixture(n_components=2, random_state=1, temperature=0.5).fit(X)
        assert m.temperature_ == 0.5
        m.set_params(temperature=None).fit(X)
        assert m.temperature_ is None
        assert m.result_.group_assignments_untempered is None


class TestTemperedAssignments:
    def test_fit_params_not_changed(self, data):
        X, _ = data
        plain = DirichletMixture(n_components=2, random_state=1).fit(X)
        temp = DirichletMixture(n_components=2, random_state=1, temperature="auto").fit(X)
        np.testing.assert_array_equal(plain.result_.parameter_estimates["Estimate"],
                                      temp.result_.parameter_estimates["Estimate"])
        np.testing.assert_array_equal(plain.result_.mixture_weights, temp.result_.mixture_weights)
        assert plain.result_.goodness_of_fit == temp.result_.goodness_of_fit
        np.testing.assert_array_equal(plain.result_.group_assignments,
                                      temp.result_.group_assignments_untempered)

    def test_temperature_one_matches_usual_responsibilities(self, data):
        X, _ = data
        plain = DirichletMixture(n_components=2, random_state=1).fit(X)
        one = DirichletMixture(n_components=2, random_state=1, temperature=1.0).fit(X)
        np.testing.assert_allclose(one.result_.group_assignments, plain.predict_proba(X),
                                   atol=1e-10)

    def test_log_odds_scaled_by_temperature(self, data):
        X, _ = data
        plain = DirichletMixture(n_components=2, random_state=1).fit(X)
        half = DirichletMixture(n_components=2, random_state=1, temperature=0.5).fit(X)
        w = plain.result_.mixture_weights
        prior = np.log(w[0] / w[1])
        p = np.clip(plain.predict_proba(X), 1e-300, 1)
        h = half.result_.group_assignments
        lo_plain = np.log(p[:, 0] / p[:, 1]) - prior
        lo_half = np.log(h[:, 0] / h[:, 1])
        ok = np.abs(lo_plain) < 500  # avoid saturated probabilities
        np.testing.assert_allclose(lo_half[ok], 0.5 * (lo_plain[ok] + prior), rtol=1e-8, atol=1e-8)

    def test_best_component_unchanged(self, data):
        X, _ = data
        plain = DirichletMixture(n_components=2, random_state=1).fit(X)
        temp = DirichletMixture(n_components=2, random_state=1, temperature=0.05).fit(X)
        np.testing.assert_array_equal(plain.predict(X), temp.predict(X))

    def test_less_confident_and_normalised(self, data):
        X, _ = data
        plain = DirichletMixture(n_components=2, random_state=1).fit(X)
        temp = DirichletMixture(n_components=2, random_state=1, temperature=0.3).fit(X)
        z = temp.result_.group_assignments
        np.testing.assert_allclose(z.sum(axis=1), 1.0)
        assert z.max(axis=1).mean() < plain.result_.group_assignments.max(axis=1).mean()

    def test_predict_proba_uses_temperature(self, data):
        X, _ = data
        m = DirichletMixture(n_components=2, random_state=1, temperature="auto").fit(X)
        np.testing.assert_allclose(m.predict_proba(X), m.result_.group_assignments, atol=1e-10)

    def test_summary_and_result(self, data):
        X, _ = data
        m = DirichletMixture(n_components=2, random_state=1, temperature=0.4).fit(X)
        assert m.result_.temperature == 0.4
        assert m.result_.summary()["temperature"] == 0.4

    def test_works_with_waic(self, data):
        X, _ = data
        plain = DirichletMixture(n_components=2, random_state=1, compute_waic=True,
                                 n_waic_draws=50).fit(X)
        temp = DirichletMixture(n_components=2, random_state=1, compute_waic=True,
                                n_waic_draws=50, temperature="auto").fit(X)
        assert plain.result_.goodness_of_fit["WAIC"] == temp.result_.goodness_of_fit["WAIC"]

    def test_clone_keeps_temperature(self):
        m = DirichletMixture(temperature="auto")
        assert clone(m).temperature == "auto"


class TestAutoTemperature:
    def test_in_range(self, data):
        X, _ = data
        m = DirichletMixture(n_components=2, random_state=1, temperature="auto").fit(X)
        assert 0.0 < m.temperature_ <= 1.0

    @pytest.mark.parametrize("k", [2, 3])
    def test_scales_with_duplicated_features(self, k):
        """Copying every feature m times inflates the log-odds m-fold; tau should follow."""
        X, _ = _dmm_data(3, n=240, s=20, k=k)
        taus = []
        for mult in (1, 2, 4):
            m = DirichletMixture(n_components=k, random_state=0, temperature="auto")
            m.fit(np.repeat(X, mult, axis=1))
            taus.append(m.temperature_)
        assert taus[1] == pytest.approx(taus[0] / 2, rel=0.2)
        assert taus[2] == pytest.approx(taus[0] / 4, rel=0.2)

    def test_single_component(self):
        assert estimate_temperature(np.zeros((10, 1)), np.array([1.0])) == 1.0

    def test_no_information_returns_one(self):
        # identical components: every log-likelihood ratio is zero
        assert estimate_temperature(np.zeros((50, 2)), np.array([0.5, 0.5])) == 1.0


class TestOneStepTemperature:
    def test_is_first_iterate(self, data):
        """One-step tau = Good's condition with the untempered assignments as weights."""
        X, _ = data
        m = DirichletMixture(n_components=2, random_state=1, temperature="one-step").fit(X)
        alpha = m.result_.parameter_estimates["Estimate"].T
        ll = component_log_likelihood(X, alpha)
        w = m.result_.mixture_weights
        z1 = tempered_responsibilities(ll, w, 1.0)
        expected = float(np.clip(_pooled_weight_of_evidence(ll, z1), MIN_TEMPERATURE, 1.0))
        assert m.temperature_ == pytest.approx(expected, rel=1e-12)
        assert estimate_temperature(ll, w, max_iter=1) == pytest.approx(expected, rel=1e-12)
        np.testing.assert_allclose(m.result_.group_assignments,
                                   tempered_responsibilities(ll, w, m.temperature_))

    def test_fit_and_argmax_unchanged(self, data):
        X, _ = data
        plain = DirichletMixture(n_components=2, random_state=1).fit(X)
        one = DirichletMixture(n_components=2, random_state=1, temperature="one-step").fit(X)
        assert 0.0 < one.temperature_ <= 1.0
        assert one.result_.temperature == one.temperature_
        np.testing.assert_array_equal(one.result_.parameter_estimates["Estimate"],
                                      plain.result_.parameter_estimates["Estimate"])
        np.testing.assert_array_equal(one.result_.group_assignments_untempered,
                                      plain.result_.group_assignments)
        np.testing.assert_array_equal(one.predict(X), plain.predict(X))
        np.testing.assert_allclose(one.predict_proba(X), one.result_.group_assignments,
                                   atol=1e-10)

    def test_does_not_drift(self):
        """The first step ignores later re-weighting: it can differ from "auto"."""
        rng = np.random.default_rng(0)
        # heavy-tailed log-likelihood ratios: the fixed point drifts below step 1
        n = 400
        lab = rng.integers(0, 2, n)
        d = rng.standard_t(1.5, n) * 3.0 + 6.0
        ll = np.column_stack([np.where(lab == 0, d, -d), np.zeros(n)])
        w = np.array([0.5, 0.5])
        one = estimate_temperature(ll, w, max_iter=1)
        auto = estimate_temperature(ll, w)
        assert one >= auto

    def test_clone_keeps_setting(self):
        assert clone(DirichletMixture(temperature="one-step")).temperature == "one-step"


class TestHelpers:
    def test_component_log_likelihood_matches_core(self, data):
        X, _ = data
        m = DirichletMixture(n_components=2, random_state=1).fit(X)
        alpha = m.result_.parameter_estimates["Estimate"]
        ll = component_log_likelihood(X, alpha.T)
        for k in range(2):
            ref = m._compute_dirichlet_multinomial_log_likelihood(X, alpha[:, k])
            np.testing.assert_allclose(ll[:, k], ref, rtol=1e-12, atol=1e-8)

    def test_tempered_responsibilities_softmax(self):
        ll = np.array([[0.0, -1.0], [-3.0, 0.0]])
        w = np.array([0.25, 0.75])
        z = tempered_responsibilities(ll, w, 1.0)
        expected = w * np.exp(ll)
        expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(z, expected)
        z = tempered_responsibilities(ll, w, 0.5)
        expected = np.sqrt(w * np.exp(ll))
        expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(z, expected)

    @pytest.mark.parametrize("bad", [0, 0.0, -0.5, 1.5, "fast", "one_step", "Auto",
                                     True, [0.5]])
    def test_invalid_temperature(self, bad, data):
        with pytest.raises(ValueError):
            validate_temperature(bad)
        X, _ = data
        with pytest.raises(ValueError):
            DirichletMixture(temperature=bad).fit(X)

    @pytest.mark.parametrize("good", [None, "auto", "one-step", 1, 1.0, 0.25,
                                      np.float64(0.5)])
    def test_valid_temperature(self, good):
        validate_temperature(good)
