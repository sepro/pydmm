"""
Tests for WAIC support: Hessian export from the C core, pointwise
log-likelihood, WAIC arithmetic and the DirichletMixture integration.

Reference values in tests/data/waic/ were computed with PyMC and arviz by
validation/waic/make_fixtures.py, so these tests do not need either package.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.special import gammaln, logsumexp
from sklearn.base import clone

from pydmm import DirichletMixture, pydmm_core
from pydmm.waic import (
    compute_waic,
    dirichlet_mixture_log_likelihood,
    observed_information,
    waic_from_log_likelihood,
)

FIXTURES = Path(__file__).parent / "data" / "waic"


def _load(name):
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"missing fixture {path}; run validation/waic/make_fixtures.py")
    return np.load(path)


@pytest.fixture
def three_cluster_data():
    rng = np.random.default_rng(0)
    protos = [np.array([8, 1, 1, 1, 1, 1.]), np.array([1, 1, 8, 4, 1, 1.]),
              np.array([1, 4, 1, 1, 6, 6.])]
    X = np.vstack([rng.multinomial(2000, rng.dirichlet(protos[i % 3] * 2))
                   for i in range(150)])
    return X.astype(np.int32)


class TestHessianExport:
    def test_not_returned_by_default(self, three_cluster_data):
        res = pydmm_core.dirichlet_fit(three_cluster_data, n_components=3, seed=1)
        assert "Hessian" not in res

    def test_shape_symmetric_positive_definite(self, three_cluster_data):
        res = pydmm_core.dirichlet_fit(three_cluster_data, n_components=3, seed=1,
                                       return_hessian=True)
        H = res["Hessian"]
        assert H.shape == (3, 6, 6)
        for k in range(3):
            np.testing.assert_allclose(H[k], H[k].T, rtol=1e-12, atol=1e-10)
            assert np.all(np.linalg.eigvalsh(H[k]) > 0)

    def test_matches_numerical_hessian(self, three_cluster_data):
        """Regression test for the operator-precedence bug in hessian()."""
        X = three_cluster_data
        res = pydmm_core.dirichlet_fit(X, n_components=3, seed=1, return_hessian=True)
        eta = nu = 0.1  # GAMMA_ITA, GAMMA_NU in the C code

        def objective(lam, pi):
            a = np.exp(lam)
            t = -(pi[:, None] * gammaln(a + X)).sum() + (pi * gammaln((a + X).sum(1))).sum()
            return (t + pi.sum() * (gammaln(a).sum() - gammaln(a.sum()))
                    + nu * a.sum() - eta * lam.sum())

        h = 1e-4
        for k in range(3):
            lam = np.log(res["Fit"]["Estimate"][:, k])
            pi = res["Group"][:, k]
            n = lam.size
            num = np.zeros((n, n))
            E = np.eye(n) * h
            for i in range(n):
                for j in range(n):
                    num[i, j] = (objective(lam + E[i] + E[j], pi)
                                 - objective(lam + E[i] - E[j], pi)
                                 - objective(lam - E[i] + E[j], pi)
                                 + objective(lam - E[i] - E[j], pi)) / (4 * h * h)
            np.testing.assert_allclose(res["Hessian"][k], num,
                                       rtol=1e-3, atol=1e-3 * np.abs(num).max())

    def test_confidence_intervals_bracket_estimate(self, three_cluster_data):
        res = pydmm_core.dirichlet_fit(three_cluster_data, n_components=3, seed=1)
        lo, est, up = (res["Fit"][k] for k in ("Lower", "Estimate", "Upper"))
        assert np.all(np.isfinite(lo)) and np.all(np.isfinite(up))
        assert np.all(lo < est) and np.all(est < up)

    def test_fit_unchanged_by_hessian_flag(self, three_cluster_data):
        a = pydmm_core.dirichlet_fit(three_cluster_data, n_components=3, seed=1)
        b = pydmm_core.dirichlet_fit(three_cluster_data, n_components=3, seed=1,
                                     return_hessian=True)
        assert a["GoodnessOfFit"] == b["GoodnessOfFit"]
        np.testing.assert_array_equal(a["Fit"]["Estimate"], b["Fit"]["Estimate"])
        np.testing.assert_array_equal(a["Group"], b["Group"])


class TestObservedInformation:
    @pytest.mark.parametrize("K", [2, 3])
    def test_matches_numerical_hessian(self, K):
        """Louis' identity result equals the finite-difference Hessian of the
        full negative log posterior in (log alpha, weight logits)."""
        rng = np.random.default_rng(3)
        protos = [np.array([6, 2, 1, 1.]), np.array([1, 1, 4, 6.]), np.array([2, 5, 2, 1.])]
        X = np.vstack([rng.multinomial(rng.integers(200, 800), rng.dirichlet(protos[i % 3] * 3))
                       for i in range(90)]).astype(np.int32)
        S = X.shape[1]
        res = pydmm_core.dirichlet_fit(X, n_components=K, seed=2, return_hessian=True)
        alpha = res["Fit"]["Estimate"].T
        w = res["Mixture"]["Weight"]
        eta_p = nu = 0.1
        n = X.sum(1)

        def neg_log_post(theta):
            lam = theta[:K * S].reshape(K, S)
            a = np.exp(lam)
            e = np.append(theta[K * S:], 0.0)
            log_w = e - logsumexp(e)
            A = a.sum(1)
            log_f = (gammaln(A)[:, None] - gammaln(A[:, None] + n[None])
                     + (gammaln(a[:, None, :] + X[None]) - gammaln(a)[:, None, :]).sum(2))
            return (-logsumexp(log_f + log_w[:, None], axis=0).sum()
                    - (eta_p * lam - nu * a).sum())

        theta = np.concatenate([np.log(alpha).ravel(), np.log(w[:-1]) - np.log(w[-1])])
        P, h = theta.size, 1e-4
        E = np.eye(P) * h
        num = np.zeros((P, P))
        for i in range(P):
            for j in range(i, P):
                num[i, j] = num[j, i] = (
                    neg_log_post(theta + E[i] + E[j]) - neg_log_post(theta + E[i] - E[j])
                    - neg_log_post(theta - E[i] + E[j]) + neg_log_post(theta - E[i] - E[j])
                ) / (4 * h * h)
        info = observed_information(X, alpha, w, res["Hessian"])
        np.testing.assert_allclose(info, num, rtol=0, atol=1e-3 * np.abs(num).max())

    def test_overfitted_model_gives_finite_waic(self, three_cluster_data):
        """Saddle/flat directions of overfitted fits are regularised."""
        dmm = DirichletMixture(8, compute_waic=True, n_waic_draws=300).fit(three_cluster_data)
        gof = dmm.result.goodness_of_fit
        assert np.isfinite(gof["WAIC"])
        assert gof["WAIC_n_regularised"] >= 0
        assert np.all(np.isfinite(dmm.result.pointwise_log_likelihood))

    def test_full_penalises_redundant_components_more_than_block(self, three_cluster_data):
        X = three_cluster_data
        res = pydmm_core.dirichlet_fit(X, n_components=5, seed=42, return_hessian=True)
        args = (X, res["Fit"]["Estimate"], res["Mixture"]["Weight"], res["Hessian"], 300, 0)
        full = compute_waic(*args, method="full")
        block = compute_waic(*args, method="block")
        assert full["p_waic"] > block["p_waic"]
        with pytest.raises(ValueError, match="unknown method"):
            compute_waic(*args, method="nope")


class TestPointwiseLogLikelihood:
    def test_matches_pymc_equal_depth(self):
        ref = _load("logp_reference.npz")
        ll = dirichlet_mixture_log_likelihood(ref["X_eq"], ref["w"], ref["alpha"])
        np.testing.assert_allclose(ll, ref["logp_eq"], rtol=1e-10, atol=1e-8)

    def test_matches_pymc_variable_depth(self):
        ref = _load("logp_reference.npz")
        ll = dirichlet_mixture_log_likelihood(ref["X_var"], ref["w"], ref["alpha"])
        np.testing.assert_allclose(ll, ref["logp_var"], rtol=1e-10, atol=1e-8)

    def test_batch_equals_single(self, three_cluster_data):
        rng = np.random.default_rng(1)
        w = rng.dirichlet(np.ones(3), size=4)
        alpha = rng.uniform(0.5, 20, size=(4, 3, 6))
        batch = dirichlet_mixture_log_likelihood(three_cluster_data, w, alpha)
        assert batch.shape == (4, three_cluster_data.shape[0])
        for d in range(4):
            single = dirichlet_mixture_log_likelihood(three_cluster_data, w[d], alpha[d])
            np.testing.assert_allclose(batch[d], single, rtol=1e-12)

    def test_zero_count_sample(self):
        X = np.array([[0, 0, 0], [3, 0, 1]], dtype=np.int32)
        ll = dirichlet_mixture_log_likelihood(X, np.array([1.0]), np.array([[1., 2., 3.]]))
        assert ll[0] == pytest.approx(0.0)
        assert np.isfinite(ll[1])


class TestWaicArithmetic:
    def test_matches_arviz(self):
        ref = _load("arith_reference.npz")
        res = waic_from_log_likelihood(ref["log_lik"])
        assert res["elpd_waic"] == pytest.approx(float(ref["elpd_waic"]), rel=1e-12)
        assert res["p_waic"] == pytest.approx(float(ref["p_waic"]), rel=1e-12)
        assert res["se"] == pytest.approx(float(ref["se"]), rel=1e-10)
        np.testing.assert_allclose(res["elpd_waic_i"], ref["elpd_waic_i"], rtol=1e-12)
        assert res["WAIC"] == -res["elpd_waic"]

    def test_prodiphy_draws_reproduce_prodiphy_waic(self):
        """prodiphy's posterior draws through pyDMM's code give prodiphy's WAIC."""
        ref = _load("swap_reference.npz")
        ll = dirichlet_mixture_log_likelihood(ref["X"], ref["w"], ref["alpha"])
        np.testing.assert_allclose(ll[:, :10], ref["log_lik_first10"], rtol=1e-10)
        res = waic_from_log_likelihood(ll)
        assert res["elpd_waic"] == pytest.approx(float(ref["elpd_waic"]), rel=1e-10)
        assert res["p_waic"] == pytest.approx(float(ref["p_waic"]), rel=1e-8)
        assert res["se"] == pytest.approx(float(ref["se"]), rel=1e-8)


class TestDirichletMixtureWaic:
    def test_off_by_default(self, simple_2cluster_data):
        dmm = DirichletMixture(n_components=2).fit(simple_2cluster_data["data"])
        assert "WAIC" not in dmm.result.goodness_of_fit
        assert dmm.result.pointwise_log_likelihood is None
        assert "WAIC" not in dmm.result.summary()

    def test_adds_metrics(self, simple_2cluster_data):
        X = simple_2cluster_data["data"]
        dmm = DirichletMixture(n_components=2, compute_waic=True, n_waic_draws=200).fit(X)
        gof = dmm.result.goodness_of_fit
        for key in ("WAIC", "elpd_waic", "p_waic", "WAIC_se", "WAIC_n_regularised"):
            assert key in gof and np.isfinite(gof[key])
        assert gof["WAIC"] == pytest.approx(-gof["elpd_waic"])
        assert gof["p_waic"] > 0
        assert dmm.result.pointwise_log_likelihood.shape == (200, X.shape[0])
        assert dmm.result.summary()["WAIC"] == gof["WAIC"]

    def test_other_outputs_unchanged(self, simple_2cluster_data):
        X = simple_2cluster_data["data"]
        a = DirichletMixture(n_components=2).fit(X).result
        b = DirichletMixture(n_components=2, compute_waic=True, n_waic_draws=50).fit(X).result
        for key in ("NLE", "LogDet", "Laplace", "BIC", "AIC"):
            assert a.goodness_of_fit[key] == b.goodness_of_fit[key]
        np.testing.assert_array_equal(a.group_assignments, b.group_assignments)

    def test_reproducible(self, simple_2cluster_data):
        X = simple_2cluster_data["data"]
        w1 = DirichletMixture(2, compute_waic=True, n_waic_draws=100).fit(X)
        w2 = DirichletMixture(2, compute_waic=True, n_waic_draws=100).fit(X)
        assert w1.result.goodness_of_fit["WAIC"] == w2.result.goodness_of_fit["WAIC"]
        w3 = DirichletMixture(2, compute_waic=True, n_waic_draws=100, random_state=7).fit(X)
        assert w3.result.goodness_of_fit["WAIC"] != w1.result.goodness_of_fit["WAIC"]

    def test_prefers_true_k_over_one(self, simple_2cluster_data):
        X = simple_2cluster_data["data"]
        waic = [DirichletMixture(k, compute_waic=True, n_waic_draws=200).fit(X)
                .result.goodness_of_fit["WAIC"] for k in (1, 2)]
        assert waic[1] < waic[0]

    def test_waic_matches_recomputation_from_pointwise(self, three_cluster_data):
        dmm = DirichletMixture(3, compute_waic=True, n_waic_draws=300).fit(three_cluster_data)
        res = waic_from_log_likelihood(dmm.result.pointwise_log_likelihood)
        assert res["WAIC"] == pytest.approx(dmm.result.goodness_of_fit["WAIC"])

    @pytest.mark.parametrize("bad", [0, 1, 2.5, "100"])
    def test_invalid_draws(self, simple_2cluster_data, bad):
        with pytest.raises(ValueError, match="n_waic_draws"):
            DirichletMixture(2, compute_waic=True, n_waic_draws=bad).fit(
                simple_2cluster_data["data"])

    def test_sklearn_params(self):
        dmm = DirichletMixture(3, compute_waic=True, n_waic_draws=500)
        params = dmm.get_params()
        assert params["compute_waic"] is True and params["n_waic_draws"] == 500
        cloned = clone(dmm)
        assert cloned.compute_waic is True and cloned.n_waic_draws == 500
