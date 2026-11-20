"""
Tests for the Dirichlet-tree multinomial mixture (DirichletTreeMixture) and Tree.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln, logsumexp
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score

from pydmm import DirichletMixture, DirichletTreeMixture, Tree, pydmm_core
from pydmm.dtmm import tree_component_log_likelihood

GAMMA_ITA = 0.1
GAMMA_NU = 0.1


def _clade_data(seed, n=160, n_clades=4, clade_size=4, depth=3000, theta_root=40.0):
    """
    Dirichlet-tree data: root over clades, clades over taxa. The two groups
    differ only in their clade totals; within-clade composition is shared.
    """
    rng = np.random.default_rng(seed)
    p0 = np.exp(rng.normal(0, 0.5, n_clades))
    shift = np.zeros(n_clades)
    shift[:2] = [1.0, -1.0]
    roots = [p0 / p0.sum(), np.exp(np.log(p0) + shift) / np.exp(np.log(p0) + shift).sum()]
    within = [rng.gamma(2.0, 1.0, clade_size) * 10 for _ in range(n_clades)]
    labels = rng.integers(0, 2, n)
    X = np.zeros((n, n_clades * clade_size), dtype=np.int32)
    for i in range(n):
        top = rng.dirichlet(roots[labels[i]] * theta_root)
        p = np.concatenate([top[c] * rng.dirichlet(within[c]) for c in range(n_clades)])
        X[i] = rng.multinomial(depth, p / p.sum())
    groups = [list(range(c * clade_size, (c + 1) * clade_size)) for c in range(n_clades)]
    return X, labels, groups


def _clade_tree(groups, n_leaves, leaf_names=None):
    parents = np.full(n_leaves + len(groups) + 1, n_leaves + len(groups))
    parents[-1] = -1
    for c, g in enumerate(groups):
        parents[g] = n_leaves + c
    return Tree.from_parents(parents, n_leaves, leaf_names)


def _accuracy(z, labels):
    pred = z.argmax(1)
    return max(np.mean(pred == labels), np.mean(pred != labels))


@pytest.fixture(scope="module")
def clade():
    X, labels, groups = _clade_data(0)
    return X, labels, _clade_tree(groups, X.shape[1])


class TestTree:
    def test_star(self):
        t = Tree.star(5)
        assert (t.n_leaves, t.n_nodes, t.n_edges) == (5, 1, 5)
        assert t.leaves(0) == [0, 1, 2, 3, 4]

    def test_from_parents_collapses_unary_nodes(self):
        # leaves 0..3; node 4 = {0, 1}; node 5 has only node 4; root 6 = {5, 2, 3}
        t = Tree.from_parents([4, 4, 6, 6, 5, 6, -1], 4)
        assert t.n_nodes == 2
        assert sorted(len(t.children(u)) for u in range(t.n_nodes)) == [2, 3]
        assert t.leaves(t.n_nodes - 1) == [0, 1, 2, 3]

    def test_post_order(self):
        t = Tree.from_parents([4, 4, 5, 5, 6, 6, -1], 4)
        for u in range(t.n_nodes):
            assert all(c < t.n_leaves + u for c in t.children(u))

    def test_from_linkage(self):
        from scipy.cluster.hierarchy import linkage
        Z = linkage(np.random.default_rng(0).normal(size=(6, 3)))
        t = Tree.from_linkage(Z)
        assert (t.n_leaves, t.n_nodes, t.n_edges) == (6, 5, 10)

    def test_from_taxonomy(self):
        tax = pd.DataFrame({
            "Family": ["A", "A", "A", "B", "B", None],
            "Genus": ["x", "x", "y", "x", np.nan, None],
        }, index=[f"f{j}" for j in range(6)])
        t = Tree.from_taxonomy(tax)
        assert t.leaf_names == [f"f{j}" for j in range(6)]
        # Genus x under A and under B are different nodes; A:y and B:x have
        # a single feature and collapse into their parent
        assert "Genus:x" in t.node_names
        assert sorted(t.node_names) == ["Family:A", "Family:B", "Genus:x", "root"]
        root = t.n_nodes - 1
        assert t.leaves(root) == list(range(6))
        assert t.leaves(t.node_names.index("Family:B")) == [3, 4]

    def test_from_correlation_is_binary(self):
        X = np.random.default_rng(1).poisson(20, size=(30, 7))
        t = Tree.from_correlation(X)
        assert t.n_nodes == 6 and t.n_edges == 12

    def test_node_counts(self, clade):
        X, _, tree = clade
        counts = tree.node_counts(X)
        root = counts[-1]
        np.testing.assert_array_equal(root.sum(axis=1), X.sum(axis=1))
        assert sum(C.shape[1] for C in counts) == tree.n_edges

    def test_reorder(self):
        t = Tree.from_parents([4, 4, 5, 5, 6, 6, -1], 4, leaf_names=["a", "b", "c", "d"])
        r = t.reorder(["d", "c", "b", "a"])
        assert r.leaf_names == ["d", "c", "b", "a"]
        assert sorted(r.label(c) for c in r.children(0)) == ["a", "b"]
        with pytest.raises(ValueError):
            t.reorder(["a", "b", "c", "e"])

    def test_edge_index(self, clade):
        _, _, tree = clade
        idx = tree.edge_index()
        assert len(idx) == tree.n_edges and idx.names == ["parent", "child"]

    @pytest.mark.parametrize("parents,n", [
        ([3, 3, -1, -1], 2),        # two roots
        ([2, 2, -1], 3),            # root is a leaf
        ([3, 3, 3, 7, -1], 3),      # parent id out of range
    ])
    def test_invalid_parents(self, parents, n):
        with pytest.raises(ValueError):
            Tree.from_parents(parents, n)

    def test_missing_leaf(self):
        with pytest.raises(ValueError):
            Tree.from_parents([3, 3, -1, -1], 3)

    def test_invalid_csr(self):
        with pytest.raises(ValueError):
            Tree([0, 2], [0, 0], 2)  # leaf used twice


class TestCExtension:
    def test_output_shapes(self, clade):
        X, _, tree = clade
        r = pydmm_core.dirichlet_tree_fit(X, tree.child_start, tree.child, n_components=3, seed=1)
        assert r["Group"].shape == (X.shape[0], 3)
        assert r["Fit"]["Estimate"].shape == (tree.n_edges, 3)
        assert r["NodeShared"].shape == (tree.n_nodes,)
        assert r["Mixture"]["Weight"].shape == (3,)
        np.testing.assert_allclose(r["Group"].sum(axis=1), 1.0)
        assert set(r["GoodnessOfFit"]) == {"NLE", "LogDet", "Laplace", "BIC", "AIC"}

    def test_star_tree_equals_dmm(self, clade):
        X = clade[0]
        dmm = DirichletMixture(n_components=2, random_state=3).fit(X)
        star = Tree.star(X.shape[1])
        r = pydmm_core.dirichlet_tree_fit(X, star.child_start, star.child, n_components=2,
                                          seed=3, node_selection=False)
        for key in ("NLE", "BIC", "AIC"):
            assert r["GoodnessOfFit"][key] == pytest.approx(dmm.result_.goodness_of_fit[key], rel=1e-8)
        np.testing.assert_allclose(r["Group"], dmm.result_.group_assignments, atol=1e-4)
        np.testing.assert_allclose(r["Fit"]["Estimate"], dmm.result_.parameter_estimates["Estimate"],
                                   rtol=1e-4)

    @pytest.mark.parametrize("selection", [False, True])
    def test_nle_matches_python_likelihood(self, clade, selection):
        """NLE from C equals the likelihood + prior recomputed in Python at the fitted values."""
        X, _, tree = clade
        r = pydmm_core.dirichlet_tree_fit(X, tree.child_start, tree.child, n_components=2,
                                          seed=0, node_selection=selection)
        alpha, w = r["Fit"]["Estimate"], r["Mixture"]["Weight"]
        shared = r["NodeShared"].astype(bool)
        ll = tree_component_log_likelihood(X, tree, alpha)
        log_multinom = gammaln(X.sum(1) + 1.0) - gammaln(X + 1.0).sum(1)
        loglik = logsumexp(ll + np.log(w), axis=1).sum() + log_multinom.sum()
        lam = [np.log(alpha[tree.child_start[u]:tree.child_start[u + 1], k])
               for k in range(2) for u in range(tree.n_nodes) if k == 0 or not shared[u]]
        lam = np.concatenate(lam)
        prior = (lam.size * (GAMMA_ITA * np.log(GAMMA_NU) - gammaln(GAMMA_ITA))
                 - GAMMA_NU * np.exp(lam).sum() + GAMMA_ITA * lam.sum())
        assert r["GoodnessOfFit"]["NLE"] == pytest.approx(-(loglik + prior), rel=1e-9)

    def test_shared_nodes_have_equal_parameters(self, clade):
        X, _, tree = clade
        r = pydmm_core.dirichlet_tree_fit(X, tree.child_start, tree.child, n_components=2, seed=0)
        alpha = r["Fit"]["Estimate"]
        for u in np.flatnonzero(r["NodeShared"]):
            block = alpha[tree.child_start[u]:tree.child_start[u + 1]]
            np.testing.assert_array_equal(block[:, 0], block[:, 1])

    @pytest.mark.parametrize("start,child", [
        ([0, 2], [0, 0]),            # leaf used twice
        ([0, 1], [0]),               # node with one child
        ([0, 3], [0, 1, -1]),        # negative child id
        ([0, 2], [0, 1]),            # leaf 2 has no parent
        ([0, 2, 3], [0, 4, 2]),      # child refers to a later node
    ])
    def test_invalid_tree_rejected(self, start, child):
        X = np.ones((5, 3), dtype=np.int32)
        with pytest.raises(ValueError):
            pydmm_core.dirichlet_tree_fit(X, np.array(start, np.int32), np.array(child, np.int32))

    def test_requires_int32(self, clade):
        X, _, tree = clade
        with pytest.raises(ValueError):
            pydmm_core.dirichlet_tree_fit(X.astype(np.int64), tree.child_start, tree.child)


class TestNodeSelection:
    def test_selects_the_differential_node(self, clade):
        """Groups differ only in clade totals: only the root should stay component-specific."""
        X, labels, tree = clade
        m = DirichletTreeMixture(n_components=2, tree=tree, random_state=0).fit(X)
        res = m.result_
        assert list(np.flatnonzero(~res.node_shared)) == [tree.n_nodes - 1]
        assert res.n_select_rounds >= 1
        assert _accuracy(res.group_assignments, labels) > 0.9

    def test_selection_off_keeps_all_nodes(self, clade):
        X, _, tree = clade
        m = DirichletTreeMixture(n_components=2, tree=tree, node_selection=False).fit(X)
        assert not m.result_.node_shared.any()
        assert m.result_.n_select_rounds == 0

    def test_selection_lowers_bic(self, clade):
        X, _, tree = clade
        on = DirichletTreeMixture(n_components=2, tree=tree).fit(X)
        off = DirichletTreeMixture(n_components=2, tree=tree, node_selection=False).fit(X)
        assert on.result_.goodness_of_fit["BIC"] < off.result_.goodness_of_fit["BIC"]

    def test_single_component(self, clade):
        X, _, tree = clade
        m = DirichletTreeMixture(n_components=1, tree=tree).fit(X)
        np.testing.assert_allclose(m.result_.group_assignments, 1.0)
        assert m.result_.n_select_rounds == 0

    def test_differential_nodes_table(self, clade):
        X, _, tree = clade
        res = DirichletTreeMixture(n_components=2, tree=tree).fit(X).result_
        table = res.differential_nodes()
        assert list(table.columns) == ["node", "n_features", "features"]
        assert len(table) == int((~res.node_shared).sum())
        assert res.summary()["n_differential_nodes"] == len(table)


class TestCollapsedStart:
    def test_restarts_from_hard_kmeans(self, capfd):
        """
        Many low-noise samples with nearly identical compositions make the soft
        k-means start give every sample the same assignments; EM then stays at
        the symmetric saddle point unless the fit restarts from hard k-means.
        """
        rng = np.random.default_rng(0)
        n, s = 150, 100
        p = rng.dirichlet(np.ones(s) * 5)
        labels = rng.integers(0, 2, n)
        X = np.zeros((n, s), np.int32)
        for i in range(n):
            q = p.copy()
            if labels[i]:
                q[:5] *= 1.5
            X[i] = rng.multinomial(20000, rng.dirichlet(q / q.sum() * 2000))
        parents = np.r_[np.repeat(np.arange(s, s + s // 5), 5), np.full(s // 5, s + s // 5), -1]
        tree = Tree.from_parents(parents, s)
        m = DirichletTreeMixture(n_components=2, tree=tree, verbose=True).fit(X)
        assert "Components identical" in capfd.readouterr().out
        z = m.result_.group_assignments
        assert np.abs(z - m.result_.mixture_weights).max() > 0.1
        assert _accuracy(z, labels) > 0.9


class TestDirichletTreeMixture:
    def test_beats_dmm_calibration_on_clade_data(self, clade):
        """On tree-structured data the DTMM's probabilities are better calibrated."""
        X, labels, tree = clade

        def log_loss(z):
            p = np.clip(z, 1e-12, 1)
            return min(-np.log(p[np.arange(len(labels)), labels]).mean(),
                       -np.log(p[np.arange(len(labels)), 1 - labels]).mean())

        dmm = DirichletMixture(n_components=2, random_state=0).fit(X)
        dtmm = DirichletTreeMixture(n_components=2, tree=tree, random_state=0).fit(X)
        assert log_loss(dtmm.result_.group_assignments) < log_loss(dmm.result_.group_assignments)

    def test_fit_predict_shapes(self, clade):
        X, _, tree = clade
        m = DirichletTreeMixture(n_components=2, tree=tree)
        pred = m.fit_predict(X)
        assert pred.shape == (X.shape[0],)
        proba = m.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        np.testing.assert_allclose(proba, m.result_.group_assignments, atol=1e-4)
        np.testing.assert_array_equal(m.predict(X), proba.argmax(axis=1))

    def test_default_tree_from_data(self, clade):
        X = clade[0]
        m = DirichletTreeMixture(n_components=2).fit(X)
        assert m.tree is None
        assert m.tree_.n_nodes == X.shape[1] - 1  # binary tree
        assert m.result_.parameter_estimates["Estimate"].shape == (m.tree_.n_edges, 2)

    def test_dataframe_columns_matched_by_name(self, clade):
        X, _, groups = _clade_data(0)
        names = [f"taxon{j}" for j in range(X.shape[1])]
        tree = _clade_tree(groups, X.shape[1], leaf_names=names)
        df = pd.DataFrame(X, columns=names)
        shuffled = df.sample(frac=1.0, axis=1, random_state=1)
        a = DirichletTreeMixture(n_components=2, tree=tree, random_state=0).fit(df)
        b = DirichletTreeMixture(n_components=2, tree=tree, random_state=0).fit(shuffled)
        assert a.result_.goodness_of_fit["NLE"] == pytest.approx(b.result_.goodness_of_fit["NLE"],
                                                                 rel=1e-6)
        assert b.tree_.leaf_names == list(shuffled.columns)
        np.testing.assert_allclose(b.predict_proba(df), b.predict_proba(shuffled), atol=1e-12)
        with pytest.raises(ValueError):
            b.predict_proba(df.drop(columns=names[0]))

    def test_tree_size_mismatch(self, clade):
        X, _, tree = clade
        with pytest.raises(ValueError):
            DirichletTreeMixture(tree=tree).fit(X[:, :-1])

    def test_invalid_tree_type(self, clade):
        with pytest.raises(ValueError):
            DirichletTreeMixture(tree="taxonomy").fit(clade[0])

    def test_predict_before_fit(self, clade):
        with pytest.raises(ValueError):
            DirichletTreeMixture().predict_proba(clade[0])

    def test_temperature(self, clade):
        X, _, tree = clade
        plain = DirichletTreeMixture(n_components=2, tree=tree).fit(X)
        temp = DirichletTreeMixture(n_components=2, tree=tree, temperature=0.5).fit(X)
        assert plain.temperature_ is None
        assert temp.temperature_ == 0.5
        np.testing.assert_array_equal(temp.result_.group_assignments_untempered,
                                      plain.result_.group_assignments)
        np.testing.assert_allclose(temp.predict_proba(X), temp.result_.group_assignments, atol=1e-10)
        np.testing.assert_array_equal(temp.predict(X), plain.predict(X))
        auto = DirichletTreeMixture(n_components=2, tree=tree, temperature="auto").fit(X)
        assert 0.0 < auto.temperature_ <= 1.0
        with pytest.raises(ValueError):
            DirichletTreeMixture(temperature=2.0).fit(X)

    def test_parameter_estimates_df(self, clade):
        X, _, tree = clade
        res = DirichletTreeMixture(n_components=2, tree=tree).fit(X).result_
        dfs = res.get_parameter_estimates_df()
        assert set(dfs) == {"Lower", "Estimate", "Upper"}
        assert dfs["Estimate"].shape == (tree.n_edges, 2)
        assert list(dfs["Estimate"].index.names) == ["parent", "child"]

    def test_component_labels(self, clade):
        X, _, tree = clade
        m = DirichletTreeMixture(n_components=2, tree=tree).fit(X)
        m.result_.set_component_labels({0: "A", 1: "B"})
        assert list(m.predict_proba(X).columns) == ["A component", "B component"]
        assert set(m.predict(X)) <= {"A", "B"}

    def test_score_is_negative_nle(self, clade):
        X, _, tree = clade
        m = DirichletTreeMixture(n_components=2, tree=tree).fit(X)
        assert m.score(X) == -m.result_.goodness_of_fit["NLE"]

    def test_reproducible(self, clade):
        X, _, tree = clade
        a = DirichletTreeMixture(n_components=2, tree=tree, random_state=5).fit(X)
        b = DirichletTreeMixture(n_components=2, tree=tree, random_state=5).fit(X)
        np.testing.assert_array_equal(a.result_.group_assignments, b.result_.group_assignments)


class TestSklearnCompatibility:
    def test_get_params(self):
        params = DirichletTreeMixture().get_params()
        assert set(params) == {"n_components", "tree", "node_selection", "verbose",
                               "random_state", "temperature"}

    def test_set_params_and_clone(self, clade):
        _, _, tree = clade
        m = DirichletTreeMixture(tree=tree).set_params(n_components=3, node_selection=False)
        c = clone(m)
        assert c.n_components == 3 and c.node_selection is False
        assert isinstance(c.tree, Tree) and c.tree.n_edges == tree.n_edges

    def test_clone_unfitted(self, clade):
        X, _, tree = clade
        m = DirichletTreeMixture(tree=tree).fit(X)
        assert not hasattr(clone(m), "result_") or clone(m).result_ is None

    def test_classes_and_feature_names(self, clade):
        X, _, tree = clade
        names = [f"t{j}" for j in range(X.shape[1])]
        m = DirichletTreeMixture(n_components=2, tree=tree).fit(pd.DataFrame(X, columns=names))
        np.testing.assert_array_equal(m.classes_, [0, 1])
        assert list(m.feature_names_in_) == names
        m.fit(X)
        assert not hasattr(m, "feature_names_in_")

    def test_cross_val_score(self, clade):
        X, _, tree = clade
        scores = cross_val_score(DirichletTreeMixture(n_components=2, tree=tree), X, cv=2)
        assert len(scores) == 2

    def test_grid_search(self, clade):
        X, _, tree = clade
        grid = GridSearchCV(DirichletTreeMixture(tree=tree), {"n_components": [1, 2]}, cv=2)
        grid.fit(X)
        assert grid.best_params_["n_components"] in (1, 2)
