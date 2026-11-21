"""
Regression tests for bugs 1-4 that were listed in BUGS.md (file removed when
all bugs were closed; see its git history).
"""

import shutil
import subprocess
import sysconfig
from pathlib import Path

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


def _build_c_harness(tmp_path, harness, inject_hessian=False):
    """Compile tests/c/<harness> with the DMM C sources; skip if that fails.

    inject_hessian renames the real hessian() to hessian_orig, so the harness
    can define its own hessian().
    """
    root = Path(__file__).resolve().parent.parent
    src = root / "src" / "pydmm"
    cc = (sysconfig.get_config_var("CC") or "cc").split()[0]
    if shutil.which(cc) is None:
        pytest.skip("no C compiler")
    node_o = tmp_path / "node.o"
    exe = tmp_path / Path(harness).stem
    rename = ["-Dhessian=hessian_orig"] if inject_hessian else []
    build = [
        [cc, "-O2", f"-I{src}", *rename, "-c", str(src / "dirichlet_node.c"),
         "-o", str(node_o)],
        [cc, "-O2", f"-I{src}", str(root / "tests" / "c" / harness),
         str(src / "dirichlet_fit_standalone.c"), str(node_o), "-o", str(exe),
         "-lgsl", "-lgslcblas", "-lm"],
    ]
    for cmd in build:
        if subprocess.run(cmd, capture_output=True).returncode != 0:
            pytest.skip("cannot compile against GSL")
    return exe


def _run_c_harness(exe):
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    return {key: float(v) for key, v in
            (line.split() for line in run.stdout.splitlines())}


class TestSingularHessian:
    """Bug 2: a singular component Hessian called exit(1) and killed Python.

    No data set is known that makes the Hessian singular, so this test
    compiles tests/c/singular_hessian.c, which replaces the Hessian of one
    component with a singular matrix (fault injection).
    """

    def test_returns_nan_instead_of_exit(self, tmp_path):
        exe = _build_c_harness(tmp_path, "singular_hessian.c", inject_hessian=True)
        values = _run_c_harness(exe)
        assert np.isfinite(values["NLE"]) and np.isfinite(values["BIC"])
        assert np.isnan(values["LogDet"]) and np.isnan(values["Laplace"])
        # only the singular component gets NaN limits
        assert np.isfinite(values["Lower0"]) and np.isfinite(values["Upper0"])
        assert np.isnan(values["Lower1"]) and np.isnan(values["Upper1"])


class TestUninitialisedStart:
    """Bug 3: the soft k-means start read aadLambda before it was set."""

    def test_fit_does_not_depend_on_heap_contents(self, tmp_path):
        exe = _build_c_harness(tmp_path, "uninitialised_start.c")
        values = _run_c_harness(exe)
        assert values["NLE0"] == values["NLE1"]
        assert values["SameGroup"] == 1


class TestMemoryLeaks:
    """Bug 4: each DirichletMixture fit kept memory (RNG, C buffers, results)."""

    def test_result_arrays_are_released(self):
        import gc
        import weakref

        X = np.random.default_rng(0).poisson(20, (50, 5)).astype(np.int32)
        r = pydmm_core.dirichlet_fit(X, n_components=2, return_hessian=True)
        refs = [weakref.ref(r["Group"]), weakref.ref(r["Mixture"]["Weight"]),
                weakref.ref(r["Hessian"])]
        refs += [weakref.ref(v) for v in r["Fit"].values()]
        del r
        gc.collect()
        assert all(ref() is None for ref in refs)

    @pytest.mark.skipif(not Path("/proc/self/statm").exists(), reason="needs /proc")
    def test_repeated_fits_do_not_grow_memory(self):
        def rss_kb():
            with open("/proc/self/statm") as f:
                return int(f.read().split()[1]) * 4

        X = np.random.default_rng(0).poisson(20, (6, 3)).astype(np.int32)
        for _ in range(300):
            pydmm_core.dirichlet_fit(X, n_components=1, seed=1)
        before = rss_kb()
        for _ in range(3000):
            pydmm_core.dirichlet_fit(X, n_components=1, seed=1)
        # before the fix: about 7 kB per fit (21 MB here)
        assert rss_kb() - before < 3000
