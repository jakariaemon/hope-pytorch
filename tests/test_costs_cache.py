"""Phase 4 gates: cached scalars equal direct synthesis to 1e-9 (Test G); incremental merge updates match a fresh rebuild; footprints; Lemma C.3 audit."""

import numpy as np
import pytest

from hope.audit import audit_merge_path
from hope.cache import LayerCache
from hope.costs import (
    dp_conv_filter,
    dp_evict,
    e_identity,
    j_evict,
    j_merge,
    j_prune,
    merge_distortion,
)
from hope.kernels import self_kernel
from hope.parent import synthesize_parent
from hope.surrogate import LayerSurrogate


def random_layer(n=12, d_in=16, d_out=8, seed=30, correlated=True):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, d_in))
    if correlated:
        w[1] = 0.9 * w[0] + 0.45 * rng.standard_normal(d_in)
        w[5] = w[4] + 0.02 * rng.standard_normal(d_in)
    return LayerSurrogate(
        w_eff=w,
        b=rng.standard_normal(n) * 0.2,
        gamma=rng.uniform(0.5, 1.5, n),
        beta=rng.standard_normal(n) * 0.25,
        w_out=rng.standard_normal((n, d_out)),
    )


class TestG:
    @pytest.mark.parametrize("mode", ["zero_bias", "exact"])
    def test_cache_matches_direct(self, mode):
        layer = random_layer()
        cache = LayerCache(random_layer(), kernel_mode=mode)
        ii, jj, j_cached, s_cached = cache.merge_costs()
        rng = np.random.default_rng(31)
        picks = rng.choice(len(ii), size=min(100, len(ii)), replace=False)
        for idx in picks:
            i, j = int(ii[idx]), int(jj[idx])
            parent = synthesize_parent(
                layer, i, j, cache.rho_hat[i, j], cache.e_rem, kernel_mode=mode
            )
            direct = j_merge(
                parent.a,
                parent.b_scalar,
                parent.s,
                cache.caps[i],
                cache.caps[j],
                cache.e_rem,
                cache.n_live,
            )
            assert j_cached[idx] == pytest.approx(direct, rel=1e-9)
            assert s_cached[idx] == pytest.approx(parent.s, rel=1e-9)

    def test_prune_cost_formula(self):
        cache = LayerCache(random_layer())
        act, costs = cache.prune_costs()
        for k, i in enumerate(act):
            expected = cache.n_live * cache.caps[i] / (cache.e_rem - cache.caps[i])
            assert costs[k] == pytest.approx(expected)


class TestCacheUpdates:
    def test_merge_update_matches_fresh_cache(self):
        cache = LayerCache(random_layer(), kernel_mode="zero_bias")
        i, j = 0, 1
        parent = cache.synthesize(i, j)
        cache.apply_merge(i, j, parent)

        surr = cache.surrogate
        keep = np.where(cache.active)[0]
        fresh = LayerCache(
            LayerSurrogate(
                w_eff=surr.w_eff[keep].copy(),
                b=surr.b[keep].copy(),
                gamma=surr.gamma[keep].copy(),
                beta=surr.beta[keep].copy(),
                w_out=surr.w_out[keep].copy(),
            ),
            kernel_mode="zero_bias",
        )
        assert np.allclose(cache.caps[keep], fresh.caps, rtol=1e-6)
        assert np.allclose(cache.k_ii[keep], fresh.k_ii, rtol=1e-9)
        assert np.allclose(cache.rho_hat[np.ix_(keep, keep)], fresh.rho_hat, atol=1e-9)
        assert np.allclose(cache.b_mat[np.ix_(keep, keep)], fresh.b_mat, rtol=1e-6, atol=1e-9)

    def test_prune_bookkeeping(self):
        cache = LayerCache(random_layer())
        e0, n0 = cache.e_rem, cache.n_live
        cap = cache.caps[3]
        cache.apply_prune(3)
        assert cache.n_live == n0 - 1
        assert cache.e_rem == pytest.approx(e0 - cap)

    def test_cache_with_dead_neuron(self):
        layer = random_layer(seed=50)
        layer.w_eff[3] = 0.0
        layer.gamma[3] = 0.0
        cache = LayerCache(layer)
        assert np.isfinite(cache.b_mat).all()
        assert np.isfinite(cache.caps).all()
        _, jp = cache.prune_costs()
        _, _, jm, s = cache.merge_costs()
        assert np.isfinite(jp).all()
        assert np.isfinite(jm).all()
        assert np.isfinite(s).all()

    def test_merge_update_with_degenerate_parent(self):
        cache = LayerCache(random_layer(seed=51))
        parent = cache.synthesize(0, 1)
        parent.w_eff = np.zeros_like(parent.w_eff)
        parent.w_raw = np.zeros_like(parent.w_raw)
        parent.gamma = 0.0
        cache.apply_merge(0, 1, parent)
        assert np.isfinite(cache.rho_hat).all()
        assert np.isfinite(cache.b_mat).all()
        others = [k for k in range(cache.surrogate.n) if k != 0 and cache.active[k]]
        assert np.all(cache.rho_hat[0, others] == 0.0)

    def test_clear_floors_capacity(self):
        cache = LayerCache(random_layer())
        cache.clear()
        assert cache.n_live == 0
        assert cache.e_rem == 1e-12


class TestCosts:
    def test_evict_cost(self):
        gamma = np.array([1.0, 0.5])
        beta = np.array([0.0, 0.5])
        ident = e_identity(gamma, beta)
        assert ident == pytest.approx(1.0 + np.sqrt(0.5))
        assert j_evict([(4, 2.0), (3, 1.5)], ident) == pytest.approx((8.0 + 4.5) / ident)

    def test_static_footprints(self):
        # ResNet-50 bottleneck symmetry, App B.2.1: both internal positions yield 13N + 4
        n = 64
        squeeze = dp_conv_filter(4 * n, 1, 1, 3 * 3 * n)
        spatial = dp_conv_filter(n, 3, 3, 1 * 1 * 4 * n)
        assert squeeze == spatial == 13 * n + 4
        assert dp_evict(100, 200, 12) == 100 + 200 + 48

    def test_prune_cost_diverges_near_extinction(self):
        j_small = j_prune(np.array([1.0]), 10.0, 5)
        j_big = j_prune(np.array([9.999999]), 10.0, 5)
        assert j_big > 1e5 * j_small

    def test_merge_distortion_identity(self):
        assert merge_distortion(2.0, 2.0, 1.0) == pytest.approx(0.0)


class TestAudit:
    def test_lemma_c3_high_correlation_holds(self):
        cache = LayerCache(random_layer(seed=40), kernel_mode="exact")
        i, j = 4, 5
        assert cache.rho_hat[i, j] > 0.5
        parent = cache.synthesize(i, j)
        report = audit_merge_path(cache, i, j, parent, steps=20)
        assert report["violations"] == 0
        assert report["min_margin"] > -1e-9

    def test_audit_report_fields(self):
        cache = LayerCache(random_layer(seed=41))
        parent = cache.synthesize(0, 1)
        report = audit_merge_path(cache, 0, 1, parent)
        assert set(report) == {"rho_ij", "violations", "min_margin", "steps"}
