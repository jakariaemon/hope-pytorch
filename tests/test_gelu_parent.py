"""GELU stage 2 gates: alpha search deployment is exact for twins, near optimal vs brute force, and distortion predictions hold."""

import numpy as np
import pytest

from hope.activations import gelu, gelu_self_kernel
from hope.kernels import pairwise_warped_correlation
from hope.parent import _mixture_stats, _objective, physical_preactivation, synthesize_parent
from hope.surrogate import LayerSurrogate

from hope.activations import get_kernels

RNG = np.random.default_rng(70)


def make_gelu_layer(rho_target=0.9, n_extra=3, d_in=12, d_out=4, seed=None, twin=False):
    rng = np.random.default_rng(seed) if seed is not None else RNG
    base = rng.standard_normal(d_in)
    if twin:
        w_j = base.copy()
    else:
        noise = rng.standard_normal(d_in)
        noise -= (noise @ base) / (base @ base) * base
        w_j = rho_target * base + np.sqrt(1 - rho_target**2) * noise * np.linalg.norm(
            base
        ) / np.linalg.norm(noise)
    w = np.vstack([base, w_j, rng.standard_normal((n_extra, d_in))])
    b = rng.standard_normal(2 + n_extra) * 0.2
    gamma = rng.uniform(0.6, 1.4, 2 + n_extra)
    beta = rng.standard_normal(2 + n_extra) * 0.2
    w_out = rng.standard_normal((2 + n_extra, d_out))
    if twin:
        b[1], gamma[1], beta[1] = b[0], gamma[0], beta[0]
        w_out[1] = w_out[0]
    return LayerSurrogate(w_eff=w, b=b, gamma=gamma, beta=beta, w_out=w_out, activation="gelu")


def synth(layer, i=0, j=1, mode="zero_bias"):
    rho = pairwise_warped_correlation(layer.w_eff, layer.gamma)
    caps = np.linalg.norm(layer.w_out, axis=1) * np.sqrt(
        gelu_self_kernel(layer.gamma, layer.beta)
    )
    return synthesize_parent(layer, i, j, rho[i, j], caps.sum(), kernel_mode=mode)


class TestDeployment:
    def test_twins_merge_lossless(self):
        layer = make_gelu_layer(seed=1, twin=True)
        parent = synth(layer)
        assert parent.distortion == pytest.approx(0.0, abs=1e-5)
        assert np.allclose(parent.w_eff, layer.w_eff[0], atol=1e-6)
        assert parent.b == pytest.approx(layer.b[0], abs=1e-6)
        assert np.allclose(parent.w_out, layer.w_out[0], atol=1e-6)
        x = np.random.default_rng(1).standard_normal((64, layer.w_eff.shape[1]))
        y_child = x @ layer.w_eff[0] + layer.b[0]
        f_child = gelu(y_child)[:, None] * layer.w_out[0]
        y_p = physical_preactivation(parent, x)
        f_parent = gelu(y_p)[:, None] * parent.w_out
        assert np.max(np.abs(f_parent - f_child)) < 1e-6

    def test_physical_forward_exactness(self):
        for seed in range(5):
            layer = make_gelu_layer(rho_target=0.8, seed=seed)
            parent = synth(layer)
            assert parent.active
            x = np.random.default_rng(seed).standard_normal((256, layer.w_eff.shape[1]))
            expected = parent.c1 * (x @ layer.w_eff[0] + layer.b[0]) + parent.c2 * (
                x @ layer.w_eff[1] + layer.b[1]
            )
            assert np.max(np.abs(physical_preactivation(parent, x) - expected)) < 1e-5

    def test_parent_capacity_equals_s(self):
        for seed in range(3):
            layer = make_gelu_layer(rho_target=0.9, seed=seed + 10)
            parent = synth(layer)
            cap = np.linalg.norm(parent.w_out) * np.sqrt(
                gelu_self_kernel(parent.gamma, parent.beta)
            )
            assert cap == pytest.approx(parent.s, rel=1e-6)


class TestDirectionQuality:
    def test_near_optimal_vs_brute_force(self):
        # stage 2 win: eigenproblem plus alpha search reaches the brute force objective
        act = get_kernels("gelu")
        worst = 1.0
        for seed in range(6):
            layer = make_gelu_layer(rho_target=0.92, seed=seed + 30)
            parent = synth(layer, mode="exact")
            rho = pairwise_warped_correlation(layer.w_eff, layer.gamma)[0, 1]
            ni = {"gamma": float(layer.gamma[0]), "beta": float(layer.beta[0])}
            nj = {"gamma": float(layer.gamma[1]), "beta": float(layer.beta[1])}
            w_aug = layer.w_aug
            g11 = float(w_aug[0] @ w_aug[0])
            g22 = float(w_aug[1] @ w_aug[1])
            o = layer.w_out[:2] @ layer.w_out[:2].T
            k_ii = float(gelu_self_kernel(ni["gamma"], ni["beta"]))
            k_jj = float(gelu_self_kernel(nj["gamma"], nj["beta"]))
            best = 0.0
            alpha0 = (g11 * g22) ** 0.25
            for theta in np.linspace(0, np.pi, 121, endpoint=False):
                q = np.array([np.cos(theta) / np.sqrt(g11), np.sin(theta) / np.sqrt(g22)])
                for alpha in alpha0 * np.logspace(-1.0, 1.0, 15):
                    for sgn in (1.0, -1.0):
                        obj = _objective(
                            act, "exact", sgn * alpha * q[0], sgn * alpha * q[1],
                            ni, nj, rho, k_ii, k_jj, o[0, 0], o[0, 1], o[1, 1],
                        )[0]
                        best = max(best, obj)
            ratio = parent.b_scalar / best
            worst = min(worst, ratio)
        print(f"\nsynthesis b vs brute force: worst ratio {worst:.5f}")
        assert worst > 0.995


class TestFidelity:
    def mc_distortion(self, layer, parent, n=2_000_000, seed=0):
        rng = np.random.default_rng(seed)
        gi, gj = abs(layer.gamma[0]), abs(layer.gamma[1])
        rho = parent.rho_hat
        z1 = rng.standard_normal(n)
        z2 = rho * z1 + np.sqrt(max(1 - rho * rho, 0.0)) * rng.standard_normal(n)
        y_i = layer.beta[0] + gi * z1
        y_j = layer.beta[1] + gj * z2
        y_p = parent.c1 * y_i + parent.c2 * y_j
        r_i, r_j, r_p = gelu(y_i), gelu(y_j), gelu(y_p)
        oi, oj, op = layer.w_out[0], layer.w_out[1], parent.w_out
        d_i = (
            np.mean(r_i**2) * (oi @ oi)
            - 2 * np.mean(r_i * r_p) * (oi @ op)
            + np.mean(r_p**2) * (op @ op)
        )
        d_j = (
            np.mean(r_j**2) * (oj @ oj)
            - 2 * np.mean(r_j * r_p) * (oj @ op)
            + np.mean(r_p**2) * (op @ op)
        )
        return np.sqrt(max(d_i + d_j, 0.0))

    @pytest.mark.parametrize("rho_target", [0.5, 0.8, 0.95])
    def test_predicted_distortion_matches_measured(self, rho_target):
        layer = make_gelu_layer(rho_target=rho_target, seed=int(rho_target * 1000))
        parent = synth(layer, mode="exact")
        measured = self.mc_distortion(layer, parent)
        assert measured == pytest.approx(parent.distortion, rel=0.02, abs=1e-3)

    def test_cache_scan_approximation_bounded(self):
        from hope.cache import LayerCache
        from hope.costs import j_merge

        layer = make_gelu_layer(rho_target=0.9, seed=99, n_extra=6)
        cache = LayerCache(
            LayerSurrogate(
                w_eff=layer.w_eff.copy(), b=layer.b.copy(), gamma=layer.gamma.copy(),
                beta=layer.beta.copy(), w_out=layer.w_out.copy(), activation="gelu",
            )
        )
        ii, jj, j_cached, _ = cache.merge_costs()
        devs = []
        for idx in range(len(ii)):
            i, j = int(ii[idx]), int(jj[idx])
            parent = cache.synthesize(i, j)
            direct = j_merge(
                parent.a, parent.b_scalar, parent.s,
                cache.caps[i], cache.caps[j], cache.e_rem, cache.n_live,
            )
            devs.append(abs(j_cached[idx] - direct) / max(direct, 1e-9))
        print(f"\ngelu cache vs jit J deviation: median {np.median(devs):.4f} max {np.max(devs):.4f}")
        assert np.median(devs) < 0.1
