"""Phase 2 gates: capacity exactly invariant under PH-1 and BN rescaling where L1 breaks (Test C); kernel predictions vs real activations (Test D)."""

import os

import numpy as np
import pytest

from hope.capacity import capacity, conv_input_vectors, conv_output_vectors, layer_capacities
from hope.kernels import self_kernel
from hope.surrogate import LayerSurrogate

RNG = np.random.default_rng(10)


def random_layer(n=8, d_in=16, d_out=12):
    return LayerSurrogate(
        w_eff=RNG.standard_normal((n, d_in)),
        b=RNG.standard_normal(n) * 0.3,
        gamma=RNG.uniform(0.4, 1.6, n) * np.sign(RNG.standard_normal(n)),
        beta=RNG.standard_normal(n) * 0.4,
        w_out=RNG.standard_normal((n, d_out)),
    )


class TestC:
    def test_ph1_rescale_invariance(self):
        layer = random_layer()
        caps, _ = layer_capacities(layer)
        lam = 3.7
        scaled = LayerSurrogate(
            w_eff=lam * layer.w_eff,
            b=lam * layer.b,
            gamma=lam * layer.gamma,
            beta=lam * layer.beta,
            w_out=layer.w_out / lam,
        )
        caps_scaled, _ = layer_capacities(scaled)
        assert np.allclose(caps_scaled, caps, rtol=1e-6)
        l1 = np.abs(layer.w_eff).sum(axis=1)
        l1_scaled = np.abs(scaled.w_eff).sum(axis=1)
        assert not np.allclose(l1_scaled, l1, rtol=1e-2)

    def test_bn_normalization_invariance(self):
        n, d = 6, 10
        w_raw = RNG.standard_normal((n, d))
        gamma = RNG.uniform(0.5, 1.5, n)
        beta = RNG.standard_normal(n) * 0.3
        mu = RNG.standard_normal(n) * 0.2
        var = RNG.uniform(0.5, 2.0, n)
        w_out = RNG.standard_normal((n, 7))
        eps = 1e-5

        layer = LayerSurrogate.from_bn(w_raw, gamma, beta, mu, var, w_out, eps)
        caps, _ = layer_capacities(layer)

        # scaling raw weights by lam with matching BN statistics leaves the function unchanged
        lam = 2.5
        var2 = lam * lam * (var + eps) - eps
        scaled = LayerSurrogate.from_bn(lam * w_raw, gamma, beta, lam * mu, var2, w_out, eps)
        caps2, _ = layer_capacities(scaled)
        assert np.allclose(scaled.w_eff, layer.w_eff, rtol=1e-9)
        assert np.allclose(scaled.b, layer.b, rtol=1e-9, atol=1e-12)
        assert np.allclose(caps2, caps, rtol=1e-9)
        l1_raw = np.abs(w_raw).sum(axis=1)
        assert not np.allclose(np.abs(lam * w_raw).sum(axis=1), l1_raw, rtol=1e-2)

    def test_capacity_matches_direct_formula(self):
        layer = random_layer()
        caps, k_ii = layer_capacities(layer)
        direct = np.linalg.norm(layer.w_out, axis=1) * np.sqrt(k_ii)
        assert np.allclose(caps, direct)

    def test_conv_vector_extraction(self):
        w_a = RNG.standard_normal((8, 3, 3, 3))
        w_b = RNG.standard_normal((5, 8, 1, 1))
        w_in = conv_input_vectors(w_a)
        assert w_in.shape == (8, 27)
        assert np.allclose(w_in[2], w_a[2].ravel())
        w_out = conv_output_vectors(w_b)
        assert w_out.shape == (8, 5)
        assert np.allclose(w_out[3], w_b[:, 3].ravel())


@pytest.mark.slow
class TestD:
    def test_resnet50_surrogate_error(self):
        imagenet_dir = os.environ.get("HOPE_IMAGENET_DIR")
        if not imagenet_dir:
            pytest.skip("set HOPE_IMAGENET_DIR to an ImageNet val directory")
        from scripts.surrogate_check import run_check

        stats = run_check(imagenet_dir, batch_size=64, device="auto")
        print("\nsurrogate per-channel rel err:", stats)
        assert stats["median"] < 0.5
