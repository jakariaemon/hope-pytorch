"""Phase 3 gates: physical exactness (Test E) and merge fidelity (Test F)."""

import numpy as np
import pytest

from hope.kernels import pairwise_warped_correlation, self_kernel
from hope.parent import physical_preactivation, synthesize_parent
from hope.surrogate import BN_EPS, LayerSurrogate

RNG = np.random.default_rng(20)


def make_pair_layer(rho_target=0.9, n_extra=3, d_in=12, d_out=4, seed=None):
    rng = np.random.default_rng(seed) if seed is not None else RNG
    base = rng.standard_normal(d_in)
    noise = rng.standard_normal(d_in)
    noise -= (noise @ base) / (base @ base) * base
    w_j = rho_target * base + np.sqrt(1 - rho_target**2) * noise * np.linalg.norm(
        base
    ) / np.linalg.norm(noise)
    w = np.vstack([base, w_j, rng.standard_normal((n_extra, d_in))])
    return LayerSurrogate(
        w_eff=w,
        b=rng.standard_normal(2 + n_extra) * 0.2,
        gamma=rng.uniform(0.6, 1.4, 2 + n_extra),
        beta=rng.standard_normal(2 + n_extra) * 0.2,
        w_out=rng.standard_normal((2 + n_extra, d_out)),
    )


def synth(layer, i=0, j=1, mode="zero_bias"):
    rho = pairwise_warped_correlation(layer.w_eff, layer.gamma)
    caps = np.linalg.norm(layer.w_out, axis=1) * np.sqrt(
        self_kernel(layer.gamma, layer.beta)
    )
    return synthesize_parent(layer, i, j, rho[i, j], caps.sum(), kernel_mode=mode)


class TestE:
    def test_physical_forward_exactness(self):
        for seed in range(5):
            layer = make_pair_layer(rho_target=0.8, seed=seed)
            parent = synth(layer)
            assert parent.active
            x = np.random.default_rng(seed).standard_normal((256, layer.w_eff.shape[1]))
            y_i = x @ layer.w_eff[0] + layer.b[0]
            y_j = x @ layer.w_eff[1] + layer.b[1]
            expected = parent.c1 * y_i + parent.c2 * y_j
            got = physical_preactivation(parent, x)
            assert np.max(np.abs(got - expected)) < 1e-5

    def test_torch_bn_forward_exactness(self):
        import torch

        layer = make_pair_layer(rho_target=0.85, seed=42)
        parent = synth(layer)
        d_in = layer.w_eff.shape[1]
        lin = torch.nn.Linear(d_in, 1, bias=False).double()
        bn = torch.nn.BatchNorm1d(1, eps=BN_EPS).double().eval()
        with torch.no_grad():
            lin.weight.copy_(torch.from_numpy(parent.w_raw[None, :]))
            bn.weight.fill_(parent.gamma)
            bn.bias.fill_(parent.beta)
            bn.running_mean.fill_(parent.mu)
            bn.running_var.fill_(parent.sigma2)
            x = torch.randn(128, d_in, dtype=torch.float64)
            got = bn(lin(x)).numpy().ravel()
        x_np = x.numpy()
        expected = parent.c1 * (x_np @ layer.w_eff[0] + layer.b[0]) + parent.c2 * (
            x_np @ layer.w_eff[1] + layer.b[1]
        )
        assert np.max(np.abs(got - expected)) < 1e-5

    def test_gamma_sign_invariance(self):
        # eq (71)-(72): flipping the sign of gamma_p leaves the deployed forward unchanged
        layer = make_pair_layer(rho_target=0.7, seed=7)
        parent = synth(layer)
        x = np.random.default_rng(7).standard_normal((64, layer.w_eff.shape[1]))
        y_pos = physical_preactivation(parent, x)
        flipped = type(parent)(**{**parent.__dict__})
        flipped.gamma = -parent.gamma
        flipped.w_raw = -parent.w_raw
        flipped.mu = -parent.mu
        y_neg = physical_preactivation(flipped, x)
        assert np.max(np.abs(y_pos - y_neg)) < 1e-9

    def test_parent_capacity_equals_s(self):
        for seed in range(3):
            layer = make_pair_layer(rho_target=0.9, seed=seed + 100)
            parent = synth(layer)
            cap = np.linalg.norm(parent.w_out) * np.sqrt(
                self_kernel(parent.gamma, parent.beta)
            )
            assert cap == pytest.approx(parent.s, rel=1e-9)

    def test_direction_matches_dense_svd(self):
        layer = make_pair_layer(rho_target=0.6, seed=11)
        parent = synth(layer)
        w_aug = layer.w_aug
        a_mat = np.outer(layer.w_out[0], w_aug[0]) + np.outer(layer.w_out[1], w_aug[1])
        _, _, vh = np.linalg.svd(a_mat)
        u_dense = vh[0]
        u_ours = parent.c1 * w_aug[0] + parent.c2 * w_aug[1]
        u_ours /= np.linalg.norm(u_ours)
        assert abs(abs(u_dense @ u_ours) - 1.0) < 1e-9

    def test_identical_twins_merge_lossless(self):
        rng = np.random.default_rng(5)
        w = rng.standard_normal(10)
        layer = LayerSurrogate(
            w_eff=np.vstack([w, w, rng.standard_normal((2, 10))]),
            b=np.array([0.1, 0.1, 0.0, 0.0]),
            gamma=np.array([1.2, 1.2, 0.8, 0.9]),
            beta=np.array([0.3, 0.3, 0.1, -0.2]),
            w_out=np.vstack([rng.standard_normal(4)] * 2 + [rng.standard_normal((2, 4))]),
        )
        parent = synth(layer)
        assert parent.distortion == pytest.approx(0.0, abs=1e-6)
        x = rng.standard_normal((32, 10))
        y_i = x @ layer.w_eff[0] + layer.b[0]
        assert np.allclose(physical_preactivation(parent, x), y_i, atol=1e-8)

    def test_inactive_parent_clamped(self):
        layer = make_pair_layer(rho_target=0.5, seed=13)
        layer.gamma = np.full_like(layer.gamma, 1e-4)
        parent = synth(layer)
        assert not parent.active
        assert np.all(parent.w_raw == 0.0)
        assert parent.sigma2 == 0.0


class TestF:
    def mc_distortion(self, layer, parent, n=2_000_000, seed=0):
        rng = np.random.default_rng(seed)
        gi, gj = abs(layer.gamma[0]), abs(layer.gamma[1])
        rho = parent.rho_hat
        z1 = rng.standard_normal(n)
        z2 = rho * z1 + np.sqrt(max(1 - rho * rho, 0.0)) * rng.standard_normal(n)
        y_i = layer.beta[0] + gi * z1
        y_j = layer.beta[1] + gj * z2
        y_p = parent.c1 * y_i + parent.c2 * y_j
        r_i, r_j, r_p = np.maximum(y_i, 0), np.maximum(y_j, 0), np.maximum(y_p, 0)
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
        layer = make_pair_layer(rho_target=rho_target, seed=int(rho_target * 100))
        parent = synth(layer, mode="exact")
        measured = self.mc_distortion(layer, parent)
        assert measured == pytest.approx(parent.distortion, rel=0.02, abs=1e-3)

    def test_distortion_vanishes_with_correlation(self):
        distortions = []
        for rho_target in [0.5, 0.99, 0.99999]:
            layer = make_pair_layer(rho_target=rho_target, seed=77)
            layer.gamma[1] = layer.gamma[0]
            layer.beta[1] = layer.beta[0]
            layer.b[1] = layer.b[0]
            layer.w_out[1] = layer.w_out[0]
            parent = synth(layer, mode="exact")
            distortions.append(parent.distortion)
        assert distortions[0] > distortions[1] > distortions[2]
        assert distortions[2] < 0.05 * distortions[0]
