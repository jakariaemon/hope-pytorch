"""GELU stage 1 gates: closed form and quadrature kernels match 40M sample Monte Carlo to 0.1 percent; properties and limits."""

import numpy as np
import pytest

from hope.activations import (
    gelu,
    gelu_boundary_identity,
    gelu_cross_kernel_exact,
    gelu_cross_kernel_zero_bias,
    gelu_interaction,
    gelu_mean,
    gelu_self_kernel,
    gelu_self_kernel_quad,
)
from hope.kernels import self_kernel as relu_self_kernel

N_MC = 40_000_000
CHUNK = 5_000_000


def mc_self(gamma, beta, seed):
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(N_MC // CHUNK):
        z = rng.standard_normal(CHUNK)
        for zs in (z, -z):
            total += np.sum(gelu(beta + abs(gamma) * zs) ** 2)
    return total / (2 * N_MC)


def mc_cross(gamma_i, beta_i, gamma_j, beta_j, rho, seed):
    rng = np.random.default_rng(seed)
    root = np.sqrt(1.0 - rho * rho)
    total = 0.0
    for _ in range(N_MC // CHUNK):
        z1 = rng.standard_normal(CHUNK)
        z2 = rho * z1 + root * rng.standard_normal(CHUNK)
        for s in (1.0, -1.0):
            total += np.sum(
                gelu(beta_i + abs(gamma_i) * s * z1) * gelu(beta_j + abs(gamma_j) * s * z2)
            )
    return total / (2 * N_MC)


class TestSelfKernel:
    @pytest.mark.parametrize(
        "gamma,beta",
        [(1.0, 0.0), (0.7, 0.3), (2.0, -1.5), (-1.2, 0.8), (0.5, 1.9), (1.0, -2.4), (3.0, 0.0)],
    )
    def test_monte_carlo(self, gamma, beta):
        closed = gelu_self_kernel(gamma, beta)
        est = mc_self(gamma, beta, seed=abs(hash((gamma, beta))) % 2**32)
        assert abs(closed - est) / max(est, 1e-6) < 1e-3

    def test_closed_form_matches_quadrature(self):
        # closed form is exact; GH-64 truncation grows with gamma, so gate by regime
        rng = np.random.default_rng(60)
        gamma = rng.uniform(0.05, 2.0, 200)
        beta = rng.uniform(-3.0, 3.0, 200)
        assert np.allclose(gelu_self_kernel(gamma, beta), gelu_self_kernel_quad(gamma, beta), rtol=1e-6)
        gamma_wide = rng.uniform(2.0, 4.0, 50)
        beta_wide = rng.uniform(-3.0, 3.0, 50)
        assert np.allclose(
            gelu_self_kernel(gamma_wide, beta_wide), gelu_self_kernel_quad(gamma_wide, beta_wide), rtol=2e-3
        )

    def test_constant_limit(self):
        for beta in [-1.0, 0.4, 2.0]:
            assert gelu_self_kernel(0.0, beta) == pytest.approx(gelu(beta) ** 2, rel=1e-12)

    def test_relu_asymptotics(self):
        # gelu approaches relu at large signal scale, so the kernels converge
        for gamma in [5.0, 20.0]:
            ratio = gelu_self_kernel(gamma, 0.0) / relu_self_kernel(gamma, 0.0)
            assert ratio == pytest.approx(1.0, abs=0.5 / gamma)


class TestCrossKernel:
    @pytest.mark.parametrize(
        "gi,bi,gj,bj,rho",
        [
            (1.0, 0.0, 1.0, 0.0, 0.6),
            (0.8, 0.4, 1.3, -0.5, 0.9),
            (1.5, -0.9, 0.6, 0.7, -0.4),
            (1.0, 1.2, 1.0, 1.2, 0.99),
            (2.0, -1.0, 0.5, -0.2, 0.0),
        ],
    )
    def test_monte_carlo(self, gi, bi, gj, bj, rho):
        closed = gelu_cross_kernel_exact(gi, bi, gj, bj, rho)
        if rho == 0.0:
            # independence factorizes exactly, a stronger gate than Monte Carlo
            assert closed == pytest.approx(
                float(gelu_mean(gi, bi)) * float(gelu_mean(gj, bj)), abs=1e-12
            )
            return
        est = mc_cross(gi, bi, gj, bj, rho, seed=abs(hash((gi, bi, gj, bj, rho))) % 2**32)
        assert abs(closed - est) / max(abs(est), abs(gi * gj) * 1e-3) < 1e-3

    def test_diagonal_consistency(self):
        for gamma, beta in [(1.0, 0.0), (0.7, 0.5), (1.3, -0.6)]:
            k_self = gelu_self_kernel(gamma, beta)
            k_cross = gelu_cross_kernel_exact(gamma, beta, gamma, beta, 1.0)
            assert k_cross == pytest.approx(k_self, rel=1e-9)

    def test_cauchy_schwarz(self):
        rng = np.random.default_rng(61)
        for _ in range(200):
            gi, gj = rng.uniform(0.2, 2.0, 2)
            bi, bj = rng.uniform(-2.0, 2.0, 2)
            rho = rng.uniform(-0.999, 0.999)
            k = gelu_cross_kernel_exact(gi, bi, gj, bj, rho)
            bound = np.sqrt(gelu_self_kernel(gi, bi) * gelu_self_kernel(gj, bj))
            assert abs(k) <= bound * (1.0 + 1e-8)

    def test_interaction_table(self):
        rho = np.linspace(-1.0, 1.0, 101)
        vals = gelu_interaction(rho)
        assert vals[-1] == pytest.approx(1.0, abs=1e-6)
        direct = gelu_cross_kernel_exact(1.0, 0.0, 1.0, 0.0, 0.37) / gelu_self_kernel(1.0, 0.0)
        assert gelu_interaction(0.37) == pytest.approx(direct, abs=1e-5)
        assert np.all(np.diff(vals) > -1e-9)

    def test_zero_bias_matches_exact_at_zero_bias(self):
        k0 = gelu_self_kernel(1.0, 0.0)
        for rho in [-0.7, 0.0, 0.5, 0.95]:
            approx = gelu_cross_kernel_zero_bias(k0, k0, rho)
            exact = gelu_cross_kernel_exact(1.0, 0.0, 1.0, 0.0, rho)
            assert approx == pytest.approx(exact, abs=1e-5)


class TestBoundaryIdentity:
    def test_k1_kp1_gap_documented(self):
        # PH-1 activations satisfy k(1) = k'(1); GELU does not, log the measured gap
        k1, k1p = gelu_boundary_identity()
        gap = abs(k1 - k1p) / k1
        print(f"\ngelu k(1)={k1:.6f} k'(1)={k1p:.6f} rel gap={gap:.4f}")
        assert 0.0 < gap < 1.0
