"""Phase 1 gates: Monte Carlo agreement (Test A) and kernel properties (Test B)."""

import numpy as np
import pytest

from hope.kernels import (
    bvn_cdf,
    cross_kernel_exact,
    cross_kernel_exact_batch,
    cross_kernel_zero_bias,
    pairwise_warped_correlation,
    relu_interaction,
    relu_mean,
    self_kernel,
    warped_correlation,
)

N_MC = 40_000_000
CHUNK = 5_000_000


def mc_self(gamma, beta, seed):
    # importance sampling restricted to the active region y > 0
    from scipy.special import ndtr, ndtri

    rng = np.random.default_rng(seed)
    g = abs(gamma)
    c = beta / g
    p = ndtr(c)
    total = 0.0
    for _ in range(N_MC // CHUNK):
        u = rng.random(CHUNK)
        z = ndtri(ndtr(-c) + u * p)
        total += np.sum((beta + g * z) ** 2)
    return p * total / N_MC


def mc_cross(gamma_i, beta_i, gamma_j, beta_j, rho, seed):
    rng = np.random.default_rng(seed)
    root = np.sqrt(1.0 - rho * rho)
    total = 0.0
    for _ in range(N_MC // CHUNK):
        z1 = rng.standard_normal(CHUNK)
        z2 = rho * z1 + root * rng.standard_normal(CHUNK)
        for s in (1.0, -1.0):
            yi = beta_i + abs(gamma_i) * s * z1
            yj = beta_j + abs(gamma_j) * s * z2
            total += np.sum(np.maximum(yi, 0.0) * np.maximum(yj, 0.0))
    return total / (2 * N_MC)


class TestA:
    @pytest.mark.parametrize(
        "gamma,beta",
        [(1.0, 0.0), (0.7, 0.3), (2.0, -1.5), (-1.2, 0.8), (0.5, 1.9), (1.0, -2.4)],
    )
    def test_self_kernel_monte_carlo(self, gamma, beta):
        closed = self_kernel(gamma, beta)
        est = mc_self(gamma, beta, seed=abs(hash((gamma, beta))) % 2**32)
        assert abs(closed - est) / max(est, 1e-6) < 1e-3

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
    def test_exact_cross_kernel_monte_carlo(self, gi, bi, gj, bj, rho):
        closed = cross_kernel_exact(gi, bi, gj, bj, rho)
        est = mc_cross(gi, bi, gj, bj, rho, seed=abs(hash((gi, bi, gj, bj, rho))) % 2**32)
        assert abs(closed - est) / max(abs(est), abs(gi * gj) * 1e-3) < 1e-3

    def test_zero_bias_domain_of_validity(self):
        # error normalized by sqrt(Kii*Kjj), the scale merge scoring operates on
        rows = []
        for c in [0.0, 0.1, 0.25, 0.45, 0.75, 1.0, 1.5]:
            worst = 0.0
            for rho in [-0.5, 0.0, 0.3, 0.6, 0.9]:
                for sign in (1.0, -1.0):
                    bi = bj = sign * c
                    exact = cross_kernel_exact(1.0, bi, 1.0, bj, rho)
                    k_ii = self_kernel(1.0, bi)
                    k_jj = self_kernel(1.0, bj)
                    approx = cross_kernel_zero_bias(k_ii, k_jj, rho)
                    worst = max(worst, abs(approx - exact) / np.sqrt(k_ii * k_jj))
            rows.append((c, worst))
            if c <= 0.1:
                assert worst < 0.03
            elif c <= 0.25:
                assert worst < 0.08
            elif c <= 0.5:
                assert worst < 0.15
        print("\nzero-bias worst err / sqrt(Kii*Kjj) vs |beta/gamma|:")
        for c, err in rows:
            print(f"  {c:4.2f}  {err:.4f}")


class TestB:
    def test_diagonal_consistency(self):
        for gamma, beta in [(1.0, 0.0), (0.7, 0.5), (1.3, -0.6)]:
            k_self = self_kernel(gamma, beta)
            k_cross = cross_kernel_exact(gamma, beta, gamma, beta, 1.0)
            assert abs(k_cross - k_self) < 1e-9 * max(k_self, 1.0)
            assert abs(cross_kernel_zero_bias(k_self, k_self, 1.0) - k_self) < 1e-12

    def test_cauchy_schwarz(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            gi, gj = rng.uniform(0.2, 2.0, 2)
            bi, bj = rng.uniform(-2.0, 2.0, 2)
            rho = rng.uniform(-0.999, 0.999)
            k = cross_kernel_exact(gi, bi, gj, bj, rho)
            bound = np.sqrt(self_kernel(gi, bi) * self_kernel(gj, bj))
            assert abs(k) <= bound * (1.0 + 1e-8)

    def test_monotone_in_rho(self):
        grid = np.linspace(-0.999, 0.999, 41)
        for gi, bi, gj, bj in [(1.0, 0.0, 1.0, 0.0), (0.8, 0.6, 1.4, -0.3)]:
            vals = [cross_kernel_exact(gi, bi, gj, bj, r) for r in grid]
            assert np.all(np.diff(vals) > -1e-9)
        approx = relu_interaction(grid)
        assert np.all(np.diff(approx) > -1e-12)

    def test_warped_correlation_properties(self):
        rng = np.random.default_rng(2)
        rho_eff = rng.uniform(-0.999, 0.999, 100)
        rho_hat = warped_correlation(rho_eff, 1.0, 1.0, 1.0, 1.0)
        assert np.all(np.abs(rho_hat) <= 1.0)
        assert np.all(np.sign(rho_hat) == np.sign(rho_eff))
        order = np.argsort(rho_eff)
        assert np.all(np.diff(rho_hat[order]) > 0)
        assert warped_correlation(1.0, 2.0, 3.0, 1.0, 1.0) == 1.0
        assert warped_correlation(-1.0, 2.0, 3.0, 1.0, 1.0) == -1.0

    def test_bvn_cdf_matches_scipy(self):
        from scipy.stats import multivariate_normal

        rng = np.random.default_rng(6)
        for _ in range(300):
            h, k = rng.uniform(-3, 3, 2)
            rho = rng.uniform(-0.999, 0.999)
            ref = multivariate_normal.cdf([h, k], mean=[0, 0], cov=[[1, rho], [rho, 1]])
            assert bvn_cdf(h, k, rho) == pytest.approx(ref, abs=1e-8)
        assert bvn_cdf(0.0, 0.0, 0.6) == pytest.approx(0.25 + np.arcsin(0.6) / (2 * np.pi))
        assert bvn_cdf(0.0, -1.3, -0.4) == pytest.approx(
            multivariate_normal.cdf([0, -1.3], mean=[0, 0], cov=[[1, -0.4], [-0.4, 1]]), abs=1e-8
        )

    def test_exact_batch_matches_scalar(self):
        rng = np.random.default_rng(8)
        gi = rng.uniform(0.2, 2.0, 500)
        gj = rng.uniform(0.2, 2.0, 500)
        bi = rng.uniform(-2.0, 2.0, 500)
        bj = rng.uniform(-2.0, 2.0, 500)
        rho = rng.uniform(-1.0, 1.0, 500)
        rho[:5] = [1.0, -1.0, 0.0, 1.0 - 1e-12, -1.0 + 1e-12]
        gi[5:8] = 0.0
        batch = cross_kernel_exact_batch(gi, bi, gj, bj, rho)
        for idx in range(500):
            ref = cross_kernel_exact(gi[idx], bi[idx], gj[idx], bj[idx], rho[idx])
            assert batch[idx] == pytest.approx(ref, rel=1e-7, abs=1e-9)

    def test_dead_neuron_limits(self):
        assert relu_mean(0.0, 0.7) == pytest.approx(0.7)
        assert relu_mean(0.0, -0.3) == 0.0
        assert self_kernel(0.0, 0.7) == pytest.approx(0.49)
        assert self_kernel(0.0, -0.3) == 0.0
        expected = 0.5 * relu_mean(1.0, 0.2)
        assert cross_kernel_exact(0.0, 0.5, 1.0, 0.2, 0.9) == pytest.approx(expected)
        assert cross_kernel_exact(1.0, 0.2, 0.0, -0.5, 0.3) == 0.0
        vals = [
            relu_mean(0.0, 0.0),
            self_kernel(0.0, 0.0),
            cross_kernel_exact(0.0, 0.0, 0.0, 0.0, 0.0),
        ]
        assert np.all(np.isfinite(vals))

    def test_dead_row_pairwise_correlation(self):
        rng = np.random.default_rng(4)
        w = rng.standard_normal((5, 8))
        w[2] = 0.0
        rho = pairwise_warped_correlation(w, rng.uniform(0.5, 1.5, 5))
        assert np.isfinite(rho).all()
        assert rho[2, 2] == 1.0
        off = [rho[2, k] for k in range(5) if k != 2]
        assert np.all(np.asarray(off) == 0.0)

    def test_pairwise_matrix(self):
        rng = np.random.default_rng(3)
        w = rng.standard_normal((6, 10))
        gamma = rng.uniform(0.5, 1.5, 6)
        rho = pairwise_warped_correlation(w, gamma)
        assert np.allclose(rho, rho.T)
        assert np.allclose(np.diag(rho), 1.0)
        w2 = np.vstack([w, 2.0 * w[0]])
        rho2 = pairwise_warped_correlation(w2, np.append(gamma, gamma[0]))
        assert rho2[0, 6] == pytest.approx(1.0)
