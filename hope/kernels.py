"""Closed-form ReLU kernels under the Gaussian surrogate. Paper App E."""

import numpy as np
from scipy.special import ndtr, owens_t
from scipy.stats import multivariate_normal

TINY = 1e-12
DEGENERATE_RHO = 1.0 - 1e-9


def _phi(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def relu_mean(gamma, beta):
    """E[relu(y)] for y ~ N(beta, gamma^2)."""
    g = np.abs(np.asarray(gamma, dtype=np.float64))
    beta = np.asarray(beta, dtype=np.float64)
    c = np.divide(beta, g, out=np.zeros_like(g), where=g > TINY)
    out = beta * ndtr(c) + g * _phi(c)
    return np.where(g > TINY, out, np.maximum(beta, 0.0))


def self_kernel(gamma, beta):
    """Self-kernel K(i,i) = E[relu(y)^2], eq (3) / (79)."""
    g = np.abs(np.asarray(gamma, dtype=np.float64))
    beta = np.asarray(beta, dtype=np.float64)
    c = np.divide(beta, g, out=np.zeros_like(g), where=g > TINY)
    out = (g * g + beta * beta) * ndtr(c) + beta * g * _phi(c)
    return np.where(g > TINY, out, np.maximum(beta, 0.0) ** 2)


def warped_correlation(rho_eff, gamma_i, gamma_j, norm_i, norm_j):
    """Warped correlation of the pairwise max-entropy surrogate, eq (80)-(81) / (4)."""
    rho_eff = np.clip(np.asarray(rho_eff, dtype=np.float64), -1.0, 1.0)
    denom = 1.0 - rho_eff * rho_eff
    safe = denom > TINY
    kappa = np.divide(rho_eff, denom, out=np.zeros_like(denom), where=safe)
    kappa = kappa * (np.abs(gamma_i) / norm_i) * (np.abs(gamma_j) / norm_j)
    kappa = np.clip(kappa, -1e150, 1e150)
    rho_hat = 2.0 * kappa / (1.0 + np.sqrt(1.0 + 4.0 * kappa * kappa))
    return np.where(safe, rho_hat, np.sign(rho_eff))


def pairwise_warped_correlation(w_eff, gamma):
    """Warped correlation matrix for all neuron pairs in a layer."""
    w = np.asarray(w_eff, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    norms = np.linalg.norm(w, axis=1)
    safe_norms = np.maximum(norms, TINY)
    rho_eff = (w @ w.T) / np.outer(safe_norms, safe_norms)
    rho_hat = warped_correlation(
        rho_eff, gamma[:, None], gamma[None, :], safe_norms[:, None], safe_norms[None, :]
    )
    dead = norms <= TINY
    rho_hat[dead, :] = 0.0
    rho_hat[:, dead] = 0.0
    np.fill_diagonal(rho_hat, 1.0)
    return rho_hat


def relu_interaction(rho):
    """Normalized arc-cosine interaction I(rho), eq (85)."""
    rho = np.clip(np.asarray(rho, dtype=np.float64), -1.0, 1.0)
    return (np.sqrt(1.0 - rho * rho) + (np.pi - np.arccos(rho)) * rho) / np.pi


def cross_kernel_zero_bias(k_ii, k_jj, rho_hat):
    """Zero-bias cross-kernel approximation, eq (5) / (85)."""
    return relu_interaction(rho_hat) * np.sqrt(np.asarray(k_ii) * np.asarray(k_jj))


def _cross_kernel_collinear(gi, bi, gj, bj, sign):
    """Exact kernel in the degenerate limit rho_hat -> +-1."""
    ci, cj = bi / gi, bj / gj
    if sign > 0:
        m = max(-ci, -cj)
        e0 = ndtr(-m)
        e1 = _phi(m)
        e2 = ndtr(-m) + m * _phi(m)
        return gi * gj * e2 + (gi * bj + gj * bi) * e1 + bi * bj * e0
    lo, hi = -ci, cj
    if hi <= lo:
        return 0.0
    e0 = ndtr(hi) - ndtr(lo)
    e1 = _phi(lo) - _phi(hi)
    e2 = (ndtr(hi) - hi * _phi(hi)) - (ndtr(lo) - lo * _phi(lo))
    return -gi * gj * e2 + (gi * bj - gj * bi) * e1 + bi * bj * e0


def cross_kernel_exact(gamma_i, beta_i, gamma_j, beta_j, rho_hat):
    """Exact biased cross-kernel via the bivariate normal CDF, eq (83). Scalar inputs."""
    gi, gj = abs(float(gamma_i)), abs(float(gamma_j))
    bi, bj = float(beta_i), float(beta_j)
    r = float(np.clip(rho_hat, -1.0, 1.0))
    if gi <= TINY or gj <= TINY:
        return max(bi, 0.0) * relu_mean(gj, bj) if gi <= TINY else relu_mean(gi, bi) * max(bj, 0.0)
    if abs(r) >= DEGENERATE_RHO:
        return _cross_kernel_collinear(gi, bi, gj, bj, np.sign(r))
    ci, cj = bi / gi, bj / gj
    s2 = 1.0 - r * r
    s = np.sqrt(s2)
    cij = (ci - r * cj) / s
    cji = (cj - r * ci) / s
    phi2 = np.exp(-(ci * ci - 2.0 * r * ci * cj + cj * cj) / (2.0 * s2)) / (2.0 * np.pi * s)
    cdf2 = multivariate_normal.cdf([ci, cj], mean=[0.0, 0.0], cov=[[1.0, r], [r, 1.0]])
    inner = (ci * cj + r) * cdf2 + ci * _phi(cj) * ndtr(cij) + cj * _phi(ci) * ndtr(cji) + s2 * phi2
    return gi * gj * inner


def bvn_cdf(h, k, rho):
    """Standard bivariate normal P(X <= h, Y <= k) via Owen's T, vectorized."""
    h, k, rho = np.broadcast_arrays(
        np.asarray(h, dtype=np.float64), np.asarray(k, dtype=np.float64), np.asarray(rho, dtype=np.float64)
    )
    h = np.where(np.abs(h) < 1e-9, 1e-9, h)
    k = np.where(np.abs(k) < 1e-9, 1e-9, k)
    rho = np.clip(rho, -1.0 + 1e-12, 1.0 - 1e-12)
    s = np.sqrt(1.0 - rho * rho)
    a_h = (k - rho * h) / (h * s)
    a_k = (h - rho * k) / (k * s)
    beta = np.where(h * k < 0, 0.5, 0.0)
    return 0.5 * (ndtr(h) + ndtr(k)) - owens_t(h, a_h) - owens_t(k, a_k) - beta


def cross_kernel_exact_batch(gamma_i, beta_i, gamma_j, beta_j, rho_hat):
    """Vectorized exact biased cross-kernel, eq (83), with degenerate and dead limits."""
    gi, gj, bi, bj, r = np.broadcast_arrays(
        np.abs(np.asarray(gamma_i, dtype=np.float64)),
        np.abs(np.asarray(gamma_j, dtype=np.float64)),
        np.asarray(beta_i, dtype=np.float64),
        np.asarray(beta_j, dtype=np.float64),
        np.clip(np.asarray(rho_hat, dtype=np.float64), -1.0, 1.0),
    )
    gis = np.maximum(gi, TINY)
    gjs = np.maximum(gj, TINY)
    ci, cj = bi / gis, bj / gjs

    rm = np.clip(r, -DEGENERATE_RHO, DEGENERATE_RHO)
    s2 = 1.0 - rm * rm
    s = np.sqrt(s2)
    cij = (ci - rm * cj) / s
    cji = (cj - rm * ci) / s
    phi2 = np.exp(-(ci * ci - 2.0 * rm * ci * cj + cj * cj) / (2.0 * s2)) / (2.0 * np.pi * s)
    main = gi * gj * (
        (ci * cj + rm) * bvn_cdf(ci, cj, rm)
        + ci * _phi(cj) * ndtr(cij)
        + cj * _phi(ci) * ndtr(cji)
        + s2 * phi2
    )

    m = np.maximum(-ci, -cj)
    pos = gi * gj * (ndtr(-m) + m * _phi(m)) + (gi * bj + gj * bi) * _phi(m) + bi * bj * ndtr(-m)
    lo, hi = -ci, cj
    e0 = ndtr(hi) - ndtr(lo)
    e1 = _phi(lo) - _phi(hi)
    e2 = (ndtr(hi) - hi * _phi(hi)) - (ndtr(lo) - lo * _phi(lo))
    neg = np.where(hi > lo, -gi * gj * e2 + (gi * bj - gj * bi) * e1 + bi * bj * e0, 0.0)

    out = np.where(r >= DEGENERATE_RHO, pos, np.where(r <= -DEGENERATE_RHO, neg, main))
    dead = (gi <= TINY) | (gj <= TINY)
    return np.where(dead, relu_mean(gi, bi) * relu_mean(gj, bj), out)


def cross_kernel(gamma_i, beta_i, gamma_j, beta_j, rho_hat, k_ii=None, k_jj=None, mode="zero_bias"):
    """Cross-kernel K(i,j) in the configured mode."""
    if mode == "exact":
        return cross_kernel_exact(gamma_i, beta_i, gamma_j, beta_j, rho_hat)
    if k_ii is None:
        k_ii = self_kernel(gamma_i, beta_i)
    if k_jj is None:
        k_jj = self_kernel(gamma_j, beta_j)
    return cross_kernel_zero_bias(k_ii, k_jj, rho_hat)
