"""GELU kernels under the Gaussian surrogate: closed-form self-kernel, quadrature cross-kernel."""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, owens_t, roots_hermite

from .kernels import DEGENERATE_RHO

GH_NODES = 64
_gh_x, _gh_w = roots_hermite(GH_NODES)
_interaction_table = None


def _phi(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def gelu(y):
    return y * ndtr(y)


def gelu_self_kernel(gamma, beta):
    """Closed form E[gelu(y)^2] for y ~ N(beta, gamma^2) via Stein identities and Owen's T."""
    beta = np.asarray(beta, dtype=np.float64)
    s2 = np.asarray(gamma, dtype=np.float64) ** 2
    r1 = 1.0 + s2
    r2 = 1.0 + 2.0 * s2
    c = beta / np.sqrt(r1)
    a_term = ndtr(c) - 2.0 * owens_t(c, 1.0 / np.sqrt(r2))
    d = beta / np.sqrt(r1 * r2)
    b_term = _phi(c) * ndtr(d) / np.sqrt(r1)
    c_term = np.exp(-beta * beta / r2) / (2.0 * np.pi * np.sqrt(r2))
    d_term = (beta * b_term + s2 * c_term) / r1
    return (beta * beta + s2) * a_term + 4.0 * beta * s2 * b_term + 2.0 * s2 * s2 * (c_term - d_term)


def gelu_self_kernel_quad(gamma, beta):
    """Gauss-Hermite reference for E[gelu(y)^2]."""
    gamma = np.atleast_1d(np.asarray(gamma, dtype=np.float64))
    beta = np.atleast_1d(np.asarray(beta, dtype=np.float64))
    y = beta[..., None] + np.abs(gamma)[..., None] * np.sqrt(2.0) * _gh_x
    return np.squeeze(gelu(y) ** 2 @ _gh_w / np.sqrt(np.pi))


def gelu_mean(gamma, beta):
    """E[gelu(y)] via Gauss-Hermite."""
    gamma = np.atleast_1d(np.asarray(gamma, dtype=np.float64))
    beta = np.atleast_1d(np.asarray(beta, dtype=np.float64))
    y = beta[..., None] + np.abs(gamma)[..., None] * np.sqrt(2.0) * _gh_x
    return np.squeeze(gelu(y) @ _gh_w / np.sqrt(np.pi))


def gelu_cross_kernel_exact(gamma_i, beta_i, gamma_j, beta_j, rho_hat):
    """E[gelu(y_i) gelu(y_j)] under the pairwise surrogate, vectorized 2D Gauss-Hermite."""
    gi, gj, bi, bj, r = np.broadcast_arrays(
        np.abs(np.asarray(gamma_i, dtype=np.float64)),
        np.abs(np.asarray(gamma_j, dtype=np.float64)),
        np.asarray(beta_i, dtype=np.float64),
        np.asarray(beta_j, dtype=np.float64),
        np.clip(np.asarray(rho_hat, dtype=np.float64), -1.0, 1.0),
    )
    shape = r.shape
    gi, gj, bi, bj, r = (a.ravel() for a in (gi, gj, bi, bj, r))

    x = np.sqrt(2.0) * _gh_x
    out = np.empty(r.size)
    for lo in range(0, r.size, 8192):
        hi = lo + 8192
        rc = np.clip(r[lo:hi], -DEGENERATE_RHO, DEGENERATE_RHO)
        root = np.sqrt(1.0 - rc * rc)
        y1 = bi[lo:hi, None] + gi[lo:hi, None] * x
        g1 = gelu(y1) * _gh_w
        z2 = rc[:, None, None] * x[:, None] + root[:, None, None] * x[None, :]
        y2 = bj[lo:hi, None, None] + gj[lo:hi, None, None] * z2
        inner = gelu(y2) @ _gh_w
        main = np.einsum("pi,pi->p", g1, inner) / np.pi

        z_col = bj[lo:hi, None] + gj[lo:hi, None] * np.sign(r[lo:hi])[:, None] * x
        coll = (gelu(y1) * gelu(z_col)) @ _gh_w / np.sqrt(np.pi)
        out[lo:hi] = np.where(np.abs(r[lo:hi]) >= DEGENERATE_RHO, coll, main)
    return out.reshape(shape) if shape else float(out[0])


def _build_interaction_table(points=2049):
    grid = np.linspace(-1.0, 1.0, points)
    k0 = float(gelu_self_kernel(1.0, 0.0))
    vals = gelu_cross_kernel_exact(1.0, 0.0, 1.0, 0.0, grid) / k0
    return grid, vals


def gelu_interaction(rho):
    """Normalized zero-bias interaction I(rho) from a cached quadrature table."""
    global _interaction_table
    if _interaction_table is None:
        _interaction_table = _build_interaction_table()
    grid, vals = _interaction_table
    return np.interp(np.clip(np.asarray(rho, dtype=np.float64), -1.0, 1.0), grid, vals)


def gelu_cross_kernel_zero_bias(k_ii, k_jj, rho_hat):
    """Zero-bias approximation K(i,j) = I(rho) sqrt(Kii Kjj), the eq (84) pattern for GELU."""
    return gelu_interaction(rho_hat) * np.sqrt(np.maximum(np.asarray(k_ii) * np.asarray(k_jj), 0.0))


def gelu_boundary_identity():
    """k(1) and k'(1) of the standardized GELU kernel; PH-1 activations satisfy k(1) = k'(1)."""
    k1 = float(gelu_self_kernel(1.0, 0.0))
    x = np.sqrt(2.0) * _gh_x
    dpsi = ndtr(x) + x * _phi(x)
    k1p = float((dpsi * dpsi) @ _gh_w / np.sqrt(np.pi))
    return k1, k1p


@dataclass(frozen=True)
class ActivationKernels:
    """Kernel function bundle for one activation; ph1 marks positive homogeneity."""

    name: str
    ph1: bool
    fn: object
    self_kernel: object
    cross_exact: object
    interaction: object


def _relu_bundle():
    from . import kernels as k

    return ActivationKernels(
        name="relu",
        ph1=True,
        fn=lambda y: np.maximum(y, 0.0),
        self_kernel=k.self_kernel,
        cross_exact=k.cross_kernel_exact_batch,
        interaction=k.relu_interaction,
    )


def _gelu_bundle():
    return ActivationKernels(
        name="gelu",
        ph1=False,
        fn=gelu,
        self_kernel=gelu_self_kernel,
        cross_exact=gelu_cross_kernel_exact,
        interaction=gelu_interaction,
    )


_BUNDLES = {}


def get_kernels(name):
    if name not in ("relu", "gelu"):
        raise ValueError(f"unsupported activation: {name}")
    if name not in _BUNDLES:
        _BUNDLES[name] = _relu_bundle() if name == "relu" else _gelu_bundle()
    return _BUNDLES[name]
