"""Optimal parent neuron synthesis and BN parameter recovery. Paper Sec 7, App C.2, D, B.5."""

from dataclasses import dataclass

import numpy as np

from .kernels import TINY, cross_kernel, self_kernel
from .surrogate import BN_EPS

E_REM_FLOOR = 1e-12


@dataclass
class Parent:
    """Physical parameters and Hilbert-space scalars of a merged parent neuron."""

    w_eff: np.ndarray
    b: float
    w_raw: np.ndarray
    w_out: np.ndarray
    gamma: float
    beta: float
    mu: float
    sigma2: float
    c1: float
    c2: float
    s: float
    a: float
    b_scalar: float
    rho_hat: float
    active: bool

    @property
    def capacity(self):
        return self.s

    @property
    def distortion(self):
        """Predicted merge distortion D, eq (54): D^2 = 2s^2 - 2bs + a."""
        return float(np.sqrt(max(2 * self.s**2 - 2 * self.b_scalar * self.s + self.a, 0.0)))


def _principal_eigvec_2x2(a, b, c):
    """Principal eigenvector of [[a, b], [b, c]]."""
    lam = 0.5 * (a + c) + np.sqrt(0.25 * (a - c) ** 2 + b * b)
    v1 = np.array([b, lam - a])
    v2 = np.array([lam - c, b])
    v = v1 if np.linalg.norm(v1) >= np.linalg.norm(v2) else v2
    n = np.linalg.norm(v)
    if n < TINY:
        return np.array([1.0, 0.0])
    return v / n


def _direction_coefficients(g11, g12, g22, o11, o12, o22):
    """Coefficients q of the optimal direction u = q1*w_i + q2*w_j, eq (14)."""
    l11 = np.sqrt(g11)
    l21 = g12 / l11
    d = g22 - l21 * l21
    if d <= 1e-12 * g22:
        return np.array([1.0 / l11, 0.0])
    l22 = np.sqrt(d)
    lmat = np.array([[l11, 0.0], [l21, l22]])
    cmat = lmat.T @ np.array([[o11, o12], [o12, o22]]) @ lmat
    p = _principal_eigvec_2x2(cmat[0, 0], cmat[0, 1], cmat[1, 1])
    return np.linalg.solve(lmat.T, p)


def _mixture_stats(c1, c2, gi, bi, gj, bj, rho_hat):
    """Marginal and pairwise stats of y_u = c1*y_i + c2*y_j under the pairwise surrogate, eq (82)."""
    gi_a, gj_a = abs(gi), abs(gj)
    cov_ij = gi_a * gj_a * rho_hat
    beta_u = c1 * bi + c2 * bj
    var_u = c1 * c1 * gi * gi + c2 * c2 * gj * gj + 2 * c1 * c2 * cov_ij
    gamma_u = np.sqrt(max(var_u, 0.0))
    cov_ui = c1 * gi * gi + c2 * cov_ij
    cov_uj = c2 * gj * gj + c1 * cov_ij
    denom_i = max(gamma_u * gi_a, TINY)
    denom_j = max(gamma_u * gj_a, TINY)
    corr_ui = float(np.clip(cov_ui / denom_i, -1.0, 1.0))
    corr_uj = float(np.clip(cov_uj / denom_j, -1.0, 1.0))
    return beta_u, gamma_u, corr_ui, corr_uj


def _objective(c1, c2, ni, nj, rho_hat, k_ii, k_jj, o11, o12, o22, kernel_mode):
    """Exact sign-check objective ||z(u)|| / sqrt(K(u,u)), eq (10)-(11)."""
    beta_u, gamma_u, corr_ui, corr_uj = _mixture_stats(
        c1, c2, ni["gamma"], ni["beta"], nj["gamma"], nj["beta"], rho_hat
    )
    k_uu = float(self_kernel(gamma_u, beta_u))
    if k_uu <= TINY:
        return 0.0, k_uu, 0.0, 0.0, 0.0
    k_ui = float(
        cross_kernel(gamma_u, beta_u, ni["gamma"], ni["beta"], corr_ui, k_uu, k_ii, kernel_mode)
    )
    k_uj = float(
        cross_kernel(gamma_u, beta_u, nj["gamma"], nj["beta"], corr_uj, k_uu, k_jj, kernel_mode)
    )
    z2 = k_ui * k_ui * o11 + 2 * k_ui * k_uj * o12 + k_uj * k_uj * o22
    z_norm = np.sqrt(max(z2, 0.0))
    return z_norm / np.sqrt(k_uu), k_uu, k_ui, k_uj, z_norm


def synthesize_parent(layer, i, j, rho_hat_ij, e_layer, kernel_mode="zero_bias", bn_eps=BN_EPS):
    """Full JIT parent synthesis for the pair (i, j), eq (12)-(18)."""
    w_aug = layer.w_aug
    wi, wj = w_aug[i], w_aug[j]
    oi, oj = layer.w_out[i], layer.w_out[j]
    ni = {"gamma": float(layer.gamma[i]), "beta": float(layer.beta[i])}
    nj = {"gamma": float(layer.gamma[j]), "beta": float(layer.beta[j])}
    rho_hat_ij = float(rho_hat_ij)

    g11, g12, g22 = float(wi @ wi), float(wi @ wj), float(wj @ wj)
    o11, o12, o22 = float(oi @ oi), float(oi @ oj), float(oj @ oj)

    k_ii = float(self_kernel(ni["gamma"], ni["beta"]))
    k_jj = float(self_kernel(nj["gamma"], nj["beta"]))
    cap_i = np.sqrt(o11 * k_ii)
    cap_j = np.sqrt(o22 * k_jj)

    q = _direction_coefficients(g11, g12, g22, o11, o12, o22)

    # eq (11): resolve the eigenvector sign ambiguity with the exact objective
    obj_pos = _objective(q[0], q[1], ni, nj, rho_hat_ij, k_ii, k_jj, o11, o12, o22, kernel_mode)
    obj_neg = _objective(-q[0], -q[1], ni, nj, rho_hat_ij, k_ii, k_jj, o11, o12, o22, kernel_mode)
    if obj_neg[0] > obj_pos[0]:
        q = -q
        obj = obj_neg
    else:
        obj = obj_pos
    _, k_uu, k_ui, k_uj, z_norm = obj

    a = float(cap_i**2 + cap_j**2)
    b_scalar = z_norm / np.sqrt(k_uu) if k_uu > TINY else 0.0
    e_rem = max(float(e_layer) - cap_i - cap_j, E_REM_FLOOR)

    # eq (40): optimal scale from cached scalars and live residual capacity
    s = (a + b_scalar * e_rem) / (2 * e_rem + b_scalar) if (2 * e_rem + b_scalar) > TINY else 0.0

    if z_norm > TINY:
        v = (k_ui * oi + k_uj * oj) / z_norm
    else:
        v = oi / max(np.sqrt(o11), TINY)

    # eq (15): input/output scale split preserving the subspace Frobenius ratio
    w_in_frob = np.sqrt(g11 + g22)
    w_out_frob = np.sqrt(o11 + o22)
    r_f = w_in_frob / max(w_out_frob, TINY)
    k_quarter = max(k_uu, TINY) ** 0.25
    scale_in = np.sqrt(max(s, 0.0) * r_f) / k_quarter
    scale_out = np.sqrt(max(s, 0.0) / r_f) / k_quarter

    c1, c2 = scale_in * q[0], scale_in * q[1]
    w_aug_p = c1 * wi + c2 * wj
    w_eff_p, b_p = w_aug_p[:-1], float(w_aug_p[-1])
    w_out_p = scale_out * v

    # eq (17)-(18) with the App B.4 warning: BN variance uses rho_hat, never raw cosine
    beta_p = c1 * ni["beta"] + c2 * nj["beta"]
    var_p = (
        c1 * c1 * ni["gamma"] ** 2
        + c2 * c2 * nj["gamma"] ** 2
        + 2 * c1 * c2 * abs(ni["gamma"]) * abs(nj["gamma"]) * rho_hat_ij
    )
    gamma_p = float(np.sqrt(max(var_p, 0.0)))
    mu_p = beta_p - b_p
    sigma2_p = max(gamma_p**2 - bn_eps, 0.0)
    active = gamma_p**2 >= bn_eps
    w_raw_p = w_eff_p.copy() if active else np.zeros_like(w_eff_p)

    return Parent(
        w_eff=w_eff_p,
        b=b_p,
        w_raw=w_raw_p,
        w_out=w_out_p,
        gamma=gamma_p,
        beta=float(beta_p),
        mu=float(mu_p),
        sigma2=float(sigma2_p),
        c1=float(c1),
        c2=float(c2),
        s=float(s),
        a=a,
        b_scalar=float(b_scalar),
        rho_hat=rho_hat_ij,
        active=active,
    )


def physical_preactivation(parent, x, bn_eps=BN_EPS):
    """Deployed BN forward y_p = gamma/sqrt(sigma^2+eps) * (w_raw^T x - mu) + beta, eq (71)."""
    z = x @ parent.w_raw - parent.mu
    return parent.gamma / np.sqrt(parent.sigma2 + bn_eps) * z + parent.beta
