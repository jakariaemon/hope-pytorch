"""Decoupled O(1) cache for greedy scans: only scalars a, b per pair. Paper App B.4."""

import numpy as np

from .costs import E_REM_FLOOR, j_merge, j_prune
from .kernels import (
    TINY,
    cross_kernel_exact_batch,
    pairwise_warped_correlation,
    relu_interaction,
    self_kernel,
    warped_correlation,
)
from .parent import synthesize_parent


class LayerCache:
    """Per-layer pair geometry cache; costs are O(1) scalar arithmetic at scan time."""

    def __init__(self, surrogate, kernel_mode="zero_bias"):
        self.surrogate = surrogate
        self.kernel_mode = kernel_mode
        n = surrogate.n
        self.active = np.ones(n, dtype=bool)
        w_aug = surrogate.w_aug
        self.g = w_aug @ w_aug.T
        self.o = surrogate.w_out @ surrogate.w_out.T
        self.rho_hat = pairwise_warped_correlation(surrogate.w_eff, surrogate.gamma)
        self.k_ii = self_kernel(surrogate.gamma, surrogate.beta)
        self.caps = np.sqrt(np.maximum(np.diag(self.o) * self.k_ii, 0.0))
        self.e_init = float(self.caps.sum())
        self.e_rem = self.e_init
        self.b_mat = np.zeros((n, n))
        if n > 1:
            ii, jj = np.triu_indices(n, k=1)
            b = self._pair_b(ii, jj)
            self.b_mat[ii, jj] = b
            self.b_mat[jj, ii] = b

    @property
    def n_live(self):
        return int(self.active.sum())

    def _cross_batch(self, gamma_u, beta_u, gamma_k, beta_k, corr, k_uu, k_kk):
        if self.kernel_mode == "zero_bias":
            return relu_interaction(corr) * np.sqrt(np.maximum(k_uu * k_kk, 0.0))
        return cross_kernel_exact_batch(gamma_u, beta_u, gamma_k, beta_k, corr)

    def _objective_batch(self, c1, c2, ii, jj, o11, o12, o22):
        surr = self.surrogate
        gi2 = surr.gamma[ii] ** 2
        gj2 = surr.gamma[jj] ** 2
        gi = np.abs(surr.gamma[ii])
        gj = np.abs(surr.gamma[jj])
        rho = self.rho_hat[ii, jj]
        cov_ij = gi * gj * rho
        beta_u = c1 * surr.beta[ii] + c2 * surr.beta[jj]
        var_u = c1 * c1 * gi2 + c2 * c2 * gj2 + 2 * c1 * c2 * cov_ij
        gamma_u = np.sqrt(np.maximum(var_u, 0.0))
        corr_ui = np.clip(
            (c1 * gi2 + c2 * cov_ij) / np.maximum(gamma_u * gi, TINY), -1.0, 1.0
        )
        corr_uj = np.clip(
            (c2 * gj2 + c1 * cov_ij) / np.maximum(gamma_u * gj, TINY), -1.0, 1.0
        )
        k_uu = self_kernel(gamma_u, beta_u)
        k_ui = self._cross_batch(gamma_u, beta_u, gi, surr.beta[ii], corr_ui, k_uu, self.k_ii[ii])
        k_uj = self._cross_batch(gamma_u, beta_u, gj, surr.beta[jj], corr_uj, k_uu, self.k_ii[jj])
        z2 = k_ui * k_ui * o11 + 2 * k_ui * k_uj * o12 + k_uj * k_uj * o22
        z_norm = np.sqrt(np.maximum(z2, 0.0))
        obj = np.where(k_uu > TINY, z_norm / np.sqrt(np.maximum(k_uu, TINY)), 0.0)
        return obj, k_uu, z_norm

    def _pair_b(self, ii, jj):
        """Cached scalar b = <psi*, f_i + f_j> for pairs, eq (40) context, vectorized eq (14)."""
        g11 = np.maximum(self.g[ii, ii], TINY)
        g12 = self.g[ii, jj]
        g22 = self.g[jj, jj]
        o11 = self.o[ii, ii]
        o12 = self.o[ii, jj]
        o22 = self.o[jj, jj]

        l11 = np.sqrt(g11)
        l21 = g12 / l11
        d = g22 - l21 * l21
        degen = d <= 1e-12 * np.maximum(g22, TINY)
        l22 = np.sqrt(np.maximum(d, TINY))

        c11 = l11 * l11 * o11 + 2 * l11 * l21 * o12 + l21 * l21 * o22
        c12 = l22 * (l11 * o12 + l21 * o22)
        c22 = l22 * l22 * o22
        lam = 0.5 * (c11 + c22) + np.sqrt(0.25 * (c11 - c22) ** 2 + c12 * c12)
        v1 = np.stack([c12, lam - c11])
        v2 = np.stack([lam - c22, c12])
        n1 = np.linalg.norm(v1, axis=0)
        n2 = np.linalg.norm(v2, axis=0)
        use1 = n1 >= n2
        p1 = np.where(use1, v1[0], v2[0])
        p2 = np.where(use1, v1[1], v2[1])
        pn = np.maximum(np.where(use1, n1, n2), TINY)
        p1, p2 = p1 / pn, p2 / pn

        q1 = p1 / l11 - p2 * l21 / (l11 * l22)
        q2 = p2 / l22
        q1 = np.where(degen, 1.0 / l11, q1)
        q2 = np.where(degen, 0.0, q2)

        # eq (11): sign check in the exact non-linearized objective
        obj_pos, k_uu_pos, z_pos = self._objective_batch(q1, q2, ii, jj, o11, o12, o22)
        obj_neg, k_uu_neg, z_neg = self._objective_batch(-q1, -q2, ii, jj, o11, o12, o22)
        neg = obj_neg > obj_pos
        k_uu = np.where(neg, k_uu_neg, k_uu_pos)
        z_norm = np.where(neg, z_neg, z_pos)
        return np.where(k_uu > TINY, z_norm / np.sqrt(np.maximum(k_uu, TINY)), 0.0)

    def prune_costs(self):
        """J_prune for all active neurons, eq (6)."""
        act = np.where(self.active)[0]
        return act, j_prune(self.caps[act], self.e_rem, self.n_live)

    def merge_costs(self):
        """J_merge and s* for all active pairs from cached scalars, eq (6), (40)."""
        act = np.where(self.active)[0]
        if act.size < 2:
            return act[:0], act[:0], np.empty(0), np.empty(0)
        ii, jj = np.triu_indices(act.size, k=1)
        ii, jj = act[ii], act[jj]
        cap_i, cap_j = self.caps[ii], self.caps[jj]
        a = cap_i * cap_i + cap_j * cap_j
        b = self.b_mat[ii, jj]
        e_rem_pair = np.maximum(self.e_rem - cap_i - cap_j, E_REM_FLOOR)
        s = (a + b * e_rem_pair) / np.maximum(2 * e_rem_pair + b, TINY)
        j = j_merge(a, b, s, cap_i, cap_j, self.e_rem, self.n_live)
        return ii, jj, j, s

    def synthesize(self, i, j):
        """JIT rank-2 synthesis for the winning pair only, App B.4."""
        return synthesize_parent(
            self.surrogate, i, j, self.rho_hat[i, j], self.e_rem, self.kernel_mode
        )

    def apply_prune(self, i):
        """Alg 2 PRUNE branch."""
        self.e_rem = max(self.e_rem - self.caps[i], E_REM_FLOOR)
        self.active[i] = False

    def apply_merge(self, i, j, parent):
        """Alg 2 MERGE branch: child i is the vessel, child j is purged; O(N) local update."""
        self.e_rem = max(self.e_rem - (self.caps[i] + self.caps[j] - parent.s), E_REM_FLOOR)
        self.active[j] = False

        surr = self.surrogate
        surr.w_eff[i] = parent.w_eff
        surr.b[i] = parent.b
        surr.gamma[i] = parent.gamma
        surr.beta[i] = parent.beta
        surr.w_out[i] = parent.w_out

        w_aug = surr.w_aug
        self.g[i, :] = w_aug @ w_aug[i]
        self.g[:, i] = self.g[i, :]
        self.o[i, :] = surr.w_out @ surr.w_out[i]
        self.o[:, i] = self.o[i, :]
        self.k_ii[i] = self_kernel(parent.gamma, parent.beta)
        self.caps[i] = parent.s

        norms = np.maximum(np.linalg.norm(surr.w_eff, axis=1), TINY)
        cos = (surr.w_eff @ surr.w_eff[i]) / (norms * norms[i])
        row = warped_correlation(cos, surr.gamma, surr.gamma[i], norms, norms[i])
        row[i] = 1.0
        dead = norms <= TINY
        row[dead] = 0.0
        self.rho_hat[i, :] = row
        self.rho_hat[:, i] = row

        others = np.where(self.active & (np.arange(surr.n) != i))[0]
        if others.size:
            b_row = self._pair_b(np.full(others.size, i), others)
            self.b_mat[i, others] = b_row
            self.b_mat[others, i] = b_row

    def clear(self):
        """Alg 2 EVICT branch: collapse the layer."""
        self.active[:] = False
        self.e_rem = E_REM_FLOOR
