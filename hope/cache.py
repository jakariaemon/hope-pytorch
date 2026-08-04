"""Decoupled O(1) cache for greedy scans: only scalars a, b per pair. Paper App B.4."""

import numpy as np

from .activations import get_kernels
from .costs import E_REM_FLOOR, j_merge, j_prune
from .kernels import TINY, pairwise_warped_correlation, warped_correlation
from .parent import synthesize_parent


class LayerCache:
    """Per-layer pair geometry cache; costs are O(1) scalar arithmetic at scan time.

    slim mode keeps only the b matrix resident after init and recomputes pair geometry
    on demand, so wide layers (Whisper large) fit in memory.
    """

    def __init__(self, surrogate, kernel_mode="zero_bias", pair_dtype=None, slim=False):
        self.surrogate = surrogate
        self.kernel_mode = kernel_mode
        self.act = get_kernels(surrogate.activation)
        self.slim = slim
        n = surrogate.n
        self.active = np.ones(n, dtype=bool)
        w_aug = surrogate.w_aug
        self.g = w_aug @ w_aug.T
        self.o = surrogate.w_out @ surrogate.w_out.T
        self.rho_hat = pairwise_warped_correlation(surrogate.w_eff, surrogate.gamma)
        self.k_ii = self.act.self_kernel(surrogate.gamma, surrogate.beta)
        self.caps = np.sqrt(np.maximum(np.diag(self.o) * self.k_ii, 0.0))
        self.e_init = float(self.caps.sum())
        self.e_rem = self.e_init
        self.b_mat = np.zeros((n, n))
        if n > 1:
            ii, jj = np.triu_indices(n, k=1)
            b = self._pair_b(ii, jj)
            self.b_mat[ii, jj] = b
            self.b_mat[jj, ii] = b
        if pair_dtype is not None:
            self.b_mat = self.b_mat.astype(pair_dtype)
        self._tri = np.triu_indices(n, k=1) if n > 1 else (np.empty(0, dtype=np.int64),) * 2
        self._tri = (self._tri[0].astype(np.int32), self._tri[1].astype(np.int32))
        if slim:
            self._g_diag = np.diag(self.g).copy()
            self._o_diag = np.diag(self.o).copy()
            self._norms = np.maximum(np.linalg.norm(surrogate.w_eff, axis=1), TINY)
            self.g = self.o = self.rho_hat = None
        elif pair_dtype is not None:
            for name in ("g", "o", "rho_hat"):
                setattr(self, name, getattr(self, name).astype(pair_dtype))

    @property
    def n_live(self):
        return int(self.active.sum())

    def _pair_geometry(self, ii, jj):
        """(g11, g12, g22, o11, o12, o22, rho) for pair index arrays."""
        if not self.slim or self.g is not None:
            return (
                self.g[ii, ii], self.g[ii, jj], self.g[jj, jj],
                self.o[ii, ii], self.o[ii, jj], self.o[jj, jj],
                self.rho_hat[ii, jj],
            )
        surr = self.surrogate
        w_aug = surr.w_aug
        g12 = np.einsum("ij,ij->i", w_aug[ii], w_aug[jj])
        o12 = np.einsum("ij,ij->i", surr.w_out[ii], surr.w_out[jj])
        cos = np.einsum("ij,ij->i", surr.w_eff[ii], surr.w_eff[jj]) / (self._norms[ii] * self._norms[jj])
        rho = warped_correlation(cos, surr.gamma[ii], surr.gamma[jj], self._norms[ii], self._norms[jj])
        return (
            self._g_diag[ii], g12, self._g_diag[jj],
            self._o_diag[ii], o12, self._o_diag[jj], rho,
        )

    def pair_rho(self, i, j):
        return float(self._pair_geometry(np.array([i]), np.array([j]))[6][0])

    def _cross_batch(self, gamma_u, beta_u, gamma_k, beta_k, corr, k_uu, k_kk):
        if self.kernel_mode == "zero_bias":
            return self.act.interaction(corr) * np.sqrt(np.maximum(k_uu * k_kk, 0.0))
        return self.act.cross_exact(gamma_u, beta_u, gamma_k, beta_k, corr)

    def _objective_batch(self, c1, c2, ii, jj, rho, o11, o12, o22):
        surr = self.surrogate
        gi2 = surr.gamma[ii] ** 2
        gj2 = surr.gamma[jj] ** 2
        gi = np.abs(surr.gamma[ii])
        gj = np.abs(surr.gamma[jj])
        cov_ij = gi * gj * rho
        beta_u = c1 * surr.beta[ii] + c2 * surr.beta[jj]
        var_u = c1 * c1 * gi2 + c2 * c2 * gj2 + 2 * c1 * c2 * cov_ij
        gamma_u = np.sqrt(np.maximum(var_u, 0.0))
        corr_ui = np.clip((c1 * gi2 + c2 * cov_ij) / np.maximum(gamma_u * gi, TINY), -1.0, 1.0)
        corr_uj = np.clip((c2 * gj2 + c1 * cov_ij) / np.maximum(gamma_u * gj, TINY), -1.0, 1.0)
        k_uu = self.act.self_kernel(gamma_u, beta_u)
        k_ui = self._cross_batch(gamma_u, beta_u, gi, surr.beta[ii], corr_ui, k_uu, self.k_ii[ii])
        k_uj = self._cross_batch(gamma_u, beta_u, gj, surr.beta[jj], corr_uj, k_uu, self.k_ii[jj])
        z2 = k_ui * k_ui * o11 + 2 * k_ui * k_uj * o12 + k_uj * k_uj * o22
        z_norm = np.sqrt(np.maximum(z2, 0.0))
        obj = np.where(k_uu > TINY, z_norm / np.sqrt(np.maximum(k_uu, TINY)), 0.0)
        return obj, k_uu, z_norm

    def _pair_b(self, ii, jj):
        """Cached scalar b = <psi*, f_i + f_j> for pairs, eq (40) context, vectorized eq (14)."""
        g11, g12, g22, o11, o12, o22, rho = self._pair_geometry(ii, jj)
        g11 = np.maximum(g11, TINY)

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
        obj_pos, k_uu_pos, z_pos = self._objective_batch(q1, q2, ii, jj, rho, o11, o12, o22)
        obj_neg, k_uu_neg, z_neg = self._objective_batch(-q1, -q2, ii, jj, rho, o11, o12, o22)
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
        if self.n_live < 2:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty, np.empty(0), np.empty(0)
        # cached full-pair indices, filtered by the live mask; float32 is enough to rank
        keep = self.active[self._tri[0]] & self.active[self._tri[1]]
        ii = self._tri[0][keep].astype(np.int64)
        jj = self._tri[1][keep].astype(np.int64)
        # float32 ranking only where the b matrix is already float16 (wide layers)
        dt = np.float32 if self.b_mat.dtype == np.float16 else np.float64
        cap_i = self.caps[ii].astype(dt)
        cap_j = self.caps[jj].astype(dt)
        a = cap_i * cap_i + cap_j * cap_j
        b = self.b_mat[ii, jj].astype(dt)
        e_rem_pair = np.maximum(dt(self.e_rem) - cap_i - cap_j, dt(E_REM_FLOOR))
        s = (a + b * e_rem_pair) / np.maximum(2 * e_rem_pair + b, dt(TINY))
        j = j_merge(a, b, s, cap_i, cap_j, dt(self.e_rem), self.n_live)
        return ii, jj, j, s

    def synthesize(self, i, j):
        """JIT rank-2 synthesis for the winning pair only, App B.4."""
        return synthesize_parent(
            self.surrogate, i, j, self.pair_rho(i, j), self.e_rem, self.kernel_mode
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

        self.k_ii[i] = self.act.self_kernel(parent.gamma, parent.beta)
        self.caps[i] = parent.s

        if self.slim:
            w_aug_i = np.concatenate([parent.w_eff, [parent.b]])
            self._g_diag[i] = float(w_aug_i @ w_aug_i)
            self._o_diag[i] = float(parent.w_out @ parent.w_out)
            self._norms[i] = max(float(np.linalg.norm(parent.w_eff)), TINY)
        else:
            w_aug = surr.w_aug
            self.g[i, :] = w_aug @ w_aug[i]
            self.g[:, i] = self.g[i, :]
            self.o[i, :] = surr.w_out @ surr.w_out[i]
            self.o[:, i] = self.o[i, :]
            norms = np.maximum(np.linalg.norm(surr.w_eff, axis=1), TINY)
            cos = (surr.w_eff @ surr.w_eff[i]) / (norms * norms[i])
            row = warped_correlation(cos, surr.gamma, surr.gamma[i], norms, norms[i])
            row[i] = 1.0
            row[norms <= TINY] = 0.0
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
