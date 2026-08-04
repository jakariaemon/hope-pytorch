"""Empirical audit of Lemma C.3: E(t) >= E(Phi_b) along the straight-line merge path."""

import numpy as np

from .costs import E_REM_FLOOR
from .parent import cross_scalar


def audit_merge_path(cache, i, j, parent, steps=20):
    """Numerically integrate E(t) for the executed merge and count bound violations, eq (45)."""
    cap_i, cap_j = cache.caps[i], cache.caps[j]
    s = parent.s
    e_rem_pair = max(cache.e_rem - cap_i - cap_j, E_REM_FLOOR)
    e_terminal = e_rem_pair + s

    surr = cache.surrogate
    k_pp = float(cache.act.self_kernel(parent.gamma, parent.beta))
    inner = {}
    for name, k in (("i", i), ("j", j)):
        corr = parent.corr_i if name == "i" else parent.corr_j
        k_pk = cross_scalar(
            cache.act,
            cache.kernel_mode,
            parent.gamma,
            parent.beta,
            surr.gamma[k],
            surr.beta[k],
            corr,
            k_pp,
            cache.k_ii[k],
        )
        inner[name] = k_pk * float(surr.w_out[k] @ parent.w_out)

    ts = np.linspace(0.0, 1.0, steps)
    e_path = np.empty(steps)
    for idx, t in enumerate(ts):
        ni = (1 - t) ** 2 * cap_i**2 + t * t * s * s + 2 * t * (1 - t) * inner["i"]
        nj = (1 - t) ** 2 * cap_j**2 + t * t * s * s + 2 * t * (1 - t) * inner["j"]
        e_path[idx] = e_rem_pair + np.sqrt(max(ni, 0.0)) + np.sqrt(max(nj, 0.0))

    tol = 1e-9 * max(e_terminal, 1.0)
    margins = e_path - e_terminal
    return {
        "rho_ij": float(cache.pair_rho(i, j)),
        "violations": int(np.sum(margins < -tol)),
        "min_margin": float(margins.min()),
        "steps": steps,
    }
