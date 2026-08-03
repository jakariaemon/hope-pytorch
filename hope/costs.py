"""Distortion costs and parameter footprints. Paper Sec 6, 8, 9, App B.2."""

import numpy as np

E_REM_FLOOR = 1e-12


def j_prune(cap, e_layer, n_live, floor=E_REM_FLOOR):
    """Pruning cost, eq (6)."""
    e_b = np.maximum(e_layer - cap, floor)
    return n_live * cap / e_b


def merge_distortion(a, b, s):
    """Projection distance D for a merge, eq (54)."""
    return np.sqrt(np.maximum(2 * s * s - 2 * b * s + a, 0.0))


def j_merge(a, b, s, cap_i, cap_j, e_layer, n_live, floor=E_REM_FLOOR):
    """Merging cost, eq (6)."""
    d = merge_distortion(a, b, s)
    e_b = np.maximum(e_layer - cap_i - cap_j + s, floor)
    return n_live * d / e_b


def e_identity(gamma, beta):
    """Parallel survival capacity of the skip connection, eq (96)."""
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    return float(np.sum(np.sqrt(gamma * gamma + beta * beta)))


def j_evict(layer_stats, e_ident, floor=E_REM_FLOOR):
    """Block eviction cost, eq (20): layer_stats is [(n_active, e_active), ...]."""
    total = sum(n * e for n, e in layer_stats)
    return total / max(e_ident, floor)


def dp_conv_filter(c_in, kh, kw, out_slice_numel):
    """Static footprint of one conv filter, App B.2.1: input weights, outgoing slice, 4 BN params."""
    return kh * kw * c_in + out_slice_numel + 4


def dp_evict(w1_numel, w2_numel, bn_channels):
    """Static footprint of block eviction, eq (37): W3 weights excluded to avoid double counting."""
    return w1_numel + w2_numel + 4 * bn_channels


def distortion_rate(j, dp_init):
    """DR = J / dP_init, eq (23)."""
    return j / max(dp_init, 1)
