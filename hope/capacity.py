"""Functional capacity of neurons and layers. Paper Sec 5, App B.1."""

import numpy as np

from .kernels import self_kernel


def capacity(w_out, k_ii):
    """||f_i||_H = ||w_out||_2 * sqrt(K(i,i)), eq (75)."""
    w_out = np.atleast_2d(np.asarray(w_out, dtype=np.float64))
    return np.linalg.norm(w_out, axis=-1) * np.sqrt(np.maximum(k_ii, 0.0))


def layer_capacities(surrogate):
    """Per-neuron capacities and self-kernels for a LayerSurrogate."""
    k_ii = self_kernel(surrogate.gamma, surrogate.beta)
    return capacity(surrogate.w_out, k_ii), k_ii


def layer_capacity(surrogate):
    """Layer capacity E(Phi) as the L1 sum of neuron capacities, Lemma C.1."""
    caps, _ = layer_capacities(surrogate)
    return float(np.sum(caps))


def conv_input_vectors(weight):
    """Per-filter flattened input weights from a conv tensor [C_out, C_in, h, w], App B.1."""
    w = np.asarray(weight, dtype=np.float64)
    return w.reshape(w.shape[0], -1)


def conv_output_vectors(next_weight):
    """Per-channel flattened outgoing weights from the next conv tensor, App B.1."""
    w = np.asarray(next_weight, dtype=np.float64)
    return np.moveaxis(w, 1, 0).reshape(w.shape[1], -1)
