"""BN-derived Gaussian surrogate for pre-activations. Paper Sec 4, App E.1."""

from dataclasses import dataclass, field

import numpy as np

BN_EPS = 1e-5


def effective_params(w_raw, gamma, beta, mu, var, eps=BN_EPS):
    """Absorb BN into effective input weights and bias, eq (1)."""
    w_raw = np.asarray(w_raw, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    scale = gamma / np.sqrt(var + eps)
    w_eff = scale[..., None] * w_raw
    b_eff = beta - scale * mu
    return w_eff, b_eff


def calibrated_params(w_raw, b_raw, mu_emp, sigma_emp):
    """Surrogate marginals for networks without BN, App E box: gamma=sigma, beta=mu+b."""
    gamma = np.asarray(sigma_emp, dtype=np.float64)
    beta = np.asarray(mu_emp, dtype=np.float64) + np.asarray(b_raw, dtype=np.float64)
    return np.asarray(w_raw, dtype=np.float64), gamma, beta


@dataclass
class LayerSurrogate:
    """Per-neuron effective parameters and surrogate marginals y_i ~ N(beta_i, gamma_i^2), eq (78)."""

    w_eff: np.ndarray  # (N, d_in)
    b: np.ndarray  # (N,)
    gamma: np.ndarray  # (N,) BN scale, |gamma| is the marginal std
    beta: np.ndarray  # (N,) BN shift, the marginal mean
    w_out: np.ndarray  # (N, d_out)

    def __post_init__(self):
        self.w_eff = np.atleast_2d(np.asarray(self.w_eff, dtype=np.float64))
        self.b = np.atleast_1d(np.asarray(self.b, dtype=np.float64))
        self.gamma = np.atleast_1d(np.asarray(self.gamma, dtype=np.float64))
        self.beta = np.atleast_1d(np.asarray(self.beta, dtype=np.float64))
        self.w_out = np.atleast_2d(np.asarray(self.w_out, dtype=np.float64))

    @property
    def n(self):
        return self.w_eff.shape[0]

    @property
    def w_aug(self):
        """Augmented effective weights [w_eff, b], Sec 7.1."""
        return np.concatenate([self.w_eff, self.b[:, None]], axis=1)

    @classmethod
    def from_bn(cls, w_raw, gamma, beta, mu, var, w_out, eps=BN_EPS):
        w_eff, b_eff = effective_params(w_raw, gamma, beta, mu, var, eps)
        return cls(w_eff=w_eff, b=b_eff, gamma=gamma, beta=beta, w_out=w_out)
