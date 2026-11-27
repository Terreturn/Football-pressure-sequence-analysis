"""Apply the fitted probability mapping to pressing sequence scores."""

from __future__ import annotations


import numpy as np


from scipy.special import softmax

ACTIVE = {
    "identity": (),
    "temperature": (0,),
    "offsets": (1, 2),
    "offsets_temperature": (0, 1, 2),
}

SCALE_FLOOR = 1e-8


def probability_inputs(probabilities):
    """Accept N by 3 probabilities in failure/neither/success order; return relative log probabilities and clipped-entry count."""
    p = np.asarray(probabilities, float)
    if (
        p.ndim != 2
        or p.shape[1] != 3
        or not len(p)
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or np.any(p > 1)
        or not np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("Invalid three-class probability matrix")
    logs = np.log(np.clip(p, 1e-12, 1))
    logs -= logs[:, [1]]
    return logs, int(np.count_nonzero(p < 1e-12))


def expand_parameters(theta, mapping):
    """Return shared scale and failure/success offsets from a dimensionless free-parameter vector; neither offset stays zero."""
    if mapping not in ACTIVE:
        raise ValueError("Unknown probability mapping")
    theta = np.asarray(theta, float)
    if theta.shape != (len(ACTIVE[mapping]),) or not np.isfinite(theta).all():
        raise ValueError("Invalid mapping parameters")
    full = np.array([1.0, 0.0, 0.0])
    full[list(ACTIVE[mapping])] = theta
    if full[0] < SCALE_FLOOR:
        raise ValueError("Nonpositive probability scale")
    return full


def map_logs(logs, theta, mapping):
    """Map relative log probabilities to N by 3 normalized probabilities; inputs and outputs are dimensionless."""
    a, failure, success = expand_parameters(theta, mapping)
    return softmax(
        a * np.asarray(logs, float) + np.array([failure, 0.0, success]), axis=1
    )


def apply_mapping(probabilities, fitted):
    """Return mapped probabilities; identity retains the original dtype and entries exactly."""
    logs, _ = probability_inputs(probabilities)
    if fitted["mapping"] == "identity":
        return np.asarray(probabilities).copy()
    return map_logs(logs, fitted["theta"], fitted["mapping"])
