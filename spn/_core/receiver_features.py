"""Join passing availability to receiver pressure and space in the existing network.

These two dimensionless geometric features add receiver-level associations to
the existing summaries. Their values are not outcome or pass probabilities."""

from __future__ import annotations

import math

import numpy as np

NAMES = ("best_forward_receiver_relief", "forward_pass_space_share")


def combine_receivers(passing, pressure, area_share, forward):
    """Return two unitless scores from receiver weights, defender-by-receiver pressures, area fractions and a forward mask.

    NaN area fractions signal unavailable geometry; passing and pressure must
    be finite in [0,1]. Empty lists represent no visible receivers, not missing
    input. Geometry availability with no receivers is handled by the wrapper.
    """
    passing, pressure = np.asarray(passing, float), np.asarray(pressure, float)
    area_share, forward = np.asarray(area_share, float), np.asarray(forward)
    n = len(passing)
    if (
        passing.ndim != 1
        or area_share.shape != (n,)
        or forward.shape != (n,)
        or forward.dtype != bool
    ):
        raise ValueError("Receiver vectors or Boolean forward mask differ")
    if pressure.ndim != 2 or pressure.shape[1] != n:
        raise ValueError("Pressure matrix does not match receivers")
    if any(
        not np.isfinite(a).all() or (a < 0).any() or (a > 1).any()
        for a in (passing, pressure)
    ):
        raise ValueError("Passing and pressure weights must lie in [0,1]")
    finite = np.isfinite(area_share)
    if (
        np.isinf(area_share).any()
        or (area_share[finite] < 0).any()
        or (area_share[finite] > 1 + 1e-10).any()
    ):
        raise ValueError("Invalid area fraction")
    freedom = np.prod(1 - pressure, axis=0)
    relief = float(np.max((passing * freedom)[forward], initial=0.0))
    space = float(np.sum((passing * area_share)[forward]))
    return dict(zip(NAMES, (relief, space)))


def receiver_features(network):
    """Return two unitless features from an audited network with meter positions and square-meter cells; never mutate it."""
    receivers, carrier = network["receivers"], network["carrier"]
    edges = {edge["dst"]: edge["w"] for edge in network["passes"]}
    if len(edges) != len(network["passes"]) or set(edges) != set(receivers):
        raise ValueError("Passing edges do not match receivers exactly")
    pressure = np.asarray(network["pressure_weights"], float)
    if pressure.shape != (len(network["defenders"]), 1 + len(receivers)):
        raise ValueError("Pressure target order is not carrier then receivers")
    visible = network["visible"]
    area_valid = (
        visible is not None
        and not visible.is_empty
        and math.isfinite(visible.area)
        and visible.area > 0
        and network["areas"][carrier] is not None
    )
    shares = [
        network["areas"][j].area / visible.area
        if area_valid and network["areas"][j] is not None
        else math.nan
        for j in receivers
    ]
    forward = np.array(
        [network["xy"][j, 0] > network["xy"][carrier, 0] for j in receivers], bool
    )
    result = combine_receivers(
        [edges[j] for j in receivers], pressure[:, 1:], shares, forward
    )
    if not area_valid:
        result[NAMES[1]] = math.nan
    return result
