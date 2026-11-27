"""Sample a reproducible set of nearby pressing positions for an observed event."""

from __future__ import annotations

from hashlib import sha256

import numpy as np

import pandas as pd

from scipy.stats import qmc


def event_seed(match_id, event_id):
    """Map two unitless stable IDs to the specified 32-bit Sobol scramble seed, independent of outcomes and row order."""
    return int(
        sha256(f"epl-sobol-area-v1|{match_id}|{event_id}".encode()).hexdigest()[:8], 16
    )


def transform(unit, origin, carrier):
    """Map unit-square fractions to radius/CCW angle/target-meter arrays; theta zero points from origin to carrier."""
    unit, origin, carrier = (
        np.asarray(unit, float),
        np.asarray(origin, float),
        np.asarray(carrier, float),
    )
    if (
        unit.ndim != 2
        or unit.shape[1] != 2
        or origin.shape != (2,)
        or carrier.shape != (2,)
        or not np.isfinite(unit).all()
        or not np.isfinite(origin).all()
        or not np.isfinite(carrier).all()
        or np.any((unit < 0) | (unit > 1))
    ):
        raise ValueError("Invalid sampling coordinates")
    delta = carrier - origin
    distance = np.linalg.norm(delta)
    if distance == 0:
        raise ValueError("undefined_carrier_bearing")
    toward = delta / distance
    across = np.array([-toward[1], toward[0]])
    radius = np.sqrt(0.25 + 3.75 * unit[:, 0])
    theta = 2 * np.pi * unit[:, 1]
    target = origin + radius[:, None] * (
        np.cos(theta)[:, None] * toward + np.sin(theta)[:, None] * across
    )
    return radius, theta, target


def candidates(match_id, event_id, origin, carrier, diagnostic=False):
    """Return 64 primary or 128 nested candidate records with meter coordinates and radian angles; no geometry filtering."""
    seed = event_seed(match_id, event_id)
    unit = qmc.Sobol(d=2, scramble=True, seed=seed).random_base2(7 if diagnostic else 6)
    radius, theta, target = transform(unit, origin, carrier)
    return pd.DataFrame(
        {
            "sample_id": np.arange(len(unit)),
            "seed": seed,
            "u": unit[:, 0],
            "v": unit[:, 1],
            "radius_m": radius,
            "theta_rad": theta,
            "target_x": target[:, 0],
            "target_y": target[:, 1],
            "dx_m": target[:, 0] - origin[0],
            "dy_m": target[:, 1] - origin[1],
            "primary": np.arange(len(unit)) < 64,
            "octant": np.minimum(7, np.floor(theta / (np.pi / 4)).astype(int)),
            "radial_band": np.where(
                radius < 1, "0.5–1", np.where(radius < 1.5, "1–1.5", "1.5–2")
            ),
        }
    )
