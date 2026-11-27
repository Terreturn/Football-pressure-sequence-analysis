"""Measure local pressing-team spacing and directional carrier space without changing the existing network."""

from __future__ import annotations

import math

import numpy as np

from shapely.geometry import Point, box

NAMES = (
    "local_defender_spread_m",
    "nearest_presser_support_m",
    "carrier_forward_space_fraction",
)

PITCH = box(0, 0, 105, 68)

TOLERANCE_M = 1e-8


def ordered_pressers(xy, carrier, defenders):
    """Return presser indices sorted by squared carrier distance then x/y, from finite meter coordinates."""
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Expected finite two-dimensional meter coordinates")
    if len(set(defenders)) != len(defenders) or carrier in defenders:
        raise ValueError("Invalid presser indices")
    return sorted(
        defenders,
        key=lambda i: (
            float(np.sum((xy[i] - xy[carrier]) ** 2)),
            float(xy[i, 0]),
            float(xy[i, 1]),
        ),
    )


def local_spread(xy, indices):
    """Return mean distance in meters from the centroid of exactly five selected points; fewer gives NaN."""
    if len(indices) < 5:
        return math.nan
    points = np.asarray(xy, float)[indices[:5]]
    return float(np.linalg.norm(points - points.mean(axis=0), axis=1).mean())


def support_spacing(xy, nearest, edges):
    """Return the median incident shape-edge length in meters, or NaN for unavailable/empty support."""
    if nearest is None or edges is None:
        return math.nan
    lengths = [
        float(np.linalg.norm(xy[a] - xy[b])) for a, b in edges if nearest in (a, b)
    ]
    return float(np.median(lengths)) if lengths else math.nan


def forward_fraction(cell, carrier_x, visible):
    """Return forward area divided by full clipped carrier-cell area, dimensionless; invalid geometry gives NaN."""
    if (
        visible is None
        or not visible.is_valid
        or visible.is_empty
        or visible.area <= 0
        or cell is None
        or not cell.is_valid
        or cell.is_empty
        or not math.isfinite(cell.area)
        or cell.area <= 0
    ):
        return math.nan
    return float(cell.intersection(box(carrier_x, 0, 105, 68)).area / cell.area)


def visibility_diagnostics(network, ordered):
    """Return camera coverage/contact flags and meter radius; NaN means the diagnostic is unavailable."""
    xy, c, visible = network["xy"], network["carrier"], network["visible"]
    radius = (
        float(np.linalg.norm(xy[ordered[4]] - xy[c])) if len(ordered) >= 5 else math.nan
    )
    result = {
        "fifth_presser_distance_m": radius,
        "five_presser_disk_observed": math.nan,
        "carrier_cell_camera_contact": math.nan,
    }
    if visible is None or not visible.is_valid or visible.is_empty or visible.area <= 0:
        return result
    hidden = PITCH.difference(visible)
    if math.isfinite(radius):
        result["five_presser_disk_observed"] = float(
            hidden.is_empty or Point(xy[c]).distance(hidden) >= radius - TOLERANCE_M
        )
    cell = network["areas"][c]
    if cell is not None and cell.is_valid and not cell.is_empty and cell.area > 0:
        camera_boundary = visible.boundary.difference(
            PITCH.boundary.buffer(TOLERANCE_M)
        )
        result["carrier_cell_camera_contact"] = float(
            not camera_boundary.is_empty
            and cell.distance(camera_boundary) <= TOLERANCE_M
        )
    return result


def spatial_features(network, graphs):
    """Return three candidate values and visibility diagnostics from meter network coordinates and legacy shape edges."""
    xy, c = network["xy"], network["carrier"]
    ordered = ordered_pressers(xy, c, network["defenders"])
    values = dict(
        zip(
            NAMES,
            (
                local_spread(xy, ordered),
                support_spacing(
                    xy, ordered[0] if ordered else None, graphs["pressing"]
                ),
                forward_fraction(
                    network["areas"][c], float(xy[c, 0]), network["visible"]
                ),
            ),
        )
    )
    return values, visibility_diagnostics(network, ordered)
