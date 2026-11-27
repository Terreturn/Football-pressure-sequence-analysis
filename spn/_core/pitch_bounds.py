"""Test player positions against pitch and visibility boundaries."""

from __future__ import annotations

import numpy as np


def canonical_meters(provider, flip=False):
    """Independently rescale provider points to meters in the frozen builder's multiply-then-divide operation order.

    Collapsing the scale to .875/.85 is algebraically equal but can move a strict
    half-meter collision comparison by a floating-point rounding unit. The coordinate
    representation is an input convention; polygon and stencil checks remain separate.
    """
    points = np.asarray(provider, dtype=float)
    if flip:
        points = np.array([120.0, 80.0]) - points
    return np.divide(np.multiply(points, [105.0, 68.0]), [120.0, 80.0])


def contains_closed(vertices, points):
    """Return closed-polygon containment using ray crossings and segment membership for points in meters."""
    vertices, points = np.asarray(vertices, float), np.atleast_2d(points).astype(float)
    inside = np.zeros(len(points), bool)
    boundary = np.zeros(len(points), bool)
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        edge = b - a
        rel = points - a
        cross = edge[0] * rel[:, 1] - edge[1] * rel[:, 0]
        dot = rel @ edge
        if np.dot(edge, edge) == 0:
            boundary |= np.linalg.norm(rel, axis=1) <= 1e-12
        else:
            boundary |= (
                (np.abs(cross) <= 1e-12) & (dot >= 0) & (dot <= np.dot(edge, edge))
            )
        crosses = (a[1] > points[:, 1]) != (b[1] > points[:, 1])
        if b[1] != a[1]:
            intersection_x = a[0] + (points[:, 1] - a[1]) * (b[0] - a[0]) / (
                b[1] - a[1]
            )
            inside ^= crosses & (points[:, 0] < intersection_x)
    return inside | boundary
