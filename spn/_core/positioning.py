"""Move one visible presser at a time and rebuild fixed-onset model inputs.

Coordinates are metric rescalings of the provider grid. Context and the observed
carrier remain fixed, so differences measure only the frozen model's response
to a separate positional change, not a joint or causal intervention."""

from __future__ import annotations

import math

import numpy as np

from shapely.geometry import Point, Polygon

from spn._core.sequence_context import context_features, sequence_incoming

from spn._core.networks import xy_m

RADII = (1.25, 2.5)


def strict_visible(frame, flip):
    """Return the recorded valid Polygon in meters; raise on unavailable/invalid geometry, without repair."""
    raw = np.asarray(frame.get("visible_area") or [], dtype=float)
    if raw.size < 6 or raw.size % 2 or not np.isfinite(raw).all():
        raise ValueError("missing_or_nonfinite_visible_polygon")
    polygon = Polygon([xy_m(p, flip) for p in raw.reshape(-1, 2)])
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise ValueError("invalid_visible_polygon")
    return polygon


def physical_players(frame, flip):
    """Return located raw-row indices and physical metric positions, retaining keepers and the frame actor."""
    indices = [
        i
        for i, p in enumerate(frame.get("freeze_frame") or [])
        if p.get("location") is not None
    ]
    xy = np.array([xy_m(frame["freeze_frame"][i]["location"], flip) for i in indices])
    if not len(xy) or xy.shape != (len(indices), 2) or not np.isfinite(xy).all():
        raise ValueError("invalid_physical_positions")
    return indices, xy


def stencil(origin, carrier):
    """Return sixteen (radius m, CCW direction index, target xy m) tuples; reject undefined zero-distance bearings."""
    delta = np.asarray(carrier, float) - origin
    distance = float(np.linalg.norm(delta))
    if not np.isfinite(distance) or distance == 0:
        raise ValueError("undefined_carrier_bearing")
    toward = delta / distance
    perpendicular = np.array([-toward[1], toward[0]])
    points = [
        (
            radius,
            direction,
            np.asarray(origin)
            + radius
            * (
                math.cos(direction * math.pi / 4) * toward
                + math.sin(direction * math.pi / 4) * perpendicular
            ),
        )
        for radius in RADII
        for direction in range(8)
    ]
    if len({tuple(p) for _, _, p in points}) != 16:
        raise ValueError("non_distinct_stencil")
    for radius, direction, target in points:
        if abs(np.linalg.norm(target - origin) - radius) > 1e-6:
            raise ValueError("wrong_stencil_radius")
        if (
            direction == 0
            and np.linalg.norm((target - origin) / radius - toward) > 1e-6
        ):
            raise ValueError("wrong_stencil_bearing")
    return points


def move_reason(target, visible, physical, node_index):
    """Return first invalidity reason, or valid, for meter positions; collision is strictly less than .5 m."""
    x, y = target
    if not (0 <= x <= 105 and 0 <= y <= 68):
        return "pitch"
    if not visible.covers(Point(float(x), float(y))):
        return "visibility"
    others = np.delete(physical, node_index, axis=0)
    if np.any(np.linalg.norm(others - target, axis=1) < 0.5):
        return "collision"
    return "valid"


def moved_frame(frame, raw_index, target, flip):
    """Copy only the moved raw player; return the frame and metric round-trip error, leaving all other records identical."""
    provider = np.asarray(target, float) / np.array([0.875, 0.85])
    if flip:
        provider = np.array([120.0, 80.0]) - provider
    players = list(frame["freeze_frame"])
    players[raw_index] = {**players[raw_index], "location": provider.tolist()}
    altered = {**frame, "freeze_frame": players}
    error = float(np.linalg.norm(xy_m(provider, flip) - target))
    if error > 1e-6:
        raise ValueError("movement_roundtrip_failed")
    return altered, error


def fixed_context(anchor, event, by_id, frames, new_convention):
    """Return unchanged incoming/context/convention values and preceding static measurements in model units."""
    previous, status = context_features(anchor, event, by_id, frames)
    incoming = sequence_incoming([anchor], by_id)[anchor["anchor_event_id"]]
    constants = {name: incoming[name] for name in ("ball_in_dist", "ball_in_angle_cos")}
    constants.update(
        new_convention=float(new_convention),
        seq_after_opponent=(
            float(anchor.get("pre_context_team_relation") == "opponent")
            if anchor.get("pre_context_status") == "selected"
            else math.nan
        ),
    )
    return {
        "constants": constants,
        "previous": previous if status == "valid" else {},
        "status": status,
    }
