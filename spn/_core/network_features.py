"""Measure pressure, passing, space and team shape in a structural pressure network.

The frozen feature contract uses unrounded player-only pressure, even though
event detection uses a rounded player-plus-boundary pressure with the same name."""

from __future__ import annotations

import math

import numpy as np

from shapely.geometry import LineString

from spn._core.networks import projection

from spn._core.pressure_events import team

FEATURES = (
    "ball_in_dist",
    "ball_in_angle_cos",
    "carrier_x_norm",
    "d_carrier_x_norm_dt",
    "carrier_touchline_m",
    "P_total",
    "d_P_total_dt",
    "effective_pressers",
    "n_open_pass",
    "best_forward_pass_w",
    "press_target_entropy",
    "mean_receiver_pressure",
    "pressure_mean_focus_entropy",
    "vor_carrier_area_share",
    "vor_forward_receiver_area_share",
    "d_vor_carrier_area_share",
    "d_vor_forward_receiver_area_share",
    "sg_n_outlets_uncrossed",
    "sg_pressed_edges_cut_frac_2m",
    "sg_press_max_edge",
    "seq_after_opponent",
    "new_convention",
)

DELTAS = {
    "d_carrier_x_norm_dt": "carrier_x_norm",
    "d_P_total_dt": "P_total",
    "d_vor_carrier_area_share": "vor_carrier_area_share",
    "d_vor_forward_receiver_area_share": "vor_forward_receiver_area_share",
}

SHAPE = ("sg_n_outlets_uncrossed", "sg_pressed_edges_cut_frac_2m", "sg_press_max_edge")


def entropy(weights, support):
    """Return normalized natural-log entropy of nonnegative unitless weights over the full support count."""
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    if total <= 0 or support <= 1:
        return 0.0
    distribution = weights[weights > 0] / total
    return -float(np.sum(distribution * np.log(distribution))) / math.log(support)


def area_features(network):
    """Return carrier and forward-receiver area fractions of camera-visible pitch, or paired NaNs."""
    empty = {
        "vor_carrier_area_share": math.nan,
        "vor_forward_receiver_area_share": math.nan,
    }
    visible = network["visible"]
    if (
        visible is None
        or visible.is_empty
        or not math.isfinite(visible.area)
        or visible.area <= 0
    ):
        return empty
    carrier = network["carrier"]
    forward = [
        i
        for i in network["receivers"]
        if network["xy"][i, 0] > network["xy"][carrier, 0]
    ]
    polygons = [network["areas"][i] for i in [carrier] + forward]
    if any(polygon is None or not math.isfinite(polygon.area) for polygon in polygons):
        return empty
    return {
        "vor_carrier_area_share": polygons[0].area / visible.area,
        "vor_forward_receiver_area_share": sum(p.area for p in polygons[1:])
        / visible.area,
    }


def actor_x(event, frame, flip=None):
    """Return clipped flagged-actor x as pitch fraction; raise if no actor location, even if network carrier was inserted."""
    reverse = (
        team(event) != team(event, "possession_team") if flip is None else bool(flip)
    )
    for player in frame.get("freeze_frame") or []:
        if player.get("actor") and player.get("location") is not None:
            x = float(player["location"][0])
            return float(np.clip(120.0 - x if reverse else x, 0, 120) / 120.0)
    raise ValueError("missing_freeze_frame_actor")


def static_features(network, event, frame, flip=None):
    """Return selected non-shape static features: meter distances, counts and dimensionless weights/fractions."""
    carrier, xy, p = network["carrier"], network["xy"], network["parameters"]
    weights = network["pressure_weights"]
    carrier_weights = weights[:, 0]
    active = carrier_weights[carrier_weights > p.epsilon]
    # Entropy and mean pressure intentionally use different threshold ties.
    target_sums = np.where(weights > p.epsilon, weights, 0.0).sum(axis=0)
    inclusive_sums = np.where(weights >= p.epsilon, weights, 0.0).sum(axis=0)
    focused = []
    for row in weights:
        retained = row[row > p.epsilon]
        if len(retained):
            # Match the documented two-step normalization of pressure edges.
            normalized = retained / sum(retained)
            focused.append(entropy(normalized, weights.shape[1]))
    forward = [
        edge["w"] for edge in network["passes"] if xy[edge["dst"], 0] > xy[carrier, 0]
    ]
    return {
        "carrier_x_norm": actor_x(event, frame, flip),
        "carrier_touchline_m": float(min(xy[carrier, 1], 68.0 - xy[carrier, 1])),
        "P_total": float(1.0 - np.prod(1.0 - carrier_weights))
        if len(carrier_weights)
        else 0.0,
        "effective_pressers": float(active.sum() ** 2 / np.sum(active**2))
        if len(active)
        else 0.0,
        "n_open_pass": sum(edge["w"] > p.open_threshold for edge in network["passes"]),
        "best_forward_pass_w": float(max(forward, default=0.0)),
        "press_target_entropy": entropy(target_sums, max(2, len(target_sums))),
        "mean_receiver_pressure": float(np.mean(inclusive_sums[1:]))
        if len(inclusive_sums) > 1
        else 0.0,
        "pressure_mean_focus_entropy": float(np.mean(focused)) if focused else 0.0,
        **area_features(network),
    }


def shape_features(network, graphs, corrected=False):
    """Return outlet count, cut-edge fraction and longest pressing edge in meters; isolate failures only in corrected mode."""
    result = dict.fromkeys(SHAPE, math.nan)
    pressed, pressing = graphs["pressed"], graphs["pressing"]
    if not corrected and (pressed is None or pressing is None):
        return result
    xy, carrier = network["xy"], network["carrier"]
    if pressing is not None:
        segments = [LineString([xy[a], xy[b]]) for a, b in pressing]
        outlets = [i for i in network["receivers"] if not network["nodes"][i]["keeper"]]
        result[SHAPE[0]] = float(
            sum(
                not any(
                    LineString([xy[carrier], xy[i]]).crosses(segment)
                    for segment in segments
                )
                for i in outlets
            )
        )
        result[SHAPE[2]] = float(
            max((np.linalg.norm(xy[a] - xy[b]) for a, b in pressing), default=math.nan)
        )
    if pressed:
        cut = 0
        for a, b in pressed:
            for defender in network["defenders"]:
                distance, t = projection(xy[defender], xy[a], xy[b], zero_t=0.0)
                if 0 <= t <= 1 and distance < 2.0:
                    cut += 1
                    break
        result[SHAPE[1]] = cut / len(pressed)
    return result
