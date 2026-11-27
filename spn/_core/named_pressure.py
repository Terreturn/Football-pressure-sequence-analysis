"""Identify named pressers in their own frames and infer carriers from close control observations.

The presser's identity is observed. Carrier transfer is an explicitly bounded
inference, audited against known actors before any model response is computed."""

from __future__ import annotations

from collections import Counter

import math

import numpy as np

from scipy.optimize import linear_sum_assignment

from spn._core.pressure_events import seconds, team

from spn._core.frame_coordinates import frame_points


def actor_index(frame):
    """Return the unique raw actor row, or None; identifiers are unitless and missing coordinates are rejected."""
    if frame is None:
        return None
    found = [
        i
        for i, p in enumerate(frame.get("freeze_frame") or [])
        if p.get("actor") and p.get("location") is not None and p.get("teammate")
    ]
    return found[0] if len(found) == 1 else None


def carrier_match(
    reference, reference_frame, target, target_frame, owner, maximum_gap=0.100
):
    """Return a unique carrier row and meter/second diagnostics under the prespecified motion allowance; no target actor flag used."""
    result = {
        "reason": "",
        "carrier_index": -1,
        "gap_seconds": math.nan,
        "distance_m": math.nan,
        "bound_m": math.nan,
        "candidates": 0,
    }
    if reference.get("period") != target.get("period") or team(reference) != owner:
        return {**result, "reason": "period_or_owner_mismatch"}
    gap = abs(seconds(reference.get("timestamp")) - seconds(target.get("timestamp")))
    result["gap_seconds"] = gap
    if not math.isfinite(gap) or gap > maximum_gap + 1e-10:
        return {**result, "reason": "outside_time_window"}
    actor = actor_index(reference_frame)
    if actor is None or target_frame is None:
        return {**result, "reason": "missing_reference_actor_or_frame"}
    try:
        _, ref, _ = frame_points(reference_frame, team(reference) != owner)
        _, dst, flags = frame_points(target_frame, team(target) != owner)
    except (ValueError, TypeError):
        return {**result, "reason": "invalid_frame"}
    candidate_rows = np.flatnonzero(flags[:, 0])
    distances = np.linalg.norm(dst[candidate_rows] - ref[actor], axis=1)
    bound = 0.5 + 12 * gap
    result.update(
        bound_m=bound,
        distance_m=float(np.min(distances)) if len(distances) else math.nan,
    )
    inside = candidate_rows[distances <= bound]
    result["candidates"] = len(inside)
    if len(inside) != 1:
        return {
            **result,
            "reason": "ambiguous_carrier" if len(inside) else "no_carrier_in_bound",
        }
    return {**result, "reason": "matched", "carrier_index": int(inside[0])}


def carrier_match_aligned(
    reference, reference_frame, target, target_frame, owner, maximum_gap=0.100
):
    """Return a carrier row unique across all feasible whole-frame matchings after an audited meter-valued offset."""
    result = carrier_match(
        reference, reference_frame, target, target_frame, owner, maximum_gap
    )
    if result["reason"] in (
        "period_or_owner_mismatch",
        "outside_time_window",
        "missing_reference_actor_or_frame",
        "invalid_frame",
    ):
        return result
    result.update(carrier_index=-1, reason="frame_incompatible")
    _, ref, rf = frame_points(reference_frame, team(reference) != owner)
    _, dst, tf = frame_points(target_frame, team(target) != owner)
    if len(ref) != len(dst) or Counter(map(tuple, rf)) != Counter(map(tuple, tf)):
        return {**result, "reason": "different_player_composition"}
    offsets = []
    for side in (False, True):
        if not np.any(rf[:, 0] == side):
            return {**result, "reason": "missing_team_for_alignment"}
        offsets.append(
            dst[tf[:, 0] == side].mean(axis=0) - ref[rf[:, 0] == side].mean(axis=0)
        )
    offset = dst.mean(axis=0) - ref.mean(axis=0)
    result.update(
        alignment_x_m=float(offset[0]),
        alignment_y_m=float(offset[1]),
        alignment_norm_m=float(np.linalg.norm(offset)),
        alignment_team_disagreement_m=float(np.linalg.norm(offsets[0] - offsets[1])),
    )
    if result["alignment_norm_m"] > 3 or result["alignment_team_disagreement_m"] > 0.5:
        return {**result, "reason": "inconsistent_frame_offset"}
    residual = np.linalg.norm(ref[:, None, :] + offset - dst[None, :, :], axis=2)
    allowed = (residual <= result["bound_m"]) & np.all(
        rf[:, None, :] == tf[None, :, :], axis=2
    )
    rr, cc = linear_sum_assignment((~allowed).astype(int))
    if not allowed[rr, cc].all():
        return {**result, "reason": "no_feasible_frame_matching"}
    source_actor = actor_index(reference_frame)
    target_actor = int(cc[np.flatnonzero(rr == source_actor)[0]])
    alternate = allowed.copy()
    alternate[source_actor, target_actor] = False
    ar, ac = linear_sum_assignment((~alternate).astype(int))
    if alternate[ar, ac].all():
        return {**result, "reason": "ambiguous_global_carrier"}
    return {
        **result,
        "reason": "matched",
        "carrier_index": target_actor,
        "distance_m": float(residual[source_actor, target_actor]),
        "unaligned_distance_m": float(
            np.linalg.norm(ref[source_actor] - dst[target_actor])
        ),
        "candidates": int(allowed[source_actor].sum()),
    }


def pressure_view(event, frame, owner, carrier, carrier_player):
    """Return a possession-oriented scoring event/frame, preserving raw row order and all physical positions in meters."""
    if team(event) == owner or carrier == actor_index(frame):
        raise ValueError("Pressure actor cannot be assigned as carrier")
    rows = []
    for i, player in enumerate(frame["freeze_frame"]):
        location = player.get("location")
        if location is None:
            raise ValueError("Missing position")
        rows.append(
            {
                **player,
                "location": [120.0 - location[0], 80.0 - location[1]],
                "teammate": not bool(player.get("teammate")),
                "actor": i == carrier,
            }
        )
    if not rows[carrier]["teammate"]:
        raise ValueError("Carrier is not on possession team")
    visible = np.asarray(frame.get("visible_area") or [], dtype=float).reshape(-1, 2)
    changed = {
        **frame,
        "freeze_frame": rows,
        "visible_area": (np.array([120.0, 80.0]) - visible).ravel().tolist(),
    }
    view = {
        **event,
        "type": {"name": "Pressure-frame scoring view"},
        "team": {"id": owner},
        "possession_team": {"id": owner},
        "player": carrier_player,
        "location": rows[carrier]["location"],
    }
    return view, changed
