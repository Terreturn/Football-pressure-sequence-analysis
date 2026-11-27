"""Measure incoming ball direction and preceding geometry at a pressing event."""

from __future__ import annotations

import math

import numpy as np

from spn._core.networks import P, build_network, xy_m

from spn._core.network_features import static_features

from spn._core.pressure_events import team


def number(value):
    """Convert a scalar or serialized numeric field to float, preserving units; invalid input becomes NaN."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def flag(value):
    """Interpret a bool or serialized bool as an indicator; missing/false values return False."""
    return value is True or str(value) in ("True", "1", "1.0")


def convention(events):
    """Return newer-convention indicator and defensive possession-start share from periods 1–2 (unitless)."""
    ordered = sorted(
        (
            event
            for event in events
            if event.get("period") in (1, 2) and event.get("possession") is not None
        ),
        key=lambda event: int(event["index"]),
    )
    previous, defensive, total = None, 0, 0
    for event in ordered:
        key = event["period"], event["possession"]
        if key != previous:
            total += 1
            defensive += (event.get("type") or {}).get("name") in ("Block", "Clearance")
        previous = key
    share = defensive / total if total else math.nan
    return (float(share >= 0.015) if total else math.nan), share


def finite_point(x, y):
    """Return a finite two-coordinate provider point or None; units are provider-grid units."""
    point = [number(x), number(y)]
    return point if all(math.isfinite(v) for v in point) else None


def incoming_direction(current, previous):
    """Return distance in meters and cosine/sine from prior to current provider point, missing within .2 grid units."""
    empty = {
        "ball_in_dist": math.nan,
        "ball_in_angle_cos": math.nan,
        "ball_in_angle_sin": math.nan,
    }
    if (
        current is None
        or previous is None
        or np.linalg.norm(np.asarray(current) - previous) <= 0.2
    ):
        return empty
    delta = xy_m(current) - xy_m(previous)
    distance = float(np.linalg.norm(delta))
    if not math.isfinite(distance) or distance <= 0:
        return empty
    angle = np.arctan2(delta[1], delta[0])
    return {
        "ball_in_dist": distance,
        "ball_in_angle_cos": float(np.cos(angle)),
        "ball_in_angle_sin": float(np.sin(angle)),
    }


def sequence_incoming(ordered, events_by_id):
    """Resolve direction for a complete ordered sequence; return per-event meter/direction values and prior-source audit."""
    points = []
    for anchor in ordered:
        point = finite_point(anchor.get("actor_x_norm"), anchor.get("actor_y"))
        if point is None:
            raw = events_by_id.get(anchor["anchor_event_id"], {}).get("location")
            point = (
                finite_point(*raw[:2]) if raw is not None and len(raw) >= 2 else None
            )
        points.append(point)
    first = ordered[0]
    selected = first.get("pre_context_status") == "selected" and flag(
        first.get("pre_context_direction_valid")
    )
    previous = finite_point(
        first.get("pre_context_ball_x"), first.get("pre_context_ball_y")
    )
    output = {}
    for ordinal, anchor in enumerate(ordered):
        values = incoming_direction(None, None)
        source, hops = "", math.nan
        status = "unresolved_same_location_chain"
        if ordinal == 0:
            candidate = {
                "ball_in_dist": number(first.get("pre_context_distance_m")),
                "ball_in_angle_cos": number(first.get("pre_context_angle_cos")),
                "ball_in_angle_sin": number(first.get("pre_context_angle_sin")),
            }
            if selected and all(math.isfinite(v) for v in candidate.values()):
                values, source, hops = (
                    candidate,
                    first.get("pre_context_event_id", ""),
                    0,
                )
                status = "pre_sequence_context"
            else:
                status = "unresolved_pre_sequence_context"
        else:
            for prior in range(ordinal - 1, -1, -1):
                candidate = incoming_direction(points[ordinal], points[prior])
                if all(math.isfinite(v) for v in candidate.values()):
                    values, source, hops = (
                        candidate,
                        ordered[prior]["anchor_event_id"],
                        ordinal - prior,
                    )
                    status = "sequence_anchor"
                    break
            if not source and selected:
                candidate = incoming_direction(points[ordinal], previous)
                if all(math.isfinite(v) for v in candidate.values()):
                    values, source, hops = (
                        candidate,
                        first.get("pre_context_event_id", ""),
                        ordinal + 1,
                    )
                    status = "pre_sequence_context_fallback"
        output[anchor["anchor_event_id"]] = {
            **values,
            "incoming_source_event": source,
            "incoming_source_kind": status,
            "incoming_anchor_hops": hops,
        }
    return output


def context_owner(frame, context_event, anchor_event):
    """Return frame-owner team ID and shared-anchor flag; matching tolerance is 1 provider unit."""
    actor = next(
        (
            p["location"]
            for p in frame.get("freeze_frame") or []
            if p.get("actor") and p.get("location") is not None
        ),
        None,
    )
    if actor is None:
        return team(context_event), False
    for event, shared in [(context_event, False), (anchor_event, True)]:
        location = event.get("location")
        if (
            location is not None
            and np.hypot(actor[0] - location[0], actor[1] - location[1]) <= 1.0
        ):
            return team(event), shared
    return None, False


def context_features(first, current_event, by_id, frames, parameters=P):
    """Return prior static values and context disposition; distances are meters and no outcomes are consulted."""
    if first.get("pre_context_status") != "selected":
        return {}, "not_selected"
    uid = first.get("pre_context_event_id")
    context, frame = by_id.get(uid), frames.get(uid)
    if context is None or frame is None:
        return {}, "missing_event_or_frame"
    owner, shared = context_owner(frame, context, current_event)
    if shared:
        return {}, "shared_anchor_frame"
    if owner is None:
        return {}, "unknown_frame_owner"
    actor_team = context.get("team") or {}
    view = (
        {**context, "possession_team": dict(actor_team)}
        if actor_team.get("id")
        else context
    )
    reverse = owner != team(current_event, "possession_team")
    try:
        network = build_network(view, frame, parameters, flip=reverse)
        return static_features(network, view, frame, flip=reverse), "valid"
    except ValueError as error:
        return {}, "invalid:" + str(error)
