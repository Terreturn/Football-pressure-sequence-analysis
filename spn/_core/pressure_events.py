"""Identify events under pressure and the preceding ball movement on a 105 by 68 meter pitch."""

from __future__ import annotations

from dataclasses import dataclass

import math

ACTIONS = frozenset({"Pass", "Carry", "Dribble"})

TOUCHES = ACTIONS | {"Shot", "Miscontrol", "Dispossessed"}

SET_PIECES = frozenset(
    {"From Free Kick", "From Corner", "From Throw In", "From Kick Off"}
)

RESTARTS = SET_PIECES | {"From Goal Kick", "From Keeper"}

ADMIN = frozenset(
    {"Injury Stoppage", "Referee Ball-Drop", "Substitution", "Player On", "Player Off"}
)

BALL_POINTS = TOUCHES | {
    "Ball Receipt*",
    "Ball Recovery",
    "Interception",
    "Duel",
    "50/50",
    "Goal Keeper",
    "Goalkeeper",
    "Clearance",
    "Shield",
    "Block",
    "Error",
}

HARD_STOPS = ADMIN | {
    "Half Start",
    "Half End",
    "Match End",
    "Ball Out",
    "Offside",
    "Foul Committed",
    "Foul Won",
}


@dataclass(frozen=True)
class Rules:
    """Hold metre/second thresholds and provider-grid tolerances from the contract."""

    distance_m: float = 6.4
    distance_sd_m: float = 2.2
    boundary_m: float = 3.2
    boundary_sd_m: float = 1.1
    pressure_threshold: float = 0.65
    own_half_x: float = 60.0
    gap_seconds: float = 5.0
    location_tolerance: float = 0.2
    time_tolerance: float = 0.01
    pre_context_events: int = 10
    pre_context_same_location_tolerance: float = 0.2
    min_displacement_m: float = 3.0
    long_switch_m: float = 25.0
    forward_degrees: float = 60.0
    backward_degrees: float = 120.0
    legacy_dribble_following: bool = True
    rounded_selection: bool = True
    collapse_context_flips: bool = False


def name(event: dict, field: str = "type") -> str:
    """Return a nested event name, or an empty string; labels have no units."""
    return (event.get(field) or {}).get("name", "")


def team(event: dict, field: str = "team") -> int | None:
    """Return the named team/player identifier, or None; identifiers are unitless."""
    return (event.get(field) or {}).get("id")


def point(value: object) -> list[float] | None:
    """Validate a provider-grid point and return two finite floats or None."""
    if not isinstance(value, (tuple, list)) or len(value) < 2:
        return None
    try:
        xy = [float(value[0]), float(value[1])]
    except (ValueError, TypeError):
        return None
    return xy if all(math.isfinite(x) for x in xy) else None


def metric(xy: list[float]) -> tuple[float, float]:
    """Rescale one provider-grid point to metres without rotating it."""
    return float(xy[0]) * 105.0 / 120.0, float(xy[1]) * 68.0 / 80.0


def distance(a: list[float], b: list[float]) -> float:
    """Return Euclidean distance, in the common units of the two points."""
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def seconds(value: object) -> float:
    """Parse a period-relative HH:MM:SS timestamp into seconds; invalid -> NaN."""
    try:
        h, m, s = value.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except (AttributeError, ValueError, TypeError):
        return math.nan


def end_seconds(event: dict) -> float:
    """Return timestamp plus nonnegative finite duration, in seconds (or NaN)."""
    start = seconds(event.get("timestamp"))
    try:
        duration = float(event.get("duration") or 0)
    except (ValueError, TypeError):
        duration = 0.0
    return start + (duration if math.isfinite(duration) and duration >= 0 else 0)


def close_time(a: float, b: float, tolerance: float) -> bool:
    """Compare two finite second values using an inclusive second tolerance."""
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance


def endpoint(event: dict) -> list[float] | None:
    """Return a Pass/Carry endpoint in provider coordinates, or None."""
    return (
        point((event.get(name(event).lower()) or {}).get("end_location"))
        if name(event) in {"Pass", "Carry"}
        else None
    )


def pass_outcome(event: dict) -> str:
    """Return the nested Pass outcome name, including when event type differs."""
    return ((event.get("pass") or {}).get("outcome") or {}).get("name", "")


def successful(event: dict) -> bool:
    """Return whether a Carry or Pass has a successful endpoint; unitless."""
    return name(event) == "Carry" or (
        name(event) == "Pass" and pass_outcome(event) in {"", "Complete", "Success"}
    )


def crosses_half(event: dict, rules: Rules) -> bool:
    """Test raw starting x or a successful endpoint against provider halfway x."""
    start, end = point(event.get("location")), endpoint(event)
    return bool(
        (start is not None and start[0] >= rules.own_half_x)
        or (successful(event) and end is not None and end[0] >= rules.own_half_x)
    )


def stopped(event: dict) -> bool:
    """Return the release's explicit out/offside/injury-clearance signal."""
    outcomes = {
        ((event.get(k) or {}).get("outcome") or {}).get("name", "")
        for k in ("pass", "shot", "duel", "goalkeeper")
    }
    return (
        event.get("out") is True
        or name(event) in {"Ball Out", "Offside"}
        or bool(outcomes & {"Out", "Off T", "Wayward", "Lost Out", "Won Out"})
        or pass_outcome(event) in {"Pass Offside", "Injury Clearance"}
    )


def keeper_control(event: dict) -> bool:
    """Identify primary-release goalkeeper control; return a boolean, no units."""
    data = event.get("goalkeeper") or {}
    outcome = (data.get("outcome") or {}).get("name", "").lower()
    return (
        name(event) in {"Goal Keeper", "Goalkeeper"}
        and (data.get("type") or {}).get("name")
        in {"Collected", "Smother", "Keeper Sweeper", "Claim"}
        and not any(
            word in outcome for word in ("lost", "fail", "unsuccess", "no touch")
        )
    )


def provisional_regain(event: dict) -> bool:
    """Identify a successful recovery/interception needing subsequent control."""
    if name(event) == "Ball Recovery":
        return not (event.get("ball_recovery") or {}).get("recovery_failure")
    outcome = (
        ((event.get("interception") or {}).get("outcome") or {}).get("name", "").lower()
    )
    return name(event) == "Interception" and not any(
        w in outcome for w in ("lost", "fail", "unsuccess")
    )


def direction(event: dict) -> tuple[float, float, float, float] | None:
    """Return Pass/Carry dx, dy, length in metres and signed angle in degrees."""
    start, end = point(event.get("location")), endpoint(event)
    if start is None or end is None:
        return None
    dx, dy = metric([end[0] - start[0], end[1] - start[1]])
    return dx, dy, math.hypot(dx, dy), math.degrees(math.atan2(dy, dx))


def dribble_point(events: list[dict], position: int, rules: Rules) -> dict:
    """Resolve a Dribble's provider-grid point and neighboring Carry evidence."""
    event = events[position]
    if name(event) != "Dribble":
        raise ValueError("Dribble required")
    raw = point(event.get("location"))
    prior = following = None
    conflict = False
    context = (
        event.get("period"),
        event.get("possession"),
        team(event),
        team(event, "player"),
    )
    if all(v is not None for v in context):
        for step in (-1, 1):
            # The corrected switch applies only to the following Carry fallback.
            if step == 1 and not rules.legacy_dribble_following:
                if ((event.get("dribble") or {}).get("outcome") or {}).get(
                    "name"
                ) != "Complete":
                    continue
            indices = (
                range(position - 1, -1, -1)
                if step == -1
                else range(position + 1, len(events))
            )
            for i in indices:
                candidate = events[i]
                if (candidate.get("period"), candidate.get("possession")) != context[
                    :2
                ]:
                    break
                same_team = team(candidate) == context[2]
                kind = name(candidate)
                if (
                    kind == "Carry"
                    and same_team
                    and team(candidate, "player") == context[3]
                ):
                    xy = (
                        endpoint(candidate)
                        if step == -1
                        else point(candidate.get("location"))
                    )
                    a, b = (
                        (end_seconds(candidate), seconds(event.get("timestamp")))
                        if step == -1
                        else (seconds(candidate.get("timestamp")), end_seconds(event))
                    )
                    if xy is not None and close_time(a, b, rules.time_tolerance):
                        if (
                            raw is not None
                            and distance(xy, raw) > rules.location_tolerance
                        ):
                            conflict = True
                        else:
                            if step == -1:
                                prior = (candidate["id"], xy)
                            else:
                                following = (candidate["id"], xy)
                            break
                if kind in TOUCHES and (step == 1 or (kind != "Carry" and same_team)):
                    break
    if (
        prior
        and following
        and distance(prior[1], following[1]) > rules.location_tolerance
    ):
        conflict = True
    chosen = prior or following
    return {
        "location": chosen[1] if chosen else raw,
        "location_source": "prior_carry_end"
        if prior
        else "following_carry_start"
        if following
        else "dribble_event"
        if raw is not None
        else "unresolved",
        "prior_carry_id": prior[0] if prior else "",
        "following_carry_id": following[0] if following else "",
        "validated_by_carry": bool(chosen),
        "location_conflict": conflict,
    }


def ball_point(events: list[dict], position: int, rules: Rules) -> list[float] | None:
    """Return a resolved action point in provider units, including Dribble repair."""
    return (
        dribble_point(events, position, rules)["location"]
        if name(events[position]) == "Dribble"
        else point(events[position].get("location"))
    )


def logistic(distance_m: float, midpoint_m: float, sd_m: float) -> float:
    """Evaluate the decreasing logistic distance weight; arguments are metres."""
    z = math.pi / (math.sqrt(3) * sd_m) * (midpoint_m - distance_m)
    # Stable tails avoid overflows without changing the weight in ordinary frames.
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def pressure(
    location: list[float] | None, frame: dict | None, rules: Rules
) -> dict | None:
    """Return noisy-OR pressure and defender geometry (metres); no defenders -> None."""
    if location is None or not frame:
        return None
    players = frame.get("freeze_frame") or []
    defenders = [
        p["location"]
        for p in players
        if p.get("location") is not None
        and not p.get("teammate")
        and not p.get("keeper")
    ]
    if not defenders:
        return None
    if any(point(p) is None for p in defenders):
        raise ValueError("Invalid defender geometry")
    carrier = metric(location)
    distances = [distance(metric(p), carrier) for p in defenders]
    no_player_pressure = math.prod(
        1 - logistic(d, rules.distance_m, rules.distance_sd_m) for d in distances
    )
    boundary_distance = min(carrier[0], 105 - carrier[0], carrier[1], 68 - carrier[1])
    boundary = logistic(boundary_distance, rules.boundary_m, rules.boundary_sd_m)
    raw = 1 - no_player_pressure * (1 - boundary)
    return {
        "P_total": round(raw, 4),
        "P_unrounded": raw,
        "nearest_def_m": round(min(distances), 2),
        "carrier_x": carrier[0],
        "carrier_y": carrier[1],
        "n_visible_players": sum(p.get("location") is not None for p in players),
        "n_in_D": sum(d <= rules.distance_m for d in distances),
        "actor_x_norm": round(location[0], 2),
        "actor_y": round(location[1], 1),
    }


def high_pressure(
    event: dict, frame: dict | None, location: list[float] | None, rules: Rules
) -> tuple[dict | None, str]:
    """Select one high-pressure event; return metric data or a single exclusion reason."""
    if name(event) not in ACTIONS:
        return None, "not_deliberate_anchor_type"
    if team(event) is None or team(event) != team(event, "possession_team"):
        return None, "event_possession_team_mismatch"
    if name(event, "play_pattern") in SET_PIECES:
        return None, "set_piece"
    if location is None:
        return None, "missing_location"
    if location[0] >= rules.own_half_x:
        return None, "not_own_half"
    if frame is None:
        return None, "missing_frame"
    metrics = pressure(location, frame, rules)
    if metrics is None:
        return None, "no_visible_outfield_defender"
    p = metrics["P_total" if rules.rounded_selection else "P_unrounded"]
    if p <= rules.pressure_threshold:
        return None, "below_pressure_threshold"
    if metrics["n_in_D"] < 1:
        return None, "no_defender_in_range"
    return metrics, "selected"


def oriented_point(
    event: dict, target_team: int | None, *, collapse_flips: bool = False
) -> tuple[list[float] | None, bool]:
    """Orient an event point to a team's attack; output remains on the provider grid."""
    xy = point(event.get("location"))
    if xy is None:
        return None, False
    actor, possession = team(event), team(event, "possession_team")
    first_flip = actor is not None and possession is not None and actor != possession
    orientation = possession or actor
    second_flip = (
        target_team is not None
        and orientation is not None
        and orientation != target_team
    )
    flipped = first_flip != second_flip
    if collapse_flips:
        return ([120 - xy[0], 80 - xy[1]] if flipped else xy), flipped
    # The released procedure makes both transforms. Algebraically cancelling two
    # flips changes floating-point values at the strict 0.2-provider-unit boundary.
    for apply_flip in (first_flip, second_flip):
        if apply_flip:
            xy = [120 - xy[0], 80 - xy[1]]
    return xy, flipped


def pre_context(
    events: list[dict],
    position: int,
    location: list[float],
    frames: dict,
    anchor_ids: set[str],
    rules: Rules,
) -> dict:
    """Find prior distinct ball-point context within ten events; distances are metres."""
    if not 0 <= position < len(events):
        raise IndexError("Invalid onset position")
    result = {
        "pre_context_status": "no_distinct_ball_point_in_window",
        "pre_context_event_id": "",
        "pre_context_skipped_same_location": 0,
        "pre_context_direction_valid": False,
    }
    if point(location) is None:
        return {**result, "pre_context_status": "invalid_anchor_location"}
    current = events[position]
    for i in range(position - 1, max(-1, position - rules.pre_context_events - 1), -1):
        event = events[i]
        if event.get("period") != current.get("period"):
            return {**result, "pre_context_status": "period_boundary"}
        if name(event) in HARD_STOPS or stopped(event):
            return {**result, "pre_context_status": "hard_stoppage"}
        if name(event) not in BALL_POINTS:
            continue
        xy, flipped = oriented_point(
            event, team(current), collapse_flips=rules.collapse_context_flips
        )
        if xy is None:
            continue
        if distance(xy, location) <= rules.pre_context_same_location_tolerance:
            result["pre_context_skipped_same_location"] += 1
            continue
        a, b = metric(xy), metric(location)
        dx, dy = b[0] - a[0], b[1] - a[1]
        angle = math.atan2(dy, dx)
        return {
            **result,
            "pre_context_status": "selected",
            "pre_context_event_id": event["id"],
            "pre_context_event_type": name(event),
            "pre_context_raw_event_position": i,
            "pre_context_timestamp": event.get("timestamp", ""),
            "pre_context_team_id": team(event),
            "pre_context_possession_team_id": team(event, "possession_team"),
            "pre_context_possession": event.get("possession"),
            "pre_context_team_relation": "same"
            if team(event) == team(current)
            else "opponent",
            "pre_context_possession_relation": "same"
            if event.get("possession") == current.get("possession")
            else "different",
            "pre_context_ball_x": xy[0],
            "pre_context_ball_y": xy[1],
            "pre_context_coordinate_flipped": flipped,
            "pre_context_raw_event_gap": position - i,
            "pre_context_time_gap_seconds": seconds(current.get("timestamp"))
            - seconds(event.get("timestamp")),
            "pre_context_has_frame": event["id"] in frames,
            "pre_context_is_previous_anchor": event["id"] in anchor_ids,
            "pre_context_related": event["id"] in (current.get("related_events") or [])
            or current["id"] in (event.get("related_events") or []),
            "pre_context_direction_valid": True,
            "pre_context_dx_m": dx,
            "pre_context_dy_m": dy,
            "pre_context_distance_m": math.hypot(dx, dy),
            "pre_context_angle_sin": math.sin(angle),
            "pre_context_angle_cos": math.cos(angle),
        }
    return result
