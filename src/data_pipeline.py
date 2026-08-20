"""StatsBomb event/360 loading, high-press detection, and sequence labels.

This module deliberately accepts independent event and 360 directories. No
competition, season, match, or local-machine path is assumed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .spn_network import PressureParams, metric_xy, total_pressure


# A high-pressure sequence is made only of deliberate, in-possession ball
# actions.  Loss-of-control and contest events remain available as raw-event
# evidence for continuity and later sequence-outcome resolution, but never
# occupy an anchor position.
CORE_ANCHOR_TYPES = frozenset({"Pass", "Carry", "Dribble"})
HP_EVENT_TYPES = CORE_ANCHOR_TYPES
# Event locations that describe the ball, or a direct contact with it, closely
# enough to serve as a pre-sequence spatial snapshot.  Off-ball annotations
# such as Pressure and Dribbled Past are deliberately absent: their location
# belongs to the defender, not the ball.
PRE_CONTEXT_BALL_POINT_TYPES = frozenset(
    {
        "Pass",
        "Carry",
        "Dribble",
        "Ball Receipt*",
        "Ball Recovery",
        "Interception",
        "Miscontrol",
        "Dispossessed",
        "Duel",
        "50/50",
        "Shot",
        "Goal Keeper",
        "Goalkeeper",
        "Clearance",
        "Shield",
        "Block",
        "Error",
    }
)
PRE_CONTEXT_HARD_STOPS = frozenset(
    {
        "Half Start",
        "Half End",
        "Match End",
        "Ball Out",
        "Offside",
        "Foul Committed",
        "Foul Won",
        "Injury Stoppage",
        "Referee Ball-Drop",
        "Substitution",
        "Player On",
        "Player Off",
    }
)
# Events that provide direct evidence that the opposing team controlled the
# ball between two otherwise connectable high-pressure candidates.
OPPONENT_ON_BALL_TYPES = {
    "Pass",
    "Carry",
    "Dribble",
    "Shot",
    "Miscontrol",
    "Dispossessed",
}
SET_PIECES = {"From Free Kick", "From Corner", "From Throw In", "From Kick Off"}
RESTART_PATTERNS = {
    "From Free Kick",
    "From Corner",
    "From Throw In",
    "From Goal Kick",
    "From Keeper",
    "From Kick Off",
}
RESTART_EVENT_TYPES = frozenset({"Pass", "Shot", "Goal Keeper", "Goalkeeper"})
ADMINISTRATIVE_STOPPAGES = {
    "Injury Stoppage",
    "Referee Ball-Drop",
    "Substitution",
    "Player On",
    "Player Off",
}
OUTCOME_CLASSES = ("success", "neutral", "fail")
EVENT_VALUE_BY_OUTPUT = {
    "success": 1.0,
    "neutral": 0.0,
    "fail": -1.0,
}
ANCHOR_TYPES = CORE_ANCHOR_TYPES
ACTION_TYPES = CORE_ANCHOR_TYPES
RETAINED_STATES = {"forward_progress", "long_switch", "contained"}
OUTCOME_BY_STATE = {
    "foul_conceded": "fail",
    "keeper_collect": "success",
    "clearance_lost": "success",
    "regain_live": "success",
    "forced_out": "success",
    "self_out": "fail",
    "throw_def": "success",
    "throw_atk": "neutral",
    "freekick_def": "success",
    "forward_progress": "fail",
    "long_switch": "fail",
    "contained": "neutral",
    "censored": "censored",
}
TERMINAL_BY_STATE = {
    "foul_conceded": True,
    "keeper_collect": True,
    "clearance_lost": True,
    "regain_live": True,
    "forced_out": True,
    "self_out": True,
    "throw_def": True,
    "throw_atk": True,
    "freekick_def": True,
    "forward_progress": False,
    "long_switch": False,
    "contained": False,
    "censored": False,
}


@dataclass(frozen=True)
class SequenceConfig:
    pressure_threshold: float = 0.65
    own_half_x: float = 60.0
    max_gap_seconds: float = 5.0
    dribble_location_tolerance: float = 0.2
    dribble_time_tolerance_seconds: float = 0.01
    pre_context_max_raw_events: int = 10
    pre_context_same_location_tolerance: float = 0.2


@dataclass(frozen=True)
class LabelConfig:
    """Public controls for semantic sequence-end context resolution."""

    min_displacement_m: float = 3.0
    long_switch_m: float = 25.0
    clearance_high_pass_m: float = 35.0
    receiver_relief_threshold: float = 0.65
    forward_angle_deg: float = 60.0
    backward_angle_deg: float = 120.0
    # Retained only so older notebooks/configuration files still construct.
    # Semantic resolution no longer uses fixed live/restart time cutoffs.
    live_resolution_seconds: float = 5.0
    restart_confirmation_seconds: float = 10.0
    goalkeeper_collect_types: tuple[str, ...] = (
        "Collected",
        "Smother",
        "Keeper Sweeper",
        "Claim",
    )


def _read_json(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _match_ids(directory: Path) -> set[str]:
    return {path.stem for path in directory.glob("*.json") if path.stem.isdigit()}


def paired_match_files(events_dir: str | Path, three_sixty_dir: str | Path) -> list[tuple[str, Path, Path]]:
    """Return matched event and 360 files named with a common match identifier."""
    events_dir, three_sixty_dir = Path(events_dir), Path(three_sixty_dir)
    if not events_dir.is_dir() or not three_sixty_dir.is_dir():
        raise FileNotFoundError("events_dir and three_sixty_dir must both be existing directories")
    common = sorted(_match_ids(events_dir) & _match_ids(three_sixty_dir), key=int)
    if not common:
        raise ValueError("No matching StatsBomb event/360 JSON filenames were found")
    return [(match_id, events_dir / f"{match_id}.json", three_sixty_dir / f"{match_id}.json") for match_id in common]


def frame_index(raw_frames: Iterable[dict]) -> dict[str, dict]:
    return {
        frame["event_uuid"]: frame
        for frame in raw_frames
        if frame.get("event_uuid") and frame.get("freeze_frame")
    }


def timestamp_seconds(value: str | None) -> float:
    if not value:
        return np.nan
    try:
        hours, minutes, seconds = value.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except (TypeError, ValueError):
        return np.nan


def _valid_location(value: object) -> bool:
    """Whether a value contains a finite StatsBomb x/y coordinate."""
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return False
    try:
        return bool(np.isfinite(float(value[0])) and np.isfinite(float(value[1])))
    except (TypeError, ValueError):
        return False


def _location_distance(left: list[float], right: list[float]) -> float:
    return float(
        np.linalg.norm(
            np.asarray(left[:2], dtype=float) - np.asarray(right[:2], dtype=float)
        )
    )


def _event_end_seconds(event: dict) -> float:
    start = timestamp_seconds(event.get("timestamp"))
    try:
        duration = float(event.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    if not np.isfinite(start):
        return np.nan
    if not np.isfinite(duration) or duration < 0.0:
        duration = 0.0
    return start + duration


def _times_close(left: float, right: float, tolerance: float) -> bool:
    return bool(
        np.isfinite(left)
        and np.isfinite(right)
        and abs(left - right) <= tolerance
    )


def _flip_statsbomb_location(location: list[float]) -> list[float]:
    """Return one StatsBomb point in the opposite team orientation."""
    return [120.0 - float(location[0]), 80.0 - float(location[1])]


def _event_location_for_team(event: dict, team_id: int | None) -> tuple[list[float] | None, bool]:
    """Express an event point in ``team_id``'s attacking orientation.

    Defensive StatsBomb annotations are first moved into their possession
    team's orientation.  A second flip is applied when that possession team is
    not the team whose sequence is being described.  The boolean records
    whether the final point differs in orientation from the raw event point.
    """
    location = event.get("location")
    if not _valid_location(location):
        return None, False
    point = [float(location[0]), float(location[1])]
    event_team_id = (event.get("team") or {}).get("id")
    possession_team_id = (event.get("possession_team") or {}).get("id")
    flipped = False
    if (
        event_team_id is not None
        and possession_team_id is not None
        and event_team_id != possession_team_id
    ):
        point = _flip_statsbomb_location(point)
        flipped = not flipped
    orientation_team_id = possession_team_id or event_team_id
    if (
        team_id is not None
        and orientation_team_id is not None
        and orientation_team_id != team_id
    ):
        point = _flip_statsbomb_location(point)
        flipped = not flipped
    return point, flipped


def _empty_pre_context(status: str, *, skipped_same_location: int = 0) -> dict:
    """Return a stable schema when no local pre-sequence snapshot is usable."""
    return {
        "pre_context_status": status,
        "pre_context_event_id": "",
        "pre_context_event_type": "",
        "pre_context_raw_event_position": np.nan,
        "pre_context_timestamp": "",
        "pre_context_team_id": np.nan,
        "pre_context_possession_team_id": np.nan,
        "pre_context_possession": np.nan,
        "pre_context_team_relation": "",
        "pre_context_possession_relation": "",
        "pre_context_ball_x": np.nan,
        "pre_context_ball_y": np.nan,
        "pre_context_raw_x": np.nan,
        "pre_context_raw_y": np.nan,
        "pre_context_coordinate_flipped": False,
        "pre_context_raw_event_gap": np.nan,
        "pre_context_time_gap_seconds": np.nan,
        "pre_context_skipped_same_location": skipped_same_location,
        "pre_context_has_frame": False,
        "pre_context_is_previous_anchor": False,
        "pre_context_related": False,
        "pre_context_direction_valid": False,
        "pre_context_dx_m": np.nan,
        "pre_context_dy_m": np.nan,
        "pre_context_distance_m": np.nan,
        "pre_context_angle_sin": np.nan,
        "pre_context_angle_cos": np.nan,
    }


def resolve_pre_sequence_context(
    events: list[dict],
    first_position: int,
    first_location: list[float],
    frames: dict[str, dict],
    config: SequenceConfig = SequenceConfig(),
    *,
    anchor_event_ids: set[str] | None = None,
) -> dict:
    """Return the nearest earlier, distinct ball point for a sequence start.

    This is deliberately a local snapshot resolver rather than a possession
    reconstruction.  It ignores team, possession and action outcome, skips
    off-ball annotations and same-location companion nodes, and stops at the
    first distinct ball/contact point.  The result is context only: it never
    becomes a sequence member or receives a target.
    """
    if not 0 <= first_position < len(events):
        raise IndexError("first_position is outside events")
    if not _valid_location(first_location):
        return _empty_pre_context("invalid_anchor_location")
    current = events[first_position]
    current_team_id = (current.get("team") or {}).get("id")
    current_period = current.get("period")
    current_possession = current.get("possession")
    current_point = [float(first_location[0]), float(first_location[1])]
    lower = max(-1, first_position - int(config.pre_context_max_raw_events) - 1)
    skipped_same_location = 0

    for position in range(first_position - 1, lower, -1):
        event = events[position]
        event_type = (event.get("type") or {}).get("name", "")
        if event.get("period") != current_period:
            return _empty_pre_context(
                "period_boundary",
                skipped_same_location=skipped_same_location,
            )
        if event_type in PRE_CONTEXT_HARD_STOPS or _stoppage_signal(event):
            return _empty_pre_context(
                "hard_stoppage",
                skipped_same_location=skipped_same_location,
            )
        if event_type not in PRE_CONTEXT_BALL_POINT_TYPES:
            continue
        point, flipped = _event_location_for_team(event, current_team_id)
        if point is None:
            continue
        if (
            _location_distance(point, current_point)
            <= config.pre_context_same_location_tolerance
        ):
            skipped_same_location += 1
            continue

        prior_metric = metric_xy(point[0], point[1])
        current_metric = metric_xy(current_point[0], current_point[1])
        delta = current_metric - prior_metric
        distance = float(np.linalg.norm(delta))
        angle = float(np.arctan2(delta[1], delta[0]))
        event_team_id = (event.get("team") or {}).get("id")
        event_possession_team_id = (event.get("possession_team") or {}).get(
            "id"
        )
        event_id = event.get("id", "")
        current_id = current.get("id", "")
        related = bool(
            event_id in (current.get("related_events") or [])
            or current_id in (event.get("related_events") or [])
        )
        return {
            "pre_context_status": "selected",
            "pre_context_event_id": event_id,
            "pre_context_event_type": event_type,
            "pre_context_raw_event_position": position,
            "pre_context_timestamp": event.get("timestamp", ""),
            "pre_context_team_id": event_team_id,
            "pre_context_possession_team_id": event_possession_team_id,
            "pre_context_possession": event.get("possession"),
            "pre_context_team_relation": (
                "same" if event_team_id == current_team_id else "opponent"
            ),
            "pre_context_possession_relation": (
                "same"
                if event.get("possession") == current_possession
                else "different"
            ),
            "pre_context_ball_x": point[0],
            "pre_context_ball_y": point[1],
            "pre_context_raw_x": float(event["location"][0]),
            "pre_context_raw_y": float(event["location"][1]),
            "pre_context_coordinate_flipped": flipped,
            "pre_context_raw_event_gap": first_position - position,
            "pre_context_time_gap_seconds": (
                timestamp_seconds(current.get("timestamp"))
                - timestamp_seconds(event.get("timestamp"))
            ),
            "pre_context_skipped_same_location": skipped_same_location,
            "pre_context_has_frame": event_id in frames,
            "pre_context_is_previous_anchor": bool(
                event_id and event_id in (anchor_event_ids or set())
            ),
            "pre_context_related": related,
            "pre_context_direction_valid": True,
            "pre_context_dx_m": float(delta[0]),
            "pre_context_dy_m": float(delta[1]),
            "pre_context_distance_m": distance,
            "pre_context_angle_sin": float(np.sin(angle)),
            "pre_context_angle_cos": float(np.cos(angle)),
        }

    return _empty_pre_context(
        "no_distinct_ball_point_in_window",
        skipped_same_location=skipped_same_location,
    )


def resolve_dribble_location(
    events: list[dict],
    position: int,
    config: SequenceConfig = SequenceConfig(),
) -> dict:
    """Resolve one Dribble at the Carry boundary where the take-on occurs.

    The preferred point is the preceding same-player Carry end. If that
    segment is absent, a successful Dribble's following Carry start is used.
    The event's own location is the final fallback. Carry matches must share
    period, possession, team, player and timestamp with the Dribble boundary.
    """
    dribble = events[position]
    if (dribble.get("type") or {}).get("name") != "Dribble":
        raise ValueError("resolve_dribble_location requires a Dribble event")

    raw_location = dribble.get("location")
    raw_is_valid = _valid_location(raw_location)
    period = dribble.get("period")
    possession = dribble.get("possession")
    team_id = (dribble.get("team") or {}).get("id")
    player_id = (dribble.get("player") or {}).get("id")
    dribble_start = timestamp_seconds(dribble.get("timestamp"))
    dribble_end = _event_end_seconds(dribble)
    spatial_tolerance = config.dribble_location_tolerance
    time_tolerance = config.dribble_time_tolerance_seconds
    conflict = False
    can_link_carry = all(
        value is not None for value in (period, possession, team_id, player_id)
    )
    deliberate_actions = {
        "Pass",
        "Carry",
        "Dribble",
        "Miscontrol",
        "Dispossessed",
        "Shot",
    }

    prior_carry = None
    prior_location = None
    for prior in reversed(events[:position]):
        if not can_link_carry:
            break
        if prior.get("period") != period or prior.get("possession") != possession:
            break
        prior_type = (prior.get("type") or {}).get("name")
        same_team = (prior.get("team") or {}).get("id") == team_id
        if prior_type != "Carry":
            if same_team and prior_type in deliberate_actions:
                break
            continue
        if not same_team:
            continue
        if (prior.get("player") or {}).get("id") != player_id:
            continue
        endpoint = (prior.get("carry") or {}).get("end_location")
        if not _valid_location(endpoint):
            continue
        if not _times_close(_event_end_seconds(prior), dribble_start, time_tolerance):
            continue
        if raw_is_valid and _location_distance(endpoint, raw_location) > spatial_tolerance:
            conflict = True
            continue
        prior_carry = prior
        prior_location = endpoint
        break

    following_carry = None
    following_location = None
    for following in events[position + 1 :]:
        if not can_link_carry:
            break
        if (
            following.get("period") != period
            or following.get("possession") != possession
        ):
            break
        following_type = (following.get("type") or {}).get("name")
        same_team = (following.get("team") or {}).get("id") == team_id
        if following_type == "Carry" and same_team:
            same_player = (following.get("player") or {}).get("id") == player_id
            start_location = following.get("location")
            if (
                same_player
                and _valid_location(start_location)
                and _times_close(
                    timestamp_seconds(following.get("timestamp")),
                    dribble_end,
                    time_tolerance,
                )
            ):
                if (
                    raw_is_valid
                    and _location_distance(start_location, raw_location)
                    > spatial_tolerance
                ):
                    conflict = True
                else:
                    following_carry = following
                    following_location = start_location
                    break
        if following_type in deliberate_actions:
            break

    if prior_location is not None:
        location = prior_location
        source = "prior_carry_end"
    elif following_location is not None:
        location = following_location
        source = "following_carry_start"
    elif raw_is_valid:
        location = raw_location
        source = "dribble_event"
    else:
        location = None
        source = "unresolved"

    if (
        prior_location is not None
        and following_location is not None
        and _location_distance(prior_location, following_location)
        > spatial_tolerance
    ):
        conflict = True

    return {
        "location": (
            [float(location[0]), float(location[1])]
            if location is not None
            else None
        ),
        "location_source": source,
        "prior_carry_id": prior_carry.get("id", "") if prior_carry else "",
        "following_carry_id": (
            following_carry.get("id", "") if following_carry else ""
        ),
        "validated_by_carry": bool(prior_carry or following_carry),
        "location_conflict": conflict,
    }


def event_is_high_press(
    event: dict,
    frame: dict,
    config: SequenceConfig,
    params: PressureParams,
    *,
    location_override: list[float] | None = None,
) -> dict | None:
    """Return detection read-outs for one in-possession high-press event."""
    if event.get("type", {}).get("name") not in HP_EVENT_TYPES:
        return None
    team_id = event.get("team", {}).get("id")
    possession_team_id = event.get("possession_team", {}).get("id")
    if team_id is None or team_id != possession_team_id:
        return None
    if (event.get("play_pattern") or {}).get("name") in SET_PIECES:
        return None
    location = (
        location_override
        if location_override is not None
        else event.get("location")
    )
    if not _valid_location(location):
        return None
    actor_x, actor_y = float(location[0]), float(location[1])
    if actor_x >= config.own_half_x:
        return None

    freeze_frame = frame.get("freeze_frame") or []
    defenders = [
        metric_xy(player["location"][0], player["location"][1])
        for player in freeze_frame
        if player.get("location") is not None
        and not player.get("teammate")
        and not player.get("keeper")
    ]
    if not defenders:
        return None

    carrier = metric_xy(location[0], location[1])
    pressure = total_pressure(
        carrier,
        defenders,
        params,
        include_boundary=True,
        use_all_lines=False,
    )["P_total"]
    distances = [float(np.linalg.norm(carrier - defender)) for defender in defenders]
    nearest = min(distances)
    n_in_range = sum(distance <= params.player_distance for distance in distances)
    if pressure <= config.pressure_threshold or n_in_range < 1:
        return None
    return {
        "P_total": round(pressure, 4),
        "nearest_def_m": round(nearest, 2),
        "carrier_x_m": float(carrier[0]),
        "carrier_y_m": float(carrier[1]),
        "n_visible_players": sum(
            player.get("location") is not None for player in freeze_frame
        ),
        "n_in_D": n_in_range,
        # Retained for backwards compatibility. This is the resolved event x;
        # for a linked Dribble it is the Carry boundary coordinate.
        "actor_x_norm": round(actor_x, 2),
        "actor_y": round(actor_y, 1),
    }


def _opponent_controlled_ball(
    events: list[dict],
    left_position: int,
    right_position: int,
    possession_team_id: int,
) -> bool:
    """Whether the opponent controlled the ball between two candidates."""
    for event in events[left_position + 1 : right_position]:
        event_team_id = (event.get("team") or {}).get("id")
        event_type = (event.get("type") or {}).get("name")
        if (
            event_team_id is not None
            and event_team_id != possession_team_id
            and event_type in OPPONENT_ON_BALL_TYPES
        ):
            return True
    return False


def _same_press_context(
    events: list[dict],
    left_position: int,
    right_position: int,
    config: SequenceConfig,
) -> bool:
    """Return whether two candidates belong to one continuous pressure spell."""
    left = events[left_position]
    right = events[right_position]
    if left.get("period") != right.get("period"):
        return False
    left_team_id = (left.get("team") or {}).get("id")
    right_team_id = (right.get("team") or {}).get("id")
    if left_team_id != right_team_id:
        return False
    right_start = timestamp_seconds(right.get("timestamp"))
    left_end = _event_end_seconds(left)
    if not np.isfinite(right_start) or not np.isfinite(left_end):
        return False
    inactive_gap = max(0.0, right_start - left_end)
    if inactive_gap > config.max_gap_seconds:
        return False
    return not _opponent_controlled_ball(
        events,
        left_position,
        right_position,
        left_team_id,
    )


def _semantic_boundary_between(
    events: list[dict],
    left_position: int,
    right_position: int,
    frames: dict[str, dict],
    config: SequenceConfig,
    params: PressureParams,
    label_config: LabelConfig,
) -> bool:
    """Whether raw context ends one pressure spell before the next anchor.

    Context events never become sequence members. An observable low-pressure
    action, a completed escape, or another decisive outcome closes the current
    high-pressure spell; the later high-pressure candidate therefore starts a
    new sequence.
    """
    left = events[left_position]
    pressed_team_id = (left.get("team") or {}).get("id")

    # The final high-pressure action can itself complete the escape. It must
    # not be hidden inside a longer sequence merely because pressure resumes
    # after the ball returns to the own half.
    if _stoppage_signal(left) or _action_crosses_half(
        left, config.own_half_x
    ):
        return True

    for position in range(left_position + 1, right_position):
        event = events[position]
        event_type = (event.get("type") or {}).get("name", "")
        event_team_id = (event.get("team") or {}).get("id")
        play_pattern = (event.get("play_pattern") or {}).get("name", "")

        if event.get("period") != left.get("period"):
            return True
        if event_type in ADMINISTRATIVE_STOPPAGES | {
            "Half End",
            "Match End",
            "Foul Committed",
            "Shot",
        }:
            return True
        if _stoppage_signal(event):
            return True
        if (
            play_pattern in RESTART_PATTERNS
            and event.get("possession") != left.get("possession")
        ):
            return True
        if _keeper_control(event, label_config):
            return True

        # Opponent stable control is already covered by
        # _opponent_controlled_ball, but retaining the explicit check keeps
        # this helper correct when used independently.
        if (
            event_team_id is not None
            and event_team_id != pressed_team_id
            and event_type in OPPONENT_ON_BALL_TYPES
        ):
            return True

        if event_type not in ACTION_TYPES or event_team_id != pressed_team_id:
            continue
        if _action_crosses_half(event, config.own_half_x):
            return True

        location = _context_location(events, position, config)
        frame = frames.get(event.get("id"))
        if location is None or frame is None:
            # Missing observation is not evidence of pressure release.
            continue
        if event_is_high_press(
            event,
            frame,
            config,
            params,
            location_override=location,
        ) is None:
            return True

    return False


def detect_sequences(
    events: list[dict],
    frames: dict[str, dict],
    config: SequenceConfig,
    params: PressureParams,
    label_config: LabelConfig = LabelConfig(),
) -> list[list[tuple[int, dict]]]:
    candidates: list[tuple[int, dict]] = []
    for position, event in enumerate(events):
        frame = frames.get(event.get("id"))
        if frame is None:
            continue
        is_dribble = (event.get("type") or {}).get("name") == "Dribble"
        location_info = (
            resolve_dribble_location(events, position, config)
            if is_dribble
            else {
                "location": event.get("location"),
                "location_source": "event_location",
                "prior_carry_id": "",
                "following_carry_id": "",
                "validated_by_carry": False,
                "location_conflict": False,
            }
        )
        metrics = event_is_high_press(
            event,
            frame,
            config,
            params,
            location_override=location_info["location"],
        )
        if metrics is not None:
            metrics.update(
                {
                    "location_source": location_info["location_source"],
                    "dribble_prior_carry_id": location_info["prior_carry_id"],
                    "dribble_following_carry_id": location_info[
                        "following_carry_id"
                    ],
                    "dribble_location_validated": location_info[
                        "validated_by_carry"
                    ],
                    "dribble_location_conflict": location_info[
                        "location_conflict"
                    ],
                }
            )
            candidates.append((position, metrics))
    sequences: list[list[tuple[int, dict]]] = []
    current: list[tuple[int, dict]] = []
    for position, metrics in candidates:
        if current:
            previous_position = current[-1][0]
            if not _same_press_context(
                events,
                previous_position,
                position,
                config,
            ) or _semantic_boundary_between(
                events,
                previous_position,
                position,
                frames,
                config,
                params,
                label_config,
            ):
                sequences.append(current)
                current = []
        current.append((position, metrics))
    if current:
        sequences.append(current)
    return sequences


def _event_direction(event: dict) -> tuple[None, float, float, float, float] | None:
    """Metric displacement of an S1 anchor action."""
    location = event.get("location")
    event_type = event.get("type", {}).get("name")
    if event_type == "Pass":
        end = (event.get("pass") or {}).get("end_location")
    elif event_type == "Carry":
        end = (event.get("carry") or {}).get("end_location")
    else:
        end = None
    if not location or not end:
        return None
    x0, y0 = float(location[0]), float(location[1])
    x1, y1 = float(end[0]), float(end[1])
    dx = (x1 - x0) * 105.0 / 120.0
    dy = (y1 - y0) * 68.0 / 80.0
    return (
        None,
        dx,
        dy,
        float(np.hypot(dx, dy)),
        float(np.degrees(np.arctan2(dy, dx))),
    )


def _make_label(
    state: str,
    direction: tuple[None, float, float, float, float] | None,
    resolution: dict | None = None,
    *,
    is_penalty: bool = False,
    is_clearance: bool = False,
    kickoff_seen: bool = False,
) -> dict:
    dx, dy, magnitude, angle = (
        direction[1:] if direction else (None, None, None, None)
    )
    return {
        "state": state,
        "outcome_tag": OUTCOME_BY_STATE[state],
        "terminal": TERMINAL_BY_STATE[state],
        "dir_dx": dx,
        "dir_dy": dy,
        "dir_mag": magnitude,
        "dir_theta_deg": angle,
        "dir_mask": 1 if state in RETAINED_STATES else 0,
        "censored": state == "censored",
        "resolution_id": resolution.get("id", "") if resolution else "",
        "resolution_pp": (
            (resolution.get("play_pattern") or {}).get("name", "")
            if resolution
            else ""
        ),
        "is_penalty": bool(is_penalty),
        "is_clearance": bool(is_clearance),
        "kickoff_seen": bool(kickoff_seen),
    }


def _is_clearance_like(event: dict, config: LabelConfig) -> bool:
    event_type = event.get("type", {}).get("name")
    if event_type == "Clearance":
        return True
    if event_type == "Pass":
        pass_data = event.get("pass") or {}
        return (
            (pass_data.get("height") or {}).get("name") == "High Pass"
            and (pass_data.get("length") or 0.0) >= config.clearance_high_pass_m
        )
    return False


def _pressure_at_event(
    event: dict,
    frames: dict[str, dict],
    params: PressureParams,
    *,
    location_override: list[float] | None = None,
) -> float | None:
    """Recompute total pressure at an event's resolved ball location."""
    location = (
        location_override
        if location_override is not None
        else event.get("location")
    )
    frame = frames.get(event.get("id"))
    if not _valid_location(location) or not frame:
        return None
    defenders = [
        metric_xy(player["location"][0], player["location"][1])
        for player in frame.get("freeze_frame") or []
        if player.get("location") is not None
        and not player.get("teammate")
        and not player.get("keeper")
    ]
    if not defenders:
        return None
    return total_pressure(
        metric_xy(location[0], location[1]),
        defenders,
        params,
        include_boundary=True,
        use_all_lines=False,
    )["P_total"]


def _receiver_relieved(
    anchor: dict,
    events: list[dict],
    position: int,
    frames: dict[str, dict],
    params: PressureParams,
    config: LabelConfig,
) -> bool | None:
    if anchor.get("type", {}).get("name") != "Pass":
        return None
    recipient_id = (
        ((anchor.get("pass") or {}).get("recipient") or {}).get("id")
    )
    if not recipient_id:
        return None
    possession = anchor.get("possession")
    for later in events[position + 1 :]:
        if later.get("possession") != possession:
            break
        if (later.get("player") or {}).get("id") == recipient_id:
            pressure = _pressure_at_event(later, frames, params)
            if pressure is not None:
                return pressure < config.receiver_relief_threshold
    return None


def _retained_state(
    anchor: dict,
    events: list[dict],
    position: int,
    frames: dict[str, dict],
    params: PressureParams,
    config: LabelConfig,
    direction: tuple[None, float, float, float, float] | None,
) -> dict:
    if direction is None:
        return _make_label("censored", direction)
    _, _, _, magnitude, angle = direction
    if magnitude < config.min_displacement_m:
        return _make_label("contained", direction)
    if abs(angle) <= config.forward_angle_deg:
        return _make_label("forward_progress", direction)
    if abs(angle) >= config.backward_angle_deg:
        return _make_label("contained", direction)
    if magnitude >= config.long_switch_m:
        if (
            _receiver_relieved(
                anchor, events, position, frames, params, config
            )
            is False
        ):
            return _make_label("contained", direction)
        return _make_label("long_switch", direction)
    return _make_label("contained", direction)


def classify_anchor(
    events: list[dict],
    position: int,
    frames: dict[str, dict],
    params: PressureParams = PressureParams(),
    config: LabelConfig = LabelConfig(),
    team_ids: set[int] | None = None,
) -> dict:
    """Apply the main v2 terminal-first state machine to one anchor."""
    anchor = events[position]
    attacking_team = anchor.get("team", {}).get("id")
    if team_ids is None:
        team_ids = {
            event.get("team", {}).get("id")
            for event in events
            if event.get("team", {}).get("id") is not None
        }
    defending_team = next(
        (team_id for team_id in team_ids if team_id != attacking_team),
        None,
    )
    possession = anchor.get("possession")
    direction = _event_direction(anchor)
    change = None
    foul_by_defender = False
    foul_is_penalty = False
    keeper_collected = False
    for later in events[position + 1 :]:
        if later.get("possession") != possession:
            change = later
            break
        event_type = later.get("type", {}).get("name")
        if (
            event_type == "Foul Committed"
            and later.get("team", {}).get("id") == defending_team
        ):
            foul_by_defender = True
            foul_is_penalty = bool(
                (later.get("foul_committed") or {}).get("penalty")
            )
        if (
            event_type == "Goalkeeper"
            and later.get("team", {}).get("id") == defending_team
            and (
                ((later.get("goalkeeper") or {}).get("type") or {}).get("name")
            )
            in config.goalkeeper_collect_types
        ):
            keeper_collected = True
        if (
            later.get("team", {}).get("id") == attacking_team
            and event_type in ACTION_TYPES
        ):
            return _retained_state(
                anchor,
                events,
                position,
                frames,
                params,
                config,
                direction,
            )

    if change is None:
        return _make_label("censored", direction)

    play_pattern = (change.get("play_pattern") or {}).get("name")
    restart_team = (change.get("possession_team") or {}).get("id")
    if play_pattern == "From Kick Off":
        return _make_label("censored", direction, kickoff_seen=True)
    if foul_by_defender or (
        play_pattern == "From Free Kick" and restart_team == attacking_team
    ):
        return _make_label(
            "foul_conceded",
            direction,
            change,
            is_penalty=foul_is_penalty,
        )
    if keeper_collected or play_pattern == "From Keeper":
        return _make_label("keeper_collect", direction, change)
    if play_pattern in ("From Corner", "From Goal Kick"):
        state = "forced_out" if restart_team == defending_team else "self_out"
        return _make_label(state, direction, change)
    if play_pattern == "From Throw In":
        state = "throw_def" if restart_team == defending_team else "throw_atk"
        return _make_label(state, direction, change)
    if play_pattern == "From Free Kick":
        return _make_label("freekick_def", direction, change)
    if restart_team == defending_team:
        if _is_clearance_like(anchor, config):
            return _make_label(
                "clearance_lost",
                direction,
                change,
                is_clearance=True,
            )
        return _make_label("regain_live", direction, change)
    return _retained_state(
        anchor,
        events,
        position,
        frames,
        params,
        config,
        direction,
    )


def label_anchor(
    events: list[dict],
    position: int,
    frames: dict[str, dict] | None = None,
    params: PressureParams = PressureParams(),
    config: LabelConfig = LabelConfig(),
    team_ids: set[int] | None = None,
) -> dict:
    """Public alias for the main-pipeline anchor classifier."""
    return classify_anchor(
        events,
        position,
        frames or {},
        params=params,
        config=config,
        team_ids=team_ids,
    )


def _format_timestamp(seconds: float) -> str:
    """Format period-relative seconds in StatsBomb timestamp form."""
    if not np.isfinite(seconds):
        return ""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining = seconds - hours * 3600 - minutes * 60
    return f"{hours:02d}:{minutes:02d}:{remaining:06.3f}"


def _context_location(
    events: list[dict],
    position: int,
    sequence_config: SequenceConfig,
) -> list[float] | None:
    event = events[position]
    if (event.get("type") or {}).get("name") == "Dribble":
        return resolve_dribble_location(events, position, sequence_config)[
            "location"
        ]
    location = event.get("location")
    if not _valid_location(location):
        return None
    return [float(location[0]), float(location[1])]


def _restart_team_id(event: dict) -> int | None:
    """Return the team actually executing a validated restart.

    The event actor is primary. Possession ownership is retained as a
    consistency check because inherited play-pattern metadata can otherwise
    make a residual event look like a restart for the wrong team.
    """
    event_team_id = (event.get("team") or {}).get("id")
    possession_team_id = (event.get("possession_team") or {}).get("id")
    if (
        event_team_id is not None
        and possession_team_id is not None
        and event_team_id != possession_team_id
    ):
        return None
    return event_team_id or possession_team_id


def _is_confirmed_restart_event(
    event: dict,
    stopped_possession: int | None,
) -> bool:
    """Whether an event actually puts a dead ball back into play.

    StatsBomb propagates ``play_pattern`` to several events in a possession.
    A Ball Receipt, Pressure, substitution, or tactical annotation carrying a
    restart pattern is therefore not itself a restart. Confirmation requires
    a restart-capable action and a new possession relative to the event that
    stopped play.
    """
    event_type = (event.get("type") or {}).get("name", "")
    play_pattern = (event.get("play_pattern") or {}).get("name", "")
    event_possession = event.get("possession")
    return bool(
        play_pattern in RESTART_PATTERNS
        and event_type in RESTART_EVENT_TYPES
        and stopped_possession is not None
        and event_possession is not None
        and event_possession != stopped_possession
    )


def _out_of_play_signal(event: dict) -> bool:
    if event.get("out") is True:
        return True
    event_type = (event.get("type") or {}).get("name")
    if event_type in {"Ball Out", "Offside"}:
        return True
    outcome_names = set()
    for key in ("pass", "shot", "duel", "goalkeeper"):
        outcome = ((event.get(key) or {}).get("outcome") or {}).get("name")
        if outcome:
            outcome_names.add(outcome)
    return bool(
        outcome_names
        & {"Out", "Off T", "Wayward", "Lost Out", "Won Out"}
    )


def _pass_outcome_name(event: dict) -> str:
    return (
        ((event.get("pass") or {}).get("outcome") or {}).get("name", "")
    )


def _action_end_location(event: dict) -> list[float] | None:
    event_type = (event.get("type") or {}).get("name")
    if event_type == "Pass":
        location = (event.get("pass") or {}).get("end_location")
    elif event_type == "Carry":
        location = (event.get("carry") or {}).get("end_location")
    else:
        location = None
    if not _valid_location(location):
        return None
    return [float(location[0]), float(location[1])]


def _action_is_successful(event: dict) -> bool:
    event_type = (event.get("type") or {}).get("name")
    if event_type == "Carry":
        return True
    if event_type != "Pass":
        return False
    # Successful StatsBomb passes normally omit the outcome field.
    return _pass_outcome_name(event) in {"", "Complete", "Success"}


def _action_crosses_half(event: dict, own_half_x: float) -> bool:
    location = event.get("location")
    if _valid_location(location) and float(location[0]) >= own_half_x:
        return True
    endpoint = _action_end_location(event)
    return bool(
        _action_is_successful(event)
        and endpoint is not None
        and float(endpoint[0]) >= own_half_x
    )


def _stoppage_signal(event: dict) -> bool:
    return _out_of_play_signal(event) or _pass_outcome_name(event) in {
        "Pass Offside",
        "Injury Clearance",
    }


def _keeper_control(event: dict, config: LabelConfig) -> bool:
    if (event.get("type") or {}).get("name") not in {
        "Goal Keeper",
        "Goalkeeper",
    }:
        return False
    goalkeeper = event.get("goalkeeper") or {}
    action = (goalkeeper.get("type") or {}).get("name")
    if action not in config.goalkeeper_collect_types:
        return False
    outcome = (goalkeeper.get("outcome") or {}).get("name", "")
    return not any(
        token in outcome.lower()
        for token in ("lost", "fail", "unsuccess", "no touch")
    )


def _regain_candidate(event: dict) -> bool:
    event_type = (event.get("type") or {}).get("name")
    if event_type == "Ball Recovery":
        return not bool(
            (event.get("ball_recovery") or {}).get("recovery_failure")
        )
    if event_type != "Interception":
        return False
    outcome = (
        ((event.get("interception") or {}).get("outcome") or {}).get(
            "name", ""
        )
    )
    return not any(
        token in outcome.lower()
        for token in ("lost", "fail", "unsuccess")
    )


def _escape_state(
    anchor: dict,
    config: LabelConfig,
) -> tuple[str, str, bool]:
    """Classify a verified low-pressure continuation from final geometry."""
    if (anchor.get("type") or {}).get("name") == "Dribble":
        return "pressure_released_unknown", "censored", False
    direction = _event_direction(anchor)
    if direction is None:
        return "pressure_released_unknown", "censored", False
    _, _, _, magnitude, angle = direction
    if magnitude < config.min_displacement_m:
        return "contained_release", "neutral", True
    if abs(angle) <= config.forward_angle_deg:
        return "forward_escape", "fail", True
    if (
        abs(angle) < config.backward_angle_deg
        and magnitude >= config.long_switch_m
    ):
        return "long_switch_escape", "fail", True
    return "contained_release", "neutral", True


def resolve_sequence_output(
    events: list[dict],
    sequence: list[tuple[int, dict]],
    frames: dict[str, dict],
    sequence_config: SequenceConfig = SequenceConfig(),
    params: PressureParams = PressureParams(),
    config: LabelConfig = LabelConfig(),
    team_ids: set[int] | None = None,
) -> dict:
    """Resolve one sequence at the first decisive semantic boundary.

    Fixed live/restart timeouts are deliberately not used. The final anchor's
    own realised result is checked first, then raw events are scanned in order
    until stable control, a restart, a terminal match event, or an
    administrative stoppage supplies a defensible result. Sequence identifiers
    are not scan boundaries: a following sequence's first anchor may confirm a
    turnover, but that first decisive event ends the scan and no later part of
    the following sequence is consumed. Unstable events are evidence only.
    Missing 360 may be skipped while the pressed team retains the same
    possession, but a later turnover is censored if pressure was never observed
    again before ownership changed.
    """
    if not sequence:
        raise ValueError("resolve_sequence_output requires a non-empty sequence")

    first_position = sequence[0][0]
    last_position = sequence[-1][0]
    first_anchor = events[first_position]
    last_anchor = events[last_position]
    pressed_team_id = (last_anchor.get("team") or {}).get("id")
    if team_ids is None:
        team_ids = {
            (event.get("team") or {}).get("id")
            for event in events
            if (event.get("team") or {}).get("id") is not None
        }
    press_team_id = next(
        (team_id for team_id in team_ids if team_id != pressed_team_id),
        None,
    )

    last_end = _event_end_seconds(last_anchor)

    context_event_count = 0
    unstable_event_count = 0
    skipped_unobservable_control_count = 0
    pending_restart = _stoppage_signal(last_anchor)
    out_signal_event = last_anchor if pending_restart else None
    clearance_seen = False
    unstable_control_seen = False
    exceptional_shot_seen = False
    regain_candidate_event: dict | None = None
    first_unobservable_control: dict | None = None
    direction = _event_direction(last_anchor)

    def finish(
        state: str,
        output: str,
        terminal: bool,
        event: dict | None = None,
        *,
        next_control: dict | None = None,
        next_location: list[float] | None = None,
        next_pressure: float | None = None,
        confidence: str = "high",
        censored_reason: str = "",
        consistency_error: str = "",
        is_penalty: bool = False,
        post_regain_shot: bool = False,
        resolution_evidence: str = "",
        restart_evidence: str = "",
    ) -> dict:
        resolution_seconds = (
            timestamp_seconds(event.get("timestamp"))
            if event is not None
            else np.nan
        )
        delay = (
            max(0.0, resolution_seconds - last_end)
            if np.isfinite(resolution_seconds) and np.isfinite(last_end)
            else np.nan
        )
        dx, dy, magnitude, angle = (
            direction[1:] if direction else (np.nan, np.nan, np.nan, np.nan)
        )
        play_pattern = (
            (event.get("play_pattern") or {}).get("name", "")
            if event is not None
            else ""
        )
        return {
            "pressed_team_id": pressed_team_id,
            "press_team_id": press_team_id,
            "anchor_count": len(sequence),
            "first_anchor_event_id": first_anchor.get("id", ""),
            "last_anchor_event_id": last_anchor.get("id", ""),
            "last_anchor_type": (last_anchor.get("type") or {}).get(
                "name", ""
            ),
            "sequence_start_timestamp": first_anchor.get("timestamp", ""),
            "sequence_end_timestamp": last_anchor.get("timestamp", ""),
            "last_anchor_end_timestamp": _format_timestamp(last_end),
            "sequence_end_state": state,
            "sequence_output": output,
            "sequence_terminal": bool(terminal),
            "resolution_event_id": event.get("id", "") if event else "",
            "resolution_event_type": (
                (event.get("type") or {}).get("name", "") if event else ""
            ),
            "resolution_team_id": (
                (event.get("team") or {}).get("id") if event else None
            ),
            "resolution_timestamp": (
                event.get("timestamp", "") if event else ""
            ),
            "resolution_delay_seconds": delay,
            "next_control_event_id": (
                next_control.get("id", "") if next_control else ""
            ),
            "next_control_team_id": (
                (next_control.get("team") or {}).get("id")
                if next_control
                else None
            ),
            "next_control_x": (
                float(next_location[0]) if next_location is not None else np.nan
            ),
            "next_control_P_total": (
                float(next_pressure) if next_pressure is not None else np.nan
            ),
            "context_event_count": context_event_count,
            "unstable_event_count": unstable_event_count,
            "skipped_unobservable_control_count": (
                skipped_unobservable_control_count
            ),
            "pending_restart_seen": bool(pending_restart),
            "out_signal_event_id": (
                out_signal_event.get("id", "") if out_signal_event else ""
            ),
            "first_unobservable_control_event_id": (
                first_unobservable_control.get("id", "")
                if first_unobservable_control
                else ""
            ),
            "resolution_confidence": confidence,
            "resolution_evidence": resolution_evidence,
            "restart_evidence": restart_evidence,
            "censored_reason": censored_reason,
            "sequence_consistency_error": consistency_error,
            "post_regain_shot": bool(post_regain_shot),
            "regain_candidate_event_id": (
                regain_candidate_event.get("id", "")
                if regain_candidate_event
                else ""
            ),
            # Compatibility aliases keep the existing S2/S3 input contract.
            "state": state,
            "outcome_tag": output,
            "terminal": bool(terminal),
            "censored": output == "censored",
            "resolution_id": event.get("id", "") if event else "",
            "resolution_pp": play_pattern,
            "dir_dx": dx,
            "dir_dy": dy,
            "dir_mag": magnitude,
            "dir_theta_deg": angle,
            "dir_mask": int(
                state
                in {"forward_escape", "long_switch_escape", "contained_release"}
            ),
            "is_penalty": bool(is_penalty),
            "is_clearance": bool(clearance_seen),
            "kickoff_seen": play_pattern == "From Kick Off",
        }

    if not np.isfinite(last_end):
        return finish(
            "unresolved",
            "censored",
            False,
            confidence="low",
            censored_reason="invalid_last_anchor_timestamp",
        )

    if _pass_outcome_name(last_anchor) == "Injury Clearance":
        return finish(
            "administrative_stoppage",
            "censored",
            False,
            last_anchor,
            confidence="low",
            censored_reason="injury_clearance",
            resolution_evidence="injury_clearance",
        )

    # A completed final action is itself result evidence. Out/offside has
    # priority because its endpoint describes where the ball left play rather
    # than a stable receiving location.
    if not pending_restart and _action_crosses_half(
        last_anchor, sequence_config.own_half_x
    ):
        endpoint = _action_end_location(last_anchor)
        return finish(
            "crossed_half",
            "fail",
            True,
            last_anchor,
            next_control=last_anchor,
            next_location=endpoint,
            resolution_evidence="final_anchor_endpoint_crossed_half",
        )

    for position in range(last_position + 1, len(events)):
        later = events[position]
        if later.get("period") != last_anchor.get("period"):
            state = (
                "unresolved_restart"
                if pending_restart
                else "unobservable_continuation"
                if skipped_unobservable_control_count
                else "unconfirmed_regain"
                if regain_candidate_event is not None
                else "unresolved_clearance"
                if clearance_seen
                else "unresolved_control"
                if unstable_control_seen
                else "exceptional_shot"
                if exceptional_shot_seen
                else "unresolved"
            )
            return finish(
                state,
                "censored",
                False,
                confidence="low",
                censored_reason="period_end",
            )

        context_event_count += 1
        event_type = (later.get("type") or {}).get("name", "")
        event_team_id = (later.get("team") or {}).get("id")
        play_pattern = (later.get("play_pattern") or {}).get("name", "")
        shot_outcome = (
            ((later.get("shot") or {}).get("outcome") or {}).get("name", "")
        )

        if _pass_outcome_name(later) == "Injury Clearance":
            return finish(
                "administrative_stoppage",
                "censored",
                False,
                later,
                confidence="low",
                censored_reason="injury_clearance",
                resolution_evidence="injury_clearance",
            )

        if event_type in {"Half End", "Match End"}:
            return finish(
                "unresolved_restart" if pending_restart else "unresolved",
                "censored",
                False,
                later,
                confidence="low",
                censored_reason="period_end",
            )

        if (
            not pending_restart
            and event_type in ADMINISTRATIVE_STOPPAGES
        ):
            return finish(
                "administrative_stoppage",
                "censored",
                False,
                later,
                confidence="low",
                censored_reason="administrative_stoppage",
            )

        # A restart is checked before ordinary control because its first event
        # is often itself a Pass. While the ball is dead, residual receipts,
        # pressure events, substitutions, and other annotations may inherit a
        # restart play pattern; they are skipped until a restart-capable event
        # begins a new possession.
        stopped_possession = (
            out_signal_event.get("possession")
            if out_signal_event is not None
            else last_anchor.get("possession")
        )
        if _is_confirmed_restart_event(later, stopped_possession):
            restart_team_id = _restart_team_id(later)
            restart_evidence = (
                "explicit_out" if pending_restart else "implicit_restart"
            )
            restart_confidence = (
                "high" if pending_restart else "medium"
            )
            if play_pattern == "From Kick Off":
                return finish(
                    "unresolved_goal_restart",
                    "censored",
                    False,
                    later,
                    confidence="low",
                    censored_reason="kickoff_without_observed_goal",
                    restart_evidence=restart_evidence,
                )
            if play_pattern == "From Free Kick":
                if restart_team_id == pressed_team_id:
                    return finish(
                        "press_foul",
                        "fail",
                        True,
                        later,
                        confidence=restart_confidence,
                        resolution_evidence="free_kick_restart",
                        restart_evidence=restart_evidence,
                    )
                if restart_team_id == press_team_id:
                    return finish(
                        "pressed_team_foul",
                        "success",
                        True,
                        later,
                        confidence=restart_confidence,
                        resolution_evidence="free_kick_restart",
                        restart_evidence=restart_evidence,
                    )
            if restart_team_id == press_team_id:
                return finish(
                    "forced_out",
                    "success",
                    True,
                    later,
                    confidence=restart_confidence,
                    resolution_evidence="restart_owner",
                    restart_evidence=restart_evidence,
                )
            if restart_team_id == pressed_team_id:
                return finish(
                    "retained_out",
                    "neutral",
                    True,
                    later,
                    confidence=restart_confidence,
                    resolution_evidence="restart_owner",
                    restart_evidence=restart_evidence,
                )
            return finish(
                "unresolved_restart",
                "censored",
                False,
                later,
                confidence="low",
                censored_reason="unknown_restart_team",
                restart_evidence=restart_evidence,
            )

        # Once play is stopped, intervening administrative events do not break
        # the causal link between the out signal and the next restart. If the
        # feed instead begins a new possession with an ordinary stable action,
        # that first action identifies restart ownership at medium confidence
        # and prevents the resolver from crossing later pressure sequences.
        if pending_restart:
            new_possession = (
                stopped_possession is not None
                and later.get("possession") is not None
                and later.get("possession") != stopped_possession
            )
            stable_live_control = (
                event_type in ACTION_TYPES or _keeper_control(later, config)
            )
            if new_possession and stable_live_control:
                restart_team_id = _restart_team_id(later)
                if restart_team_id == press_team_id:
                    return finish(
                        "forced_out",
                        "success",
                        True,
                        later,
                        next_control=later,
                        next_location=_context_location(
                            events, position, sequence_config
                        ),
                        confidence="medium",
                        resolution_evidence="restart_owner_from_live_control",
                        restart_evidence="inferred_new_possession_control",
                    )
                if restart_team_id == pressed_team_id:
                    return finish(
                        "retained_out",
                        "neutral",
                        True,
                        later,
                        next_control=later,
                        next_location=_context_location(
                            events, position, sequence_config
                        ),
                        confidence="medium",
                        resolution_evidence="restart_owner_from_live_control",
                        restart_evidence="inferred_new_possession_control",
                    )
                return finish(
                    "unresolved_restart",
                    "censored",
                    False,
                    later,
                    confidence="low",
                    censored_reason="inconsistent_live_control_after_out",
                    restart_evidence="inferred_new_possession_control",
                )
            continue

        # Confirmed goals, fouls and post-regain shots are immediate outcomes.
        if event_type == "Shot" and shot_outcome == "Goal":
            if event_team_id == pressed_team_id:
                return finish(
                    "goal_conceded",
                    "fail",
                    True,
                    later,
                    resolution_evidence="goal",
                )
            if event_team_id == press_team_id:
                return finish(
                    "regain_control",
                    "success",
                    True,
                    later,
                    post_regain_shot=True,
                    resolution_evidence="press_team_shot",
                )
        if event_type == "Foul Committed":
            penalty = bool(
                (later.get("foul_committed") or {}).get("penalty")
            )
            if event_team_id == press_team_id:
                return finish(
                    "press_foul",
                    "fail",
                    True,
                    later,
                    is_penalty=penalty,
                    resolution_evidence="foul",
                )
            if event_team_id == pressed_team_id:
                return finish(
                    "pressed_team_foul",
                    "success",
                    True,
                    later,
                    is_penalty=penalty,
                    resolution_evidence="foul",
                )
        if event_type == "Shot" and event_team_id == press_team_id:
            return finish(
                "regain_control",
                "success",
                True,
                later,
                post_regain_shot=True,
                resolution_evidence="press_team_shot",
            )

        if _stoppage_signal(later):
            pending_restart = True
            out_signal_event = later
            exceptional_shot_seen = exceptional_shot_seen or event_type == "Shot"
            continue

        if _keeper_control(later, config):
            if event_team_id == press_team_id:
                return finish(
                    "keeper_collect",
                    "success",
                    True,
                    later,
                    resolution_evidence="keeper_control",
                )
            if event_team_id == pressed_team_id:
                location = _context_location(events, position, sequence_config)
                return finish(
                    "pressed_keeper_control",
                    "neutral",
                    True,
                    later,
                    next_control=later,
                    next_location=location,
                    resolution_evidence="keeper_control",
                )

        if event_type == "Shot":
            exceptional_shot_seen = True
            unstable_event_count += 1
            continue

        if _regain_candidate(later):
            regain_candidate_event = later
            unstable_event_count += 1
            continue

        if event_type in ACTION_TYPES:
            location = _context_location(events, position, sequence_config)
            if event_team_id == press_team_id:
                if skipped_unobservable_control_count:
                    return finish(
                        "ambiguous_turnover_after_unobservable_control",
                        "censored",
                        False,
                        later,
                        next_control=later,
                        next_location=location,
                        confidence="low",
                        censored_reason="unobservable_control_before_turnover",
                        resolution_evidence="press_team_stable_control",
                    )
                state = "clearance_lost" if clearance_seen else "regain_control"
                return finish(
                    state,
                    "success",
                    True,
                    later,
                    next_control=later,
                    next_location=location,
                    resolution_evidence="press_team_stable_control",
                )
            if event_team_id != pressed_team_id:
                continue
            frame = frames.get(later.get("id"))
            high_metrics = None
            pressure = None
            if location is not None and frame is not None:
                high_metrics = event_is_high_press(
                    later,
                    frame,
                    sequence_config,
                    params,
                    location_override=location,
                )
                pressure = _pressure_at_event(
                    later, frames, params, location_override=location
                )
            belongs_to_same_sequence = _same_press_context(
                events, last_position, position, sequence_config
            )
            if high_metrics is not None and belongs_to_same_sequence:
                return finish(
                    "continued_pressure",
                    "censored",
                    False,
                    later,
                    next_control=later,
                    next_location=location,
                    next_pressure=pressure,
                    confidence="low",
                    censored_reason="missing_anchor_inside_sequence",
                    consistency_error="missing_anchor_inside_sequence",
                )
            if high_metrics is not None and _opponent_controlled_ball(
                events,
                last_position,
                position,
                pressed_team_id,
            ):
                return finish(
                    "pressure_restarted_after_opponent_touch",
                    "censored",
                    False,
                    later,
                    next_control=later,
                    next_location=location,
                    next_pressure=pressure,
                    confidence="low",
                    censored_reason="opponent_touch_split_sequence",
                )
            if high_metrics is not None:
                return finish(
                    "new_pressure_spell",
                    "censored",
                    False,
                    later,
                    next_control=later,
                    next_location=location,
                    next_pressure=pressure,
                    confidence="low",
                    censored_reason="inactive_gap_before_new_pressure",
                )
            if _action_crosses_half(later, sequence_config.own_half_x):
                endpoint = _action_end_location(later)
                evidence_location = (
                    endpoint
                    if endpoint is not None
                    and float(endpoint[0]) >= sequence_config.own_half_x
                    else location
                )
                return finish(
                    "crossed_half",
                    "fail",
                    True,
                    later,
                    next_control=later,
                    next_location=evidence_location,
                    resolution_evidence="stable_action_crossed_half",
                )
            if location is None or frame is None or pressure is None:
                skipped_unobservable_control_count += 1
                if first_unobservable_control is None:
                    first_unobservable_control = later
                continue
            state, output, terminal = _escape_state(last_anchor, config)
            return finish(
                state,
                output,
                terminal,
                later,
                next_control=later,
                next_location=location,
                next_pressure=pressure,
                confidence="medium" if output != "censored" else "low",
                censored_reason=(
                    "final_anchor_has_no_endpoint"
                    if output == "censored"
                    else ""
                ),
                resolution_evidence=(
                    "pressure_released"
                    if pressure <= sequence_config.pressure_threshold
                    else "not_high_by_full_anchor_rule"
                ),
            )

        if event_type in {
            "Miscontrol",
            "Dispossessed",
            "Duel",
            "50/50",
            "Block",
            "Clearance",
            "Shield",
            "Pressure",
        }:
            unstable_event_count += 1
            clearance_seen = clearance_seen or event_type == "Clearance"
            unstable_control_seen = unstable_control_seen or event_type in {
                "Miscontrol",
                "Dispossessed",
            }
            continue

    if pending_restart:
        state = "unresolved_restart"
    elif skipped_unobservable_control_count:
        state = "unobservable_continuation"
    elif regain_candidate_event is not None:
        state = "unconfirmed_regain"
    elif clearance_seen:
        state = "unresolved_clearance"
    elif unstable_control_seen:
        state = "unresolved_control"
    elif exceptional_shot_seen:
        state = "exceptional_shot"
    else:
        state = "unresolved"
    return finish(
        state,
        "censored",
        False,
        confidence="low",
        censored_reason="data_end",
    )


def _action_progress(event: dict, config: LabelConfig) -> dict:
    """Describe an anchor's realised Pass/Carry geometry without labelling it.

    Progress is deliberately an auxiliary description.  Under the event-level
    scheme, a forward action whose next sequence anchor is still high-pressure
    remains ``continued_pressure`` rather than becoming a pressure failure.
    """
    direction = _event_direction(event)
    duration = event.get("duration")
    try:
        duration = float(duration or 0.0)
    except (TypeError, ValueError):
        duration = np.nan
    if direction is None:
        return {
            "action_dx": np.nan,
            "action_dy": np.nan,
            "action_distance": np.nan,
            "action_angle_deg": np.nan,
            "action_duration_seconds": duration,
            "action_speed_mps": np.nan,
            "action_progress_state": "not_applicable",
        }

    _, dx, dy, distance, angle = direction
    if distance < config.min_displacement_m:
        progress = "short_control"
    elif abs(angle) <= config.forward_angle_deg:
        progress = "forward_progress"
    elif abs(angle) >= config.backward_angle_deg:
        progress = "backward_progress"
    elif distance >= config.long_switch_m:
        progress = "long_switch"
    else:
        progress = "lateral_progress"
    speed = distance / duration if np.isfinite(duration) and duration > 0.0 else np.nan
    return {
        "action_dx": dx,
        "action_dy": dy,
        "action_distance": distance,
        "action_angle_deg": angle,
        "action_duration_seconds": duration,
        "action_speed_mps": speed,
        "action_progress_state": progress,
    }


def _event_outcome_details(event: dict) -> dict:
    """Return action-level outcomes; these never override a pressure label."""
    event_type = (event.get("type") or {}).get("name", "")
    pass_outcome = _pass_outcome_name(event) if event_type == "Pass" else ""
    dribble_outcome = (
        ((event.get("dribble") or {}).get("outcome") or {}).get("name", "")
        if event_type == "Dribble"
        else ""
    )
    return {
        "pass_outcome": pass_outcome,
        "pass_completed": (
            _action_is_successful(event) if event_type == "Pass" else None
        ),
        "dribble_outcome": dribble_outcome,
        "take_on_success": (
            True
            if dribble_outcome == "Complete"
            else False
            if dribble_outcome == "Incomplete"
            else None
        ),
    }


def _anchor_location_from_row(row: dict) -> list[float] | None:
    """Return the resolved high-pressure location stored for one anchor row."""
    x, y = row.get("actor_x_norm"), row.get("actor_y")
    if x is None or y is None:
        return None
    try:
        x, y = float(x), float(y)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return [x, y]


def _linked_carry_to_dribble(current: dict, next_row: dict) -> bool:
    """Whether the following Dribble was explicitly located from this Carry."""
    if current.get("id") != (next_row.get("dribble_prior_carry_id") or ""):
        return False
    return bool(next_row.get("event_type") == "Dribble")


def _sequence_transition_details(
    current: dict,
    current_row: dict,
    next_event: dict,
    next_row: dict,
    sequence_config: SequenceConfig,
) -> dict:
    """Describe a transition to the next *sequence* anchor.

    The next anchor always proves that high pressure continued.  It is only a
    valid endpoint pressure snapshot when it occurs at the current Pass/Carry
    endpoint, so endpoint deltas are withheld for non-direct transitions.
    """
    current_type = (current.get("type") or {}).get("name", "")
    next_location = _anchor_location_from_row(next_row)
    endpoint = _action_end_location(current)
    current_end = _event_end_seconds(current)
    next_start = timestamp_seconds(next_event.get("timestamp"))
    time_gap = (
        next_start - current_end
        if np.isfinite(next_start) and np.isfinite(current_end)
        else np.nan
    )
    endpoint_error = (
        _location_distance(endpoint, next_location)
        if endpoint is not None and next_location is not None
        else np.nan
    )
    direct = bool(
        current_type in {"Pass", "Carry"}
        and endpoint is not None
        and next_location is not None
        and _times_close(
            current_end,
            next_start,
            sequence_config.dribble_time_tolerance_seconds,
        )
        and endpoint_error <= sequence_config.dribble_location_tolerance
    )
    if direct:
        p_start = float(current_row["P_total"])
        p_end = float(next_row["P_total"])
        nearest_start = float(current_row["nearest_def_m"])
        nearest_end = float(next_row["nearest_def_m"])
        n_in_start = float(current_row["n_in_D"])
        n_in_end = float(next_row["n_in_D"])
        visible_start = float(current_row["n_visible_players"])
        visible_end = float(next_row["n_visible_players"])
        scope = "direct_next_anchor"
    else:
        p_start = float(current_row["P_total"])
        p_end = np.nan
        nearest_start = float(current_row["nearest_def_m"])
        nearest_end = np.nan
        n_in_start = float(current_row["n_in_D"])
        n_in_end = np.nan
        visible_start = float(current_row["n_visible_players"])
        visible_end = np.nan
        scope = "next_anchor_not_direct_endpoint"

    next_dribble_outcome = ""
    next_take_on_success = None
    carry_to_dribble = _linked_carry_to_dribble(current, next_row)
    if carry_to_dribble:
        next_dribble_outcome = (
            ((next_event.get("dribble") or {}).get("outcome") or {}).get(
                "name", ""
            )
        )
        next_take_on_success = (
            True
            if next_dribble_outcome == "Complete"
            else False
            if next_dribble_outcome == "Incomplete"
            else None
        )

    return {
        "next_anchor_event_id": next_event.get("id", ""),
        "next_anchor_type": (next_event.get("type") or {}).get("name", ""),
        "next_anchor_timestamp": next_event.get("timestamp", ""),
        "next_anchor_x": (
            float(next_location[0]) if next_location is not None else np.nan
        ),
        "next_anchor_y": (
            float(next_location[1]) if next_location is not None else np.nan
        ),
        "next_anchor_P_total": float(next_row["P_total"]),
        "transition_time_gap_seconds": time_gap,
        "endpoint_distance_error": endpoint_error,
        "direct_transition": direct,
        "transition_scope": scope,
        "P_start": p_start,
        "P_end": p_end,
        "delta_P": p_end - p_start if direct else np.nan,
        "nearest_defender_start": nearest_start,
        "nearest_defender_end": nearest_end,
        "delta_nearest_defender": (
            nearest_end - nearest_start if direct else np.nan
        ),
        "n_in_D_start": n_in_start,
        "n_in_D_end": n_in_end,
        "delta_n_in_D": n_in_end - n_in_start if direct else np.nan,
        "n_visible_players_start": visible_start,
        "n_visible_players_end": visible_end,
        "carry_to_dribble": carry_to_dribble,
        "next_dribble_outcome": next_dribble_outcome,
        "next_take_on_success": next_take_on_success,
    }


def _terminal_transition_details(current_row: dict) -> dict:
    """Return explicit empty next-anchor fields for a sequence-final event."""
    p_start = float(current_row["P_total"])
    return {
        "next_anchor_event_id": "",
        "next_anchor_type": "",
        "next_anchor_timestamp": "",
        "next_anchor_x": np.nan,
        "next_anchor_y": np.nan,
        "next_anchor_P_total": np.nan,
        "transition_time_gap_seconds": np.nan,
        "endpoint_distance_error": np.nan,
        "direct_transition": False,
        "transition_scope": "sequence_terminal",
        "P_start": p_start,
        "P_end": np.nan,
        "delta_P": np.nan,
        "nearest_defender_start": float(current_row["nearest_def_m"]),
        "nearest_defender_end": np.nan,
        "delta_nearest_defender": np.nan,
        "n_in_D_start": float(current_row["n_in_D"]),
        "n_in_D_end": np.nan,
        "delta_n_in_D": np.nan,
        "n_visible_players_start": float(current_row["n_visible_players"]),
        "n_visible_players_end": np.nan,
        "carry_to_dribble": False,
        "next_dribble_outcome": "",
        "next_take_on_success": None,
    }


def label_sequence_anchors(
    events: list[dict],
    sequence: list[tuple[int, dict]],
    sequence_rows: list[dict],
    sequence_output: dict,
    sequence_config: SequenceConfig = SequenceConfig(),
    label_config: LabelConfig = LabelConfig(),
) -> list[dict]:
    """Attach one common final-output target to every sequence anchor.

    ``outcome_tag`` is the regression target class and therefore equals the
    sequence's final ``sequence_output`` on every row.  Its numeric target is
    ``event_value`` (success=1, neutral=0, fail=-1); censored rows receive NaN
    and are excluded through ``value_model_eligible``.  Every sequence has unit
    total training weight, split evenly over its anchors.

    Immediate process semantics remain separate: a non-final anchor is
    ``continued_pressure`` with ``transition_outcome_tag=neutral`` and the
    final anchor retains the semantic sequence-end state.  These audit fields
    explain the transition but are not regression targets.

    Future-anchor data are returned as audit-only label evidence.  S3 does not
    include those fields in the feature matrix, preventing look-ahead leakage.
    """
    if len(sequence) != len(sequence_rows):
        raise ValueError("sequence and sequence_rows must have equal length")
    final_output = sequence_output["sequence_output"]
    event_value = EVENT_VALUE_BY_OUTPUT.get(final_output, np.nan)
    training_weight = 1.0 / len(sequence)
    target = {
        "future_output_target": final_output,
        "outcome_tag": final_output,
        "event_value": event_value,
        "training_weight": training_weight,
        "value_model_eligible": final_output in EVENT_VALUE_BY_OUTPUT,
    }
    labels: list[dict] = []
    for index, ((position, _), row) in enumerate(zip(sequence, sequence_rows)):
        current = events[position]
        is_last = index == len(sequence) - 1
        action = {
            **_action_progress(current, label_config),
            **_event_outcome_details(current),
        }
        if is_last:
            core = {
                "event_state": sequence_output["sequence_end_state"],
                "state": sequence_output["state"],
                "transition_outcome_tag": sequence_output["outcome_tag"],
                "terminal": bool(sequence_output["terminal"]),
                "censored": bool(sequence_output["censored"]),
                "event_label_source": "sequence_output",
                "is_sequence_last": True,
                "resolution_event_id": sequence_output["resolution_event_id"],
                "resolution_event_type": sequence_output[
                    "resolution_event_type"
                ],
                "resolution_team_id": sequence_output["resolution_team_id"],
                "resolution_id": sequence_output["resolution_id"],
                "resolution_pp": sequence_output["resolution_pp"],
            }
            transition = _terminal_transition_details(row)
        else:
            next_position, _ = sequence[index + 1]
            next_event = events[next_position]
            next_row = sequence_rows[index + 1]
            if (
                current.get("period") != next_event.get("period")
                or (current.get("team") or {}).get("id")
                != (next_event.get("team") or {}).get("id")
            ):
                raise ValueError("adjacent anchors in one sequence are inconsistent")
            core = {
                "event_state": "continued_pressure",
                "state": "continued_pressure",
                "transition_outcome_tag": "neutral",
                "terminal": False,
                "censored": False,
                "event_label_source": "next_sequence_anchor",
                "is_sequence_last": False,
                "resolution_event_id": next_event.get("id", ""),
                "resolution_event_type": (
                    next_event.get("type") or {}
                ).get("name", ""),
                "resolution_team_id": (next_event.get("team") or {}).get("id"),
                "resolution_id": next_event.get("id", ""),
                "resolution_pp": (
                    next_event.get("play_pattern") or {}
                ).get("name", ""),
            }
            transition = _sequence_transition_details(
                current,
                row,
                next_event,
                next_row,
                sequence_config,
            )
        labels.append({**row, **core, **target, **action, **transition})
    return labels


def build_sequences(
    events_dir: str | Path,
    three_sixty_dir: str | Path,
    config: SequenceConfig = SequenceConfig(),
    params: PressureParams = PressureParams(),
    label_config: LabelConfig = LabelConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create high-pressure anchors and final-output regression targets.

    ``sequences.csv`` contains sequence membership and detector read-outs.
    ``labels.csv`` has the same anchor-level keys. Every anchor inherits its
    sequence's final outcome class, numeric event value, and 1/anchor_count
    training weight. Non-final anchors still retain an immediate
    ``continued_pressure`` process state for auditability.
    """
    sequence_rows: list[dict] = []
    label_rows: list[dict] = []
    for match_id, event_path, frame_path in paired_match_files(events_dir, three_sixty_dir):
        events = sorted(_read_json(event_path), key=lambda event: event["index"])
        frames = frame_index(_read_json(frame_path))
        team_ids = {
            event.get("team", {}).get("id")
            for event in events
            if event.get("team", {}).get("id") is not None
        }
        detected_sequences = detect_sequences(
            events,
            frames,
            config,
            params,
            label_config,
        )
        detected_anchor_ids = {
            events[position].get("id", "")
            for detected_sequence in detected_sequences
            for position, _ in detected_sequence
        }
        for sequence_id, sequence in enumerate(detected_sequences, start=1):
            current_sequence_rows: list[dict] = []
            first_position, first_metrics = sequence[0]
            pre_context = resolve_pre_sequence_context(
                events,
                first_position,
                [
                    float(first_metrics["actor_x_norm"]),
                    float(first_metrics["actor_y"]),
                ],
                frames,
                config,
                anchor_event_ids=detected_anchor_ids,
            )
            for sequence_position, (event_position, metrics) in enumerate(sequence):
                event = events[event_position]
                event_type = event.get("type", {}).get("name")
                row = {
                    "match_id": str(match_id),
                    "seq_id": sequence_id,
                    "ev_pos": sequence_position,
                    "anchor_event_id": event["id"],
                    "raw_event_position": event_position,
                    "period": event.get("period"),
                    "timestamp": event.get("timestamp"),
                    "pressed_team_id": (event.get("team") or {}).get("id"),
                    "pressed_team_name": (event.get("team") or {}).get("name"),
                    "team_name": event.get("team", {}).get("name"),
                    "possession_team_name": event.get("possession_team", {}).get("name"),
                    "event_type": event_type,
                    **metrics,
                    **(
                        pre_context
                        if sequence_position == 0
                        else _empty_pre_context("not_sequence_start")
                    ),
                }
                sequence_rows.append(row)
                current_sequence_rows.append(row)
            outcome = resolve_sequence_output(
                events,
                sequence,
                frames,
                sequence_config=config,
                params=params,
                config=label_config,
                team_ids=team_ids,
            )
            press_team_id = outcome["press_team_id"]
            press_team_name = next(
                (
                    (event.get("team") or {}).get("name")
                    for event in events
                    if (event.get("team") or {}).get("id") == press_team_id
                ),
                None,
            )
            for label in label_sequence_anchors(
                events,
                sequence,
                current_sequence_rows,
                outcome,
                config,
                label_config,
            ):
                # Preserve the semantic sequence result and regression target
                # on every event while keeping state/terminal event-specific.
                label_rows.append(
                    {
                        **label,
                        **outcome,
                        **{
                            key: label[key]
                            for key in (
                                "event_state",
                                "state",
                                "outcome_tag",
                                "terminal",
                                "censored",
                                "event_label_source",
                                "is_sequence_last",
                                "resolution_event_id",
                                "resolution_event_type",
                                "resolution_team_id",
                                "resolution_id",
                                "resolution_pp",
                            )
                        },
                        "press_team_name": press_team_name,
                    }
                )
    sequences = pd.DataFrame(sequence_rows)
    labels = pd.DataFrame(label_rows)
    if sequences.empty:
        raise ValueError(
            "No high-pressure events were detected in the supplied paired data"
        )
    return sequences, labels


def save_stage1(sequences: pd.DataFrame, labels: pd.DataFrame, output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_path, label_path = output_dir / "sequences.csv", output_dir / "labels.csv"
    sequences.to_csv(sequence_path, index=False)
    labels.to_csv(label_path, index=False)
    return sequence_path, label_path
