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

from .hpn_network import PressureParams, metric_xy, total_pressure


HP_EVENT_TYPES = {"Pass", "Carry", "Miscontrol", "Dribble"}
SET_PIECES = {"From Free Kick", "From Corner", "From Throw In", "From Kick Off"}
OUTCOME_CLASSES = ("success", "neutral", "fail")
ANCHOR_TYPES = {"Pass", "Carry", "Dribble", "Shot"}
ACTION_TYPES = {"Pass", "Carry", "Dribble", "Shot"}
RETAINED_STATES = {"forward_progress", "long_switch", "contained"}
OUTCOME_BY_STATE = {
    "shot": "fail",
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
    "shot": True,
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


@dataclass(frozen=True)
class LabelConfig:
    """Public controls for the main-pipeline anchor state machine."""

    min_displacement_m: float = 3.0
    long_switch_m: float = 25.0
    clearance_high_pass_m: float = 35.0
    receiver_relief_threshold: float = 0.65
    forward_angle_deg: float = 60.0
    backward_angle_deg: float = 120.0
    goalkeeper_collect_types: tuple[str, ...] = (
        "Collected",
        "Smother",
        "Keeper Sweeper",
        "Claim",
        "Punch",
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


def _attack_normalised_location(event: dict, x: float, y: float) -> tuple[float, float]:
    """Express a point with the possession team attacking towards increasing x."""
    if event.get("team", {}).get("id") != event.get("possession_team", {}).get("id"):
        return 120.0 - float(x), 80.0 - float(y)
    return float(x), float(y)


def event_is_high_press(event: dict, frame: dict, config: SequenceConfig, params: PressureParams) -> dict | None:
    """Return main-pipeline detection read-outs for one high-press event."""
    if event.get("type", {}).get("name") not in HP_EVENT_TYPES:
        return None
    if (event.get("play_pattern") or {}).get("name") in SET_PIECES:
        return None
    location = event.get("location")
    if not location:
        return None
    actor_x_norm, actor_y = _attack_normalised_location(event, location[0], location[1])
    if actor_x_norm >= config.own_half_x:
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
        "actor_x_norm": round(actor_x_norm, 2),
        "actor_y": round(actor_y, 1),
    }


def _same_press_context(left: dict, right: dict, config: SequenceConfig) -> bool:
    if left.get("period") != right.get("period"):
        return False
    if left.get("team", {}).get("id") != right.get("team", {}).get("id"):
        return False
    return timestamp_seconds(right.get("timestamp")) - timestamp_seconds(left.get("timestamp")) <= config.max_gap_seconds


def detect_sequences(events: list[dict], frames: dict[str, dict], config: SequenceConfig, params: PressureParams) -> list[list[tuple[int, dict]]]:
    candidates: list[tuple[int, dict]] = []
    for position, event in enumerate(events):
        frame = frames.get(event.get("id"))
        if frame is None:
            continue
        metrics = event_is_high_press(event, frame, config, params)
        if metrics is not None:
            candidates.append((position, metrics))
    sequences: list[list[tuple[int, dict]]] = []
    current: list[tuple[int, dict]] = []
    for position, metrics in candidates:
        if current and not _same_press_context(events[current[-1][0]], events[position], config):
            sequences.append(current)
            current = []
        current.append((position, metrics))
    if current:
        sequences.append(current)
    return sequences


def _event_direction(event: dict) -> tuple[None, float, float, float, float] | None:
    """Attack-normalised metric displacement of the anchor action."""
    location = event.get("location")
    event_type = event.get("type", {}).get("name")
    if event_type == "Pass":
        end = (event.get("pass") or {}).get("end_location")
    elif event_type == "Carry":
        end = (event.get("carry") or {}).get("end_location")
    elif event_type == "Shot":
        end = (event.get("shot") or {}).get("end_location")
    else:
        end = None
    if not location or not end:
        return None
    x0, y0 = _attack_normalised_location(event, location[0], location[1])
    x1, y1 = _attack_normalised_location(event, end[0], end[1])
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
) -> float | None:
    """Recompute total pressure at a possible pass receiver's event."""
    location = event.get("location")
    frame = frames.get(event.get("id"))
    if not location or not frame:
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
    if anchor.get("type", {}).get("name") == "Shot":
        return _make_label("shot", direction)

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


def build_sequences(
    events_dir: str | Path,
    three_sixty_dir: str | Path,
    config: SequenceConfig = SequenceConfig(),
    params: PressureParams = PressureParams(),
    label_config: LabelConfig = LabelConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create sequence frames and main-v2 anchor labels from paired user data."""
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
        for sequence_id, sequence in enumerate(detect_sequences(events, frames, config, params), start=1):
            for sequence_position, (event_position, metrics) in enumerate(sequence):
                event = events[event_position]
                event_type = event.get("type", {}).get("name")
                row = {
                    "match_id": str(match_id),
                    "seq_id": sequence_id,
                    "ev_pos": sequence_position,
                    "anchor_event_id": event["id"],
                    "period": event.get("period"),
                    "timestamp": event.get("timestamp"),
                    "team_name": event.get("team", {}).get("name"),
                    "possession_team_name": event.get("possession_team", {}).get("name"),
                    "event_type": event_type,
                    **metrics,
                }
                sequence_rows.append(row)
                if event_type not in ANCHOR_TYPES:
                    continue
                label_rows.append(
                    {
                        **row,
                        **classify_anchor(
                            events,
                            event_position,
                            frames,
                            params=params,
                            config=label_config,
                            team_ids=team_ids,
                        ),
                    }
                )
    sequences = pd.DataFrame(sequence_rows)
    labels = pd.DataFrame(label_rows)
    if sequences.empty:
        raise ValueError(
            "No high-pressure events were detected in the supplied paired data"
        )
    if labels.empty:
        labels = pd.DataFrame(
            columns=[
                *sequences.columns,
                "state",
                "outcome_tag",
                "terminal",
                "dir_dx",
                "dir_dy",
                "dir_mag",
                "dir_theta_deg",
                "dir_mask",
                "censored",
                "resolution_id",
                "resolution_pp",
                "is_penalty",
                "is_clearance",
                "kickoff_seen",
            ]
        )
    return sequences, labels


def save_stage1(sequences: pd.DataFrame, labels: pd.DataFrame, output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_path, label_path = output_dir / "sequences.csv", output_dir / "labels.csv"
    sequences.to_csv(sequence_path, index=False)
    labels.to_csv(label_path, index=False)
    return sequence_path, label_path
