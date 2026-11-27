"""Read public StatsBomb inputs and resolve complete pressing sequences."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from .pressure_events import Rules
from .sequences import sequence_anchors
from .networks import PressureParams


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
    # Accepted for configuration compatibility; semantic resolution does not
    # use fixed live/restart time cutoffs.
    live_resolution_seconds: float = 5.0
    restart_confirmation_seconds: float = 10.0
    goalkeeper_collect_types: tuple[str, ...] = (
        "Collected",
        "Smother",
        "Keeper Sweeper",
        "Claim",
    )


@dataclass(frozen=True)
class MatchData:
    """One normalised StatsBomb event/360 match held fully in memory."""

    match_id: str
    events: list[dict]
    frames: dict[str, dict]
    raw_frame_count: int
    warnings: tuple[str, ...] = ()


def _json_records(source: str | Path | list[dict], label: str) -> list[dict]:
    if isinstance(source, list):
        records = source
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"missing {label} JSON: {path}")
        records = _read_json(path)
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError(f"{label} JSON must be an array of objects")
    return records


def load_match_data(
    events_source: str | Path | list[dict],
    three_sixty_source: str | Path | list[dict],
    match_id: str | None = None,
) -> MatchData:
    """Load and validate one StatsBomb-schema event/360 pair.

    Sources may be file paths or already-decoded JSON arrays.  A file stem is
    used as the match identifier when ``match_id`` is omitted.
    """
    raw_events = _json_records(events_source, "events")
    raw_frames = _json_records(three_sixty_source, "three-sixty")
    if not raw_events:
        raise ValueError("events JSON is empty")
    event_ids = [event.get("id") for event in raw_events]
    if any(not event_id for event_id in event_ids):
        raise ValueError("every event must contain a non-empty id")
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("event ids must be unique within a match")
    if any("index" not in event for event in raw_events):
        raise ValueError("every event must contain its native index")
    try:
        events = sorted(raw_events, key=lambda event: int(event["index"]))
    except (TypeError, ValueError) as error:
        raise ValueError("event indexes must be integer-like") from error
    resolved_match_id = str(match_id or "")
    if not resolved_match_id and not isinstance(events_source, list):
        resolved_match_id = Path(events_source).stem
    if not resolved_match_id:
        raise ValueError("match_id is required when events are supplied in memory")
    event_ids = set(event_ids)
    # Some earlier 360 exports assigned different UUIDs to the two files for the same match.
    matched = sum(frame.get("event_uuid") in event_ids for frame in raw_frames)
    warnings = []
    if raw_frames and matched < 0.95 * len(raw_frames):
        frames = link_by_location(events, raw_frames)
        warnings.append(f"Relinked {len(frames)} of {len(raw_frames)} frames using actor locations")
    else:
        frames = _link_by_uuid(raw_frames, event_ids)
        if matched < len(raw_frames):
            warnings.append(f"Ignored {len(raw_frames) - matched} frames with unknown event UUIDs")
    if not frames:
        warnings.append("no usable freeze frames were indexed")
    return MatchData(resolved_match_id, events, frames, len(raw_frames), tuple(warnings))


def _link_by_uuid(frames, event_ids):
    linked = {}
    seen = set()
    for frame in frames:
        uid = frame.get("event_uuid")
        if uid not in event_ids:
            continue
        if uid in seen:
            raise ValueError("Two frames reference the same event UUID")
        seen.add(uid)
        if frame.get("freeze_frame"):
            linked[uid] = frame
    return linked


def link_by_location(events, frames):
    """Recover misassigned frame UUIDs from actor positions on the provider's 120 by 80 grid."""
    located = [event for event in events if event.get("location")]
    linked = {}
    cursor = 0
    for frame in frames:
        actor = next(
            (
                p["location"]
                for p in frame.get("freeze_frame", [])
                if p.get("actor") and p.get("location")
            ),
            None,
        )
        if actor is None:
            continue
        # Allow a small overlap between search windows because adjacent frames can be out of order.
        for index in range(cursor, min(cursor + 60, len(located))):
            event = located[index]
            x, y = event["location"][:2]
            if (
                event["id"] not in linked
                and abs(x - actor[0]) <= 0.051
                and abs(y - actor[1]) <= 0.051
            ):
                linked[event["id"]] = dict(frame, event_uuid=event["id"])
                cursor = max(cursor, index - 3)
                break
    return linked


def _read_json(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _match_ids(directory: Path) -> set[str]:
    return {path.stem for path in directory.glob("*.json") if path.stem.isdigit()}


def paired_match_files(
    events_dir: str | Path, three_sixty_dir: str | Path
) -> list[tuple[str, Path, Path]]:
    """Return matched event and 360 files named with a common match identifier."""
    events_dir, three_sixty_dir = Path(events_dir), Path(three_sixty_dir)
    if not events_dir.is_dir() or not three_sixty_dir.is_dir():
        raise FileNotFoundError("events_dir and three_sixty_dir must both be existing directories")
    common = sorted(_match_ids(events_dir) & _match_ids(three_sixty_dir), key=int)
    if not common:
        raise ValueError("No matching StatsBomb event/360 JSON filenames were found")
    return [
        (match_id, events_dir / f"{match_id}.json", three_sixty_dir / f"{match_id}.json")
        for match_id in common
    ]


ANCHOR_KEYS = ["match_id", "seq_id", "ev_pos", "anchor_event_id"]


def build_sequences_from_match(
    match: MatchData,
    config: SequenceConfig = SequenceConfig(),
    params: PressureParams = PressureParams(),
    label_config: LabelConfig = LabelConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return detected anchors and their sequence-level outcomes before eligibility filtering."""
    rules = Rules(
        distance_m=params.player_distance,
        distance_sd_m=params.player_sd,
        boundary_m=params.boundary_distance,
        boundary_sd_m=params.boundary_sd,
        pressure_threshold=config.pressure_threshold,
        own_half_x=config.own_half_x,
        gap_seconds=config.max_gap_seconds,
        location_tolerance=config.dribble_location_tolerance,
        time_tolerance=config.dribble_time_tolerance_seconds,
        pre_context_events=config.pre_context_max_raw_events,
        pre_context_same_location_tolerance=config.pre_context_same_location_tolerance,
        min_displacement_m=label_config.min_displacement_m,
        long_switch_m=label_config.long_switch_m,
        forward_degrees=label_config.forward_angle_deg,
        backward_degrees=label_config.backward_angle_deg,
    )
    anchors = sequence_anchors(match.events, match.frames, rules)
    teams = {
        e["team"]["id"]: e["team"].get("name")
        for e in match.events
        if (e.get("team") or {}).get("id") is not None
    }
    base = ANCHOR_KEYS + [
        "raw_event_position",
        "period",
        "timestamp",
        "event_type",
        "pressed_team_id",
        "pressed_team_name",
        "team_name",
        "possession_team_name",
        "actor_x_norm",
        "actor_y",
        "P_total",
        "P_unrounded",
        "nearest_def_m",
        "carrier_x",
        "carrier_y",
        "n_visible_players",
        "n_in_D",
        "is_onset",
        "anchor_count",
    ]
    if not anchors:
        target_fields = [
            "press_team_id",
            "press_team_name",
            "outcome_tag",
            "value_model_eligible",
            "training_weight",
            "terminal",
            "censored",
            "state",
            "sequence_output",
            "sequence_end_state",
        ]
        return pd.DataFrame(columns=base), pd.DataFrame(columns=base + target_fields)
    full = pd.DataFrame.from_records(anchors)
    full.insert(0, "match_id", match.match_id)
    full["pressed_team_name"] = full.pressed_team_id.map(teams)
    full["team_name"] = full["possession_team_name"] = full["pressed_team_name"]
    full["press_team_name"] = full.press_team_id.map(teams)
    full["resolution_id"] = full["resolution_event_id"]
    context = [c for c in full if c.startswith(("pre_context_", "dribble_"))]
    return full[base + context].copy(), full


def build_sequences(
    events_dir,
    three_sixty_dir,
    config=SequenceConfig(),
    params=PressureParams(),
    label_config=LabelConfig(),
):
    """Read paired match directories using the same linker and sequence rules as single-match inference."""
    matches = [
        build_sequences_from_match(
            load_match_data(events, frames, mid), config, params, label_config
        )
        for mid, events, frames in paired_match_files(events_dir, three_sixty_dir)
    ]
    sequences, labels = (pd.concat(part, ignore_index=True) for part in zip(*matches))
    if sequences.empty:
        raise ValueError("No high-pressure events were detected in the supplied paired data")
    return sequences, labels


def save_stage1(sequences: pd.DataFrame, labels: pd.DataFrame, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_path, label_path = output_dir / "sequences.csv", output_dir / "labels.csv"
    sequences.to_csv(sequence_path, index=False)
    labels.to_csv(label_path, index=False)
    return sequence_path, label_path
