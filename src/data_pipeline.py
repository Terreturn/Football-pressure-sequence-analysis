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

from .hpn_network import PressureParams, frame_players, total_player_pressure


HP_EVENT_TYPES = {"Pass", "Carry", "Miscontrol", "Dribble"}
SET_PIECES = {"From Free Kick", "From Corner", "From Throw In", "From Kick Off"}
OUTCOME_CLASSES = ("success", "neutral", "fail")


@dataclass(frozen=True)
class SequenceConfig:
    pressure_threshold: float = 0.65
    nearest_defender_m: float = 6.4
    own_half_x: float = 60.0
    max_gap_seconds: float = 5.0
    forward_progress_m: float = 25.0
    lookahead_events: int = 12


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


def event_is_high_press(event: dict, frame: dict, config: SequenceConfig, params: PressureParams) -> dict | None:
    """Return detection read-outs when an event is a high-press anchor."""
    if event.get("type", {}).get("name") not in HP_EVENT_TYPES:
        return None
    if event.get("play_pattern", {}).get("name") in SET_PIECES or not event.get("location"):
        return None
    players, carrier_index = frame_players(event, frame)
    if carrier_index is None:
        return None
    carrier = players[carrier_index]["xy"]
    if carrier[0] >= config.own_half_x * 105.0 / 120.0:
        return None
    defenders = [player["xy"] for player in players if player["role"] == "defender" and not player["keeper"]]
    if not defenders:
        return None
    pressure = total_player_pressure(carrier, defenders, params)
    nearest = min(float(np.linalg.norm(carrier - defender)) for defender in defenders)
    if pressure < config.pressure_threshold or nearest > config.nearest_defender_m:
        return None
    return {
        "P_total": pressure,
        "nearest_def_m": nearest,
        "carrier_x_m": float(carrier[0]),
        "carrier_y_m": float(carrier[1]),
        "n_visible_players": len(players),
    }


def _same_press_context(left: dict, right: dict, config: SequenceConfig) -> bool:
    if left.get("period") != right.get("period"):
        return False
    if left.get("possession_team", {}).get("id") != right.get("possession_team", {}).get("id"):
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


def label_anchor(events: list[dict], position: int, config: SequenceConfig) -> tuple[str, bool]:
    """Create a transparent terminal-first label for new user data."""
    anchor = events[position]
    possession_id = anchor.get("possession_team", {}).get("id")
    pressing_id = anchor.get("team", {}).get("id")
    anchor_location = anchor.get("location") or [0.0, 0.0]
    for later in events[position + 1 : position + 1 + config.lookahead_events]:
        if later.get("period") != anchor.get("period"):
            break
        team_id = later.get("team", {}).get("id")
        event_type = later.get("type", {}).get("name")
        if team_id == pressing_id and team_id != possession_id:
            return "success", True
        if team_id == possession_id:
            if event_type == "Shot":
                return "fail", True
            location = later.get("location")
            if location and location[0] - anchor_location[0] >= config.forward_progress_m:
                return "fail", False
    return "neutral", False


def build_sequences(
    events_dir: str | Path,
    three_sixty_dir: str | Path,
    config: SequenceConfig = SequenceConfig(),
    params: PressureParams = PressureParams(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create frame-level sequence and anchor-label tables from user-provided data."""
    sequence_rows: list[dict] = []
    label_rows: list[dict] = []
    for match_id, event_path, frame_path in paired_match_files(events_dir, three_sixty_dir):
        events = _read_json(event_path)
        frames = frame_index(_read_json(frame_path))
        for sequence_id, sequence in enumerate(detect_sequences(events, frames, config, params), start=1):
            for event_position, metrics in sequence:
                event = events[event_position]
                row = {
                    "match_id": str(match_id),
                    "seq_id": sequence_id,
                    "ev_pos": event_position,
                    "anchor_event_id": event["id"],
                    "period": event.get("period"),
                    "timestamp": event.get("timestamp"),
                    "team_name": event.get("team", {}).get("name"),
                    "possession_team_name": event.get("possession_team", {}).get("name"),
                    "event_type": event.get("type", {}).get("name"),
                    **metrics,
                }
                sequence_rows.append(row)
                outcome, terminal = label_anchor(events, event_position, config)
                label_rows.append({**row, "outcome_tag": outcome, "terminal": terminal})
    sequences = pd.DataFrame(sequence_rows)
    labels = pd.DataFrame(label_rows)
    if labels.empty:
        raise ValueError("No high-press anchors were detected in the supplied paired data")
    return sequences, labels


def save_stage1(sequences: pd.DataFrame, labels: pd.DataFrame, output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_path, label_path = output_dir / "sequences.csv", output_dir / "labels.csv"
    sequences.to_csv(sequence_path, index=False)
    labels.to_csv(label_path, index=False)
    return sequence_path, label_path
