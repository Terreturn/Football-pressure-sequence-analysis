"""S3/V4 feature construction from user-supplied StatsBomb event and 360 data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .hpn_network import P_EPS, PressureParams, build_hpn_network, frame_players, metric_xy, sigmoid_pressure


V4_STATIC_FEATURES = [
    "carrier_x_norm", "P_total", "effective_pressers", "mean_receiver_pressure",
    "frac_receivers_pressed", "press_target_entropy", "weighted_angular_dispersion",
    "carrier_boundary_pressure", "n_active_carrier_boundaries",
    "shared_boundary_outlet_pressure", "best_forward_pass_w", "n_open_pass",
    "escape_capacity", "ball_in_dist", "ball_in_angle_sin", "ball_in_angle_cos",
]
V4_TEMPORAL_FEATURES = [
    "d_P_total_dt", "d_carrier_x_norm_dt", "d_carrier_boundary_pressure_dt",
    "d_shared_boundary_outlet_pressure_dt", "d_escape_capacity_dt",
]
V4_ALL_FEATURES = V4_STATIC_FEATURES + V4_TEMPORAL_FEATURES
V4_SELECTED_FEATURES = [
    "ball_in_dist", "ball_in_angle_cos", "d_carrier_x_norm_dt",
    "carrier_x_norm", "best_forward_pass_w", "P_total", "n_open_pass",
    "press_target_entropy", "n_active_carrier_boundaries", "escape_capacity",
    "ball_in_angle_sin", "d_P_total_dt", "mean_receiver_pressure",
    "d_carrier_boundary_pressure_dt", "weighted_angular_dispersion",
]
V4_TOP_15 = V4_SELECTED_FEATURES


def _events_and_frames(events_dir: str | Path, three_sixty_dir: str | Path, match_id: str) -> tuple[list[dict], dict[str, dict]]:
    with (Path(events_dir) / f"{match_id}.json").open(encoding="utf-8") as handle:
        events = sorted(json.load(handle), key=lambda event: event["index"])
    with (Path(three_sixty_dir) / f"{match_id}.json").open(encoding="utf-8") as handle:
        frames = {frame["event_uuid"]: frame for frame in json.load(handle) if frame.get("freeze_frame")}
    return events, frames


def _incoming_ball(events: list[dict], position: int, flip: bool) -> tuple[float, float, float]:
    current = events[position].get("location")
    if not current:
        return np.nan, np.nan, np.nan
    for prior in reversed(events[:position]):
        if prior.get("possession") != events[position].get("possession"):
            break
        if not prior.get("location"):
            continue
        delta = metric_xy(*current, flip=flip) - metric_xy(*prior["location"], flip=flip)
        angle = float(np.arctan2(delta[1], delta[0]))
        return float(np.linalg.norm(delta)), float(np.sin(angle)), float(np.cos(angle))
    return np.nan, np.nan, np.nan


def _legacy_carrier_x_norm(event: dict, frame: dict) -> float | None:
    """Return the main V4 carrier-x source: the clipped freeze-frame actor."""
    defending_event = (
        event.get("team", {}).get("id")
        != event.get("possession_team", {}).get("id")
    )
    for player in frame.get("freeze_frame") or []:
        if not player.get("actor") or player.get("location") is None:
            continue
        x = float(player["location"][0])
        if defending_event:
            x = 120.0 - x
        return float(np.clip(x, 0.0, 120.0) / 120.0)
    return None


def network_features(
    network: dict,
    events: list[dict],
    position: int,
    params: PressureParams = PressureParams(),
    frame: dict | None = None,
) -> dict:
    xy, carrier_i = network["xy"], network["carrier_index"]
    carrier = xy[carrier_i]
    defenders = network["defenders"]
    receivers = network["attackers"]
    carrier_weights = np.array([
        sigmoid_pressure(params.k_player, params.player_distance, np.linalg.norm(xy[index] - carrier))
        for index in defenders
    ])
    # V3-derived read-outs use the strict edge threshold.
    active = carrier_weights[carrier_weights > P_EPS]
    receiver_pressure = []
    # V4 supplement read-outs use the inclusive threshold in the main version.
    receiver_pressure_inclusive = []
    for receiver in receivers:
        values = [sigmoid_pressure(params.k_player, params.player_distance, np.linalg.norm(xy[index] - xy[receiver])) for index in defenders]
        receiver_pressure.append(float(sum(value for value in values if value > P_EPS)))
        receiver_pressure_inclusive.append(
            float(sum(value for value in values if value >= P_EPS))
        )
    target_pressure = np.array([active.sum(), *receiver_pressure], dtype=float)
    total = target_pressure.sum()
    distribution = target_pressure[target_pressure > 0] / total if total else np.array([])
    entropy = float(-(distribution * np.log(distribution)).sum() / np.log(max(2, len(target_pressure)))) if len(distribution) else 0.0
    angles = np.array([np.arctan2(xy[index, 1] - carrier[1], xy[index, 0] - carrier[0]) for index in defenders])
    angular_mask = carrier_weights >= P_EPS
    angular_weights = carrier_weights[angular_mask]
    directional = abs(np.sum(angular_weights * np.exp(1j * angles[angular_mask])) / angular_weights.sum()) if len(angular_weights) else 1.0
    carrier_boundary = [edge["p"] for edge in network["boundary_edges"] if edge["on_carrier"]]
    receiver_boundary = []
    for receiver in receivers:
        values = [edge["p"] for edge in network["boundary_edges"] if edge["dst"] == receiver]
        receiver_boundary.append(1.0 - np.prod([1.0 - value for value in values]) if values else 0.0)
    pass_edges = network["pass_edges"]
    forward = [edge["w"] for edge in pass_edges if xy[edge["dst"], 0] > carrier[0]]
    escape = [edge["w"] for edge in pass_edges if edge["w"] >= .5]
    flip = network["event"].get("team", {}).get("id") != network["event"].get("possession_team", {}).get("id")
    distance, sine, cosine = _incoming_ball(events, position, flip)
    carrier_x_norm = (
        _legacy_carrier_x_norm(network["event"], frame)
        if frame is not None
        else float(carrier[0] / 105.0)
    )
    if carrier_x_norm is None:
        raise ValueError("The main V4 carrier_x_norm requires a freeze-frame actor")
    return {
        "carrier_x_norm": carrier_x_norm,
        "P_total": 1.0 - np.prod(1.0 - carrier_weights) if len(carrier_weights) else 0.0,
        "effective_pressers": float(active.sum() ** 2 / np.sum(active ** 2)) if len(active) and np.sum(active ** 2) else 0.0,
        "mean_receiver_pressure": float(np.mean(receiver_pressure_inclusive)) if receiver_pressure_inclusive else 0.0,
        "frac_receivers_pressed": float(np.mean(np.asarray(receiver_pressure) > .5)) if receiver_pressure else 0.0,
        "press_target_entropy": entropy,
        "weighted_angular_dispersion": float(1.0 - directional),
        "carrier_boundary_pressure": float(1.0 - np.prod([1.0 - value for value in carrier_boundary])) if carrier_boundary else 0.0,
        "n_active_carrier_boundaries": len(carrier_boundary),
        "shared_boundary_outlet_pressure": float(np.mean(receiver_boundary)) if receiver_boundary else 0.0,
        "best_forward_pass_w": float(max(forward, default=0.0)),
        "n_open_pass": int(sum(edge["w"] > .5 for edge in pass_edges)),
        "escape_capacity": float(np.mean(escape)) if escape else 0.0,
        "ball_in_dist": distance, "ball_in_angle_sin": sine, "ball_in_angle_cos": cosine,
    }


def add_temporal_features(table: pd.DataFrame) -> pd.DataFrame:
    table = table.sort_values(["match_id", "seq_id", "ev_pos"]).copy()
    sources = {
        "d_P_total_dt": "P_total", "d_carrier_x_norm_dt": "carrier_x_norm",
        "d_carrier_boundary_pressure_dt": "carrier_boundary_pressure",
        "d_shared_boundary_outlet_pressure_dt": "shared_boundary_outlet_pressure",
        "d_escape_capacity_dt": "escape_capacity",
    }
    for target, source in sources.items():
        table[target] = table.groupby(["match_id", "seq_id"], sort=False)[source].diff()
    return table


def build_v4_feature_table(labels: pd.DataFrame, events_dir: str | Path, three_sixty_dir: str | Path, params: PressureParams = PressureParams()) -> pd.DataFrame:
    rows = []
    for match_id, group in labels.groupby(labels["match_id"].astype(str), sort=True):
        events, frames = _events_and_frames(events_dir, three_sixty_dir, match_id)
        by_id = {event["id"]: (position, event) for position, event in enumerate(events)}
        match_teams = sorted({
            event.get("team", {}).get("name")
            for event in events
            if event.get("team", {}).get("name")
        })
        for _, label in group.iterrows():
            item = by_id.get(label["anchor_event_id"])
            frame = frames.get(label["anchor_event_id"])
            if item is None or frame is None:
                continue
            position, event = item
            try:
                values = network_features(
                    build_hpn_network(event, frame, params),
                    events,
                    position,
                    params,
                    frame,
                )
            except ValueError:
                continue
            pressed_team = label.get("possession_team_name") or label.get("team_name")
            press_team = next(
                (team for team in match_teams if team != pressed_team),
                None,
            ) if len(match_teams) == 2 else None
            rows.append({
                "match_id": str(match_id), "seq_id": int(label["seq_id"]), "ev_pos": int(label["ev_pos"]),
                "anchor_event_id": label["anchor_event_id"], "team_name": label.get("team_name"),
                "pressed_team": pressed_team, "press_team": press_team,
                "outcome_tag": label["outcome_tag"], "terminal": bool(label["terminal"]), **values,
            })
    if not rows:
        raise ValueError("No V4 feature rows could be built from the supplied labels and freeze frames")
    return add_temporal_features(pd.DataFrame(rows))


def save_feature_table(table: pd.DataFrame, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "v4_features.parquet"
    table.to_parquet(path, index=False)
    return path
