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
V4_TOP_16 = [
    "ball_in_dist", "ball_in_angle_cos", "n_open_pass", "d_carrier_x_norm_dt",
    "carrier_x_norm", "best_forward_pass_w", "n_active_carrier_boundaries",
    "press_target_entropy", "escape_capacity", "ball_in_angle_sin", "d_P_total_dt",
    "mean_receiver_pressure", "P_total", "d_carrier_boundary_pressure_dt",
    "effective_pressers", "weighted_angular_dispersion",
]


def _events_and_frames(events_dir: str | Path, three_sixty_dir: str | Path, match_id: str) -> tuple[list[dict], dict[str, dict]]:
    with (Path(events_dir) / f"{match_id}.json").open(encoding="utf-8") as handle:
        events = json.load(handle)
    with (Path(three_sixty_dir) / f"{match_id}.json").open(encoding="utf-8") as handle:
        frames = {frame["event_uuid"]: frame for frame in json.load(handle) if frame.get("freeze_frame")}
    return events, frames


def _incoming_ball(events: list[dict], position: int, flip: bool) -> tuple[float, float, float]:
    current = events[position].get("location")
    if not current:
        return np.nan, np.nan, np.nan
    for prior in reversed(events[:position]):
        if prior.get("possession") != events[position].get("possession") or not prior.get("location"):
            continue
        delta = metric_xy(*current, flip=flip) - metric_xy(*prior["location"], flip=flip)
        angle = float(np.arctan2(delta[1], delta[0]))
        return float(np.linalg.norm(delta)), float(np.sin(angle)), float(np.cos(angle))
    return np.nan, np.nan, np.nan


def network_features(network: dict, events: list[dict], position: int, params: PressureParams = PressureParams()) -> dict:
    xy, carrier_i = network["xy"], network["carrier_index"]
    carrier = xy[carrier_i]
    defenders = network["defenders"]
    receivers = network["attackers"]
    carrier_weights = np.array([
        sigmoid_pressure(params.k_player, params.player_distance, np.linalg.norm(xy[index] - carrier))
        for index in defenders
    ])
    active = carrier_weights[carrier_weights > P_EPS]
    receiver_pressure = []
    for receiver in receivers:
        values = [sigmoid_pressure(params.k_player, params.player_distance, np.linalg.norm(xy[index] - xy[receiver])) for index in defenders]
        receiver_pressure.append(float(sum(value for value in values if value > P_EPS)))
    target_pressure = np.array([active.sum(), *receiver_pressure], dtype=float)
    total = target_pressure.sum()
    distribution = target_pressure[target_pressure > 0] / total if total else np.array([])
    entropy = float(-(distribution * np.log(distribution)).sum() / np.log(max(2, len(target_pressure)))) if len(distribution) else 0.0
    angles = np.array([np.arctan2(xy[index, 1] - carrier[1], xy[index, 0] - carrier[0]) for index in defenders])
    directional = abs(np.sum(active * np.exp(1j * angles[carrier_weights > P_EPS])) / active.sum()) if len(active) else 1.0
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
    return {
        "carrier_x_norm": carrier[0] / 105.0,
        "P_total": 1.0 - np.prod(1.0 - carrier_weights) if len(carrier_weights) else 0.0,
        "effective_pressers": float(active.sum() ** 2 / np.sum(active ** 2)) if len(active) and np.sum(active ** 2) else 0.0,
        "mean_receiver_pressure": float(np.mean(receiver_pressure)) if receiver_pressure else 0.0,
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
        for _, label in group.iterrows():
            item = by_id.get(label["anchor_event_id"])
            frame = frames.get(label["anchor_event_id"])
            if item is None or frame is None:
                continue
            position, event = item
            try:
                values = network_features(build_hpn_network(event, frame, params), events, position, params)
            except ValueError:
                continue
            rows.append({
                "match_id": str(match_id), "seq_id": int(label["seq_id"]), "ev_pos": int(label["ev_pos"]),
                "anchor_event_id": label["anchor_event_id"], "team_name": label.get("team_name"),
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
