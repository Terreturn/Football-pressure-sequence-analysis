"""S3/SPN feature construction from user-supplied StatsBomb event and 360 data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .spn_network import P_EPS, PressureParams, build_spn_network, frame_players, metric_xy, sigmoid_pressure


SPN_BASE_STATIC_FEATURES = [
    "carrier_x_norm", "P_total", "effective_pressers", "mean_receiver_pressure",
    "frac_receivers_pressed", "press_target_entropy", "weighted_angular_dispersion",
    "carrier_boundary_pressure", "n_active_carrier_boundaries",
    "shared_boundary_outlet_pressure", "best_forward_pass_w", "n_open_pass",
    "escape_capacity", "ball_in_dist", "ball_in_angle_sin", "ball_in_angle_cos",
]
SPN_NETWORK_STATIC_FEATURES = [
    "vor_carrier_area_share", "vor_forward_receiver_area_share",
    "pressure_edge_density", "pressure_mean_focus_entropy",
    "del_cross_role_edge_ratio", "del_carrier_degree_norm",
]
SPN_STATIC_FEATURES = SPN_BASE_STATIC_FEATURES + SPN_NETWORK_STATIC_FEATURES
SPN_BASE_TEMPORAL_FEATURES = [
    "d_P_total_dt", "d_carrier_x_norm_dt", "d_carrier_boundary_pressure_dt",
    "d_shared_boundary_outlet_pressure_dt", "d_escape_capacity_dt",
]
SPN_NETWORK_TEMPORAL_FEATURES = [
    "d_vor_carrier_area_share", "d_vor_forward_receiver_area_share",
]
SPN_TEMPORAL_FEATURES = SPN_BASE_TEMPORAL_FEATURES + SPN_NETWORK_TEMPORAL_FEATURES
SPN_TEMPORAL_SOURCES = {
    "d_P_total_dt": "P_total",
    "d_carrier_x_norm_dt": "carrier_x_norm",
    "d_carrier_boundary_pressure_dt": "carrier_boundary_pressure",
    "d_shared_boundary_outlet_pressure_dt": "shared_boundary_outlet_pressure",
    "d_escape_capacity_dt": "escape_capacity",
    "d_vor_carrier_area_share": "vor_carrier_area_share",
    "d_vor_forward_receiver_area_share": "vor_forward_receiver_area_share",
}
SPN_ALL_FEATURES = SPN_STATIC_FEATURES + SPN_TEMPORAL_FEATURES
SPN_SELECTED_FEATURES = [
    "ball_in_dist", "ball_in_angle_cos", "d_carrier_x_norm_dt",
    "carrier_x_norm", "best_forward_pass_w", "P_total", "n_open_pass",
    "press_target_entropy", "escape_capacity",
    "ball_in_angle_sin", "d_P_total_dt", "mean_receiver_pressure",
    "d_carrier_boundary_pressure_dt", "weighted_angular_dispersion",
    *SPN_NETWORK_STATIC_FEATURES,
    *SPN_NETWORK_TEMPORAL_FEATURES,
]
SEQUENCE_CONTEXT_LOCATION_TOLERANCE = 0.2


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
    """Return the main SPN carrier-x source: the clipped freeze-frame actor."""
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


def _voronoi_area_features(network: dict) -> dict[str, float]:
    """Return visible-area-normalised carrier and forward-outlet control."""
    empty = {
        "vor_carrier_area_share": np.nan,
        "vor_forward_receiver_area_share": np.nan,
    }
    visible_area = network.get("visible_area")
    if visible_area is None or getattr(visible_area, "is_empty", True):
        return empty
    try:
        visible_size = float(visible_area.area)
    except (AttributeError, TypeError, ValueError):
        return empty
    if not np.isfinite(visible_size) or visible_size <= 0.0:
        return empty

    areas = network.get("areas") or []
    carrier_i = int(network["carrier_index"])
    if carrier_i >= len(areas) or areas[carrier_i] is None:
        return empty
    try:
        carrier_size = float(areas[carrier_i].area)
    except (AttributeError, TypeError, ValueError):
        return empty

    xy = network["xy"]
    carrier_x = float(xy[carrier_i, 0])
    forward_sizes = []
    for receiver in network.get("attackers", []):
        if float(xy[receiver, 0]) <= carrier_x:
            continue
        if receiver >= len(areas) or areas[receiver] is None:
            return empty
        try:
            forward_sizes.append(float(areas[receiver].area))
        except (AttributeError, TypeError, ValueError):
            return empty
    values = np.asarray([carrier_size, *forward_sizes], dtype=float)
    if not np.isfinite(values).all():
        return empty
    return {
        "vor_carrier_area_share": carrier_size / visible_size,
        "vor_forward_receiver_area_share": sum(forward_sizes) / visible_size,
    }


def _pressure_graph_features(network: dict) -> dict[str, float]:
    """Summarise thresholded pressure coverage and per-defender focus."""
    defenders = set(network.get("defenders", []))
    targets = {int(network["carrier_index"]), *network.get("attackers", [])}
    denominator = len(defenders) * len(targets)
    by_defender: dict[int, list[float]] = {}
    active_pairs = set()
    for edge in network.get("pressure_edges", []):
        source, target = edge.get("src"), edge.get("dst")
        if source not in defenders or target not in targets:
            continue
        active_pairs.add((int(source), int(target)))
        try:
            weight = float(edge["w"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(weight) and weight > 0.0:
            by_defender.setdefault(int(source), []).append(weight)

    entropies = []
    if len(targets) > 1:
        normaliser = float(np.log(len(targets)))
        for weights in by_defender.values():
            distribution = np.asarray(weights, dtype=float)
            distribution /= distribution.sum()
            entropy = -float(np.sum(distribution * np.log(distribution)))
            entropies.append(entropy / normaliser)
    elif by_defender:
        entropies = [0.0] * len(by_defender)
    return {
        "pressure_edge_density": (
            len(active_pairs) / denominator if denominator else 0.0
        ),
        "pressure_mean_focus_entropy": (
            float(np.mean(entropies)) if entropies else 0.0
        ),
    }


def _delaunay_graph_features(network: dict) -> dict[str, float]:
    """Return carrier centrality and attack/defence mixing in the spatial web."""
    xy = network["xy"]
    node_count = len(xy)
    carrier_i = int(network["carrier_index"])
    topology = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in network.get("topology", set())
        if len(edge) == 2
        and 0 <= int(edge[0]) < node_count
        and 0 <= int(edge[1]) < node_count
        and int(edge[0]) != int(edge[1])
    }
    carrier_degree = sum(carrier_i in edge for edge in topology)

    players = network.get("players") or []
    if len(players) == node_count:
        possession_nodes = {
            index
            for index, player in enumerate(players)
            if player.get("role") in {"carrier", "attacker"}
        }
        defence_nodes = {
            index
            for index, player in enumerate(players)
            if player.get("role") == "defender"
        }
    else:
        possession_nodes = {carrier_i, *network.get("attackers", [])}
        defence_nodes = set(network.get("defenders", []))
    cross_edges = sum(
        (left in possession_nodes and right in defence_nodes)
        or (right in possession_nodes and left in defence_nodes)
        for left, right in topology
    )
    return {
        "del_cross_role_edge_ratio": (
            cross_edges / len(topology) if topology else 0.0
        ),
        "del_carrier_degree_norm": (
            carrier_degree / (node_count - 1) if node_count > 1 else 0.0
        ),
    }


def network_features(
    network: dict,
    events: list[dict],
    position: int,
    params: PressureParams = PressureParams(),
    frame: dict | None = None,
    *,
    resolve_raw_incoming: bool = False,
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
    # SPN supplement read-outs use the inclusive threshold in the main version.
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
    distance, sine, cosine = (
        _incoming_ball(events, position, flip)
        if resolve_raw_incoming
        else (np.nan, np.nan, np.nan)
    )
    carrier_x_norm = (
        _legacy_carrier_x_norm(network["event"], frame)
        if frame is not None
        else float(carrier[0] / 105.0)
    )
    if carrier_x_norm is None:
        raise ValueError("The main SPN carrier_x_norm requires a freeze-frame actor")
    structural = {
        **_voronoi_area_features(network),
        **_pressure_graph_features(network),
        **_delaunay_graph_features(network),
    }
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
        **structural,
    }


def add_temporal_features(table: pd.DataFrame) -> pd.DataFrame:
    """Add discrete current-minus-previous differences between adjacent anchors."""
    table = table.sort_values(["match_id", "seq_id", "ev_pos"]).copy()
    groups = table.groupby(["match_id", "seq_id"], sort=False)
    adjacent_anchor = groups["ev_pos"].diff().eq(1)
    for target, source in SPN_TEMPORAL_SOURCES.items():
        values = groups[source].diff().where(adjacent_anchor)
        context_column = f"pre_context_{source}"
        if context_column in table:
            first_with_context = (
                table["ev_pos"].eq(0)
                & table.get(
                    "pre_context_spn_valid",
                    pd.Series(False, index=table.index),
                ).eq(True)  # noqa: E712
                & pd.to_numeric(
                    table[context_column], errors="coerce"
                ).notna()
            )
            values = values.mask(
                first_with_context,
                table[source] - table[context_column],
            )
        table[target] = values
    return table


def _finite_point(x, y) -> list[float] | None:
    """Return a two-dimensional StatsBomb point when both values are finite."""
    try:
        point = [float(x), float(y)]
    except (TypeError, ValueError):
        return None
    return point if np.isfinite(point).all() else None


def _sequence_anchor_point(
    anchor: pd.Series,
    by_id: dict[str, tuple[int, dict]],
) -> list[float] | None:
    """Use S1's resolved anchor point, falling back to the raw event location."""
    point = _finite_point(
        anchor.get("actor_x_norm"),
        anchor.get("actor_y"),
    )
    if point is not None:
        return point
    item = by_id.get(anchor.get("anchor_event_id", ""))
    location = item[1].get("location") if item is not None else None
    return (
        _finite_point(location[0], location[1])
        if isinstance(location, (list, tuple)) and len(location) >= 2
        else None
    )


def _incoming_from_context_points(
    current: list[float] | None,
    source: list[float] | None,
) -> dict[str, float]:
    """Return source-to-current direction, or NaN for an unresolved context."""
    empty = {
        "ball_in_dist": np.nan,
        "ball_in_angle_sin": np.nan,
        "ball_in_angle_cos": np.nan,
    }
    if current is None or source is None:
        return empty
    if float(np.linalg.norm(np.asarray(current) - np.asarray(source))) <= (
        SEQUENCE_CONTEXT_LOCATION_TOLERANCE
    ):
        return empty
    delta = metric_xy(*current) - metric_xy(*source)
    distance = float(np.linalg.norm(delta))
    if not np.isfinite(distance) or distance <= 0.0:
        return empty
    angle = float(np.arctan2(delta[1], delta[0]))
    return {
        "ball_in_dist": distance,
        "ball_in_angle_sin": float(np.sin(angle)),
        "ball_in_angle_cos": float(np.cos(angle)),
    }


def _incoming_is_resolved(incoming: dict[str, float]) -> bool:
    try:
        return all(np.isfinite(float(value)) for value in incoming.values())
    except (TypeError, ValueError):
        return False


def _sequence_context_incoming(
    anchors: pd.DataFrame,
    by_id: dict[str, tuple[int, dict]],
) -> dict[tuple[int, int, str], dict]:
    """Resolve incoming direction only through the sequence context chain.

    The first anchor consumes the pre-sequence context already resolved by
    S1.  Every later anchor starts at the immediately preceding sequence
    anchor. Same-location anchors are followed recursively until a distinct
    sequence point is found, with the sequence's pre-context as the final
    fallback. Raw events between anchors are never searched here. If the
    chain is exhausted, all three incoming fields remain NaN.
    """
    result: dict[tuple[int, int, str], dict] = {}
    for sequence_id, sequence in anchors.groupby("seq_id", sort=False):
        ordered = sequence.sort_values("ev_pos").reset_index(drop=True)
        points = [
            _sequence_anchor_point(anchor, by_id)
            for _, anchor in ordered.iterrows()
        ]
        first = ordered.iloc[0]
        pre_context_point = _finite_point(
            first.get("pre_context_ball_x"),
            first.get("pre_context_ball_y"),
        )
        pre_context_selected = (
            first.get("pre_context_status") == "selected"
            and bool(first.get("pre_context_direction_valid", False))
        )

        for ordinal, anchor in ordered.iterrows():
            key = (
                int(sequence_id),
                int(anchor["ev_pos"]),
                anchor["anchor_event_id"],
            )
            current = points[ordinal]
            audit = {
                "ball_in_context_status": "unresolved",
                "ball_in_context_source": "",
                "ball_in_context_event_id": "",
                "ball_in_context_event_type": "",
                "ball_in_context_anchor_hops": np.nan,
            }

            if ordinal == 0:
                incoming = {
                    "ball_in_dist": anchor.get("pre_context_distance_m", np.nan),
                    "ball_in_angle_sin": anchor.get("pre_context_angle_sin", np.nan),
                    "ball_in_angle_cos": anchor.get("pre_context_angle_cos", np.nan),
                }
                if pre_context_selected and _incoming_is_resolved(incoming):
                    audit.update(
                        {
                            "ball_in_context_status": "resolved",
                            "ball_in_context_source": "pre_sequence_context",
                            "ball_in_context_event_id": first.get(
                                "pre_context_event_id", ""
                            ),
                            "ball_in_context_event_type": first.get(
                                "pre_context_event_type", ""
                            ),
                            "ball_in_context_anchor_hops": 0,
                        }
                    )
                else:
                    incoming = _incoming_from_context_points(None, None)
                    audit["ball_in_context_status"] = (
                        "unresolved_pre_sequence_context"
                    )
                result[key] = {**incoming, **audit}
                continue

            source_ordinal = None
            for prior in range(ordinal - 1, -1, -1):
                candidate = _incoming_from_context_points(
                    current, points[prior]
                )
                if _incoming_is_resolved(candidate):
                    source_ordinal = prior
                    break
            if source_ordinal is not None:
                source_anchor = ordered.iloc[source_ordinal]
                incoming = _incoming_from_context_points(
                    current, points[source_ordinal]
                )
                audit.update(
                    {
                        "ball_in_context_status": "resolved",
                        "ball_in_context_source": "sequence_anchor",
                        "ball_in_context_event_id": source_anchor[
                            "anchor_event_id"
                        ],
                        "ball_in_context_event_type": source_anchor.get(
                            "event_type", ""
                        ),
                        "ball_in_context_anchor_hops": (
                            ordinal - source_ordinal
                        ),
                    }
                )
            elif pre_context_selected:
                incoming = _incoming_from_context_points(
                    current, pre_context_point
                )
                if np.isfinite(incoming["ball_in_dist"]):
                    audit.update(
                        {
                            "ball_in_context_status": "resolved",
                            "ball_in_context_source": (
                                "pre_sequence_context_fallback"
                            ),
                            "ball_in_context_event_id": first.get(
                                "pre_context_event_id", ""
                            ),
                            "ball_in_context_event_type": first.get(
                                "pre_context_event_type", ""
                            ),
                            "ball_in_context_anchor_hops": ordinal + 1,
                        }
                    )
                else:
                    audit["ball_in_context_status"] = (
                        "unresolved_same_location_chain"
                    )
            else:
                incoming = _incoming_from_context_points(None, None)
                audit["ball_in_context_status"] = (
                    "unresolved_same_location_chain"
                )
            result[key] = {**incoming, **audit}
    return result


def _context_actor_event(event: dict) -> dict:
    """Treat the selected ball-contact actor as the context carrier.

    Most selected context events are already in-possession actions.  For a
    defensive contact such as an interception or block, this small view keeps
    teammate/opponent roles centred on the event actor rather than applying
    the normal defensive-event inversion.  The original event is untouched.
    """
    actor_team = event.get("team") or {}
    if not actor_team.get("id"):
        return event
    return {**event, "possession_team": dict(actor_team)}


def build_spn_feature_table(
    sequences: pd.DataFrame,
    labels: pd.DataFrame,
    events_dir: str | Path,
    three_sixty_dir: str | Path,
    params: PressureParams = PressureParams(),
) -> pd.DataFrame:
    """Build all anchor SPNs and join final-output regression targets.

    A sequence-first anchor uses S1's nearest distinct pre-context ball point
    for incoming direction. Later anchors resolve direction only through the
    preceding sequence-anchor chain; same-location anchors are followed back
    to the first distinct anchor or the sequence pre-context. Raw intervening
    events are never scanned, and an exhausted chain produces NaN direction
    fields. If the exact pre-context event also has a usable 360 frame, its SPN
    supplies the previous value for first-anchor temporal differences. Later
    differences remain current high-pressure anchor minus the immediately
    preceding sequence anchor, even when their locations are equal. Context
    itself is never emitted as a feature row.
    S1 supplies one target per anchor: every event in a sequence inherits the
    same final output, numeric event value, and 1/anchor_count training weight.
    Immediate event states remain audit fields and are never SPN features.
    """
    required_sequence_columns = {
        "match_id",
        "seq_id",
        "ev_pos",
        "anchor_event_id",
    }
    required_label_columns = {
        "match_id",
        "seq_id",
        "ev_pos",
        "anchor_event_id",
        "outcome_tag",
        "terminal",
        "future_output_target",
        "event_value",
        "training_weight",
        "value_model_eligible",
    }
    if missing := required_sequence_columns - set(sequences.columns):
        raise ValueError(f"sequences is missing required columns: {sorted(missing)}")
    if missing := required_label_columns - set(labels.columns):
        raise ValueError(f"labels is missing required columns: {sorted(missing)}")

    anchors = sequences.copy()
    anchors["match_id"] = anchors["match_id"].astype(str)
    targets = labels.copy()
    targets["match_id"] = targets["match_id"].astype(str)
    label_key = ["match_id", "seq_id", "ev_pos", "anchor_event_id"]
    if targets.duplicated(label_key).any():
        raise ValueError("labels must contain at most one row per sequence anchor")
    anchor_keys = anchors[label_key].drop_duplicates()
    target_keys = targets[label_key].drop_duplicates()
    matched_keys = anchor_keys.merge(target_keys, on=label_key, how="inner")
    if len(anchor_keys) != len(target_keys) or len(matched_keys) != len(anchor_keys):
        raise ValueError("labels must contain exactly one row for every sequence anchor")
    rows = []
    for match_id, group in anchors.groupby("match_id", sort=True):
        events, frames = _events_and_frames(events_dir, three_sixty_dir, match_id)
        by_id = {event["id"]: (position, event) for position, event in enumerate(events)}
        sequence_incoming = _sequence_context_incoming(group, by_id)
        match_teams = sorted({
            event.get("team", {}).get("name")
            for event in events
            if event.get("team", {}).get("name")
        })
        for _, anchor in group.iterrows():
            item = by_id.get(anchor["anchor_event_id"])
            frame = frames.get(anchor["anchor_event_id"])
            if item is None or frame is None:
                continue
            position, event = item
            try:
                values = network_features(
                    build_spn_network(event, frame, params),
                    events,
                    position,
                    params,
                    frame,
                    resolve_raw_incoming=False,
                )
            except ValueError:
                continue
            context_values = {
                f"pre_context_{source}": np.nan
                for source in SPN_TEMPORAL_SOURCES.values()
            }
            context_spn_valid = False
            context_id = anchor.get("pre_context_event_id", "")
            if (
                int(anchor["ev_pos"]) == 0
                and anchor.get("pre_context_status") == "selected"
                and isinstance(context_id, str)
                and context_id
            ):
                context_item = by_id.get(context_id)
                context_frame = frames.get(context_id)
                if context_item is not None and context_frame is not None:
                    context_position, context_event = context_item
                    context_event_view = _context_actor_event(context_event)
                    try:
                        source_values = network_features(
                            build_spn_network(
                                context_event_view,
                                context_frame,
                                params,
                            ),
                            events,
                            context_position,
                            params,
                            context_frame,
                            resolve_raw_incoming=False,
                        )
                    except ValueError:
                        pass
                    else:
                        context_values = {
                            f"pre_context_{source}": source_values[source]
                            for source in SPN_TEMPORAL_SOURCES.values()
                        }
                        context_spn_valid = True

            incoming = sequence_incoming[
                (
                    int(anchor["seq_id"]),
                    int(anchor["ev_pos"]),
                    anchor["anchor_event_id"],
                )
            ]
            values.update(
                {
                    feature: incoming[feature]
                    for feature in (
                        "ball_in_dist",
                        "ball_in_angle_sin",
                        "ball_in_angle_cos",
                    )
                }
            )
            pressed_team = (
                anchor.get("possession_team_name") or anchor.get("team_name")
            )
            press_team = next(
                (team for team in match_teams if team != pressed_team),
                None,
            ) if len(match_teams) == 2 else None
            pre_context_metadata = {
                column: anchor[column]
                for column in anchors.columns
                if column.startswith("pre_context_")
            }
            rows.append({
                "match_id": str(match_id), "seq_id": int(anchor["seq_id"]), "ev_pos": int(anchor["ev_pos"]),
                "anchor_event_id": anchor["anchor_event_id"],
                "event_type": anchor.get("event_type") or (event.get("type") or {}).get("name"),
                "period": anchor.get("period"), "timestamp": anchor.get("timestamp"),
                "team_name": anchor.get("team_name"),
                "pressed_team": pressed_team, "press_team": press_team,
                **pre_context_metadata,
                **context_values,
                "pre_context_spn_valid": context_spn_valid,
                **{
                    column: value
                    for column, value in incoming.items()
                    if column.startswith("ball_in_context_")
                },
                **values,
            })
    if not rows:
        raise ValueError(
            "No SPN feature rows could be built from the supplied sequences "
            "and freeze frames"
        )
    anchor_features = add_temporal_features(pd.DataFrame(rows))
    target_columns = [
        "match_id",
        "seq_id",
        "ev_pos",
        "anchor_event_id",
        "outcome_tag",
        "terminal",
        "future_output_target",
        "event_value",
        "training_weight",
        "value_model_eligible",
    ]
    optional_target_columns = [
        column
        for column in (
            "event_state",
            "state",
            "transition_outcome_tag",
            "censored",
            "event_label_source",
            "is_sequence_last",
            "sequence_end_state",
            "sequence_output",
            "anchor_count",
        )
        if column in targets.columns
    ]
    labelled_rows = anchor_features.merge(
        targets[target_columns + optional_target_columns],
        on=label_key,
        how="inner",
        validate="one_to_one",
    )
    if labelled_rows.empty:
        raise ValueError(
            "No sequence anchors matched between sequences and event labels"
        )
    return labelled_rows.sort_values(["match_id", "seq_id", "ev_pos"]).reset_index(drop=True)


def save_feature_table(table: pd.DataFrame, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "spn_features.parquet"
    table.to_parquet(path, index=False)
    return path
