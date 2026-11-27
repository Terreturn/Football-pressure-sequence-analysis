"""Build the ordered 26 SPN inputs from network geometry and preceding events."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import LineString, box

from .data_processing import ANCHOR_KEYS, MatchData, load_match_data
from .networks import (
    P,
    PressureParams,
    build_network,
    network_parameters,
    projection,
    team_shapes,
    xy_m,
)
from .pressure_events import team


# Ordered inputs from the sequence fit, followed by the receiver and space features.
ARCHIVED_INPUTS = [
    "ball_in_dist",
    "ball_in_angle_cos",
    "carrier_x_norm",
    "d_carrier_x_norm_dt",
    "carrier_touchline_m",
    "P_total",
    "d_P_total_dt",
    "effective_pressers",
    "n_open_pass",
    "best_forward_pass_w",
    "press_target_entropy",
    "mean_receiver_pressure",
    "pressure_mean_focus_entropy",
    "vor_carrier_area_share",
    "vor_forward_receiver_area_share",
    "d_vor_carrier_area_share",
    "d_vor_forward_receiver_area_share",
    "sg_n_outlets_uncrossed",
    "sg_pressed_edges_cut_frac_2m",
    "sg_press_max_edge",
    "seq_after_opponent",
]

DELTAS = {
    "d_carrier_x_norm_dt": "carrier_x_norm",
    "d_P_total_dt": "P_total",
    "d_vor_carrier_area_share": "vor_carrier_area_share",
    "d_vor_forward_receiver_area_share": "vor_forward_receiver_area_share",
}

SHAPE = ("sg_n_outlets_uncrossed", "sg_pressed_edges_cut_frac_2m", "sg_press_max_edge")


RECEIVER_INPUTS = ("best_forward_receiver_relief", "forward_pass_space_share")


SPACE_INPUTS = (
    "local_defender_spread_m",
    "nearest_presser_support_m",
    "carrier_forward_space_fraction",
)

ADDED_INPUTS = list(RECEIVER_INPUTS) + list(SPACE_INPUTS)
INPUTS = ARCHIVED_INPUTS + ADDED_INPUTS


SEQUENCE_COLUMNS = [
    "match_id",
    "seq_id",
    "ev_pos",
    "anchor_event_id",
    "pressing_team_id",
    "outcome_tag",
    "training_weight",
]


def build_feature_table(match_data: MatchData, sequences, labels, params=PressureParams()):
    """Measure every anchor of each resolved, representable sequence in this match."""
    # CSV readers may infer integer match IDs while MatchData uses the provider ID as text.
    sequences = sequences.assign(match_id=sequences.match_id.astype(str))
    labels = labels.assign(match_id=labels.match_id.astype(str))
    targets = [
        "press_team_id",
        "press_team_name",
        "training_weight",
        "outcome_tag",
        "value_model_eligible",
    ]
    if "anchor_count" not in sequences:
        targets.append("anchor_count")
    anchors = sequences.merge(
        labels[ANCHOR_KEYS + targets],
        on=ANCHOR_KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not anchors["_merge"].eq("both").all():
        raise ValueError("Sequence and label tables must describe the same anchors")
    if not anchors.match_id.astype(str).eq(str(match_data.match_id)).all():
        raise ValueError("Feature inputs must belong to the supplied match")

    by_id = {event["id"]: event for event in match_data.events}
    rows = []
    excluded_incomplete = excluded_censored = 0
    parameters = network_parameters(params)
    groups = anchors.sort_values(["seq_id", "ev_pos"]).groupby("seq_id", sort=False)
    for _, group in groups:
        if not group["value_model_eligible"].isin([True, "True"]).all():
            excluded_censored += 1
            continue
        if (
            group.ev_pos.tolist() != list(range(len(group)))
            or not group.anchor_count.eq(len(group)).all()
        ):
            raise ValueError("Incomplete sequence membership in input anchor table")
        records = group.to_dict("records")
        measured = sequence_rows(records, by_id, match_data.frames, match_data.match_id, parameters)
        if not measured:
            excluded_incomplete += 1
            continue
        for row, anchor in zip(measured, records):
            row.update(
                period=anchor["period"],
                timestamp=anchor["timestamp"],
                event_type=anchor["event_type"],
                pressed_team=anchor["pressed_team_name"],
                press_team=anchor["press_team_name"],
            )
        rows.extend(measured)

    table = pd.DataFrame.from_records(
        rows,
        columns=SEQUENCE_COLUMNS
        + INPUTS
        + ["period", "timestamp", "event_type", "pressed_team", "press_team"],
    )
    audit_fields = [
        column
        for column in (
            "terminal",
            "future_output_target",
            "event_value",
            "value_model_eligible",
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
        if column in labels
    ]
    table = table.merge(
        labels[ANCHOR_KEYS + audit_fields], on=ANCHOR_KEYS, how="left", validate="one_to_one"
    )
    table.attrs["s3_completeness_audit"] = {
        "expected_anchors": len(anchors),
        "expected_sequences": groups.ngroups,
        "retained_anchors": len(table),
        "excluded_incomplete_sequences": excluded_incomplete,
        "excluded_censored_sequences": excluded_censored,
    }
    return table


def sequence_rows(anchors, events_by_id, frames, match_id, parameters=P):
    """Measure the 26 inputs in order for one resolved sequence. Return an empty list if any frame is unusable.

    The previous frame supplies spatial differences. At onset, a usable pre-sequence
    event can supply that reference. Each returned row retains the outcome and the
    sequence-normalized training weight. MODEL.md gives the units by column.
    """
    first = anchors[0]
    incoming = sequence_incoming(anchors, events_by_id)
    previous, _ = context_features(
        first, events_by_id[first["anchor_event_id"]], events_by_id, frames, parameters
    )
    preceding_opponent = after_opponent(first)
    rows = []
    for anchor in anchors:
        event_id = anchor["anchor_event_id"]
        event = events_by_id[event_id]
        frame = frames.get(event_id)
        if frame is None:
            return []
        try:
            network = build_network(event, frame, parameters)
            current = static_features(network, event, frame)
        except ValueError:
            # Eligibility applies to the whole sequence, even if its onset was representable.
            return []
        rows.append(
            {
                "match_id": str(match_id),
                "seq_id": anchor["seq_id"],
                "ev_pos": anchor["ev_pos"],
                "anchor_event_id": event_id,
                "pressing_team_id": anchor["press_team_id"],
                "outcome_tag": anchor["outcome_tag"],
                "training_weight": anchor["training_weight"],
                "seq_after_opponent": preceding_opponent,
                **frame_features(network, current, previous, corrected_shape=False),
                **incoming[event_id],
            }
        )
        previous = current
    return rows


def frame_features(network, current, previous, *, corrected_shape):
    """Combine frame measurements using the shape convention of the relevant fitted model."""
    graphs, _ = team_shapes(network, corrected=corrected_shape)
    changes = {
        name: current[source] - previous.get(source, math.nan)
        for name, source in DELTAS.items()
    }
    return {
        **current,
        **changes,
        **shape_features(network, graphs, corrected=corrected_shape),
        **receiver_features(network),
        **spatial_features(network, graphs),
    }


# Pressure, passing and available space in one frame.

def entropy(weights, support):
    """Return normalized natural-log entropy of nonnegative unitless weights over the full support count."""
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    if total <= 0 or support <= 1:
        return 0.0
    distribution = weights[weights > 0] / total
    return -float(np.sum(distribution * np.log(distribution))) / math.log(support)


def area_features(network):
    """Return carrier and forward-receiver area fractions of camera-visible pitch, or paired NaNs."""
    empty = {
        "vor_carrier_area_share": math.nan,
        "vor_forward_receiver_area_share": math.nan,
    }
    visible = network["visible"]
    if (
        visible is None
        or visible.is_empty
        or not math.isfinite(visible.area)
        or visible.area <= 0
    ):
        return empty
    carrier = network["carrier"]
    forward = [
        i
        for i in network["receivers"]
        if network["xy"][i, 0] > network["xy"][carrier, 0]
    ]
    polygons = [network["areas"][i] for i in [carrier] + forward]
    if any(polygon is None or not math.isfinite(polygon.area) for polygon in polygons):
        return empty
    return {
        "vor_carrier_area_share": polygons[0].area / visible.area,
        "vor_forward_receiver_area_share": sum(p.area for p in polygons[1:])
        / visible.area,
    }


def actor_x(event, frame, flip=None):
    """Return clipped flagged-actor x as pitch fraction; raise if no actor location, even if network carrier was inserted."""
    reverse = (
        team(event) != team(event, "possession_team") if flip is None else bool(flip)
    )
    for player in frame.get("freeze_frame") or []:
        if player.get("actor") and player.get("location") is not None:
            x = float(player["location"][0])
            return float(np.clip(120.0 - x if reverse else x, 0, 120) / 120.0)
    raise ValueError("missing_freeze_frame_actor")


def static_features(network, event, frame, flip=None):
    """Return selected non-shape static features: meter distances, counts and dimensionless weights/fractions."""
    carrier, xy, p = network["carrier"], network["xy"], network["parameters"]
    weights = network["pressure_weights"]
    carrier_weights = weights[:, 0]
    active = carrier_weights[carrier_weights > p.epsilon]
    # Entropy and mean pressure intentionally use different threshold ties.
    target_sums = np.where(weights > p.epsilon, weights, 0.0).sum(axis=0)
    inclusive_sums = np.where(weights >= p.epsilon, weights, 0.0).sum(axis=0)
    focused = []
    for row in weights:
        retained = row[row > p.epsilon]
        if len(retained):
            # Match the documented two-step normalization of pressure edges.
            normalized = retained / sum(retained)
            focused.append(entropy(normalized, weights.shape[1]))
    forward = [
        edge["w"] for edge in network["passes"] if xy[edge["dst"], 0] > xy[carrier, 0]
    ]
    return {
        "carrier_x_norm": actor_x(event, frame, flip),
        "carrier_touchline_m": float(min(xy[carrier, 1], 68.0 - xy[carrier, 1])),
        "P_total": float(1.0 - np.prod(1.0 - carrier_weights))
        if len(carrier_weights)
        else 0.0,
        "effective_pressers": float(active.sum() ** 2 / np.sum(active**2))
        if len(active)
        else 0.0,
        "n_open_pass": sum(edge["w"] > p.open_threshold for edge in network["passes"]),
        "best_forward_pass_w": float(max(forward, default=0.0)),
        "press_target_entropy": entropy(target_sums, max(2, len(target_sums))),
        "mean_receiver_pressure": float(np.mean(inclusive_sums[1:]))
        if len(inclusive_sums) > 1
        else 0.0,
        "pressure_mean_focus_entropy": float(np.mean(focused)) if focused else 0.0,
        **area_features(network),
    }


def shape_features(network, graphs, corrected=False):
    """Return outlet count, cut-edge fraction and longest pressing edge in meters; isolate failures only in corrected mode."""
    result = dict.fromkeys(SHAPE, math.nan)
    pressed, pressing = graphs["pressed"], graphs["pressing"]
    if not corrected and (pressed is None or pressing is None):
        return result
    xy, carrier = network["xy"], network["carrier"]
    if pressing is not None:
        segments = [LineString([xy[a], xy[b]]) for a, b in pressing]
        outlets = [i for i in network["receivers"] if not network["nodes"][i]["keeper"]]
        result[SHAPE[0]] = float(
            sum(
                not any(
                    LineString([xy[carrier], xy[i]]).crosses(segment)
                    for segment in segments
                )
                for i in outlets
            )
        )
        result[SHAPE[2]] = float(
            max((np.linalg.norm(xy[a] - xy[b]) for a, b in pressing), default=math.nan)
        )
    if pressed:
        cut = 0
        for a, b in pressed:
            for defender in network["defenders"]:
                distance, t = projection(xy[defender], xy[a], xy[b], zero_t=0.0)
                if 0 <= t <= 1 and distance < 2.0:
                    cut += 1
                    break
        result[SHAPE[1]] = cut / len(pressed)
    return result


def combine_receivers(passing, pressure, area_share, forward):
    """Return two unitless scores from receiver weights, defender-by-receiver pressures, area fractions and a forward mask.

    NaN area fractions signal unavailable geometry; passing and pressure must
    be finite in [0,1]. Empty lists represent no visible receivers, not missing
    input. Geometry availability with no receivers is handled by the wrapper.
    """
    passing, pressure = np.asarray(passing, float), np.asarray(pressure, float)
    area_share, forward = np.asarray(area_share, float), np.asarray(forward)
    n = len(passing)
    if (
        passing.ndim != 1
        or area_share.shape != (n,)
        or forward.shape != (n,)
        or forward.dtype != bool
    ):
        raise ValueError("Receiver vectors or Boolean forward mask differ")
    if pressure.ndim != 2 or pressure.shape[1] != n:
        raise ValueError("Pressure matrix does not match receivers")
    if any(
        not np.isfinite(a).all() or (a < 0).any() or (a > 1).any()
        for a in (passing, pressure)
    ):
        raise ValueError("Passing and pressure weights must lie in [0,1]")
    finite = np.isfinite(area_share)
    if (
        np.isinf(area_share).any()
        or (area_share[finite] < 0).any()
        or (area_share[finite] > 1 + 1e-10).any()
    ):
        raise ValueError("Invalid area fraction")
    freedom = np.prod(1 - pressure, axis=0)
    relief = float(np.max((passing * freedom)[forward], initial=0.0))
    space = float(np.sum((passing * area_share)[forward]))
    return dict(zip(RECEIVER_INPUTS, (relief, space)))


def receiver_features(network):
    """Return two unitless features from an audited network with meter positions and square-meter cells; never mutate it."""
    receivers, carrier = network["receivers"], network["carrier"]
    edges = {edge["dst"]: edge["w"] for edge in network["passes"]}
    if len(edges) != len(network["passes"]) or set(edges) != set(receivers):
        raise ValueError("Passing edges do not match receivers exactly")
    pressure = np.asarray(network["pressure_weights"], float)
    if pressure.shape != (len(network["defenders"]), 1 + len(receivers)):
        raise ValueError("Pressure target order is not carrier then receivers")
    visible = network["visible"]
    area_valid = (
        visible is not None
        and not visible.is_empty
        and math.isfinite(visible.area)
        and visible.area > 0
        and network["areas"][carrier] is not None
    )
    shares = [
        network["areas"][j].area / visible.area
        if area_valid and network["areas"][j] is not None
        else math.nan
        for j in receivers
    ]
    forward = np.array(
        [network["xy"][j, 0] > network["xy"][carrier, 0] for j in receivers], bool
    )
    result = combine_receivers(
        [edges[j] for j in receivers], pressure[:, 1:], shares, forward
    )
    if not area_valid:
        result[RECEIVER_INPUTS[1]] = math.nan
    return result


def ordered_pressers(xy, carrier, defenders):
    """Return presser indices sorted by squared carrier distance then x/y, from finite meter coordinates."""
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Expected finite two-dimensional meter coordinates")
    if len(set(defenders)) != len(defenders) or carrier in defenders:
        raise ValueError("Invalid presser indices")
    return sorted(
        defenders,
        key=lambda i: (
            float(np.sum((xy[i] - xy[carrier]) ** 2)),
            float(xy[i, 0]),
            float(xy[i, 1]),
        ),
    )


def local_spread(xy, indices):
    """Return mean distance in meters from the centroid of exactly five selected points; fewer gives NaN."""
    if len(indices) < 5:
        return math.nan
    points = np.asarray(xy, float)[indices[:5]]
    return float(np.linalg.norm(points - points.mean(axis=0), axis=1).mean())


def support_spacing(xy, nearest, edges):
    """Return the median incident shape-edge length in meters, or NaN for unavailable/empty support."""
    if nearest is None or edges is None:
        return math.nan
    lengths = [
        float(np.linalg.norm(xy[a] - xy[b])) for a, b in edges if nearest in (a, b)
    ]
    return float(np.median(lengths)) if lengths else math.nan


def forward_fraction(cell, carrier_x, visible):
    """Return forward area divided by full clipped carrier-cell area, dimensionless; invalid geometry gives NaN."""
    if (
        visible is None
        or not visible.is_valid
        or visible.is_empty
        or visible.area <= 0
        or cell is None
        or not cell.is_valid
        or cell.is_empty
        or not math.isfinite(cell.area)
        or cell.area <= 0
    ):
        return math.nan
    return float(cell.intersection(box(carrier_x, 0, 105, 68)).area / cell.area)


def spatial_features(network, graphs):
    """Measure local spread, support spacing and forward space on the supplied shape graph."""
    xy, c = network["xy"], network["carrier"]
    ordered = ordered_pressers(xy, c, network["defenders"])
    return dict(
        zip(
            SPACE_INPUTS,
            (
                local_spread(xy, ordered),
                support_spacing(
                    xy, ordered[0] if ordered else None, graphs["pressing"]
                ),
                forward_fraction(
                    network["areas"][c], float(xy[c, 0]), network["visible"]
                ),
            ),
        )
    )


# Direction and changes relative to the preceding event.

def number(value):
    """Convert a scalar or serialized numeric field to float, preserving units; invalid input becomes NaN."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def flag(value):
    """Interpret a bool or serialized bool as an indicator; missing/false values return False."""
    return value is True or str(value) in ("True", "1", "1.0")


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


def after_opponent(anchor):
    """Encode preceding opponent play, leaving unavailable context missing."""
    if anchor.get("pre_context_status") != "selected":
        return math.nan
    return float(anchor.get("pre_context_team_relation") == "opponent")


def build_spn_feature_table(
    sequences,
    labels,
    events_dir=None,
    three_sixty_dir=None,
    params=PressureParams(),
    *,
    match_data=None,
):
    if match_data is not None:
        return build_feature_table(match_data, sequences, labels, params)
    if events_dir is None or three_sixty_dir is None:
        raise ValueError("Supply a match or both input directories")
    tables = []
    for mid, group in sequences.groupby("match_id", sort=True):
        match = load_match_data(
            Path(events_dir) / f"{mid}.json", Path(three_sixty_dir) / f"{mid}.json", str(mid)
        )
        tables.append(
            build_feature_table(
                match, group, labels.loc[labels.match_id.astype(str).eq(str(mid))], params
            )
        )
    if not tables:
        return pd.DataFrame(
            columns=SEQUENCE_COLUMNS
            + INPUTS
            + ["period", "timestamp", "event_type", "pressed_team", "press_team"]
        )
    return pd.concat(tables, ignore_index=True)


def save_feature_table(table, output_dir):
    path = Path(output_dir) / "spn_features.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return path
