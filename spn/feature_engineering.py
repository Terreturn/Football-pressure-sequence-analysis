"""Read paired StatsBomb event and 360 files and construct the sequence model's inputs."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from spn._core.sequence_context import context_features, sequence_incoming
from spn._core.network_features import (
    DELTAS,
    FEATURES,
    shape_features,
    static_features,
)
from spn._core.networks import P, build_network, team_shapes
from spn._core.receiver_features import NAMES as RECEIVER_INPUTS, receiver_features
from .data_processing import ANCHOR_KEYS, MatchData, load_match_data
from ._network import PressureParams, network_parameters
from spn._core.spatial_features import NAMES as SPACE_INPUTS, spatial_features

# XGBoost stores columns in this order. The first 21 came from the archived sequence matrix.
ARCHIVED_INPUTS = [name for name in FEATURES if name != "new_convention"]
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


def sequence_rows(anchors, events_by_id, frames, match_id, parameters=P):
    """Measure the 26 inputs in order for one resolved sequence. Return an empty list if any frame is unusable.

    The previous frame supplies spatial differences. At onset, a usable pre-sequence
    event can supply that reference. Each returned row retains the outcome and the
    sequence-normalized training weight. FEATURES.md gives the units by column.
    """
    first = anchors[0]
    incoming = sequence_incoming(anchors, events_by_id)
    previous, _ = context_features(
        first, events_by_id[first["anchor_event_id"]], events_by_id, frames, parameters
    )
    after_opponent = (
        float(first.get("pre_context_team_relation") == "opponent")
        if first.get("pre_context_status") == "selected"
        else math.nan
    )
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
        changes = {
            name: current[source] - previous.get(source, math.nan)
            for name, source in DELTAS.items()
        }
        # The pooled sequence fit used the original team shape graphs on this same frame.
        graphs, _ = team_shapes(network, corrected=False)
        spatial, _ = spatial_features(network, graphs)
        rows.append(
            {
                "match_id": str(match_id),
                "seq_id": anchor["seq_id"],
                "ev_pos": anchor["ev_pos"],
                "anchor_event_id": event_id,
                "pressing_team_id": anchor["press_team_id"],
                "outcome_tag": anchor["outcome_tag"],
                "training_weight": anchor["training_weight"],
                "seq_after_opponent": after_opponent,
                **current,
                **changes,
                **incoming[event_id],
                **shape_features(network, graphs),
                **receiver_features(network),
                **spatial,
            }
        )
        previous = current
    return rows


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
