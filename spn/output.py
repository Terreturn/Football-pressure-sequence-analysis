"""Compact JSON and CSV serializers for the public SPN pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .model import KEY_COLUMNS, SPNModel


EMPTY_PREDICTION_COLUMNS = KEY_COLUMNS + [
    "period", "timestamp", "event_type", "pressed_team", "press_team",
    "observed_outcome", "state", "terminal", "censored",
    "sequence_end_state", "sequence_output", "evaluation_eligible",
    "p_fail", "p_neutral", "p_success", "predicted_class", "confidence",
    "model_id", "model_version",
]
EMPTY_SEQUENCE_COLUMNS = [
    "match_id", "seq_id", "anchor_count", "start_timestamp", "end_timestamp",
    "pressed_team", "press_team", "observed_outcome", "sequence_end_state",
    "sequence_output",
]


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def prediction_table(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    model: SPNModel,
) -> pd.DataFrame:
    """Join anchor metadata, observed labels, and frozen-model probabilities."""
    metadata = [
        column
        for column in (
            *KEY_COLUMNS,
            "period",
            "timestamp",
            "event_type",
            "pressed_team",
            "press_team",
        )
        if column in features.columns
    ]
    table = features[metadata].merge(
        predictions,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    label_columns = [
        column
        for column in (
            *KEY_COLUMNS,
            "outcome_tag",
            "state",
            "terminal",
            "censored",
            "sequence_end_state",
            "sequence_output",
        )
        if column in labels.columns
    ]
    table = table.merge(
        labels[label_columns],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    table = table.rename(columns={"outcome_tag": "observed_outcome"})
    table["evaluation_eligible"] = table.get(
        "observed_outcome", pd.Series(index=table.index, dtype=object)
    ).isin(model.class_names)
    table["model_id"] = model.config["model_id"]
    table["model_version"] = model.config["model_version"]
    return table.sort_values(["match_id", "seq_id", "ev_pos"]).reset_index(drop=True)


def sequence_table(
    sequences: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Return one compact audit row per detected pressure sequence."""
    if sequences.empty:
        return pd.DataFrame(
            columns=[
                "match_id", "seq_id", "anchor_count", "start_timestamp",
                "end_timestamp", "pressed_team", "press_team", "observed_outcome",
            ]
        )
    rows: list[dict[str, Any]] = []
    for (match_id, seq_id), group in sequences.groupby(
        ["match_id", "seq_id"], sort=True
    ):
        ordered = group.sort_values("ev_pos")
        label_group = labels.loc[
            labels["match_id"].astype(str).eq(str(match_id))
            & labels["seq_id"].eq(seq_id)
        ]
        first_label = label_group.iloc[0] if not label_group.empty else {}
        rows.append(
            {
                "match_id": str(match_id),
                "seq_id": int(seq_id),
                "anchor_count": int(len(ordered)),
                "start_timestamp": ordered.iloc[0].get("timestamp"),
                "end_timestamp": ordered.iloc[-1].get("timestamp"),
                "pressed_team": ordered.iloc[0].get("pressed_team_name"),
                "press_team": first_label.get("press_team_name"),
                "observed_outcome": first_label.get("outcome_tag"),
                "sequence_end_state": first_label.get("sequence_end_state"),
                "sequence_output": first_label.get("sequence_output"),
            }
        )
    return pd.DataFrame(rows)


def build_result(
    *,
    match: Any,
    sequences: pd.DataFrame,
    labels: pd.DataFrame,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    model: SPNModel,
    status: str = "ok",
) -> dict[str, Any]:
    """Build the stable public result envelope."""
    scored = prediction_table(features, labels, predictions, model)
    sequence_summary = sequence_table(sequences, labels)
    completeness = features.attrs.get("s3_completeness_audit", {})
    audit = {
        "event_count": len(match.events),
        "frame_count": int(match.raw_frame_count),
        "indexed_frame_count": len(match.frames),
        "pressure_anchor_count": int(len(sequences)),
        "sequence_count": int(
            sequences[["match_id", "seq_id"]].drop_duplicates().shape[0]
        ) if not sequences.empty else 0,
        "labelled_anchor_count": int(len(labels)),
        "feature_anchor_count": int(len(features)),
        "excluded_incomplete_sequences": int(
            completeness.get("excluded_incomplete_sequences", 0)
        ),
        "prediction_count": int(len(scored)),
        "warnings": list(match.warnings),
    }
    return _clean_json(
        {
            "schema_version": "spn-output/1.0",
            "status": status,
            "match_id": match.match_id,
            "model": {
                "model_id": model.config["model_id"],
                "model_version": model.config["model_version"],
                "probability_layer": model.config["probability_layer"],
            },
            "audit": audit,
            "sequences": sequence_summary.to_dict("records"),
            "predictions": scored.to_dict("records"),
        }
    )


def empty_result(match: Any, model: SPNModel) -> dict[str, Any]:
    """Return a valid envelope when no pressure sequence is detected."""
    return _clean_json(
        {
            "schema_version": "spn-output/1.0",
            "status": "no_sequences",
            "match_id": match.match_id,
            "model": {
                "model_id": model.config["model_id"],
                "model_version": model.config["model_version"],
                "probability_layer": model.config["probability_layer"],
            },
            "audit": {
                "event_count": len(match.events),
                "frame_count": int(match.raw_frame_count),
                "indexed_frame_count": len(match.frames),
                "pressure_anchor_count": 0,
                "sequence_count": 0,
                "labelled_anchor_count": 0,
                "feature_anchor_count": 0,
                "excluded_incomplete_sequences": 0,
                "prediction_count": 0,
                "warnings": list(match.warnings),
            },
            "sequences": [],
            "predictions": [],
        }
    )


def save_result(result: dict[str, Any], output_dir: str | Path) -> None:
    """Write result.json plus the two compact public tables."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    predictions = pd.DataFrame(result["predictions"])
    if predictions.empty:
        predictions = pd.DataFrame(columns=EMPTY_PREDICTION_COLUMNS)
    predictions.to_csv(target / "predictions.csv", index=False)
    sequences = pd.DataFrame(result["sequences"])
    if sequences.empty:
        sequences = pd.DataFrame(columns=EMPTY_SEQUENCE_COLUMNS)
    sequences.to_csv(target / "sequences.csv", index=False)
