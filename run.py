#!/usr/bin/env python3
"""Run the frozen public SPN pipeline on paired StatsBomb JSON data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from spn._network import PressureParams
from spn.data_processing import (
    LabelConfig,
    SequenceConfig,
    build_sequences_from_match,
    load_match_data,
)
from spn.feature_engineering import build_feature_table
from spn.model import SPNModel
from spn.output import build_result, empty_result, save_result


def _runtime_configs(model: SPNModel) -> tuple[PressureParams, SequenceConfig, LabelConfig]:
    pressure = PressureParams(**model.config["pressure"])
    sequence = SequenceConfig(**model.config["sequence"])
    labels = LabelConfig(**model.config["labels"])
    return pressure, sequence, labels


def run_pipeline(
    events_source: str | Path | list[dict],
    three_sixty_source: str | Path | list[dict],
    output_dir: str | Path,
    match_id: str | None = None,
    *,
    model: SPNModel | None = None,
) -> dict[str, Any]:
    """Process one match and write JSON/CSV inference outputs."""
    frozen_model = model or SPNModel()
    pressure, sequence_config, label_config = _runtime_configs(frozen_model)
    match = load_match_data(
        events_source,
        three_sixty_source,
        match_id=match_id,
    )
    sequences, labels = build_sequences_from_match(
        match,
        config=sequence_config,
        params=pressure,
        label_config=label_config,
    )
    if sequences.empty:
        result = empty_result(match, frozen_model)
        save_result(result, output_dir)
        return result
    features = build_feature_table(
        match,
        sequences,
        labels,
        params=pressure,
    )
    predictions = frozen_model.predict(features)
    result = build_result(
        match=match,
        sequences=sequences,
        labels=labels,
        features=features,
        predictions=predictions,
        model=frozen_model,
    )
    save_result(result, output_dir)
    return result


def _paired_json_files(events_dir: Path, frames_dir: Path) -> list[tuple[str, Path, Path]]:
    if not events_dir.is_dir() or not frames_dir.is_dir():
        raise FileNotFoundError("events-dir and three-sixty-dir must both exist")
    events = {path.stem: path for path in events_dir.glob("*.json")}
    frames = {path.stem: path for path in frames_dir.glob("*.json")}
    common = sorted(set(events) & set(frames))
    if not common:
        raise ValueError("no paired event/360 JSON filenames were found")
    return [(match_id, events[match_id], frames[match_id]) for match_id in common]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, help="one StatsBomb event JSON file")
    parser.add_argument("--three-sixty", type=Path, help="one StatsBomb 360 JSON file")
    parser.add_argument("--match-id", help="optional match identifier for single-file mode")
    parser.add_argument("--events-dir", type=Path, help="directory of event JSON files")
    parser.add_argument("--three-sixty-dir", type=Path, help="directory of 360 JSON files")
    parser.add_argument("--output", type=Path, required=True, help="output directory")
    args = parser.parse_args()
    single = args.events is not None or args.three_sixty is not None
    batch = args.events_dir is not None or args.three_sixty_dir is not None
    if single == batch:
        parser.error("choose either single-file mode or directory mode")
    model = SPNModel()
    if single:
        if args.events is None or args.three_sixty is None:
            parser.error("--events and --three-sixty must be supplied together")
        result = run_pipeline(
            args.events,
            args.three_sixty,
            args.output,
            match_id=args.match_id,
            model=model,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "match_id": result["match_id"],
                    "prediction_count": result["audit"]["prediction_count"],
                    "output": str(args.output.resolve()),
                },
                ensure_ascii=False,
            )
        )
        return
    if args.events_dir is None or args.three_sixty_dir is None:
        parser.error("--events-dir and --three-sixty-dir must be supplied together")
    rows = []
    for match_id, event_path, frame_path in _paired_json_files(
        args.events_dir, args.three_sixty_dir
    ):
        match_output = args.output / match_id
        result = run_pipeline(
            event_path,
            frame_path,
            match_output,
            match_id=match_id,
            model=model,
        )
        rows.append(
            {
                "match_id": match_id,
                "status": result["status"],
                "prediction_count": result["audit"]["prediction_count"],
                "output": str(match_output.resolve()),
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "batch_result.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
