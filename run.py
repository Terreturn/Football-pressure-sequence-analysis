#!/usr/bin/env python3
"""Run the frozen public SPN pipeline on paired StatsBomb JSON data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from spn._network import PressureParams
from spn.positioning import PositioningModel
from spn.analysis import team_values
from spn.data_processing import (
    LabelConfig,
    SequenceConfig,
    build_sequences_from_match,
    load_match_data,
)
from spn.feature_engineering import build_feature_table
from spn.model import KEY_COLUMNS, SPNModel
from spn.output import _clean_json, build_result, empty_result, save_features, save_result


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
    export_features: bool = False,
    task: str = "sequence",
    positioning_model: PositioningModel | None = None,
) -> dict[str, Any]:
    """Score sequence outcomes, recorded-presser positions, or both in one match."""
    if task not in {"sequence", "positioning", "both"}:
        raise ValueError("task must be sequence, positioning or both")
    match = load_match_data(events_source, three_sixty_source, match_id=match_id)
    positions = None
    if task in {"positioning", "both"}:
        position_model = positioning_model or PositioningModel(model.config_path if model else None)
        positions = position_model.predict(match)
    target = Path(output_dir)
    if task == "positioning":
        target.mkdir(parents=True, exist_ok=True)
        positions.to_csv(target / "pressures.csv", index=False)
        return {"status": "ok", "match_id": match.match_id, "model": {"model_id": position_model.model_id},
                "audit": {"position_count": len(positions)}, "pressures": positions.to_dict("records")}

    frozen_model = model or SPNModel()
    pressure, sequence_config, label_config = _runtime_configs(frozen_model)
    sequences, labels = build_sequences_from_match(
        match,
        config=sequence_config,
        params=pressure,
        label_config=label_config,
    )
    features = build_feature_table(match, sequences, labels, params=pressure)
    predictions = frozen_model.predict(features)
    if sequences.empty:
        result = empty_result(match, frozen_model)
    else:
        result = build_result(match=match, sequences=sequences, labels=labels,
                              features=features, predictions=predictions, model=frozen_model)
    scored = predictions.merge(features[[*KEY_COLUMNS, "pressing_team_id"]],
                               on=KEY_COLUMNS, validate="one_to_one")
    teams = team_values(scored)
    result["teams"] = _clean_json(teams.to_dict("records"))
    if positions is not None:
        result["audit"]["position_count"] = len(positions)
        result["pressures"] = positions.to_dict("records")
    save_result(result, target)
    teams.to_csv(target / "teams.csv", index=False)
    if positions is not None:
        positions.to_csv(target / "pressures.csv", index=False)
    if export_features:
        save_features(features, frozen_model, target)
    return result


def _paired_json_files(events_dir: Path, frames_dir: Path) -> list[tuple[str, Path, Path]]:
    if not events_dir.is_dir() or not frames_dir.is_dir():
        raise FileNotFoundError("events-dir and three-sixty-dir must both exist")
    events = {
        path.stem: path for path in events_dir.glob("*.json")
        if path.is_file() and not path.name.startswith("._")
    }
    frames = {
        path.stem: path for path in frames_dir.glob("*.json")
        if path.is_file() and not path.name.startswith("._")
    }
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
    parser.add_argument("--export-features", action="store_true", help="also export the 26 model inputs for each scored event")
    parser.add_argument("--task", choices=["sequence", "positioning", "both"], default="sequence",
                        help="score sequence outcomes, individual positioning, or both")
    args = parser.parse_args()
    single = args.events is not None or args.three_sixty is not None
    batch = args.events_dir is not None or args.three_sixty_dir is not None
    if single == batch:
        parser.error("choose either single-file mode or directory mode")
    model = SPNModel() if args.task != "positioning" else None
    position_model = PositioningModel() if args.task != "sequence" else None
    if single:
        if args.events is None or args.three_sixty is None:
            parser.error("--events and --three-sixty must be supplied together")
        result = run_pipeline(
            args.events,
            args.three_sixty,
            args.output,
            match_id=args.match_id,
            model=model,
            export_features=args.export_features,
            task=args.task, positioning_model=position_model,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "match_id": result["match_id"],
                    **{key: result["audit"][key] for key in ("prediction_count", "position_count") if key in result["audit"]},
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
            export_features=args.export_features,
            task=args.task, positioning_model=position_model,
        )
        rows.append(
            {
                "match_id": match_id,
                "status": result["status"],
                **{key: result["audit"][key] for key in ("prediction_count", "position_count") if key in result["audit"]},
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
