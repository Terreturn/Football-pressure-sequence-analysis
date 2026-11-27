"""End-to-end regression checks using invented StatsBomb records and the frozen fits."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from run import run_pipeline
from spn import (
    SPNModel,
    PositioningModel,
    build_feature_table,
    build_sequences_from_match,
    load_match_data,
    score_positions,
)


def fixture():
    """Return four made-up events and one linked frame on the 120 by 80 provider grid."""
    events = []
    for index, kind, side, x, end in [
        (1, "Pass", 1, 32, 35),
        (2, "Ball Recovery", 2, 84, 84),
        (3, "Pass", 2, 84, 89),
        (4, "Carry", 2, 89, 92),
    ]:
        event = {
            "id": f"invented-{index}",
            "index": index,
            "type": {"name": kind},
            "team": {"id": side},
            "possession_team": {"id": side},
            "possession": side,
            "player": {"id": 10 * side},
            "timestamp": f"00:00:0{index}.000",
            "period": 1,
            "duration": 0,
            "play_pattern": {"name": "Regular Play"},
            "location": [x, 41],
        }
        if kind in ("Pass", "Carry"):
            event[kind.lower()] = {"end_location": [end, 41]}
        events.append(event)
    players = []
    for i, (x, y) in enumerate(
        [(32, 41), (36, 20), (42, 49), (22, 58), (34, 41), (39, 50), (37, 27), (22, 52)]
    ):
        players.append({"actor": i == 0, "teammate": i < 4, "keeper": False, "location": [x, y]})
    frame = {
        "event_uuid": "invented-1",
        "freeze_frame": players,
        "visible_area": [10, 5, 55, 5, 55, 70, 10, 70],
    }
    return events, {"invented-1": frame}


def pressure_fixture():
    # The first player holds the ball through the time of the recorded pressure.
    events, frames = fixture()
    reference = events[0]
    reference.update(type={"name": "Carry"}, duration=0.4, carry={"end_location": [32, 41]})
    reference.pop("pass")
    frame = frames[reference["id"]]
    pressure = {
        **reference,
        "id": "named-pressure",
        "index": 2,
        "timestamp": "00:00:01.200",
        "duration": 0,
        "type": {"name": "Pressure"},
        "team": {"id": 2},
        "player": {"id": 27},
        "location": [86, 39],
    }
    pressure.pop("carry")
    for event in events[1:]:
        event["index"] += 1
    target = deepcopy(frame)
    target["event_uuid"] = pressure["id"]
    for i, player in enumerate(target["freeze_frame"]):
        x, y = player["location"]
        player.update(location=[120 - x, 80 - y], teammate=not player["teammate"], actor=i == 4)
    target["visible_area"] = [
        120 - v if i % 2 == 0 else 80 - v for i, v in enumerate(frame["visible_area"])
    ]
    return [reference, pressure, *events[1:]], [frame, target]


def test_sequence_pipeline_and_export(tmp_path):
    events, frames = fixture()
    result = run_pipeline(events, list(frames.values()), tmp_path, "example", export_features=True)
    assert result["model"]["model_id"] == "spn"
    (row,) = result["predictions"]
    assert row["observed_outcome"] == "success" and row["ev_pos"] == 0
    np.testing.assert_allclose(
        [row["p_fail"], row["p_neutral"], row["p_success"]],
        [0.28576742688035073, 0.5344030283422778, 0.17982954477737143],
        atol=1e-8,
        rtol=0,
    )
    assert result["teams"][0]["expected_press_value"] == pytest.approx(
        row["p_success"] - row["p_fail"]
    )
    metadata = json.loads((tmp_path / "features_metadata.json").read_text())
    assert metadata["probability_layer"] == "temperature"
    features = pd.read_csv(tmp_path / "features.csv")
    assert len(metadata["features"]) == 26 and list(features.columns[4:]) == metadata["features"]
    np.testing.assert_allclose(SPNModel().predict(features).confidence, [row["confidence"]])


def test_censored_in_audit_and_empty_position_output(tmp_path):
    events, frames = fixture()
    result = run_pipeline(events[:1], list(frames.values()), tmp_path, "test")
    assert len(result["sequences"]) == result["audit"]["excluded_censored_sequences"] == 1
    assert result["predictions"] == result["teams"] == []
    result = run_pipeline(events, [], tmp_path / "none", "test", task="both")
    assert result["status"] == "no_sequences" and result["pressures"] == []
    assert "position_quality_pp" in pd.read_csv(tmp_path / "none" / "pressures.csv").columns


def test_relink_and_duplicate_uuids():
    events, frames = fixture()
    (frame,) = frames.values()
    stale = {**frame, "event_uuid": "old-export-id"}
    match = load_match_data(events, [stale], "test")
    assert list(match.frames) == [events[0]["id"]] and match.warnings
    with pytest.raises(ValueError, match="Two frames"):
        load_match_data(events, [frame, frame], "test")


def test_missing_later_frame_excludes_whole_sequence():
    events, frames = fixture()
    second = {**events[0], "id": "second", "index": 2, "timestamp": "00:00:01.100"}
    for event in events[1:]:
        event["index"] += 1
    frame = {**frames["invented-1"], "event_uuid": "second"}
    match = load_match_data([events[0], second, *events[1:]], [*frames.values(), frame], "pair")
    anchors, labels = build_sequences_from_match(match)
    assert anchors.ev_pos.tolist() == [0, 1]
    del match.frames["second"]
    features = build_feature_table(match, anchors, labels)
    assert (
        features.empty
        and features.attrs["s3_completeness_audit"]["excluded_incomplete_sequences"] == 1
    )


@pytest.mark.parametrize(
    "match_id, observed, alternative, expected_pp",
    [
        ("invented", 0.11292282342910767, 0.12115053554050256, -0.8227712111394891),
        ("3999361", 0.11053077131509781, 0.11540490230087375, -0.48741309857759374),
    ],
)
def test_recorded_presser_against_paper_folds(match_id, observed, alternative, expected_pp):
    events, frames = pressure_fixture()
    match = load_match_data(events, frames, match_id)
    (row,) = score_positions(match).to_dict("records")
    assert (row["event_id"], row["player_id"], row["e2"], row["valid_alternatives"]) == (
        "named-pressure",
        27,
        1,
        61,
    )
    np.testing.assert_allclose(
        [row["observed_probability"], row["alternative_probability"], row["position_quality_pp"]],
        [observed, alternative, expected_pp],
        atol=1e-7,
        rtol=0,
    )


@pytest.mark.parametrize(
    "event_location, expected_ids",
    [
        ([86.1, 39.20000076293945], ["named-pressure"]),
        ([86.2, 39.2], []),
        ([86.1, 39.3], []),
    ],
)
def test_presser_coordinates_match_at_one_decimal(event_location, expected_ids):
    events, frames = pressure_fixture()
    events[1]["location"] = event_location
    actor = next(player for player in frames[1]["freeze_frame"] if player["actor"])
    actor["location"] = [86.0999984741211, 39.2]
    match = load_match_data(events, frames, "invented")
    assert score_positions(match)["event_id"].tolist() == expected_ids


def test_unnamed_pressure_is_excluded():
    events, frames = pressure_fixture()
    events[1].pop("player")
    match = load_match_data(events, frames, "invented")
    assert PositioningModel().predict(match).empty
