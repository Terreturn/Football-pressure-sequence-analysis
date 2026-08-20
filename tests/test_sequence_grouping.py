from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUBLIC_ROOT))

from src.data_pipeline import (
    OPPONENT_ON_BALL_TYPES,
    SequenceConfig,
    build_sequences,
    detect_sequences,
    frame_index,
    resolve_dribble_location,
    resolve_pre_sequence_context,
)
from src.spn_network import PressureParams


CONFIG = SequenceConfig()
PARAMS = PressureParams()


def event(
    event_id: str,
    index: int,
    timestamp: str,
    *,
    team_id: int,
    possession_team_id: int,
    location: list[float],
    event_type: str = "Pass",
) -> dict:
    return {
        "id": event_id,
        "index": index,
        "period": 1,
        "timestamp": timestamp,
        "type": {"name": event_type},
        "play_pattern": {"name": "Regular Play"},
        "location": location,
        "team": {"id": team_id, "name": f"Team {team_id}"},
        "possession_team": {
            "id": possession_team_id,
            "name": f"Team {possession_team_id}",
        },
    }


def freeze_frame(event_id: str, carrier: list[float]) -> dict:
    return {
        "event_uuid": event_id,
        "freeze_frame": [
            {
                "location": carrier,
                "teammate": True,
                "actor": True,
                "keeper": False,
            },
            {
                "location": [carrier[0] + 1.0, carrier[1]],
                "teammate": False,
                "actor": False,
                "keeper": False,
            },
        ],
    }


def control_event(
    event_id: str,
    index: int,
    timestamp: str,
    *,
    event_type: str,
    location: list[float] | None,
    duration: float = 0.0,
    end_location: list[float] | None = None,
    player_id: int = 10,
) -> dict:
    item = event(
        event_id,
        index,
        timestamp,
        team_id=1,
        possession_team_id=1,
        location=location,
        event_type=event_type,
    )
    item.update(
        {
            "possession": 7,
            "player": {"id": player_id},
            "duration": duration,
        }
    )
    if event_type == "Carry":
        item["carry"] = {"end_location": end_location}
    elif event_type == "Dribble":
        item["dribble"] = {"outcome": {"name": "Complete"}}
    return item


class SequenceGroupingTests(unittest.TestCase):
    def test_pre_context_uses_nearest_distinct_ball_point(self) -> None:
        events = [
            event(
                "older-carry",
                1,
                "00:00:01.000",
                team_id=1,
                possession_team_id=1,
                location=[10.0, 20.0],
                event_type="Carry",
            ),
            event(
                "nearest-receipt",
                2,
                "00:00:02.000",
                team_id=1,
                possession_team_id=1,
                location=[25.0, 20.0],
                event_type="Ball Receipt*",
            ),
            event(
                "same-point-recovery",
                3,
                "00:00:03.000",
                team_id=1,
                possession_team_id=1,
                location=[30.0, 20.0],
                event_type="Ball Recovery",
            ),
            event(
                "first-anchor",
                4,
                "00:00:03.000",
                team_id=1,
                possession_team_id=1,
                location=[30.0, 20.0],
            ),
        ]
        for item in events:
            item["possession"] = 7

        result = resolve_pre_sequence_context(
            events,
            3,
            [30.0, 20.0],
            frame_index([freeze_frame("nearest-receipt", [25.0, 20.0])]),
            CONFIG,
        )

        self.assertEqual(result["pre_context_status"], "selected")
        self.assertEqual(
            result["pre_context_event_id"], "nearest-receipt"
        )
        self.assertEqual(result["pre_context_event_type"], "Ball Receipt*")
        self.assertEqual(result["pre_context_skipped_same_location"], 1)
        self.assertTrue(result["pre_context_has_frame"])
        self.assertGreater(result["pre_context_distance_m"], 0.0)

    def test_pre_context_accepts_opponent_point_and_flips_coordinates(self) -> None:
        opponent = event(
            "opponent-pass",
            1,
            "00:00:01.000",
            team_id=2,
            possession_team_id=2,
            location=[100.0, 40.0],
        )
        opponent["possession"] = 3
        current = event(
            "first-anchor",
            2,
            "00:00:02.000",
            team_id=1,
            possession_team_id=1,
            location=[30.0, 40.0],
        )
        current["possession"] = 4

        result = resolve_pre_sequence_context(
            [opponent, current],
            1,
            [30.0, 40.0],
            {},
            CONFIG,
        )

        self.assertEqual(result["pre_context_status"], "selected")
        self.assertEqual(result["pre_context_team_relation"], "opponent")
        self.assertEqual(result["pre_context_possession_relation"], "different")
        self.assertEqual(result["pre_context_ball_x"], 20.0)
        self.assertEqual(result["pre_context_ball_y"], 40.0)
        self.assertTrue(result["pre_context_coordinate_flipped"])
        self.assertAlmostEqual(result["pre_context_angle_cos"], 1.0)

    def test_pre_context_does_not_scan_through_hard_stoppage(self) -> None:
        prior = event(
            "prior",
            1,
            "00:00:01.000",
            team_id=1,
            possession_team_id=1,
            location=[10.0, 20.0],
        )
        stoppage = event(
            "foul",
            2,
            "00:00:02.000",
            team_id=1,
            possession_team_id=1,
            location=[20.0, 20.0],
            event_type="Foul Committed",
        )
        current = event(
            "first-anchor",
            3,
            "00:00:03.000",
            team_id=1,
            possession_team_id=1,
            location=[30.0, 20.0],
        )

        result = resolve_pre_sequence_context(
            [prior, stoppage, current],
            2,
            [30.0, 20.0],
            {},
            CONFIG,
        )

        self.assertEqual(result["pre_context_status"], "hard_stoppage")
        self.assertFalse(result["pre_context_direction_valid"])

    def test_dribble_location_prefers_prior_carry_and_records_both_links(self) -> None:
        events = [
            control_event(
                "prior-carry",
                1,
                "00:00:01.000",
                event_type="Carry",
                location=[10.0, 20.0],
                duration=1.0,
                end_location=[20.0, 20.0],
            ),
            control_event(
                "dribble",
                2,
                "00:00:02.000",
                event_type="Dribble",
                location=[20.0, 20.0],
            ),
            control_event(
                "following-carry",
                3,
                "00:00:02.000",
                event_type="Carry",
                location=[20.0, 20.0],
                duration=1.0,
                end_location=[25.0, 20.0],
            ),
        ]

        result = resolve_dribble_location(events, 1, CONFIG)

        self.assertEqual(result["location"], [20.0, 20.0])
        self.assertEqual(result["location_source"], "prior_carry_end")
        self.assertEqual(result["prior_carry_id"], "prior-carry")
        self.assertEqual(result["following_carry_id"], "following-carry")
        self.assertTrue(result["validated_by_carry"])
        self.assertFalse(result["location_conflict"])

    def test_dribble_location_falls_back_to_following_carry(self) -> None:
        events = [
            control_event(
                "dribble",
                1,
                "00:00:02.000",
                event_type="Dribble",
                location=[20.0, 20.0],
            ),
            control_event(
                "following-carry",
                2,
                "00:00:02.000",
                event_type="Carry",
                location=[20.0, 20.0],
                duration=1.0,
                end_location=[25.0, 20.0],
            ),
        ]

        result = resolve_dribble_location(events, 0, CONFIG)

        self.assertEqual(result["location_source"], "following_carry_start")
        self.assertEqual(result["following_carry_id"], "following-carry")

    def test_dribble_location_falls_back_to_raw_event(self) -> None:
        dribble = control_event(
            "dribble",
            1,
            "00:00:02.000",
            event_type="Dribble",
            location=[20.0, 20.0],
        )
        dribble["dribble"]["outcome"] = {"name": "Incomplete"}

        result = resolve_dribble_location([dribble], 0, CONFIG)

        self.assertEqual(result["location"], [20.0, 20.0])
        self.assertEqual(result["location_source"], "dribble_event")
        self.assertFalse(result["validated_by_carry"])

    def test_dribble_location_marks_spatial_carry_conflict(self) -> None:
        events = [
            control_event(
                "prior-carry",
                1,
                "00:00:01.000",
                event_type="Carry",
                location=[10.0, 20.0],
                duration=1.0,
                end_location=[25.0, 20.0],
            ),
            control_event(
                "dribble",
                2,
                "00:00:02.000",
                event_type="Dribble",
                location=[20.0, 20.0],
            ),
        ]

        result = resolve_dribble_location(events, 1, CONFIG)

        self.assertEqual(result["location_source"], "dribble_event")
        self.assertTrue(result["location_conflict"])

    def test_detection_uses_prior_carry_when_dribble_location_is_missing(self) -> None:
        events = [
            control_event(
                "prior-carry",
                1,
                "00:00:01.000",
                event_type="Carry",
                location=[10.0, 20.0],
                duration=1.0,
                end_location=[20.0, 20.0],
            ),
            control_event(
                "dribble",
                2,
                "00:00:02.000",
                event_type="Dribble",
                location=None,
            ),
        ]
        frames = frame_index([freeze_frame("dribble", [20.0, 20.0])])

        sequences = detect_sequences(events, frames, CONFIG, PARAMS)

        self.assertEqual([len(sequence) for sequence in sequences], [1])
        metrics = sequences[0][0][1]
        self.assertEqual(metrics["actor_x_norm"], 20.0)
        self.assertEqual(metrics["location_source"], "prior_carry_end")
        self.assertEqual(metrics["dribble_prior_carry_id"], "prior-carry")
        self.assertTrue(metrics["dribble_location_validated"])

    def test_opponent_on_ball_actions_split_sequence_despite_stale_possession(self) -> None:
        for opponent_event_type in OPPONENT_ON_BALL_TYPES:
            with self.subTest(opponent_event_type=opponent_event_type):
                events = [
                    event(
                        "first",
                        1,
                        "00:00:01.000",
                        team_id=1,
                        possession_team_id=1,
                        location=[20.0, 1.0],
                    ),
                    event(
                        "opponent-control",
                        2,
                        "00:00:02.000",
                        team_id=2,
                        possession_team_id=1,
                        location=[40.0, 1.0],
                        event_type=opponent_event_type,
                    ),
                    event(
                        "second",
                        3,
                        "00:00:04.000",
                        team_id=1,
                        possession_team_id=1,
                        location=[21.0, 1.0],
                    ),
                ]
                frames = frame_index(
                    [
                        freeze_frame("first", events[0]["location"]),
                        freeze_frame("second", events[2]["location"]),
                    ]
                )

                sequences = detect_sequences(events, frames, CONFIG, PARAMS)

                self.assertEqual([len(sequence) for sequence in sequences], [1, 1])

    def test_pressure_and_ball_receipt_do_not_split_sequence(self) -> None:
        events = [
            event(
                "first",
                1,
                "00:00:01.000",
                team_id=1,
                possession_team_id=1,
                location=[20.0, 1.0],
            ),
            event(
                "opponent-pressure",
                2,
                "00:00:02.000",
                team_id=2,
                possession_team_id=1,
                location=[20.0, 1.0],
                event_type="Pressure",
            ),
            event(
                "receipt",
                3,
                "00:00:02.500",
                team_id=1,
                possession_team_id=1,
                location=[21.0, 1.0],
                event_type="Ball Receipt*",
            ),
            event(
                "second",
                4,
                "00:00:04.000",
                team_id=1,
                possession_team_id=1,
                location=[21.0, 1.0],
            ),
        ]
        frames = frame_index(
            [
                freeze_frame("first", events[0]["location"]),
                freeze_frame("second", events[3]["location"]),
            ]
        )

        sequences = detect_sequences(events, frames, CONFIG, PARAMS)

        self.assertEqual([len(sequence) for sequence in sequences], [2])

    def test_gap_is_measured_from_previous_action_end(self) -> None:
        first = event(
            "first",
            1,
            "00:00:01.000",
            team_id=1,
            possession_team_id=1,
            location=[20.0, 1.0],
        )
        first["duration"] = 4.0
        second = event(
            "second",
            2,
            "00:00:09.000",
            team_id=1,
            possession_team_id=1,
            location=[21.0, 1.0],
        )
        frames = frame_index(
            [
                freeze_frame("first", first["location"]),
                freeze_frame("second", second["location"]),
            ]
        )

        sequences = detect_sequences([first, second], frames, CONFIG, PARAMS)

        self.assertEqual([len(sequence) for sequence in sequences], [2])

    def test_observable_low_pressure_action_splits_high_pressure_spells(self) -> None:
        events = [
            event(
                "first-high",
                1,
                "00:00:01.000",
                team_id=1,
                possession_team_id=1,
                location=[20.0, 20.0],
            ),
            event(
                "low-context",
                2,
                "00:00:02.000",
                team_id=1,
                possession_team_id=1,
                location=[25.0, 20.0],
                event_type="Carry",
            ),
            event(
                "second-high",
                3,
                "00:00:04.000",
                team_id=1,
                possession_team_id=1,
                location=[30.0, 20.0],
            ),
        ]
        low_frame = freeze_frame("low-context", events[1]["location"])
        low_frame["freeze_frame"][1]["location"] = [100.0, 20.0]
        frames = frame_index(
            [
                freeze_frame("first-high", events[0]["location"]),
                low_frame,
                freeze_frame("second-high", events[2]["location"]),
            ]
        )

        sequences = detect_sequences(events, frames, CONFIG, PARAMS)

        self.assertEqual([len(sequence) for sequence in sequences], [1, 1])
        self.assertEqual(
            [events[sequence[0][0]]["id"] for sequence in sequences],
            ["first-high", "second-high"],
        )

    def test_unobservable_context_does_not_claim_pressure_release(self) -> None:
        events = [
            event(
                "first-high",
                1,
                "00:00:01.000",
                team_id=1,
                possession_team_id=1,
                location=[20.0, 20.0],
            ),
            event(
                "unknown-context",
                2,
                "00:00:02.000",
                team_id=1,
                possession_team_id=1,
                location=[25.0, 20.0],
                event_type="Carry",
            ),
            event(
                "second-high",
                3,
                "00:00:04.000",
                team_id=1,
                possession_team_id=1,
                location=[30.0, 20.0],
            ),
        ]
        frames = frame_index(
            [
                freeze_frame("first-high", events[0]["location"]),
                freeze_frame("second-high", events[2]["location"]),
            ]
        )

        sequences = detect_sequences(events, frames, CONFIG, PARAMS)

        self.assertEqual([len(sequence) for sequence in sequences], [2])

    def test_completed_halfway_escape_splits_before_later_pressure(self) -> None:
        first = control_event(
            "escaped",
            1,
            "00:00:01.000",
            event_type="Carry",
            location=[50.0, 20.0],
            duration=4.0,
            end_location=[65.0, 20.0],
        )
        second = control_event(
            "pressure-returned",
            2,
            "00:00:08.000",
            event_type="Carry",
            location=[50.0, 20.0],
            end_location=[52.0, 20.0],
        )
        frames = frame_index(
            [
                freeze_frame("escaped", first["location"]),
                freeze_frame("pressure-returned", second["location"]),
            ]
        )

        sequences = detect_sequences([first, second], frames, CONFIG, PARAMS)

        self.assertEqual([len(sequence) for sequence in sequences], [1, 1])

    def test_non_possession_event_is_excluded_when_event_team_differs(self) -> None:
        events = [
            event(
                "first",
                1,
                "00:00:01.000",
                team_id=1,
                possession_team_id=1,
                location=[20.0, 1.0],
            ),
            event(
                "second",
                2,
                "00:00:04.000",
                team_id=2,
                possession_team_id=1,
                location=[90.0, 1.0],
            ),
        ]
        frames = frame_index(
            [
                freeze_frame("first", events[0]["location"]),
                freeze_frame("second", events[1]["location"]),
            ]
        )
        sequences = detect_sequences(events, frames, CONFIG, PARAMS)
        self.assertEqual(len(sequences), 1)
        self.assertEqual(len(sequences[0]), 1)

    def test_non_possession_event_is_excluded_from_sequence(self) -> None:
        events = [
            event(
                "first",
                1,
                "00:00:01.000",
                team_id=1,
                possession_team_id=1,
                location=[20.0, 1.0],
            ),
            event(
                "second",
                2,
                "00:00:04.000",
                team_id=1,
                possession_team_id=2,
                location=[90.0, 1.0],
            ),
        ]
        frames = frame_index(
            [
                freeze_frame("first", events[0]["location"]),
                freeze_frame("second", events[1]["location"]),
            ]
        )
        sequences = detect_sequences(events, frames, CONFIG, PARAMS)
        self.assertEqual(len(sequences), 1)
        self.assertEqual(len(sequences[0]), 1)

    def test_build_sequences_sorts_by_index_and_uses_local_event_positions(self) -> None:
        earlier = event(
            "earlier",
            1,
            "00:00:01.000",
            team_id=1,
            possession_team_id=1,
            location=[20.0, 1.0],
        )
        later = event(
            "later",
            2,
            "00:00:04.000",
            team_id=1,
            possession_team_id=1,
            location=[21.0, 1.0],
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events_dir, frames_dir = root / "events", root / "360"
            events_dir.mkdir()
            frames_dir.mkdir()
            (events_dir / "1.json").write_text(
                json.dumps([later, earlier]),
                encoding="utf-8",
            )
            (frames_dir / "1.json").write_text(
                json.dumps(
                    [
                        freeze_frame("later", later["location"]),
                        freeze_frame("earlier", earlier["location"]),
                    ]
                ),
                encoding="utf-8",
            )

            sequences, _ = build_sequences(
                events_dir,
                frames_dir,
                CONFIG,
                PARAMS,
            )

        self.assertEqual(sequences["anchor_event_id"].tolist(), ["earlier", "later"])
        self.assertEqual(sequences["seq_id"].tolist(), [1, 1])
        self.assertEqual(sequences["ev_pos"].tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
