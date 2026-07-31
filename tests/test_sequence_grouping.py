from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUBLIC_ROOT))

from src.data_pipeline import SequenceConfig, build_sequences, detect_sequences, frame_index
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
) -> dict:
    return {
        "id": event_id,
        "index": index,
        "period": 1,
        "timestamp": timestamp,
        "type": {"name": "Pass"},
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


class SequenceGroupingTests(unittest.TestCase):
    def test_different_event_teams_split_even_when_possession_team_matches(self) -> None:
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
        self.assertEqual(len(sequences), 2)

    def test_same_event_team_connects_even_when_possession_team_differs(self) -> None:
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
        self.assertEqual(len(sequences[0]), 2)

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
