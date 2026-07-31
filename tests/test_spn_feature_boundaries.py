from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUBLIC_ROOT))

from src.spn_features import (  # noqa: E402
    _events_and_frames,
    _incoming_ball,
    _legacy_carrier_x_norm,
    network_features,
)
from src.spn_network import PressureParams, metric_xy  # noqa: E402


def event(
    event_id: str,
    index: int,
    *,
    possession: int = 10,
    team_id: int = 1,
    possession_team_id: int = 1,
    location: list[float] | None = None,
) -> dict:
    return {
        "id": event_id,
        "index": index,
        "possession": possession,
        "team": {"id": team_id},
        "possession_team": {"id": possession_team_id},
        "location": location or [20.0, 40.0],
    }


def minimal_network(anchor: dict, xy: np.ndarray) -> dict:
    return {
        "event": anchor,
        "xy": xy,
        "carrier_index": 0,
        "attackers": [],
        "defenders": [],
        "boundary_edges": [],
        "pass_edges": [],
    }


class MainFeatureBoundaryTests(unittest.TestCase):
    def test_event_loader_sorts_by_native_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events_dir, frames_dir = root / "events", root / "360"
            events_dir.mkdir()
            frames_dir.mkdir()
            (events_dir / "1.json").write_text(
                json.dumps([event("later", 2), event("earlier", 1)]),
                encoding="utf-8",
            )
            (frames_dir / "1.json").write_text("[]", encoding="utf-8")

            events, _ = _events_and_frames(events_dir, frames_dir, "1")

        self.assertEqual([item["id"] for item in events], ["earlier", "later"])

    def test_incoming_ball_search_stops_at_possession_boundary(self) -> None:
        events = [
            event("old-same-id", 1, possession=10, location=[10.0, 40.0]),
            event("boundary", 2, possession=11, location=[15.0, 40.0]),
            event("anchor", 3, possession=10, location=[20.0, 40.0]),
        ]

        distance, sine, cosine = _incoming_ball(events, 2, flip=False)

        self.assertTrue(np.isnan(distance))
        self.assertTrue(np.isnan(sine))
        self.assertTrue(np.isnan(cosine))

    def test_carrier_x_uses_clipped_freeze_frame_actor(self) -> None:
        anchor = event("anchor", 1, location=[20.0, 40.0])
        frame = {
            "freeze_frame": [
                {"actor": True, "location": [30.0, 40.0]},
            ]
        }
        network = minimal_network(
            anchor,
            np.asarray([metric_xy(20.0, 40.0)]),
        )

        values = network_features(
            network,
            [anchor],
            0,
            frame=frame,
        )

        self.assertAlmostEqual(values["carrier_x_norm"], 30.0 / 120.0)

        defending_anchor = event(
            "defending",
            2,
            team_id=2,
            possession_team_id=1,
        )
        outside_frame = {
            "freeze_frame": [
                {"actor": True, "location": [-10.0, 40.0]},
            ]
        }
        self.assertEqual(
            _legacy_carrier_x_norm(defending_anchor, outside_frame),
            1.0,
        )

    def test_supplement_pressure_threshold_is_inclusive(self) -> None:
        anchor = event("anchor", 1)
        xy = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [-1.0, 0.0],
            ]
        )
        network = {
            **minimal_network(anchor, xy),
            "attackers": [1],
            "defenders": [2, 3],
        }
        params = PressureParams(player_distance=1.0)

        with patch("src.spn_features.P_EPS", 0.5):
            values = network_features(
                network,
                [anchor],
                0,
                params=params,
            )

        self.assertEqual(values["effective_pressers"], 0.0)
        self.assertAlmostEqual(values["mean_receiver_pressure"], 1.0)
        self.assertAlmostEqual(values["weighted_angular_dispersion"], 1.0)

    def test_missing_actor_is_not_used_for_main_carrier_x(self) -> None:
        anchor = event("anchor", 1)
        network = minimal_network(
            anchor,
            np.asarray([metric_xy(20.0, 40.0)]),
        )

        with self.assertRaisesRegex(ValueError, "freeze-frame actor"):
            network_features(
                network,
                [anchor],
                0,
                frame={"freeze_frame": []},
            )


if __name__ == "__main__":
    unittest.main()
