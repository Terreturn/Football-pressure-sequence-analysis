from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUBLIC_ROOT))

from src.spn_network import (  # noqa: E402
    PITCH_L,
    PITCH_W,
    build_spn_network,
    frame_players,
    metric_xy,
)


def event(
    location: list[float] | None,
    *,
    team_id: int = 1,
    possession_team_id: int = 1,
) -> dict:
    output = {
        "id": "event",
        "team": {"id": team_id},
        "possession_team": {"id": possession_team_id},
    }
    if location is not None:
        output["location"] = location
    return output


def frame(
    actor_location: list[float],
    *,
    visible_area: list[float] | None = None,
) -> dict:
    output = {
        "event_uuid": "event",
        "freeze_frame": [
            {
                "location": actor_location,
                "teammate": True,
                "actor": True,
                "keeper": False,
            },
            {
                "location": [25.0, 25.0],
                "teammate": True,
                "actor": False,
                "keeper": False,
            },
            {
                "location": [35.0, 55.0],
                "teammate": True,
                "actor": False,
                "keeper": False,
            },
            {
                "location": [23.0, 40.0],
                "teammate": False,
                "actor": False,
                "keeper": False,
            },
            {
                "location": [42.0, 42.0],
                "teammate": False,
                "actor": False,
                "keeper": False,
            },
        ],
    }
    if visible_area is not None:
        output["visible_area"] = visible_area
    return output


class SPNNetworkConstructionTests(unittest.TestCase):
    def test_event_location_is_authoritative_for_carrier_node(self) -> None:
        source_event = event([20.0, 40.0])
        network = build_spn_network(
            source_event,
            frame(
                [90.0, 70.0],
                visible_area=[0.0, 0.0, 60.0, 0.0, 60.0, 80.0, 0.0, 80.0],
            ),
        )
        expected = metric_xy(*source_event["location"])
        np.testing.assert_allclose(
            network["xy"][network["carrier_index"]],
            expected,
        )

    def test_voronoi_areas_are_clipped_to_camera_visible_pitch(self) -> None:
        network = build_spn_network(
            event([20.0, 40.0]),
            frame(
                [20.0, 40.0],
                visible_area=[0.0, 0.0, 60.0, 0.0, 60.0, 80.0, 0.0, 80.0],
            ),
        )
        expected_visible_area = (60.0 * PITCH_L / 120.0) * PITCH_W
        actual = sum(
            area.area
            for area in network["areas"]
            if area is not None
        )
        self.assertAlmostEqual(actual, expected_visible_area, places=5)
        self.assertIsNotNone(network["visible_area"])

    def test_pressure_edges_include_within_defender_normalisation(self) -> None:
        network = build_spn_network(
            event([20.0, 40.0]),
            frame([20.0, 40.0]),
        )
        for defender in network["defenders"]:
            edges = [
                edge
                for edge in network["pressure_edges"]
                if edge["src"] == defender
            ]
            if edges:
                self.assertAlmostEqual(sum(edge["w"] for edge in edges), 1.0)
                total = sum(edge["p"] for edge in edges)
                for edge in edges:
                    self.assertAlmostEqual(edge["w"], edge["p"] / total)

    def test_defending_event_actor_is_carrier_and_roles_are_inverted(self) -> None:
        source_event = event(
            [90.0, 40.0],
            team_id=2,
            possession_team_id=1,
        )
        source_frame = frame([88.0, 40.0])
        players, carrier = frame_players(source_event, source_frame)
        self.assertIsNotNone(carrier)
        self.assertEqual(players[carrier]["role"], "carrier")
        np.testing.assert_allclose(
            players[carrier]["xy"],
            metric_xy(*source_event["location"], flip=True),
        )
        self.assertTrue(
            all(
                player["role"] == "defender"
                for index, player in enumerate(players)
                if index != carrier and player["actor"] is False and source_frame["freeze_frame"][index]["teammate"]
            )
        )
        network = build_spn_network(source_event, source_frame)
        self.assertNotIn(network["carrier_index"], network["defenders"])

    def test_user_data_without_optional_geometry_uses_safe_fallbacks(self) -> None:
        source_frame = frame([20.0, 40.0])
        network = build_spn_network(event(None), source_frame)
        np.testing.assert_allclose(
            network["xy"][network["carrier_index"]],
            metric_xy(20.0, 40.0),
        )
        self.assertIsNone(network["visible_area"])
        actual = sum(
            area.area
            for area in network["areas"]
            if area is not None
        )
        self.assertAlmostEqual(actual, PITCH_L * PITCH_W, places=5)


if __name__ == "__main__":
    unittest.main()
