from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUBLIC_ROOT))

from src.data_pipeline import SequenceConfig, event_is_high_press
from src.hpn_network import PressureParams


CONFIG = SequenceConfig()
PARAMS = PressureParams()


def event(
    event_id: str,
    location: list[float],
    *,
    event_type: str = "Pass",
    play_pattern: str = "Regular Play",
    team_id: int = 1,
    possession_team_id: int = 1,
) -> dict:
    return {
        "id": event_id,
        "type": {"name": event_type},
        "play_pattern": {"name": play_pattern},
        "location": location,
        "team": {"id": team_id},
        "possession_team": {"id": possession_team_id},
    }


def frame(
    event_id: str,
    actor_location: list[float],
    opponents: list[tuple[list[float], bool]],
) -> dict:
    return {
        "event_uuid": event_id,
        "freeze_frame": [
            {
                "location": actor_location,
                "teammate": True,
                "actor": True,
                "keeper": False,
            },
            *[
                {
                    "location": location,
                    "teammate": False,
                    "actor": False,
                    "keeper": keeper,
                }
                for location, keeper in opponents
            ],
        ],
    }


class HighPressDetectionTests(unittest.TestCase):
    def test_nearest_boundary_pressure_is_included(self) -> None:
        carrier = [20.0, 1.0]
        defender = [carrier[0] + PARAMS.player_distance * 120.0 / 105.0, carrier[1]]
        result = event_is_high_press(
            event("boundary", carrier),
            frame("boundary", carrier, [(defender, False)]),
            CONFIG,
            PARAMS,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["P_total"], 0.9898)
        self.assertEqual(result["nearest_def_m"], 6.4)
        self.assertEqual(result["n_in_D"], 1)

    def test_pressure_equal_to_threshold_is_rejected_after_rounding(self) -> None:
        target = CONFIG.pressure_threshold
        distance = PARAMS.player_distance - math.log(target / (1.0 - target)) / PARAMS.k_player
        carrier = [30.0, 40.0]
        defender = [carrier[0] + distance * 120.0 / 105.0, carrier[1]]
        result = event_is_high_press(
            event("threshold", carrier),
            frame("threshold", carrier, [(defender, False)]),
            CONFIG,
            PARAMS,
        )
        self.assertIsNone(result)

    def test_event_location_is_authoritative_over_freeze_frame_actor(self) -> None:
        carrier = [20.0, 20.0]
        defender = [carrier[0] + 1.0, carrier[1]]
        result = event_is_high_press(
            event("actor-location", carrier),
            frame("actor-location", [100.0, 20.0], [(defender, False)]),
            CONFIG,
            PARAMS,
        )
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["carrier_x_m"], 17.5)
        self.assertEqual(result["actor_x_norm"], 20.0)

    def test_non_possession_event_does_not_count_actor_as_defender(self) -> None:
        carrier = [90.0, 40.0]
        distant_opponent = [70.0, 40.0]
        result = event_is_high_press(
            event(
                "defending-event",
                carrier,
                team_id=2,
                possession_team_id=1,
            ),
            frame("defending-event", carrier, [(distant_opponent, False)]),
            CONFIG,
            PARAMS,
        )
        self.assertIsNone(result)

    def test_goalkeeper_is_not_a_valid_nearby_defender(self) -> None:
        carrier = [20.0, 20.0]
        result = event_is_high_press(
            event("keeper", carrier),
            frame("keeper", carrier, [([21.0, 20.0], True)]),
            CONFIG,
            PARAMS,
        )
        self.assertIsNone(result)

    def test_halfway_line_is_excluded(self) -> None:
        carrier = [60.0, 40.0]
        result = event_is_high_press(
            event("halfway", carrier),
            frame("halfway", carrier, [([61.0, 40.0], False)]),
            CONFIG,
            PARAMS,
        )
        self.assertIsNone(result)

    def test_configured_restart_is_excluded(self) -> None:
        carrier = [20.0, 20.0]
        result = event_is_high_press(
            event("corner", carrier, play_pattern="From Corner"),
            frame("corner", carrier, [([21.0, 20.0], False)]),
            CONFIG,
            PARAMS,
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
