from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUBLIC_ROOT))

from src.press_analysis import attach_sequence_change, team_summary  # noqa: E402


class PressAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scored = pd.DataFrame(
            [
                {"match_id": "1", "seq_id": 1, "ev_pos": 0, "team_name": "A", "outcome_tag": "neutral", "p_success": .1, "p_fail": .3, "P_total": .5, "escape_capacity": .4, "carrier_boundary_pressure": .2, "n_open_pass": 2},
                {"match_id": "1", "seq_id": 1, "ev_pos": 1, "team_name": "A", "outcome_tag": "success", "p_success": .3, "p_fail": .2, "P_total": .6, "escape_capacity": .3, "carrier_boundary_pressure": .3, "n_open_pass": 1},
                {"match_id": "1", "seq_id": 2, "ev_pos": 0, "team_name": "B", "outcome_tag": "fail", "p_success": .1, "p_fail": .4, "P_total": .4, "escape_capacity": .5, "carrier_boundary_pressure": .1, "n_open_pass": 3},
                {"match_id": "2", "seq_id": 1, "ev_pos": 0, "team_name": "A", "outcome_tag": "neutral", "p_success": .2, "p_fail": .2, "P_total": .5, "escape_capacity": .4, "carrier_boundary_pressure": .2, "n_open_pass": 2},
                {"match_id": "2", "seq_id": 1, "ev_pos": 1, "team_name": "A", "outcome_tag": "success", "p_success": .4, "p_fail": .1, "P_total": .7, "escape_capacity": .2, "carrier_boundary_pressure": .4, "n_open_pass": 1},
                {"match_id": "2", "seq_id": 2, "ev_pos": 0, "team_name": "B", "outcome_tag": "fail", "p_success": .1, "p_fail": .5, "P_total": .3, "escape_capacity": .6, "carrier_boundary_pressure": .1, "n_open_pass": 4},
            ]
        )

    def test_possession_team_is_mapped_to_opponent_pressing_team(self) -> None:
        attached = attach_sequence_change(self.scored)
        self.assertTrue((attached.loc[attached["team_name"] == "A", "press_team"] == "B").all())
        self.assertTrue((attached.loc[attached["team_name"] == "B", "press_team"] == "A").all())

    def test_sequence_identity_includes_match(self) -> None:
        summary = team_summary(self.scored, min_sequences=1).set_index("team_name")
        self.assertEqual(summary.loc["B", "n_sequences"], 2)
        self.assertAlmostEqual(summary.loc["B", "press_efficiency"], .30)


if __name__ == "__main__":
    unittest.main()
