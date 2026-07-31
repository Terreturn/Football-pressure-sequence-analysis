from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUBLIC_ROOT))

from src.data_pipeline import (  # noqa: E402
    LabelConfig,
    SequenceConfig,
    build_sequences,
    classify_anchor,
)
from src.hpn_network import PressureParams  # noqa: E402


PARAMS = PressureParams()
LABEL_CONFIG = LabelConfig()


def event(
    event_id: str,
    index: int,
    *,
    event_type: str = "Pass",
    team_id: int = 1,
    possession_team_id: int | None = None,
    possession: int = 10,
    play_pattern: str = "Regular Play",
    location: list[float] | None = None,
    end_location: list[float] | None = None,
) -> dict:
    possession_team_id = (
        team_id if possession_team_id is None else possession_team_id
    )
    item = {
        "id": event_id,
        "index": index,
        "period": 1,
        "timestamp": f"00:00:{index:02d}.000",
        "minute": 0,
        "second": index,
        "type": {"name": event_type},
        "play_pattern": {"name": play_pattern},
        "location": location or [20.0, 40.0],
        "team": {"id": team_id, "name": f"Team {team_id}"},
        "possession_team": {
            "id": possession_team_id,
            "name": f"Team {possession_team_id}",
        },
        "possession": possession,
        "player": {"id": 100 + index, "name": f"Player {index}"},
    }
    if event_type == "Pass":
        item["pass"] = {"end_location": end_location or [40.0, 40.0]}
    elif event_type == "Carry":
        item["carry"] = {"end_location": end_location or [40.0, 40.0]}
    elif event_type == "Shot":
        item["shot"] = {"end_location": end_location or [120.0, 40.0]}
    return item


def possession_change(
    index: int,
    *,
    restart_team: int = 2,
    play_pattern: str = "Regular Play",
) -> dict:
    return event(
        f"change-{index}",
        index,
        event_type="Ball Receipt*",
        team_id=restart_team,
        possession_team_id=restart_team,
        possession=11,
        play_pattern=play_pattern,
    )


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


class MainStateMachineTests(unittest.TestCase):
    def classify(self, events: list[dict], position: int = 0) -> dict:
        return classify_anchor(
            events,
            position,
            {},
            params=PARAMS,
            config=LABEL_CONFIG,
            team_ids={1, 2},
        )

    def test_retained_forward_action_is_nonterminal_failure(self) -> None:
        result = self.classify(
            [
                event("anchor", 1, end_location=[40.0, 40.0]),
                event("next-action", 2, event_type="Carry"),
            ]
        )
        self.assertEqual(result["state"], "forward_progress")
        self.assertEqual(result["outcome_tag"], "fail")
        self.assertFalse(result["terminal"])
        self.assertEqual(result["dir_mask"], 1)

    def test_retained_backward_action_is_contained(self) -> None:
        result = self.classify(
            [
                event("anchor", 1, end_location=[10.0, 40.0]),
                event("next-action", 2, event_type="Carry"),
            ]
        )
        self.assertEqual(
            (result["state"], result["outcome_tag"], result["terminal"]),
            ("contained", "neutral", False),
        )

    def test_live_turnover_is_terminal_success(self) -> None:
        result = self.classify(
            [event("anchor", 1), possession_change(2)]
        )
        self.assertEqual(
            (result["state"], result["outcome_tag"], result["terminal"]),
            ("regain_live", "success", True),
        )

    def test_defending_foul_precedes_failure_label(self) -> None:
        foul = event(
            "foul",
            2,
            event_type="Foul Committed",
            team_id=2,
            possession_team_id=1,
        )
        foul["foul_committed"] = {"penalty": True}
        result = self.classify(
            [
                event("anchor", 1),
                foul,
                possession_change(
                    3, restart_team=1, play_pattern="From Free Kick"
                ),
            ]
        )
        self.assertEqual(result["state"], "foul_conceded")
        self.assertEqual(result["outcome_tag"], "fail")
        self.assertTrue(result["terminal"])
        self.assertTrue(result["is_penalty"])

    def test_throw_in_restart_team_determines_outcome(self) -> None:
        defending_restart = self.classify(
            [
                event("anchor", 1),
                possession_change(
                    2, restart_team=2, play_pattern="From Throw In"
                ),
            ]
        )
        attacking_restart = self.classify(
            [
                event("anchor", 1),
                possession_change(
                    2, restart_team=1, play_pattern="From Throw In"
                ),
            ]
        )
        self.assertEqual(
            (
                defending_restart["state"],
                defending_restart["outcome_tag"],
                attacking_restart["state"],
                attacking_restart["outcome_tag"],
            ),
            ("throw_def", "success", "throw_atk", "neutral"),
        )
        self.assertTrue(defending_restart["terminal"])
        self.assertTrue(attacking_restart["terminal"])

    def test_high_long_pass_lost_is_clearance_success(self) -> None:
        anchor = event("anchor", 1)
        anchor["pass"].update(
            {"height": {"name": "High Pass"}, "length": 40.0}
        )
        result = self.classify([anchor, possession_change(2)])
        self.assertEqual(result["state"], "clearance_lost")
        self.assertEqual(result["outcome_tag"], "success")
        self.assertTrue(result["is_clearance"])

    def test_no_resolution_is_censored(self) -> None:
        result = self.classify([event("anchor", 1)])
        self.assertEqual(
            (
                result["state"],
                result["outcome_tag"],
                result["terminal"],
                result["censored"],
            ),
            ("censored", "censored", False, True),
        )

    def test_build_labels_only_main_anchor_types(self) -> None:
        pass_event = event("pass", 1)
        miscontrol = event("miscontrol", 2, event_type="Miscontrol")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events_dir, frames_dir = root / "events", root / "360"
            events_dir.mkdir()
            frames_dir.mkdir()
            (events_dir / "1.json").write_text(
                json.dumps([pass_event, miscontrol]),
                encoding="utf-8",
            )
            (frames_dir / "1.json").write_text(
                json.dumps(
                    [
                        freeze_frame("pass", pass_event["location"]),
                        freeze_frame("miscontrol", miscontrol["location"]),
                    ]
                ),
                encoding="utf-8",
            )

            sequences, labels = build_sequences(
                events_dir,
                frames_dir,
                SequenceConfig(),
                PARAMS,
                LABEL_CONFIG,
            )

        self.assertEqual(len(sequences), 2)
        self.assertEqual(labels["anchor_event_id"].tolist(), ["pass"])
        self.assertEqual(labels["event_type"].tolist(), ["Pass"])
        self.assertIn("state", labels.columns)
        self.assertIn("resolution_id", labels.columns)

    def test_miscontrol_only_sequence_is_valid_but_has_no_anchor_label(self) -> None:
        miscontrol = event("miscontrol", 1, event_type="Miscontrol")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events_dir, frames_dir = root / "events", root / "360"
            events_dir.mkdir()
            frames_dir.mkdir()
            (events_dir / "1.json").write_text(
                json.dumps([miscontrol]),
                encoding="utf-8",
            )
            (frames_dir / "1.json").write_text(
                json.dumps(
                    [freeze_frame("miscontrol", miscontrol["location"])]
                ),
                encoding="utf-8",
            )

            sequences, labels = build_sequences(
                events_dir,
                frames_dir,
                SequenceConfig(),
                PARAMS,
                LABEL_CONFIG,
            )

        self.assertEqual(len(sequences), 1)
        self.assertTrue(labels.empty)
        self.assertIn("state", labels.columns)
        self.assertIn("outcome_tag", labels.columns)


if __name__ == "__main__":
    unittest.main()
