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

from src.data_pipeline import (  # noqa: E402
    EVENT_VALUE_BY_OUTPUT,
    LabelConfig,
    SequenceConfig,
    build_sequences,
    classify_anchor,
    resolve_sequence_output,
)
from src.spn_network import PressureParams  # noqa: E402


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
    def test_final_output_value_mapping(self) -> None:
        self.assertEqual(
            EVENT_VALUE_BY_OUTPUT,
            {"success": 1.0, "neutral": 0.0, "fail": -1.0},
        )

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

    def test_build_sequences_and_labels_only_core_anchor_types(self) -> None:
        pass_event = event("pass", 1)
        carry_event = event("carry", 2, event_type="Carry")
        dribble_event = event("dribble", 3, event_type="Dribble")
        miscontrol = event("miscontrol", 4, event_type="Miscontrol")
        shot = event("shot", 5, event_type="Shot")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events_dir, frames_dir = root / "events", root / "360"
            events_dir.mkdir()
            frames_dir.mkdir()
            (events_dir / "1.json").write_text(
                json.dumps(
                    [pass_event, carry_event, dribble_event, miscontrol, shot]
                ),
                encoding="utf-8",
            )
            (frames_dir / "1.json").write_text(
                json.dumps(
                    [
                        freeze_frame("pass", pass_event["location"]),
                        freeze_frame("carry", carry_event["location"]),
                        freeze_frame("dribble", dribble_event["location"]),
                        freeze_frame("miscontrol", miscontrol["location"]),
                        freeze_frame("shot", shot["location"]),
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

        self.assertEqual(sequences["anchor_event_id"].tolist(), ["pass", "carry", "dribble"])
        self.assertEqual(
            labels["anchor_event_id"].tolist(), ["pass", "carry", "dribble"]
        )
        self.assertEqual(labels["event_type"].tolist(), ["Pass", "Carry", "Dribble"])
        self.assertEqual(labels["anchor_count"].tolist(), [3, 3, 3])
        self.assertEqual(labels["first_anchor_event_id"].tolist(), ["pass"] * 3)
        self.assertEqual(labels["last_anchor_event_id"].tolist(), ["dribble"] * 3)
        self.assertEqual(
            labels["state"].tolist()[:2],
            ["continued_pressure", "continued_pressure"],
        )
        self.assertTrue(
            (labels["outcome_tag"] == labels["sequence_output"]).all()
        )
        self.assertEqual(
            labels["transition_outcome_tag"].tolist()[:2],
            ["neutral", "neutral"],
        )
        self.assertTrue(labels["event_value"].isna().all())
        self.assertFalse(labels["value_model_eligible"].any())
        self.assertTrue(
            np.allclose(labels["training_weight"], [1 / 3, 1 / 3, 1 / 3])
        )
        self.assertEqual(labels["terminal"].tolist()[:2], [False, False])
        self.assertEqual(
            labels["resolution_event_id"].tolist()[:2], ["carry", "dribble"]
        )
        self.assertTrue(labels["is_sequence_last"].tolist()[-1])
        self.assertIn("state", labels.columns)
        self.assertIn("resolution_id", labels.columns)
        self.assertIn("sequence_end_state", labels.columns)

    def test_every_anchor_inherits_success_value_and_sequence_weight(self) -> None:
        first = event("first", 1)
        second = event("second", 2, event_type="Carry")
        press_control = event(
            "press-control",
            3,
            event_type="Carry",
            team_id=2,
            possession_team_id=2,
            possession=11,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events_dir, frames_dir = root / "events", root / "360"
            events_dir.mkdir()
            frames_dir.mkdir()
            (events_dir / "1.json").write_text(
                json.dumps([first, second, press_control]), encoding="utf-8"
            )
            (frames_dir / "1.json").write_text(
                json.dumps(
                    [
                        freeze_frame("first", first["location"]),
                        freeze_frame("second", second["location"]),
                    ]
                ),
                encoding="utf-8",
            )

            _, labels = build_sequences(
                events_dir,
                frames_dir,
                SequenceConfig(),
                PARAMS,
                LABEL_CONFIG,
            )

        self.assertEqual(labels["outcome_tag"].tolist(), ["success", "success"])
        self.assertEqual(
            labels["future_output_target"].tolist(), ["success", "success"]
        )
        self.assertEqual(labels["event_value"].tolist(), [1.0, 1.0])
        self.assertEqual(labels["training_weight"].tolist(), [0.5, 0.5])
        self.assertTrue(labels["value_model_eligible"].all())
        self.assertEqual(
            labels["transition_outcome_tag"].tolist(), ["neutral", "success"]
        )

    def test_carry_to_dribble_uses_the_dribble_as_a_direct_endpoint(self) -> None:
        carry = event(
            "carry",
            1,
            event_type="Carry",
            location=[20.0, 40.0],
            end_location=[30.0, 40.0],
        )
        carry["duration"] = 1.0
        dribble = event(
            "dribble",
            2,
            event_type="Dribble",
            location=[30.0, 40.0],
        )
        dribble["player"] = carry["player"]
        dribble["dribble"] = {"outcome": {"name": "Complete"}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events_dir, frames_dir = root / "events", root / "360"
            events_dir.mkdir()
            frames_dir.mkdir()
            (events_dir / "1.json").write_text(
                json.dumps([carry, dribble]), encoding="utf-8"
            )
            (frames_dir / "1.json").write_text(
                json.dumps(
                    [
                        freeze_frame("carry", carry["location"]),
                        freeze_frame("dribble", dribble["location"]),
                    ]
                ),
                encoding="utf-8",
            )

            _, labels = build_sequences(
                events_dir,
                frames_dir,
                SequenceConfig(),
                PARAMS,
                LABEL_CONFIG,
            )

        carry_label = labels.iloc[0]
        self.assertEqual(carry_label["state"], "continued_pressure")
        self.assertTrue(carry_label["direct_transition"])
        self.assertTrue(carry_label["carry_to_dribble"])
        self.assertEqual(carry_label["next_dribble_outcome"], "Complete")
        self.assertTrue(carry_label["next_take_on_success"])
        self.assertFalse(carry_label["terminal"])

    def test_miscontrol_only_input_has_no_core_sequence(self) -> None:
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

            with self.assertRaisesRegex(ValueError, "No high-pressure events"):
                build_sequences(
                    events_dir,
                    frames_dir,
                    SequenceConfig(),
                    PARAMS,
                    LABEL_CONFIG,
                )


class SequenceContextResolverTests(unittest.TestCase):
    def resolve(
        self,
        events: list[dict],
        frames: dict[str, dict] | None = None,
        *,
        sequence: list[tuple[int, dict]] | None = None,
        label_config: LabelConfig = LABEL_CONFIG,
    ) -> dict:
        return resolve_sequence_output(
            events,
            sequence or [(0, {})],
            frames or {},
            sequence_config=SequenceConfig(),
            params=PARAMS,
            config=label_config,
            team_ids={1, 2},
        )

    def test_press_team_stable_control_is_success(self) -> None:
        result = self.resolve(
            [
                event("anchor", 1),
                event(
                    "press-carry",
                    2,
                    event_type="Carry",
                    team_id=2,
                    possession_team_id=2,
                    possession=11,
                ),
            ]
        )
        self.assertEqual(
            (
                result["sequence_end_state"],
                result["sequence_output"],
                result["resolution_event_id"],
            ),
            ("regain_control", "success", "press-carry"),
        )

    def test_miscontrol_waits_for_stable_control(self) -> None:
        result = self.resolve(
            [
                event("anchor", 1),
                event("lost-touch", 2, event_type="Miscontrol"),
                event(
                    "press-pass",
                    3,
                    team_id=2,
                    possession_team_id=2,
                    possession=11,
                ),
            ]
        )
        self.assertEqual(result["sequence_output"], "success")
        self.assertEqual(result["resolution_event_id"], "press-pass")
        self.assertEqual(result["unstable_event_count"], 1)

    def test_crossing_half_is_failure_without_360(self) -> None:
        next_control = event(
            "crossed-half",
            2,
            event_type="Carry",
            location=[60.0, 40.0],
        )
        result = self.resolve([event("anchor", 1), next_control])
        self.assertEqual(
            (result["sequence_end_state"], result["sequence_output"]),
            ("crossed_half", "fail"),
        )

    def test_low_pressure_control_uses_final_anchor_geometry(self) -> None:
        next_control = event(
            "released",
            2,
            event_type="Carry",
            location=[35.0, 40.0],
        )
        frame = freeze_frame("released", next_control["location"])
        frame["freeze_frame"][1]["location"] = [110.0, 40.0]
        result = self.resolve(
            [event("anchor", 1, end_location=[45.0, 40.0]), next_control],
            {"released": frame},
        )
        self.assertEqual(
            (result["sequence_end_state"], result["sequence_output"]),
            ("forward_escape", "fail"),
        )
        self.assertLessEqual(result["next_control_P_total"], 0.65)

    def test_high_pressure_control_reports_missing_anchor(self) -> None:
        next_control = event("still-pressed", 2, event_type="Carry")
        result = self.resolve(
            [event("anchor", 1), next_control],
            {"still-pressed": freeze_frame("still-pressed", [20.0, 40.0])},
        )
        self.assertEqual(result["sequence_output"], "censored")
        self.assertEqual(
            result["sequence_consistency_error"],
            "missing_anchor_inside_sequence",
        )

    def test_opponent_unstable_touch_explains_high_pressure_split(self) -> None:
        opponent_touch = event(
            "press-miscontrol",
            2,
            event_type="Miscontrol",
            team_id=2,
            possession_team_id=1,
        )
        next_control = event("still-pressed", 3, event_type="Carry")
        result = self.resolve(
            [event("anchor", 1), opponent_touch, next_control],
            {"still-pressed": freeze_frame("still-pressed", [20.0, 40.0])},
        )
        self.assertEqual(
            result["sequence_end_state"],
            "pressure_restarted_after_opponent_touch",
        )
        self.assertEqual(result["sequence_consistency_error"], "")

    def test_out_signal_opens_only_restart_extension(self) -> None:
        out_event = event("out", 4, event_type="Miscontrol")
        out_event["out"] = True
        restart = event(
            "throw",
            12,
            team_id=2,
            possession_team_id=2,
            possession=11,
            play_pattern="From Throw In",
        )
        result = self.resolve([event("anchor", 1), out_event, restart])
        self.assertEqual(
            (result["sequence_end_state"], result["sequence_output"]),
            ("forced_out", "success"),
        )
        self.assertTrue(result["pending_restart_seen"])

    def test_inherited_play_pattern_is_not_a_new_restart(self) -> None:
        anchor = event(
            "anchor",
            1,
            play_pattern="From Goal Kick",
            end_location=[45.0, 40.0],
        )
        next_control = event(
            "same-possession-carry",
            2,
            event_type="Carry",
            play_pattern="From Goal Kick",
            location=[35.0, 40.0],
        )
        frame = freeze_frame(
            "same-possession-carry", next_control["location"]
        )
        frame["freeze_frame"][1]["location"] = [110.0, 40.0]
        result = self.resolve(
            [anchor, next_control],
            {"same-possession-carry": frame},
        )
        self.assertEqual(
            (result["sequence_end_state"], result["sequence_output"]),
            ("forward_escape", "fail"),
        )

    def test_dead_ball_skips_inherited_events_until_actual_restart(self) -> None:
        anchor = event("anchor", 1, play_pattern="From Goal Kick")
        out_event = event(
            "out",
            2,
            event_type="Miscontrol",
            play_pattern="From Goal Kick",
        )
        out_event["out"] = True
        inherited_receipt = event(
            "receipt",
            3,
            event_type="Ball Receipt*",
            play_pattern="From Goal Kick",
        )
        inherited_pressure = event(
            "pressure",
            4,
            event_type="Pressure",
            team_id=2,
            possession_team_id=1,
            play_pattern="From Goal Kick",
        )
        substitution = event(
            "substitution",
            5,
            event_type="Substitution",
            team_id=2,
            possession_team_id=1,
            play_pattern="From Goal Kick",
        )
        restart = event(
            "throw",
            20,
            team_id=2,
            possession_team_id=2,
            possession=11,
            play_pattern="From Throw In",
        )
        result = self.resolve(
            [
                anchor,
                out_event,
                inherited_receipt,
                inherited_pressure,
                substitution,
                restart,
            ]
        )
        self.assertEqual(
            (
                result["sequence_end_state"],
                result["sequence_output"],
                result["resolution_event_id"],
                result["resolution_event_type"],
            ),
            ("forced_out", "success", "throw", "Pass"),
        )
        self.assertEqual(result["restart_evidence"], "explicit_out")

    def test_dead_ball_ignores_restart_pattern_in_same_possession_pass(self) -> None:
        anchor = event("anchor", 1, play_pattern="From Goal Kick")
        anchor["out"] = True
        inherited_pass = event(
            "inherited-pass",
            2,
            play_pattern="From Goal Kick",
            possession=10,
        )
        restart = event(
            "throw",
            3,
            team_id=2,
            possession_team_id=2,
            possession=11,
            play_pattern="From Throw In",
        )
        result = self.resolve([anchor, inherited_pass, restart])
        self.assertEqual(result["resolution_event_id"], "throw")
        self.assertEqual(result["sequence_output"], "success")

    def test_dead_ball_stops_at_first_new_possession_live_control(self) -> None:
        anchor = event("anchor", 1)
        anchor["out"] = True
        inherited_receipt = event(
            "receipt",
            2,
            event_type="Ball Receipt*",
            play_pattern="From Goal Kick",
        )
        first_live_control = event(
            "first-live-control",
            3,
            event_type="Carry",
            team_id=2,
            possession_team_id=2,
            possession=11,
            play_pattern="Regular Play",
        )
        unrelated_later_restart = event(
            "later-throw",
            20,
            team_id=1,
            possession_team_id=1,
            possession=12,
            play_pattern="From Throw In",
        )
        result = self.resolve(
            [
                anchor,
                inherited_receipt,
                first_live_control,
                unrelated_later_restart,
            ]
        )
        self.assertEqual(
            (
                result["sequence_end_state"],
                result["sequence_output"],
                result["resolution_event_id"],
            ),
            ("forced_out", "success", "first-live-control"),
        )
        self.assertEqual(
            result["restart_evidence"],
            "inferred_new_possession_control",
        )

    def test_restart_actor_and_possession_owner_must_agree(self) -> None:
        anchor = event("anchor", 1)
        anchor["out"] = True
        inconsistent_restart = event(
            "throw",
            2,
            team_id=2,
            possession_team_id=1,
            possession=11,
            play_pattern="From Throw In",
        )
        result = self.resolve([anchor, inconsistent_restart])
        self.assertEqual(result["sequence_output"], "censored")
        self.assertEqual(result["sequence_end_state"], "unresolved_restart")
        self.assertEqual(result["censored_reason"], "unknown_restart_team")

    def test_injury_clearance_is_administrative_censoring(self) -> None:
        anchor = event("anchor", 1)
        anchor["pass"]["outcome"] = {"name": "Injury Clearance"}
        later_restart = event(
            "throw",
            20,
            team_id=2,
            possession_team_id=2,
            possession=11,
            play_pattern="From Throw In",
        )
        result = self.resolve([anchor, later_restart])
        self.assertEqual(result["sequence_output"], "censored")
        self.assertEqual(result["censored_reason"], "injury_clearance")
        self.assertEqual(result["resolution_event_id"], "anchor")

    def test_late_live_event_resolves_without_an_arbitrary_timeout(self) -> None:
        late_control = event(
            "late-regain",
            7,
            event_type="Carry",
            team_id=2,
            possession_team_id=2,
            possession=11,
        )
        result = self.resolve([event("anchor", 1), late_control])
        self.assertEqual(
            (result["sequence_end_state"], result["sequence_output"]),
            ("regain_control", "success"),
        )
        self.assertEqual(result["resolution_event_id"], "late-regain")
        self.assertEqual(result["context_event_count"], 1)

    def test_final_anchor_endpoint_crosses_half_immediately(self) -> None:
        result = self.resolve(
            [event("anchor", 1, end_location=[70.0, 40.0])]
        )
        self.assertEqual(
            (result["sequence_end_state"], result["sequence_output"]),
            ("crossed_half", "fail"),
        )
        self.assertEqual(
            result["resolution_evidence"],
            "final_anchor_endpoint_crossed_half",
        )
        self.assertEqual(result["resolution_event_id"], "anchor")

    def test_missing_360_is_skipped_until_deterministic_escape(self) -> None:
        unobservable = event(
            "unobservable",
            2,
            event_type="Carry",
            location=[30.0, 40.0],
            end_location=[35.0, 40.0],
        )
        crossed = event(
            "crossed",
            3,
            location=[45.0, 40.0],
            end_location=[70.0, 40.0],
        )
        result = self.resolve([event("anchor", 1), unobservable, crossed])
        self.assertEqual(result["sequence_output"], "fail")
        self.assertEqual(result["resolution_event_id"], "crossed")
        self.assertEqual(result["skipped_unobservable_control_count"], 1)

    def test_administrative_stoppage_blocks_unrelated_late_control(self) -> None:
        stoppage = event("injury", 2, event_type="Injury Stoppage")
        late_control = event(
            "later-regain",
            20,
            event_type="Carry",
            team_id=2,
            possession_team_id=2,
            possession=11,
        )
        result = self.resolve(
            [event("anchor", 1), stoppage, late_control]
        )
        self.assertEqual(result["sequence_output"], "censored")
        self.assertEqual(
            result["censored_reason"], "administrative_stoppage"
        )
        self.assertEqual(result["resolution_event_id"], "injury")

    def test_final_anchor_out_waits_for_delayed_restart(self) -> None:
        anchor = event("anchor", 1)
        anchor["out"] = True
        restart = event(
            "throw",
            30,
            team_id=2,
            possession_team_id=2,
            possession=11,
            play_pattern="From Throw In",
        )
        result = self.resolve([anchor, restart])
        self.assertEqual(
            (result["sequence_end_state"], result["sequence_output"]),
            ("forced_out", "success"),
        )
        self.assertEqual(result["out_signal_event_id"], "anchor")
        self.assertEqual(result["restart_evidence"], "explicit_out")

    def test_not_high_by_full_anchor_rule_is_a_pressure_release(self) -> None:
        next_control = event(
            "boundary-only",
            2,
            event_type="Carry",
            location=[35.0, 40.0],
        )
        with patch(
            "src.data_pipeline.event_is_high_press", return_value=None
        ), patch("src.data_pipeline._pressure_at_event", return_value=0.8):
            result = self.resolve(
                [event("anchor", 1, end_location=[45.0, 40.0]), next_control],
                {"boundary-only": freeze_frame("boundary-only", [35.0, 40.0])},
            )
        self.assertEqual(result["sequence_output"], "fail")
        self.assertEqual(
            result["resolution_evidence"], "not_high_by_full_anchor_rule"
        )

    def test_adjacent_opponent_anchor_confirms_regain_and_stops_scan(self) -> None:
        next_pressure = event(
            "next-pressure",
            2,
            event_type="Carry",
            team_id=2,
            possession_team_id=2,
            possession=11,
        )
        later_regain = event(
            "later-regain",
            3,
            event_type="Carry",
            team_id=1,
            possession_team_id=1,
            possession=12,
        )
        result = self.resolve(
            [event("anchor", 1), next_pressure, later_regain]
        )
        self.assertEqual(result["sequence_output"], "success")
        self.assertEqual(result["sequence_end_state"], "regain_control")
        self.assertEqual(result["resolution_event_id"], "next-pressure")

    def test_same_team_new_pressure_stops_before_its_escape_result(self) -> None:
        next_pressure = event(
            "new-pressure-spell",
            10,
            end_location=[70.0, 40.0],
        )
        result = self.resolve(
            [event("anchor", 1), next_pressure],
            {"new-pressure-spell": freeze_frame("new-pressure-spell", [20.0, 40.0])},
        )
        self.assertEqual(result["sequence_output"], "censored")
        self.assertEqual(result["sequence_end_state"], "new_pressure_spell")
        self.assertEqual(
            result["censored_reason"], "inactive_gap_before_new_pressure"
        )

    def test_unobservable_retention_blocks_later_turnover_attribution(self) -> None:
        unobservable = event(
            "unobservable-retention",
            2,
            event_type="Carry",
            end_location=[30.0, 40.0],
        )
        later_turnover = event(
            "later-turnover",
            3,
            team_id=2,
            possession_team_id=2,
            possession=11,
        )
        result = self.resolve(
            [event("anchor", 1), unobservable, later_turnover]
        )
        self.assertEqual(result["sequence_output"], "censored")
        self.assertEqual(
            result["sequence_end_state"],
            "ambiguous_turnover_after_unobservable_control",
        )
        self.assertEqual(
            result["censored_reason"],
            "unobservable_control_before_turnover",
        )
        self.assertEqual(result["resolution_event_id"], "later-turnover")


if __name__ == "__main__":
    unittest.main()
