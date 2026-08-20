from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from shapely.geometry import box


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUBLIC_ROOT))

from src.spn_features import (  # noqa: E402
    SPN_NETWORK_STATIC_FEATURES,
    SPN_NETWORK_TEMPORAL_FEATURES,
    SPN_STATIC_FEATURES,
    SPN_SELECTED_FEATURES,
    SPN_TEMPORAL_FEATURES,
    _events_and_frames,
    _incoming_ball,
    _legacy_carrier_x_norm,
    _sequence_context_incoming,
    build_spn_feature_table,
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
    def test_new_network_features_are_selected_for_modeling(self) -> None:
        expected = {
            "vor_carrier_area_share",
            "vor_forward_receiver_area_share",
            "pressure_edge_density",
            "pressure_mean_focus_entropy",
            "del_cross_role_edge_ratio",
            "del_carrier_degree_norm",
            "d_vor_carrier_area_share",
            "d_vor_forward_receiver_area_share",
        }

        self.assertEqual(
            expected,
            set(SPN_NETWORK_STATIC_FEATURES + SPN_NETWORK_TEMPORAL_FEATURES),
        )
        self.assertTrue(expected.issubset(SPN_SELECTED_FEATURES))

    def test_removed_features_are_not_selected_for_modeling(self) -> None:
        removed = {
            "frac_receivers_pressed",
            "n_active_carrier_boundaries",
            "shared_boundary_outlet_pressure",
            "d_shared_boundary_outlet_pressure_dt",
        }

        self.assertTrue(removed.isdisjoint(SPN_SELECTED_FEATURES))

    def test_incoming_direction_follows_only_sequence_context_chain(self) -> None:
        anchors = pd.DataFrame(
            [
                {
                    "seq_id": 1,
                    "ev_pos": 0,
                    "anchor_event_id": "anchor-1",
                    "event_type": "Carry",
                    "actor_x_norm": 20.0,
                    "actor_y": 40.0,
                    "pre_context_status": "selected",
                    "pre_context_event_id": "context",
                    "pre_context_event_type": "Pass",
                    "pre_context_direction_valid": True,
                    "pre_context_ball_x": 10.0,
                    "pre_context_ball_y": 40.0,
                    "pre_context_distance_m": 8.75,
                    "pre_context_angle_sin": 0.0,
                    "pre_context_angle_cos": 1.0,
                },
                {
                    "seq_id": 1,
                    "ev_pos": 1,
                    "anchor_event_id": "anchor-2",
                    "event_type": "Dribble",
                    "actor_x_norm": 30.0,
                    "actor_y": 40.0,
                },
                {
                    "seq_id": 1,
                    "ev_pos": 2,
                    "anchor_event_id": "anchor-3",
                    "event_type": "Carry",
                    "actor_x_norm": 30.0,
                    "actor_y": 40.0,
                },
            ]
        )

        incoming = _sequence_context_incoming(anchors, {})

        first = incoming[(1, 0, "anchor-1")]
        direct = incoming[(1, 1, "anchor-2")]
        traced = incoming[(1, 2, "anchor-3")]
        self.assertEqual(first["ball_in_context_source"], "pre_sequence_context")
        self.assertEqual(direct["ball_in_context_event_id"], "anchor-1")
        self.assertEqual(direct["ball_in_context_anchor_hops"], 1)
        self.assertEqual(traced["ball_in_context_event_id"], "anchor-1")
        self.assertEqual(traced["ball_in_context_anchor_hops"], 2)
        self.assertAlmostEqual(traced["ball_in_dist"], 8.75)
        self.assertEqual(traced["ball_in_angle_sin"], 0.0)
        self.assertEqual(traced["ball_in_angle_cos"], 1.0)

    def test_unresolved_sequence_context_chain_returns_nan(self) -> None:
        anchors = pd.DataFrame(
            [
                {
                    "seq_id": 1,
                    "ev_pos": 0,
                    "anchor_event_id": "anchor-1",
                    "event_type": "Carry",
                    "actor_x_norm": 20.0,
                    "actor_y": 40.0,
                    "pre_context_status": "hard_stoppage",
                    "pre_context_direction_valid": False,
                },
                {
                    "seq_id": 1,
                    "ev_pos": 1,
                    "anchor_event_id": "anchor-2",
                    "event_type": "Pass",
                    "actor_x_norm": 20.0,
                    "actor_y": 40.0,
                },
            ]
        )

        incoming = _sequence_context_incoming(anchors, {})

        for position, anchor_id in enumerate(("anchor-1", "anchor-2")):
            values = incoming[(1, position, anchor_id)]
            self.assertTrue(np.isnan(values["ball_in_dist"]))
            self.assertTrue(np.isnan(values["ball_in_angle_sin"]))
            self.assertTrue(np.isnan(values["ball_in_angle_cos"]))
        self.assertEqual(
            incoming[(1, 1, "anchor-2")]["ball_in_context_status"],
            "unresolved_same_location_chain",
        )

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

    def test_network_structure_features_use_visible_area_and_graph_edges(self) -> None:
        anchor = event("anchor", 1)
        xy = np.asarray(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [1.0, 1.0],
                [-1.0, 0.0],
            ]
        )
        network = {
            **minimal_network(anchor, xy),
            "players": [
                {"role": "carrier"},
                {"role": "attacker"},
                {"role": "defender"},
                {"role": "defender"},
            ],
            "attackers": [1],
            "defenders": [2, 3],
            "visible_area": box(0.0, 0.0, 10.0, 10.0),
            "areas": [
                box(0.0, 0.0, 2.0, 5.0),
                box(2.0, 0.0, 8.0, 5.0),
                box(0.0, 5.0, 5.0, 10.0),
                box(5.0, 5.0, 10.0, 10.0),
            ],
            "topology": {(0, 1), (0, 2), (1, 2), (2, 3), (0, 3)},
            "pressure_edges": [
                {"src": 2, "dst": 0, "w": 0.75},
                {"src": 2, "dst": 1, "w": 0.25},
                {"src": 3, "dst": 0, "w": 1.0},
            ],
            "pass_edges": [{"dst": 1, "w": 0.0}],
        }

        values = network_features(network, [anchor], 0)

        self.assertAlmostEqual(values["vor_carrier_area_share"], 0.1)
        self.assertAlmostEqual(values["vor_forward_receiver_area_share"], 0.3)
        self.assertAlmostEqual(values["pressure_edge_density"], 0.75)
        expected_entropy = (
            -(0.75 * np.log(0.75) + 0.25 * np.log(0.25))
            / np.log(2.0)
            / 2.0
        )
        self.assertAlmostEqual(
            values["pressure_mean_focus_entropy"], expected_entropy
        )
        self.assertAlmostEqual(values["del_cross_role_edge_ratio"], 0.6)
        self.assertAlmostEqual(values["del_carrier_degree_norm"], 1.0)

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

    def test_sequence_anchors_create_temporal_features_before_final_label_join(self) -> None:
        events = [
            event("anchor-1", 1, location=[20.0, 40.0]),
            event("low-context", 2, location=[25.0, 40.0]),
            event("anchor-2", 3, location=[30.0, 40.0]),
        ]
        sequences = pd.DataFrame(
            [
                {
                    "match_id": "1",
                    "seq_id": 1,
                    "ev_pos": 0,
                    "anchor_event_id": "anchor-1",
                    "team_name": "Team 1",
                    "possession_team_name": "Team 1",
                },
                {
                    "match_id": "1",
                    "seq_id": 1,
                    "ev_pos": 1,
                    "anchor_event_id": "anchor-2",
                    "team_name": "Team 1",
                    "possession_team_name": "Team 1",
                },
            ]
        )
        labels = pd.DataFrame(
            [
                {
                    "match_id": "1",
                    "seq_id": 1,
                    "ev_pos": 0,
                    "anchor_event_id": "anchor-1",
                    "outcome_tag": "fail",
                    "future_output_target": "fail",
                    "event_value": -1.0,
                    "training_weight": 0.5,
                    "value_model_eligible": True,
                    "terminal": False,
                    "event_state": "continued_pressure",
                    "transition_outcome_tag": "neutral",
                    "is_sequence_last": False,
                    "anchor_count": 2,
                },
                {
                    "match_id": "1",
                    "seq_id": 1,
                    "ev_pos": 1,
                    "anchor_event_id": "anchor-2",
                    "outcome_tag": "fail",
                    "future_output_target": "fail",
                    "event_value": -1.0,
                    "training_weight": 0.5,
                    "value_model_eligible": True,
                    "terminal": True,
                    "event_state": "crossed_half",
                    "transition_outcome_tag": "fail",
                    "is_sequence_last": True,
                    "anchor_count": 2,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events_dir, frames_dir = root / "events", root / "360"
            events_dir.mkdir()
            frames_dir.mkdir()
            (events_dir / "1.json").write_text(
                json.dumps(events), encoding="utf-8"
            )
            (frames_dir / "1.json").write_text(
                json.dumps(
                    [
                        {"event_uuid": "anchor-1", "freeze_frame": [{}]},
                        {"event_uuid": "anchor-2", "freeze_frame": [{}]},
                    ]
                ),
                encoding="utf-8",
            )

            def values(network, *_args, **_kwargs):
                value = 1.0 if network["event"]["id"] == "anchor-1" else 3.0
                return {feature: value for feature in SPN_STATIC_FEATURES}

            with patch(
                "src.spn_features.build_spn_network",
                side_effect=lambda source, *_args, **_kwargs: {"event": source},
            ), patch("src.spn_features.network_features", side_effect=values):
                table = build_spn_feature_table(
                    sequences,
                    labels,
                    events_dir,
                    frames_dir,
                )

        self.assertEqual(table["anchor_event_id"].tolist(), ["anchor-1", "anchor-2"])
        self.assertEqual(table["outcome_tag"].tolist(), ["fail", "fail"])
        self.assertEqual(table["event_value"].tolist(), [-1.0, -1.0])
        self.assertEqual(table["training_weight"].tolist(), [0.5, 0.5])
        self.assertTrue(table.loc[0, SPN_TEMPORAL_FEATURES].isna().all())
        self.assertTrue(table.loc[1, SPN_TEMPORAL_FEATURES].notna().all())
        self.assertTrue((table.loc[1, SPN_TEMPORAL_FEATURES] == 2.0).all())

    def test_sequence_first_anchor_uses_selected_pre_context(self) -> None:
        events = [
            event("context", 1, location=[10.0, 40.0]),
            event("anchor-1", 2, location=[20.0, 40.0]),
            event("anchor-2", 3, location=[30.0, 40.0]),
        ]
        sequences = pd.DataFrame(
            [
                {
                    "match_id": "1",
                    "seq_id": 1,
                    "ev_pos": 0,
                    "anchor_event_id": "anchor-1",
                    "team_name": "Team 1",
                    "possession_team_name": "Team 1",
                    "pre_context_status": "selected",
                    "pre_context_event_id": "context",
                    "pre_context_direction_valid": True,
                    "pre_context_distance_m": 8.75,
                    "pre_context_angle_sin": 0.0,
                    "pre_context_angle_cos": 1.0,
                },
                {
                    "match_id": "1",
                    "seq_id": 1,
                    "ev_pos": 1,
                    "anchor_event_id": "anchor-2",
                    "team_name": "Team 1",
                    "possession_team_name": "Team 1",
                    "pre_context_status": "not_sequence_start",
                    "pre_context_event_id": "",
                    "pre_context_direction_valid": False,
                },
            ]
        )
        labels = pd.DataFrame(
            [
                {
                    "match_id": "1",
                    "seq_id": 1,
                    "ev_pos": position,
                    "anchor_event_id": anchor_id,
                    "outcome_tag": "fail",
                    "future_output_target": "fail",
                    "event_value": -1.0,
                    "training_weight": 0.5,
                    "value_model_eligible": True,
                    "terminal": position == 1,
                }
                for position, anchor_id in enumerate(("anchor-1", "anchor-2"))
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events_dir, frames_dir = root / "events", root / "360"
            events_dir.mkdir()
            frames_dir.mkdir()
            (events_dir / "1.json").write_text(
                json.dumps(events), encoding="utf-8"
            )
            (frames_dir / "1.json").write_text(
                json.dumps(
                    [
                        {"event_uuid": event_id, "freeze_frame": [{}]}
                        for event_id in ("context", "anchor-1", "anchor-2")
                    ]
                ),
                encoding="utf-8",
            )

            def values(network, *_args, **_kwargs):
                value = {
                    "context": 0.0,
                    "anchor-1": 1.0,
                    "anchor-2": 3.0,
                }[network["event"]["id"]]
                return {feature: value for feature in SPN_STATIC_FEATURES}

            with patch(
                "src.spn_features.build_spn_network",
                side_effect=lambda source, *_args, **_kwargs: {"event": source},
            ), patch("src.spn_features.network_features", side_effect=values):
                table = build_spn_feature_table(
                    sequences,
                    labels,
                    events_dir,
                    frames_dir,
                )

        self.assertTrue(table.loc[0, "pre_context_spn_valid"])
        self.assertTrue((table.loc[0, SPN_TEMPORAL_FEATURES] == 1.0).all())
        self.assertTrue((table.loc[1, SPN_TEMPORAL_FEATURES] == 2.0).all())
        self.assertEqual(table.loc[0, "ball_in_dist"], 8.75)
        self.assertEqual(table.loc[0, "ball_in_angle_sin"], 0.0)
        self.assertEqual(table.loc[0, "ball_in_angle_cos"], 1.0)
        self.assertEqual(table.loc[1, "ball_in_dist"], 8.75)
        self.assertEqual(table.loc[1, "ball_in_angle_sin"], 0.0)
        self.assertEqual(table.loc[1, "ball_in_angle_cos"], 1.0)
        self.assertEqual(
            table.loc[1, "ball_in_context_source"],
            "sequence_anchor",
        )
        self.assertEqual(
            table.loc[1, "ball_in_context_event_id"],
            "anchor-1",
        )
        self.assertEqual(table.loc[1, "ball_in_context_anchor_hops"], 1)


if __name__ == "__main__":
    unittest.main()
