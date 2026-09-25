from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spn.analysis import (
    InsufficientDataError, bootstrap_team_efficiency, build_sequence_values,
    build_step_values, driver_table, load_analysis_inputs, prepare_predictions,
    team_efficiency_tables,
)
from spn.model import KEY_COLUMNS


class ContractModel:
    features = ["P_total"]
    config = {"model_id": "contract-test", "model_version": "1.0", "probability_layer": "raw"}

    def predict(self, features):
        table = features[KEY_COLUMNS].copy()
        table["p_fail"] = 0.4
        table["p_success"] = 0.1 + 0.1 * features.P_total
        table["p_neutral"] = 1 - table.p_fail - table.p_success
        return table


def sequence_fixture():
    frame = pd.DataFrame([
        ("1", 1, 0, "A", "success", 0.6, 0.2, 0.2),
        ("1", 1, 1, "A", "success", 0.5, 0.2, 0.3),
        ("1", 1, 2, "A", "success", 0.2, 0.3, 0.5),
        ("2", 1, 0, "A", "fail", 0.7, 0.2, 0.1),
        ("3", 1, 0, "B", "neutral", 0.4, 0.4, 0.2),
        ("3", 1, 1, "B", "neutral", 0.3, 0.4, 0.3),
        ("4", 1, 0, "B", "censored", 0.3, 0.4, 0.3),
    ], columns=["match_id", "seq_id", "ev_pos", "press_team", "observed_outcome", "p_fail", "p_neutral", "p_success"])
    frame["anchor_event_id"] = [f"event-{i}" for i in range(len(frame))]
    frame["evaluation_eligible"] = frame.observed_outcome.ne("censored")
    return frame


class AnalysisTests(unittest.TestCase):
    def test_existing_sequence_value_and_weighting(self):
        prepared, audit = prepare_predictions(sequence_fixture())
        self.assertEqual(audit["eligible_sequences"], 3)
        sequences = build_sequence_values(build_step_values(prepared))
        success = sequences.loc[sequences.observed_outcome.eq("success")].iloc[0]
        self.assertAlmostEqual(success.process_value, 0.7)
        self.assertAlmostEqual(success.terminal_value, 0.5)
        teams, _ = team_efficiency_tables(sequences, [0.5])
        a = teams.loc[teams.press_team.eq("A")].iloc[0]
        self.assertAlmostEqual(a.absolute_efficiency, 0.3)
        self.assertAlmostEqual(a.relative_efficiency, 1 / 30)
        self.assertAlmostEqual(np.average(teams.relative_efficiency, weights=teams.n_sequences), 0)

    def test_bootstrap_reproducible_and_uses_team_matches(self):
        predictions, _ = prepare_predictions(sequence_fixture())
        sequences = build_sequence_values(build_step_values(predictions))
        one = bootstrap_team_efficiency(sequences, iterations=25, random_state=4)
        two = bootstrap_team_efficiency(sequences, iterations=25, random_state=4)
        pd.testing.assert_frame_equal(one, two)
        self.assertEqual(one.set_index("press_team").bootstrap_matches.to_dict(), {"A": 2, "B": 1})

    def test_empty_and_censored_input_has_explicit_status(self):
        for frame in [sequence_fixture().iloc[:0], sequence_fixture().iloc[-1:]]:
            with self.assertRaises(InsufficientDataError):
                prepare_predictions(frame)

    def test_single_anchor_sequences_cannot_define_drivers(self):
        predictions = sequence_fixture().query("ev_pos == 0 and observed_outcome != 'censored'").copy()
        features = predictions[KEY_COLUMNS].assign(P_total=[0.2, 0.5, 0.8])
        with self.assertRaises(InsufficientDataError):
            driver_table(predictions, features, ["P_total"])

    def test_constant_feature_does_not_destroy_varying_feature_regression(self):
        predictions, _ = prepare_predictions(sequence_fixture())
        features = predictions[KEY_COLUMNS].copy()
        features["P_total"] = [0.1, 0.2, 0.7, 0.4, 0.2, 0.6]
        features["n_open_pass"] = 2.0
        drivers, _, audit = driver_table(predictions, features, ["P_total", "n_open_pass"])
        self.assertTrue(np.isfinite(audit["joint_r2"]))
        self.assertTrue(np.isnan(drivers.set_index("feature").loc["n_open_pass", "pearson_r_vt"]))

    def test_noninteger_position_rejected(self):
        frame = sequence_fixture(); frame["ev_pos"] = frame.ev_pos.astype(float); frame.loc[1, "ev_pos"] = 1.2
        with self.assertRaisesRegex(ValueError, "integers"):
            prepare_predictions(frame)

    def test_plot_preserves_interval_even_when_it_excludes_point_estimate(self):
        from spn.plotting import plot_intensity_efficiency, plt
        table = pd.DataFrame({
            "press_team": ["A", "B"], "carrier_pressure_intensity": [12.0, 15.0],
            "relative_efficiency_per_100_sequences": [-1.0, 1.0],
            "relative_efficiency_ci_low_per_100": [-0.9, 0.8],
            "relative_efficiency_ci_high_per_100": [-0.7, 1.2],
            "marker_size": [150.0, 520.0], "successful_sequence_rate": [0.2, 0.4],
            "n_sequences": [100, 300],
        })
        figure = plot_intensity_efficiency(table, {"intensity_median": 13.5})
        try:
            segments = figure.axes[0].collections[0].get_segments()
            np.testing.assert_allclose(segments[0], [[12.0, -0.9], [12.0, -0.7]])
        finally:
            plt.close(figure)


class InputContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = ContractModel()
        self.features = pd.DataFrame([
            ("00042", 1, 0, "a", 0.1), ("00042", 1, 1, "b", 0.8),
            ("match-alpha", 1, 0, "c", 0.2),
        ], columns=[*KEY_COLUMNS, "P_total"])
        self.predictions = self.model.predict(self.features).assign(
            press_team=["A", "A", "B"], observed_outcome=["success", "success", "fail"],
            evaluation_eligible=True, model_id="contract-test", model_version="1.0",
        )
        self.features.to_csv(self.root / "features.csv", index=False)
        self.predictions.to_csv(self.root / "predictions.csv", index=False)
        self.predictions.groupby(["match_id", "seq_id"]).size().rename("anchor_count").to_csv(self.root / "sequences.csv")

    def tearDown(self):
        self.temp.cleanup()

    def test_public_csv_keys_and_feature_alignment(self):
        predictions, features, audit = load_analysis_inputs(self.root, self.model)
        self.assertEqual(predictions.match_id.unique().tolist(), ["00042", "match-alpha"])
        self.assertEqual(features.anchor_event_id.tolist(), ["a", "b", "c"])
        self.assertEqual(audit["sequence_counts_checked"], 2)
        self.assertLess(audit["probability_parity_max_abs_difference"], 1e-12)

    def test_mixed_models_rejected(self):
        self.predictions.loc[0, "model_id"] = "wrong-model"
        self.predictions.to_csv(self.root / "predictions.csv", index=False)
        with self.assertRaisesRegex(ValueError, "different or unidentified"):
            load_analysis_inputs(self.root, self.model)

    def test_truncated_tail_rejected_against_sequence_export(self):
        self.predictions.drop(index=1).to_csv(self.root / "predictions.csv", index=False)
        with self.assertRaisesRegex(ValueError, "truncated"):
            load_analysis_inputs(self.root, self.model)

    def test_wrong_features_rejected_by_probability_parity(self):
        self.features.loc[0, "P_total"] = 0.9
        self.features.to_csv(self.root / "features.csv", index=False)
        with self.assertRaisesRegex(ValueError, "do not reproduce"):
            load_analysis_inputs(self.root, self.model)

    def test_missing_feature_exports_explained(self):
        (self.root / "features.csv").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "export-features"):
            load_analysis_inputs(self.root, self.model)


if __name__ == "__main__":
    unittest.main()
