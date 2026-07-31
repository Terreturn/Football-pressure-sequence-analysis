from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUBLIC_ROOT))

from src.hpn_model import (  # noqa: E402
    FINAL_V4_XGB_PARAMS,
    apply_isotonic,
    fit_isotonic,
    grouped_calibration_comparison,
    make_model,
)


class PublicModelTests(unittest.TestCase):
    def test_xgboost_defaults_match_final_main_contract(self) -> None:
        model = make_model("xgb")
        parameters = model.get_params()
        for name, value in FINAL_V4_XGB_PARAMS.items():
            self.assertEqual(parameters[name], value)

    def test_isotonic_preserves_the_probability_simplex(self) -> None:
        probability = np.asarray([
            [.70, .25, .05], [.20, .65, .15], [.10, .20, .70],
            [.60, .30, .10], [.20, .20, .60], [.15, .70, .15],
        ])
        truth = np.asarray([0, 1, 2, 0, 2, 1])
        calibrated = apply_isotonic(probability, fit_isotonic(probability, truth))
        self.assertTrue(np.isfinite(calibrated).all())
        self.assertTrue(np.allclose(calibrated.sum(axis=1), 1.0))

    def test_nested_grouped_calibration_returns_both_layers(self) -> None:
        rows = []
        for match in range(6):
            for klass, label in enumerate(["fail", "neutral", "success"]):
                for repetition in range(2):
                    rows.append({
                        "match_id": str(match),
                        "outcome_tag": label,
                        "feature_a": klass + repetition * .1 + match * .01,
                        "feature_b": 2 - klass + repetition * .2,
                    })
        result = grouped_calibration_comparison(
            pd.DataFrame(rows),
            ["feature_a", "feature_b"],
            "logistic",
            folds=3,
            inner_folds=2,
        )
        self.assertEqual(set(result["calibration"]), {"raw", "isotonic"})
        self.assertTrue(np.isfinite(result["log_loss"]).all())


if __name__ == "__main__":
    unittest.main()
