"""Frozen SPN XGBoost inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import Booster, DMatrix, XGBClassifier

from .feature_engineering import INPUTS
from ._core.probability_mapping import apply_mapping


KEY_COLUMNS = ["match_id", "seq_id", "ev_pos", "anchor_event_id"]


MODEL_DIR = Path(__file__).resolve().parents[1] / "model"


def verified_path(path: Path, digest: str) -> Path:
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"SPN model file checksum mismatch: {path.name}")
    return path


def load_booster(path):
    """Load a saved tree model and check the names and order of its 26 inputs."""
    model = Booster(params={"nthread": 2}, model_file=str(path))
    if model.feature_names != INPUTS:
        raise ValueError(f"Unexpected feature list in {path}")
    return model


def probabilities(model, table):
    """Score the 26 named columns. Return native probabilities in training order.

    A sequence model returns an event-by-three array (failure, neither, success).
    A regain model returns one probability per event. NaN is XGBoost's native
    encoding for missing geometry or context.
    """
    features = table.loc[:, INPUTS].astype(float)
    if features.columns.duplicated().any() or np.isinf(features.to_numpy()).any():
        raise ValueError("Duplicate feature columns or infinite values")
    return model.predict(DMatrix(features, nthread=2))


class SPNModel:
    """Load the public native artifact and enforce its feature contract."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        self.config_path = Path(config_path or MODEL_DIR / "model_config.json")
        if not self.config_path.is_file():
            raise FileNotFoundError(f"missing model config: {self.config_path}")
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.model_path = Path(
            model_path or self.config_path.parent / self.config.get("artifact_file", "model.ubj")
        )
        if not self.model_path.is_file():
            raise FileNotFoundError(f"missing model artifact: {self.model_path}")
        verified_path(self.model_path, self.config["artifact_sha256"])
        self.features = list(self.config["features"])
        self.class_names = list(self.config["class_names"])
        if self.features != INPUTS or self.class_names != ["fail", "neutral", "success"]:
            raise ValueError("Unexpected SPN feature or class order")
        mapping_file = self.config["temperature_file"]
        mapping_path = verified_path(
            self.config_path.parent / mapping_file, self.config["artifact_checksums"][mapping_file]
        )
        self.mapping = json.loads(mapping_path.read_text())
        self.model = XGBClassifier(n_jobs=2)
        self.model.load_model(self.model_path)
        if self.model.get_booster().feature_names != self.features:
            raise ValueError("Model artifact and configured feature order differ")

    def predict(self, feature_table: pd.DataFrame) -> pd.DataFrame:
        """Return one temperature-calibrated three-class probability row per modelled anchor."""
        missing_keys = sorted(set(KEY_COLUMNS) - set(feature_table.columns))
        if missing_keys:
            raise ValueError(f"feature table is missing anchor keys: {missing_keys}")
        missing_features = sorted(set(self.features) - set(feature_table.columns))
        if missing_features:
            raise ValueError(f"feature table is missing model features: {missing_features}")
        if feature_table.empty:
            return pd.DataFrame(
                columns=KEY_COLUMNS
                + [f"p_{name}" for name in self.class_names]
                + ["predicted_class", "confidence"]
            )
        probability = apply_mapping(
            probabilities(self.model.get_booster(), feature_table), self.mapping
        )
        result = feature_table[KEY_COLUMNS].copy().reset_index(drop=True)
        for index, name in enumerate(self.class_names):
            result[f"p_{name}"] = probability[:, index]
        prediction = probability.argmax(axis=1)
        result["predicted_class"] = [self.class_names[index] for index in prediction]
        result["confidence"] = probability.max(axis=1)
        return result
