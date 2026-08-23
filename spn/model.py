"""Frozen current-SPN XGBoost inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


KEY_COLUMNS = ["match_id", "seq_id", "ev_pos", "anchor_event_id"]


class SPNModel:
    """Load the public native artifact and enforce its feature contract."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        release_root = Path(__file__).resolve().parents[1]
        self.config_path = Path(config_path or release_root / "model" / "model_config.json")
        if not self.config_path.is_file():
            raise FileNotFoundError(f"missing model config: {self.config_path}")
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.model_path = Path(
            model_path
            or self.config_path.parent / self.config.get("artifact_file", "model.ubj")
        )
        if not self.model_path.is_file():
            raise FileNotFoundError(f"missing model artifact: {self.model_path}")
        expected_hash = str(self.config.get("artifact_sha256", ""))
        observed_hash = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        if not expected_hash or observed_hash != expected_hash:
            raise ValueError("model artifact SHA-256 does not match model_config.json")
        self.features = list(self.config["features"])
        self.class_names = list(self.config["class_names"])
        if self.class_names != ["fail", "neutral", "success"]:
            raise ValueError("public model class order must be fail/neutral/success")
        self.model = XGBClassifier()
        self.model.load_model(self.model_path)
        artifact_features = self.model.get_booster().feature_names
        if artifact_features is not None and list(artifact_features) != self.features:
            raise ValueError("model artifact and configured feature order differ")

    def predict(self, feature_table: pd.DataFrame) -> pd.DataFrame:
        """Return one raw three-class probability row per modelled anchor."""
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
        matrix = (
            feature_table[self.features]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
        )
        probability = np.asarray(self.model.predict_proba(matrix), dtype=float)
        if probability.shape != (len(feature_table), len(self.class_names)):
            raise RuntimeError("model returned an unexpected probability shape")
        if not np.isfinite(probability).all():
            raise RuntimeError("model returned non-finite probabilities")
        row_sum = probability.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 0.0):
            raise RuntimeError("model returned a zero probability row")
        probability = probability / row_sum
        result = feature_table[KEY_COLUMNS].copy().reset_index(drop=True)
        for index, name in enumerate(self.class_names):
            result[f"p_{name}"] = probability[:, index]
        prediction = probability.argmax(axis=1)
        result["predicted_class"] = [self.class_names[index] for index in prediction]
        result["confidence"] = probability.max(axis=1)
        return result
