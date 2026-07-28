"""Grouped V4 model selection, calibration, scoring, and robustness helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


CLASS_NAMES = ["fail", "neutral", "success"]


def feature_diagnostics(table: pd.DataFrame, features: list[str]) -> dict[str, pd.DataFrame]:
    frame = table[features].copy()
    correlations = frame.corr(numeric_only=True)
    pairs = (
        correlations.where(np.triu(np.ones(correlations.shape), 1).astype(bool))
        .stack().rename("correlation").abs().sort_values(ascending=False).reset_index()
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b"})
    )
    return {
        "missing": frame.isna().mean().rename("missing_rate").sort_values(ascending=False).to_frame(),
        "variance": frame.var(numeric_only=True).rename("variance").sort_values().to_frame(),
        "correlations": correlations,
        "correlation_pairs": pairs,
    }


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    one_hot = label_binarize(y, classes=np.arange(3))
    success = one_hot[:, 2]
    return {
        "log_loss": float(log_loss(y, probability, labels=np.arange(3))),
        "macro_auc": float(roc_auc_score(one_hot, probability, average="macro")),
        "success_auc": float(roc_auc_score(success, probability[:, 2])),
    }


def make_model(name: str, balanced: bool = False, random_state: int = 42):
    if name == "logistic":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=4000, multi_class="multinomial", random_state=random_state)),
        ])
    return XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss", tree_method="hist",
        n_estimators=500, max_depth=6, learning_rate=.035, subsample=.8, colsample_bytree=.8,
        min_child_weight=3, reg_lambda=3, n_jobs=-1, random_state=random_state,
    )


def grouped_model_comparison(table: pd.DataFrame, features: list[str], folds: int = 5, random_state: int = 42) -> tuple[pd.DataFrame, dict[str, dict]]:
    clean = table[table["outcome_tag"].isin(CLASS_NAMES)].copy()
    y = clean["outcome_tag"].map({name: index for index, name in enumerate(CLASS_NAMES)}).to_numpy()
    groups, X = clean["match_id"].astype(str).to_numpy(), clean[features]
    folds = min(folds, len(np.unique(groups)))
    if folds < 2:
        raise ValueError("At least two matches are required for grouped model validation")
    results, fitted = [], {}
    for name, balanced in [("logistic", False), ("xgb_unweighted", False), ("xgb_balanced", True)]:
        oof = np.zeros((len(clean), 3))
        for train, valid in GroupKFold(n_splits=folds).split(X, y, groups):
            model = make_model("logistic" if name == "logistic" else "xgb", balanced, random_state)
            if balanced and name != "logistic":
                model.fit(X.iloc[train], y[train], sample_weight=compute_sample_weight("balanced", y[train]))
            else:
                model.fit(X.iloc[train], y[train])
            oof[valid] = model.predict_proba(X.iloc[valid])
        results.append({"model": name, **_metrics(y, oof)})
        final = make_model("logistic" if name == "logistic" else "xgb", balanced, random_state)
        if balanced and name != "logistic":
            final.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
        else:
            final.fit(X, y)
        fitted[name] = {"model": final, "oof_probability": oof}
    return pd.DataFrame(results).sort_values("log_loss"), fitted


def fit_isotonic(probability: np.ndarray, y: np.ndarray) -> list[IsotonicRegression]:
    return [IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(probability[:, index], y == index) for index in range(3)]


def apply_isotonic(probability: np.ndarray, calibrators: list[IsotonicRegression]) -> np.ndarray:
    value = np.column_stack([calibrator.predict(probability[:, index]) for index, calibrator in enumerate(calibrators)])
    value = np.clip(value, 1e-8, 1.0)
    return value / value.sum(axis=1, keepdims=True)


def save_bundle(path, model, features: list[str], calibration: str = "raw", calibrators=None) -> None:
    joblib.dump({"model": model, "features": features, "class_names": CLASS_NAMES, "calibration": calibration, "calibrators": calibrators}, path)


def score_bundle(bundle_path, table: pd.DataFrame) -> pd.DataFrame:
    bundle = joblib.load(bundle_path)
    probability = bundle["model"].predict_proba(table[bundle["features"]])
    if bundle.get("calibration") == "isotonic":
        probability = apply_isotonic(probability, bundle["calibrators"])
    scored = table.copy()
    for index, name in enumerate(CLASS_NAMES):
        scored[f"p_{name}"] = probability[:, index]
    return scored


def run_robustness(
    baseline: pd.DataFrame,
    rebuild: Callable[[dict], pd.DataFrame],
    scenarios: dict[str, dict],
    features: list[str],
) -> pd.DataFrame:
    """Rebuild feature tables under user-selected parameter scenarios."""
    rows = []
    for name, parameters in scenarios.items():
        candidate = rebuild(parameters)
        shared = baseline.merge(candidate, on=["match_id", "seq_id", "ev_pos"], suffixes=("_base", "_candidate"))
        for feature in features:
            left, right = shared[f"{feature}_base"], shared[f"{feature}_candidate"]
            rows.append({"scenario": name, "feature": feature, "spearman": left.corr(right, method="spearman"), "mean_abs_change": float((left - right).abs().mean())})
    return pd.DataFrame(rows)
