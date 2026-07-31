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
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


CLASS_NAMES = ["fail", "neutral", "success"]
DEFAULT_RANDOM_STATE = 202405
FINAL_V4_XGB_PARAMS = {
    "n_estimators": 778,
    "max_depth": 7,
    "learning_rate": 0.02327096042208491,
    "subsample": 0.8829039205580804,
    "colsample_bytree": 0.8719057429987207,
    "min_child_weight": 3.735532759635943,
    "gamma": 1.4416405196512205,
    "reg_lambda": 3.80847296998745,
}
FINAL_V4_LOGISTIC_C = 0.007498942093324558


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


def _normalise(probability: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=float), 1e-8, 1.0)
    return value / value.sum(axis=1, keepdims=True)


def _ece_binary(probability: np.ndarray, truth: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probability >= lower) & (
            probability <= upper if upper == 1.0 else probability < upper
        )
        if mask.any():
            result += float(mask.mean() * abs(truth[mask].mean() - probability[mask].mean()))
    return result


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = _normalise(probability)
    one_hot = label_binarize(y, classes=np.arange(3))
    success = one_hot[:, 2]
    prediction = probability.argmax(axis=1)
    return {
        "log_loss": float(log_loss(y, probability, labels=np.arange(3))),
        "brier": float(np.mean(np.sum((probability - one_hot) ** 2, axis=1))),
        "top_label_ece": _ece_binary(
            probability.max(axis=1), (prediction == y).astype(float)
        ),
        "success_ece": _ece_binary(probability[:, 2], success),
        "macro_auc": float(roc_auc_score(one_hot, probability, average="macro")),
        "success_auc": float(roc_auc_score(success, probability[:, 2])),
        "success_auprc": float(average_precision_score(success, probability[:, 2])),
    }


def make_model(
    name: str,
    balanced: bool = False,
    random_state: int = DEFAULT_RANDOM_STATE,
    xgb_params: dict | None = None,
):
    """Build a public model with final-main defaults and user overrides."""
    if name == "logistic":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=FINAL_V4_LOGISTIC_C,
                    max_iter=4000,
                    random_state=random_state,
                ),
            ),
        ])
    parameters = {**FINAL_V4_XGB_PARAMS, **(xgb_params or {})}
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=random_state,
        **parameters,
    )


def _fit_model(
    name: str,
    balanced: bool,
    X: pd.DataFrame,
    y: np.ndarray,
    random_state: int,
    xgb_params: dict | None,
):
    model = make_model(
        "logistic" if name == "logistic" else "xgb",
        balanced,
        random_state,
        xgb_params,
    )
    if balanced and name != "logistic":
        model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)
    return model


def _model_definition(name: str) -> tuple[str, bool]:
    definitions = {
        "logistic": ("logistic", False),
        "xgb_unweighted": ("xgb", False),
        "xgb_balanced": ("xgb", True),
    }
    if name not in definitions:
        raise ValueError(f"unknown model: {name}")
    return definitions[name]


def grouped_model_comparison(
    table: pd.DataFrame,
    features: list[str],
    folds: int = 5,
    random_state: int = DEFAULT_RANDOM_STATE,
    xgb_params: dict | None = None,
) -> tuple[pd.DataFrame, dict[str, dict]]:
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
            model = _fit_model(
                name, balanced, X.iloc[train], y[train], random_state, xgb_params
            )
            oof[valid] = model.predict_proba(X.iloc[valid])
        results.append({"model": name, **_metrics(y, oof)})
        final = _fit_model(name, balanced, X, y, random_state, xgb_params)
        fitted[name] = {"model": final, "oof_probability": oof}
    return pd.DataFrame(results).sort_values("log_loss"), fitted


def fit_isotonic(probability: np.ndarray, y: np.ndarray) -> list[IsotonicRegression]:
    return [IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(probability[:, index], y == index) for index in range(3)]


def apply_isotonic(probability: np.ndarray, calibrators: list[IsotonicRegression]) -> np.ndarray:
    value = np.column_stack([calibrator.predict(probability[:, index]) for index, calibrator in enumerate(calibrators)])
    return _normalise(value)


def grouped_calibration_comparison(
    table: pd.DataFrame,
    features: list[str],
    model_name: str,
    folds: int = 5,
    inner_folds: int = 3,
    random_state: int = DEFAULT_RANDOM_STATE,
    xgb_params: dict | None = None,
) -> pd.DataFrame:
    """Compare raw and isotonic probabilities without reusing validation labels.

    Each outer-match fold is scored by a model and isotonic map trained without
    that fold.  The isotonic map itself is fitted on inner-fold OOF scores from
    the outer-training matches, mirroring the final main workflow.
    """
    clean = table.loc[table["outcome_tag"].isin(CLASS_NAMES)].copy()
    y = clean["outcome_tag"].map({name: index for index, name in enumerate(CLASS_NAMES)}).to_numpy()
    groups = clean["match_id"].astype(str).to_numpy()
    X = clean[features].reset_index(drop=True)
    outer_folds = min(folds, len(np.unique(groups)))
    if outer_folds < 2:
        raise ValueError("At least two matches are required for grouped calibration validation")
    base_name, balanced = _model_definition(model_name)
    raw_oof = np.empty((len(clean), len(CLASS_NAMES)))
    isotonic_oof = np.empty_like(raw_oof)
    for train, valid in GroupKFold(n_splits=outer_folds).split(X, y, groups):
        X_train, y_train, groups_train = X.iloc[train], y[train], groups[train]
        available_inner = len(np.unique(groups_train))
        current_inner_folds = min(inner_folds, available_inner)
        if current_inner_folds < 2:
            raise ValueError("Each outer calibration fold needs at least two training matches")
        inner_probability = np.empty((len(train), len(CLASS_NAMES)))
        for inner_train, inner_valid in GroupKFold(n_splits=current_inner_folds).split(
            X_train, y_train, groups_train
        ):
            model = _fit_model(
                base_name,
                balanced,
                X_train.iloc[inner_train],
                y_train[inner_train],
                random_state,
                xgb_params,
            )
            inner_probability[inner_valid] = model.predict_proba(X_train.iloc[inner_valid])
        calibrators = fit_isotonic(inner_probability, y_train)
        model = _fit_model(
            base_name, balanced, X_train, y_train, random_state, xgb_params
        )
        raw_probability = model.predict_proba(X.iloc[valid])
        raw_oof[valid] = raw_probability
        isotonic_oof[valid] = apply_isotonic(raw_probability, calibrators)
    return pd.DataFrame(
        [
            {"calibration": "raw", **_metrics(y, raw_oof)},
            {"calibration": "isotonic", **_metrics(y, isotonic_oof)},
        ]
    ).sort_values("log_loss", ignore_index=True)


def save_bundle(
    path,
    model,
    features: list[str],
    calibration: str = "raw",
    calibrators=None,
    calibration_metadata: dict | None = None,
) -> None:
    if calibration not in {"raw", "isotonic"}:
        raise ValueError("calibration must be 'raw' or 'isotonic'")
    if calibration == "isotonic" and not calibrators:
        raise ValueError("isotonic bundles require fitted calibrators")
    joblib.dump(
        {
            "model": model,
            "features": features,
            "class_names": CLASS_NAMES,
            "calibration": calibration,
            "calibrators": calibrators,
            "calibration_metadata": calibration_metadata or {"fit_scope": "raw"},
        },
        path,
    )


def score_bundle(bundle_path, table: pd.DataFrame, *, apply_calibration: bool = True) -> pd.DataFrame:
    bundle = joblib.load(bundle_path)
    probability = bundle["model"].predict_proba(table[bundle["features"]])
    if apply_calibration and bundle.get("calibration") == "isotonic":
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
