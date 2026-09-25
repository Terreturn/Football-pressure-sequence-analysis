"""SPN descriptive analysis shared by the public plot entry point.

Sequence-value and team-analysis definitions are implemented here for the plots.
Raw probabilities come from the included model.
No training, calibration, external season tables or private paths are needed.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .model import KEY_COLUMNS, SPNModel

OUTCOMES = ("fail", "neutral", "success")
SEQUENCE_KEYS = ["match_id", "seq_id"]
ANCHOR_KEYS = [*SEQUENCE_KEYS, "ev_pos"]
PROBABILITY_COLUMNS = ["p_fail", "p_neutral", "p_success"]


class InsufficientDataError(ValueError):
    """Valid input has insufficient observations for a requested analysis."""


FEATURE_LABELS = {
    "carrier_x_norm": "Carrier x position",
    "P_total": "Carrier total pressure",
    "effective_pressers": "Effective pressers",
    "mean_receiver_pressure": "Receiver pressure",
    "press_target_entropy": "Press-target entropy",
    "weighted_angular_dispersion": "Angular dispersion",
    "n_active_carrier_boundaries": "Active boundary constraints",
    "best_forward_pass_w": "Best forward-pass quality",
    "n_open_pass": "Open passing options",
    "ball_in_dist": "Incoming-ball distance",
    "ball_in_angle_cos": "Incoming-angle cosine",
    "vor_carrier_area_share": "Carrier Voronoi-area share",
    "vor_forward_receiver_area_share": "Forward-receiver Voronoi-area share",
    "pressure_mean_focus_entropy": "Defender focus entropy",
}


def prepare_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate the public-output contract and retain complete labelled sequences."""
    required = {
        *ANCHOR_KEYS,
        "press_team",
        "observed_outcome",
        *PROBABILITY_COLUMNS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"prediction table is missing columns: {missing}")

    if frame.empty:
        raise InsufficientDataError("no modelled anchors are available")
    if frame[[*ANCHOR_KEYS, "press_team", "observed_outcome"]].isna().any().any():
        raise ValueError("prediction identifiers, teams and outcomes must not be missing")

    table = frame.copy()
    table["match_id"] = table["match_id"].astype(str)
    for column in ("seq_id", "ev_pos"):
        values = pd.to_numeric(table[column], errors="raise")
        if not np.isfinite(values).all() or not values.eq(np.floor(values)).all():
            raise ValueError(f"{column} must contain integers")
        table[column] = values.astype(int)
    for column in PROBABILITY_COLUMNS:
        table[column] = pd.to_numeric(table[column], errors="raise").astype(float)

    probability = table[PROBABILITY_COLUMNS].to_numpy(float)
    if not np.isfinite(probability).all():
        raise ValueError("probabilities contain non-finite values")
    if (probability < -1e-12).any() or (probability > 1.0 + 1e-12).any():
        raise ValueError("probabilities fall outside [0, 1]")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("probability rows do not sum to one")
    if table.duplicated(ANCHOR_KEYS).any():
        duplicate = table.loc[table.duplicated(ANCHOR_KEYS, keep=False), ANCHOR_KEYS].head()
        raise ValueError(f"duplicate anchor keys found:\n{duplicate.to_string(index=False)}")

    table = table.sort_values(ANCHOR_KEYS, kind="stable").reset_index(drop=True)
    grouped = table.groupby(SEQUENCE_KEYS, sort=False, observed=True)
    outcome_count = grouped["observed_outcome"].nunique(dropna=False)
    team_count = grouped["press_team"].nunique(dropna=False)
    if outcome_count.ne(1).any():
        raise ValueError("observed_outcome must be constant within each sequence")
    if team_count.ne(1).any():
        raise ValueError("press_team must be constant within each sequence")

    layout = grouped["ev_pos"].agg(["min", "max", "size"])
    if layout["min"].ne(0).any() or layout["max"].ne(layout["size"] - 1).any():
        raise ValueError("every exported sequence must contain contiguous ev_pos values from zero")

    outcomes = grouped["observed_outcome"].first()
    eligible_index = outcomes.loc[outcomes.isin(OUTCOMES)].index
    eligible_keys = pd.MultiIndex.from_frame(table[SEQUENCE_KEYS]).isin(eligible_index)
    eligible = table.loc[eligible_keys].copy().reset_index(drop=True)
    if eligible.empty:
        raise InsufficientDataError("no complete fail/neutral/success sequences are available")

    if "evaluation_eligible" in eligible.columns:
        flag = eligible["evaluation_eligible"]
        if flag.dtype != bool:
            flag = flag.astype(str).str.lower().map({"true": True, "false": False})
        if flag.isna().any() or not flag.all():
            raise ValueError("labelled sequences disagree with evaluation_eligible")

    audit = {
        "input_anchors": int(len(table)),
        "input_sequences": int(len(outcomes)),
        "eligible_anchors": int(len(eligible)),
        "eligible_sequences": int(len(eligible_index)),
        "excluded_censored_sequences": int((~outcomes.isin(OUTCOMES)).sum()),
        "matches": int(eligible["match_id"].nunique()),
        "teams": int(eligible["press_team"].nunique()),
    }
    return eligible, audit


def build_step_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Add state, forward-transition, and unweighted terminal VAEP components."""
    table = frame.sort_values(ANCHOR_KEYS, kind="stable").copy().reset_index(drop=True)
    grouped = table.groupby(SEQUENCE_KEYS, sort=False, observed=True)
    table["state_value"] = table["p_success"] - table["p_fail"]
    table["next_state_value"] = grouped["state_value"].shift(-1)
    table["transition_vaep"] = table["next_state_value"] - table["state_value"]
    table["is_sequence_end"] = grouped.cumcount(ascending=False).eq(0)
    table["terminal_vaep"] = np.nan

    last = table["is_sequence_end"]
    table.loc[last & table["observed_outcome"].eq("success"), "terminal_vaep"] = table.loc[
        last & table["observed_outcome"].eq("success"), "p_success"
    ]
    table.loc[last & table["observed_outcome"].eq("neutral"), "terminal_vaep"] = 0.0
    table.loc[last & table["observed_outcome"].eq("fail"), "terminal_vaep"] = -table.loc[
        last & table["observed_outcome"].eq("fail"), "p_fail"
    ]
    if table.loc[last, "terminal_vaep"].isna().any():
        raise ValueError("a labelled sequence has no terminal VAEP value")
    return table


def build_sequence_values(step_values: pd.DataFrame) -> pd.DataFrame:
    """Collapse anchor values to one auditable row per pressure sequence."""
    table = step_values.copy()
    table["transition_for_sum"] = table["transition_vaep"].fillna(0.0)
    table["terminal_for_sum"] = table["terminal_vaep"].fillna(0.0)
    aggregation: dict[str, tuple[str, str]] = {
        "press_team": ("press_team", "first"),
        "observed_outcome": ("observed_outcome", "first"),
        "n_anchors": ("ev_pos", "size"),
        "process_value": ("transition_for_sum", "sum"),
        "terminal_value": ("terminal_for_sum", "sum"),
        "start_state_value": ("state_value", "first"),
        "end_state_value": ("state_value", "last"),
    }
    if "sequence_end_state" in table.columns:
        aggregation["sequence_end_state"] = ("sequence_end_state", "last")
    elif "state" in table.columns:
        aggregation["sequence_end_state"] = ("state", "last")

    sequence = (
        table.groupby(SEQUENCE_KEYS, as_index=False, sort=True, observed=True)
        .agg(**aggregation)
        .sort_values(SEQUENCE_KEYS, kind="stable")
        .reset_index(drop=True)
    )
    telescope_error = (
        sequence["process_value"]
        - (sequence["end_state_value"] - sequence["start_state_value"])
    ).abs()
    if float(telescope_error.max()) > 1e-10:
        raise ValueError("within-sequence transition values fail the telescoping identity")
    return sequence


def team_efficiency_tables(
    sequences: pd.DataFrame,
    terminal_weights: Iterable[float],
    *,
    baseline_weight: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return league-centred team rankings and terminal-weight diagnostics.

    ``absolute_efficiency`` retains the raw composite value for audit.  The
    primary ``relative_efficiency`` subtracts the sequence-weighted league mean
    at the same terminal weight, so zero represents league-average performance.
    """
    weights = sorted({float(weight) for weight in terminal_weights})
    if not weights:
        raise ValueError("at least one terminal weight is required")
    if any(weight < 0.0 or weight > 1.0 for weight in weights):
        raise ValueError("terminal weights must be within [0, 1]")
    if baseline_weight not in weights:
        raise ValueError("baseline_weight must be included in terminal_weights")

    base = sequences.groupby("press_team", observed=True).agg(
        n_sequences=("seq_id", "size"),
        n_matches=("match_id", "nunique"),
        process_value_per_seq=("process_value", "mean"),
        terminal_value_per_seq=("terminal_value", "mean"),
        successful_sequence_rate=("observed_outcome", lambda value: value.eq("success").mean()),
        neutral_sequence_rate=("observed_outcome", lambda value: value.eq("neutral").mean()),
        failed_sequence_rate=("observed_outcome", lambda value: value.eq("fail").mean()),
    )

    tables: list[pd.DataFrame] = []
    for weight in weights:
        current = base.copy()
        current["terminal_weight"] = weight
        current["weighted_terminal_value_per_seq"] = weight * current["terminal_value_per_seq"]
        current["absolute_efficiency"] = (
            current["process_value_per_seq"] + current["weighted_terminal_value_per_seq"]
        )
        league_efficiency = float(
            np.average(
                current["absolute_efficiency"],
                weights=current["n_sequences"].to_numpy(float),
            )
        )
        current["league_absolute_efficiency"] = league_efficiency
        current["relative_efficiency"] = current["absolute_efficiency"] - league_efficiency
        current["relative_efficiency_per_100_sequences"] = 100.0 * current["relative_efficiency"]
        current["rank"] = current["relative_efficiency"].rank(
            ascending=False, method="min"
        ).astype(int)
        tables.append(current.reset_index())
    team = pd.concat(tables, ignore_index=True).sort_values(
        ["terminal_weight", "rank", "press_team"], kind="stable"
    )

    baseline = team.loc[team["terminal_weight"].eq(baseline_weight)].set_index("press_team")
    baseline_rank = baseline["rank"]
    baseline_efficiency = baseline["relative_efficiency"]
    baseline_top_five = set(baseline.nsmallest(min(5, len(baseline)), "rank").index)
    summary_rows: list[dict[str, float | int]] = []
    for weight in weights:
        current = team.loc[team["terminal_weight"].eq(weight)].set_index("press_team")
        rank_change = current["rank"] - baseline_rank
        top_five = set(current.nsmallest(min(5, len(current)), "rank").index)
        rho = (
            spearmanr(current["relative_efficiency"], baseline_efficiency.reindex(current.index)).statistic
            if len(current) >= 2 and current["relative_efficiency"].nunique() > 1
            and baseline_efficiency.nunique() > 1 else np.nan
        )
        sequence_weights = current["n_sequences"].to_numpy(float)
        summary_rows.append(
            {
                "terminal_weight": weight,
                "league_absolute_efficiency": float(
                    np.average(current["absolute_efficiency"], weights=sequence_weights)
                ),
                "league_relative_efficiency": float(
                    np.average(current["relative_efficiency"], weights=sequence_weights)
                ),
                "mean_process_component": float(
                    np.average(current["process_value_per_seq"], weights=sequence_weights)
                ),
                "mean_weighted_terminal_component": float(
                    np.average(current["weighted_terminal_value_per_seq"], weights=sequence_weights)
                ),
                "above_average_teams": int(current["relative_efficiency"].gt(0.0).sum()),
                "spearman_vs_baseline": float(rho),
                "mean_absolute_rank_change": float(rank_change.abs().mean()),
                "max_absolute_rank_change": int(rank_change.abs().max()),
                "teams_moved_at_least_2": int(rank_change.abs().ge(2).sum()),
                "top_five_overlap": int(len(top_five.intersection(baseline_top_five))),
            }
        )
    return team.reset_index(drop=True), pd.DataFrame(summary_rows)


def bootstrap_team_efficiency(
    sequences: pd.DataFrame,
    *,
    terminal_weight: float = 0.5,
    iterations: int = 2_000,
    random_state: int = 20260824,
) -> pd.DataFrame:
    """Match-bootstrap absolute and league-relative 95% efficiency intervals.

    Matches are resampled within each pressing team. At every iteration, the
    sequence-weighted league mean is recomputed and subtracted from all team
    draws.
    """
    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    valued = sequences.copy()
    valued["sequence_efficiency"] = (
        valued["process_value"] + terminal_weight * valued["terminal_value"]
    )
    team_match = valued.groupby(
        ["press_team", "match_id"], as_index=False, observed=True
    ).agg(
        sum_efficiency=("sequence_efficiency", "sum"),
        n_sequences=("seq_id", "size"),
    )

    rng = np.random.default_rng(random_state)
    draws_by_team: dict[str, np.ndarray] = {}
    counts_by_team: dict[str, np.ndarray] = {}
    point_by_team: dict[str, float] = {}
    match_count_by_team: dict[str, int] = {}
    for team_name, matches in team_match.groupby("press_team", sort=True, observed=True):
        matches = matches.reset_index(drop=True)
        sample_index = rng.integers(0, len(matches), size=(iterations, len(matches)))
        efficiency_sum = matches["sum_efficiency"].to_numpy(float)[sample_index].sum(axis=1)
        sequence_sum = matches["n_sequences"].to_numpy(float)[sample_index].sum(axis=1)
        draws = efficiency_sum / sequence_sum
        point = float(matches["sum_efficiency"].sum() / matches["n_sequences"].sum())
        team_key = str(team_name)
        draws_by_team[team_key] = draws
        counts_by_team[team_key] = sequence_sum
        point_by_team[team_key] = point
        match_count_by_team[team_key] = int(len(matches))

    team_names = sorted(draws_by_team)
    draw_numerator = np.zeros(iterations, dtype=float)
    draw_denominator = np.zeros(iterations, dtype=float)
    for team_name in team_names:
        draw_numerator += draws_by_team[team_name] * counts_by_team[team_name]
        draw_denominator += counts_by_team[team_name]
    league_draws = draw_numerator / draw_denominator
    league_point = float(
        valued["sequence_efficiency"].sum() / len(valued)
    )

    rows: list[dict[str, float | int | str]] = []
    for team_name in team_names:
        absolute_draws = draws_by_team[team_name]
        relative_draws = absolute_draws - league_draws
        absolute_point = point_by_team[team_name]
        relative_point = absolute_point - league_point
        rows.append(
            {
                "press_team": team_name,
                "bootstrap_matches": match_count_by_team[team_name],
                "bootstrap_iterations": int(iterations),
                "bootstrap_league_absolute_efficiency": league_point,
                "bootstrap_absolute_efficiency": absolute_point,
                "absolute_efficiency_ci_low": float(np.quantile(absolute_draws, 0.025)),
                "absolute_efficiency_ci_high": float(np.quantile(absolute_draws, 0.975)),
                "bootstrap_relative_efficiency": relative_point,
                "relative_efficiency_ci_low": float(np.quantile(relative_draws, 0.025)),
                "relative_efficiency_ci_high": float(np.quantile(relative_draws, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def gain_table(model: Any, features: list[str]) -> pd.DataFrame:
    booster = model.get_booster()
    mean_gain = booster.get_score(importance_type="gain")
    split_count = booster.get_score(importance_type="weight")
    total_gain = booster.get_score(importance_type="total_gain")

    table = pd.DataFrame(
        {
            "feature": features,
            "mean_gain": [float(mean_gain.get(feature, 0.0)) for feature in features],
            "total_gain": [float(total_gain.get(feature, 0.0)) for feature in features],
            "split_count": [int(split_count.get(feature, 0)) for feature in features],
        }
    )
    gain_sum = float(table["mean_gain"].sum())
    if gain_sum <= 0:
        raise ValueError("the model has no positive gain importance")
    table["relative_gain"] = table["mean_gain"] / gain_sum
    table = table.sort_values(
        ["relative_gain", "feature"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    table.insert(0, "rank", table.index + 1)
    return table


def step_label(dps: float, dpf: float, transition_vt: float, threshold: float) -> str:
    success = 1 if dps > threshold else (-1 if dps < -threshold else 0)
    fail = 1 if dpf > threshold else (-1 if dpf < -threshold else 0)
    if success == 0 and fail == 0:
        return "Negligible"
    if success == 1 and fail <= 0:
        return "Effective"
    if success <= 0 and fail == 1:
        return "Beaten"
    if success == 1 and fail == 1:
        return "Risky-fav" if transition_vt > 0 else "Risky-adv"
    return "De-escalation"


def driver_table(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    level_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    valued = build_step_values(predictions).merge(
        features[[*KEY_COLUMNS, *level_features]],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    feature_missing_counts = {
        feature: int(count)
        for feature, count in valued[level_features].isna().sum().items()
        if count > 0
    }

    valued = valued.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)
    groups = valued.groupby(["match_id", "seq_id"], sort=False)
    valued["dps"] = groups["p_success"].shift(-1) - valued["p_success"]
    valued["dpf"] = groups["p_fail"].shift(-1) - valued["p_fail"]

    change_columns = []
    for feature in level_features:
        column = f"change__{feature}"
        valued[column] = groups[feature].shift(-1) - valued[feature]
        change_columns.append(column)

    transition = valued["transition_vaep"].notna()
    if not transition.any():
        raise InsufficientDataError("v_t drivers require sequences with at least two anchors")
    threshold = float(
        np.percentile(
            np.r_[
                valued.loc[transition, "dps"].abs().to_numpy(float),
                valued.loc[transition, "dpf"].abs().to_numpy(float),
            ],
            25,
        )
    )
    valued["step_label"] = None
    valued.loc[transition, "step_label"] = [
        step_label(dps, dpf, vt, threshold)
        for dps, dpf, vt in valued.loc[
            transition, ["dps", "dpf", "transition_vaep"]
        ].itertuples(index=False, name=None)
    ]

    complete = valued.dropna(subset=["transition_vaep", *change_columns]).copy()
    transition_candidates = int(transition.sum())
    if complete.empty:
        raise InsufficientDataError("no complete adjacent transitions remain for driver analysis")
    rows = []
    for feature, column in zip(level_features, change_columns):
        effective_mean = complete.loc[
            complete["step_label"].eq("Effective"), column
        ].mean()
        beaten_mean = complete.loc[complete["step_label"].eq("Beaten"), column].mean()
        rows.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "pearson_r_vt": (complete[column].corr(complete["transition_vaep"])
                    if complete[column].nunique() > 1 and complete["transition_vaep"].nunique() > 1 else np.nan),
                "effective_mean": effective_mean,
                "beaten_mean": beaten_mean,
                "effective_minus_beaten": effective_mean - beaten_mean,
            }
        )
    drivers = pd.DataFrame(rows)

    matrix = complete[change_columns]
    standard_deviation = matrix.std()
    usable = standard_deviation.index[standard_deviation.gt(0) & np.isfinite(standard_deviation)]
    coefficients = np.full(len(level_features), np.nan)
    if len(complete) < 2 or not len(usable) or complete["transition_vaep"].nunique() < 2:
        raise InsufficientDataError("v_t drivers require varying features and transition values")
    from sklearn.linear_model import LinearRegression

    standardized = (matrix[usable] - matrix[usable].mean()) / standard_deviation[usable]
    regression = LinearRegression().fit(standardized, complete["transition_vaep"])
    for column, coefficient in zip(usable, regression.coef_):
        coefficients[change_columns.index(column)] = coefficient
    joint_r2 = float(regression.score(standardized, complete["transition_vaep"]))
    drivers["standardized_x_reg_beta"] = coefficients
    drivers["n_transitions"] = len(complete)
    drivers["direction"] = np.where(
        drivers["pearson_r_vt"].gt(0), "positive", "negative"
    )
    drivers = drivers.sort_values(
        ["pearson_r_vt", "feature"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    drivers.insert(0, "rank", drivers.index + 1)

    change_table = complete[
        [
            *KEY_COLUMNS,
            "press_team",
            "p_fail",
            "p_success",
            "dps",
            "dpf",
            "transition_vaep",
            "step_label",
            *change_columns,
        ]
    ].copy()
    audit = {
        "eligible_anchors": len(valued),
        "transition_candidates": transition_candidates,
        "complete_transitions": len(complete),
        "excluded_incomplete_feature_transitions": transition_candidates - len(complete),
        "complete_transition_share": len(complete) / transition_candidates,
        "level_feature_missing_anchor_counts": feature_missing_counts,
        "sequences_with_transitions": int(
            complete[["match_id", "seq_id"]].drop_duplicates().shape[0]
        ),
        "matches_with_transitions": int(complete["match_id"].nunique()),
        "step_threshold": threshold,
        "joint_r2": joint_r2,
    }
    return drivers, change_table, audit


def build_team_table(
    steps: pd.DataFrame, teams: pd.DataFrame, features: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    anchors = steps[[*KEY_COLUMNS, "press_team"]].merge(
        features[[*KEY_COLUMNS, "P_total"]],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if anchors["_merge"].ne("both").any():
        raise ValueError("feature cache does not cover every valued anchor")
    anchors = anchors.drop(columns="_merge")
    if anchors["press_team"].isna().any() or anchors["P_total"].isna().any():
        raise ValueError("press_team and P_total must be complete")

    intensity = (
        anchors.groupby("press_team", observed=True)
        .agg(
            n_matches_from_anchors=("match_id", "nunique"),
            n_anchors=("P_total", "size"),
            mean_P_total=("P_total", "mean"),
            total_P_total=("P_total", "sum"),
        )
        .reset_index()
    )
    intensity["anchors_per_match"] = (
        intensity["n_anchors"] / intensity["n_matches_from_anchors"]
    )
    intensity["carrier_pressure_intensity"] = (
        intensity["anchors_per_match"] * intensity["mean_P_total"]
    )
    direct_intensity = intensity["total_P_total"] / intensity["n_matches_from_anchors"]
    if not np.allclose(
        intensity["carrier_pressure_intensity"], direct_intensity, rtol=0, atol=1e-12
    ):
        raise AssertionError("intensity identity failed")

    columns = [
        "press_team",
        "n_sequences",
        "n_matches",
        "successful_sequence_rate",
        "process_value_per_seq",
        "terminal_value_per_seq",
        "terminal_weight",
        "relative_efficiency_per_100_sequences",
        "relative_efficiency_ci_low_per_100",
        "relative_efficiency_ci_high_per_100",
    ]
    table = teams[columns].merge(
        intensity, on="press_team", how="outer", validate="one_to_one", indicator=True
    )
    if table["_merge"].ne("both").any():
        unmatched = table.loc[table["_merge"].ne("both"), ["press_team", "_merge"]]
        raise ValueError(f"team coverage mismatch:\n{unmatched.to_string(index=False)}")
    table = table.drop(columns="_merge")
    if not (table["n_matches"].astype(int) == table["n_matches_from_anchors"]).all():
        raise ValueError("team match counts disagree between efficiency and anchor data")
    if table["terminal_weight"].nunique() != 1:
        raise ValueError("plot requires a single terminal weight")

    size_min, size_max = 150.0, 520.0
    count_min = float(table["n_sequences"].min())
    count_max = float(table["n_sequences"].max())
    if count_max == count_min:
        table["marker_size"] = (size_min + size_max) / 2
    else:
        scaled = (table["n_sequences"] - count_min) / (count_max - count_min)
        table["marker_size"] = size_min + (size_max - size_min) * scaled

    table["intensity_rank"] = table["carrier_pressure_intensity"].rank(
        ascending=False, method="min"
    ).astype(int)
    table["efficiency_rank"] = table[
        "relative_efficiency_per_100_sequences"
    ].rank(ascending=False, method="min").astype(int)
    table = table.sort_values(
        ["relative_efficiency_per_100_sequences", "press_team"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)

    x = table["carrier_pressure_intensity"]
    y = table["relative_efficiency_per_100_sequences"]
    audit: dict[str, object] = {
        "anchor_rows": int(len(anchors)),
        "matches": int(anchors["match_id"].nunique()),
        "teams": int(len(table)),
        "team_matches_min": int(table["n_matches"].min()),
        "team_matches_max": int(table["n_matches"].max()),
        "terminal_weight": float(table["terminal_weight"].iloc[0]),
        "league_weighted_relative_efficiency_per_100": float(
            np.average(y, weights=table["n_sequences"])
        ),
        "intensity_median": float(x.median()),
        "pearson_r": float(x.corr(y, method="pearson")) if len(x) > 1 and x.nunique() > 1 and y.nunique() > 1 else np.nan,
        "spearman_rho": float(x.corr(y, method="spearman")) if len(x) > 1 and x.nunique() > 1 and y.nunique() > 1 else np.nan,
    }
    return table, audit


def _csv_paths(source: str | Path, filename: str) -> list[Path]:
    source = Path(source)
    if source.is_file():
        if source.suffix.lower() != ".csv":
            raise ValueError("public plot inputs must be CSV files or public output directories")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    direct = source / filename
    paths = [direct] if direct.is_file() else sorted(source.rglob(filename))
    paths = [p for p in paths if not any(part.startswith("._") for part in p.parts)]
    if not paths:
        raise FileNotFoundError(f"no {filename} files found under {source}")
    return paths


def _read_csv(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={"match_id": str, "anchor_event_id": str})
    for column in ("match_id", "anchor_event_id"):
        if column in table and table[column].isna().any():
            raise ValueError(f"missing {column} in {path}")
    return table


def _check_model_columns(table: pd.DataFrame, model: SPNModel, source: Path) -> None:
    for column in ("model_id", "model_version"):
        if column not in table:
            raise ValueError(f"{source} is missing {column}; use public predictions.csv outputs")
        if table[column].isna().any() or not table[column].astype(str).eq(str(model.config[column])).all():
            raise ValueError(f"{source} contains predictions from a different or unidentified model")
    if "probability_layer" in table and not table["probability_layer"].eq("raw").all():
        raise ValueError("plot analysis requires the frozen model's raw probabilities")


def load_analysis_inputs(
    source: str | Path,
    model: SPNModel,
    *,
    features_source: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load public exports, verify complete sequences, and align model features.

    Prediction exports without a feature file remain usable when one is
    supplied explicitly. No raw events, private feature caches or
    competition-specific tables are required.
    """
    paths = _csv_paths(source, "predictions.csv")
    frames = []
    counts_checked = 0
    for path in paths:
        table = _read_csv(path)
        _check_model_columns(table, model, path)
        envelope_path = path.parent / "result.json"
        if envelope_path.is_file():
            envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
            for field in ("model_id", "model_version", "probability_layer"):
                if envelope.get("model", {}).get(field) != model.config[field]:
                    raise ValueError(f"model contract differs in {envelope_path}")
        sequence_path = path.parent / "sequences.csv"
        if sequence_path.is_file() and not table.empty:
            sequences = _read_csv(sequence_path)
            required = [*SEQUENCE_KEYS, "anchor_count"]
            if not set(required).issubset(sequences):
                raise ValueError(f"{sequence_path} is missing sequence counts")
            expected = sequences[required]
            if expected.duplicated(SEQUENCE_KEYS).any():
                raise ValueError(f"duplicate sequence identifiers in {sequence_path}")
            actual = table.groupby(SEQUENCE_KEYS).size().rename("exported_count").reset_index()
            check = actual.merge(expected, on=SEQUENCE_KEYS, how="left", validate="one_to_one")
            if not check["exported_count"].eq(check["anchor_count"]).all():
                raise ValueError(f"truncated or unmatched sequence in {path}")
            counts_checked += len(check)
        if not table.empty:
            frames.append(table)
    if not frames:
        raise InsufficientDataError("all prediction exports are empty")
    raw = pd.concat(frames, ignore_index=True)
    prepared, audit = prepare_predictions(raw)
    if "anchor_event_id" not in prepared:
        raise ValueError("predictions must contain anchor_event_id for feature alignment")

    feature_paths = (
        _csv_paths(features_source, "features.csv")
        if features_source is not None else [path.parent / "features.csv" for path in paths]
    )
    feature_frames = []
    for path in feature_paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {path}; rerun run.py with --export-features, or supply --features"
            )
        table = _read_csv(path)
        required = [*KEY_COLUMNS, *model.features]
        missing = sorted(set(required) - set(table))
        if missing:
            raise ValueError(f"{path} is missing model features or keys: {missing}")
        metadata_path = path.parent / "features_metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected = {
                "model_id": model.config["model_id"],
                "model_version": model.config["model_version"],
                "probability_layer": "raw",
                "model_artifact_sha256": model.config["artifact_sha256"],
                "model_config_sha256": hashlib.sha256(model.config_path.read_bytes()).hexdigest(),
                "features": model.features,
                "rows": len(table),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise ValueError(f"feature provenance or file contents differ in {path}")
        if not table.empty:
            feature_frames.append(table[required])
    if not feature_frames:
        raise ValueError("predictions exist but all feature exports are empty")
    features = pd.concat(feature_frames, ignore_index=True)
    if features.duplicated(KEY_COLUMNS).any():
        raise ValueError("duplicate anchor identifiers in feature exports")
    selected = prepared[KEY_COLUMNS].merge(
        features, on=KEY_COLUMNS, how="left", validate="one_to_one", indicator=True,
    )
    if selected["_merge"].ne("both").any():
        raise ValueError("features do not cover every eligible prediction anchor")
    selected = selected.drop(columns="_merge")
    selected[model.features] = selected[model.features].astype(float).replace([np.inf, -np.inf], np.nan)
    repeated = model.predict(selected)
    difference = float(np.max(np.abs(
        repeated[PROBABILITY_COLUMNS].to_numpy(float) - prepared[PROBABILITY_COLUMNS].to_numpy(float)
    )))
    if difference > 1e-6:
        raise ValueError("features and raw probabilities do not reproduce the supplied model")
    return prepared, selected, {
        **audit,
        "sequence_counts_checked": counts_checked,
        "prediction_files": [str(path.resolve()) for path in paths],
        "feature_files": [str(path.resolve()) for path in feature_paths],
        "probability_parity_max_abs_difference": difference,
    }
