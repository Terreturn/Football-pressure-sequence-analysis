"""Dataset-driven S5 aggregation helpers; no league or fixed-team assumptions."""
from __future__ import annotations

import pandas as pd


def attach_pressing_team(scored: pd.DataFrame) -> pd.DataFrame:
    """Attach the defending/pressing team to each possession-team anchor."""
    data = scored.copy()
    if "pressed_team" not in data:
        data["pressed_team"] = data["team_name"]
    if "press_team" in data and data["press_team"].notna().all():
        return data
    teams_by_match = data.groupby("match_id")["pressed_team"].unique()
    invalid = teams_by_match.map(len).ne(2)
    if invalid.any():
        examples = ", ".join(map(str, teams_by_match.index[invalid][:5]))
        raise ValueError(
            "S5 requires exactly two teams per match to infer the pressing team; "
            f"invalid matches include: {examples}"
        )
    opponents = teams_by_match.to_dict()
    data["press_team"] = [
        next(team for team in opponents[match_id] if team != pressed_team)
        for match_id, pressed_team in zip(data["match_id"], data["pressed_team"])
    ]
    return data


def attach_sequence_change(scored: pd.DataFrame) -> pd.DataFrame:
    scored = attach_pressing_team(scored)
    scored = scored.sort_values(["match_id", "seq_id", "ev_pos"]).copy()
    scored["d_p_success"] = scored.groupby(["match_id", "seq_id"], sort=False)["p_success"].diff()
    scored["d_p_fail"] = scored.groupby(["match_id", "seq_id"], sort=False)["p_fail"].diff()
    scored["v_t"] = scored["d_p_success"] - scored["d_p_fail"]
    scored["sequence_key"] = (
        scored["match_id"].astype(str)
        + "_"
        + scored["seq_id"].astype(str)
    )
    return scored


def team_summary(scored: pd.DataFrame, min_sequences: int = 5) -> pd.DataFrame:
    data = attach_sequence_change(scored)
    grouped = data.groupby("press_team", dropna=True)
    summary = grouped.agg(
        n_frames=("p_success", "size"),
        mean_success_probability=("p_success", "mean"),
        mean_pressure=("P_total", "mean"),
        mean_escape_capacity=("escape_capacity", "mean"),
        mean_boundary_pressure=("carrier_boundary_pressure", "mean"),
        mean_open_passes=("n_open_pass", "mean"),
    )
    sequences = data.groupby(
        ["press_team", "sequence_key"],
        as_index=False,
    ).agg(
        sequence_value=("v_t", "sum"),
        final_outcome=("outcome_tag", "last"),
    )
    sequence_summary = sequences.groupby("press_team").agg(
        n_sequences=("sequence_key", "size"),
        press_efficiency=("sequence_value", "mean"),
        regain_rate=("final_outcome", lambda values: (values == "success").mean()),
    )
    summary = summary.join(sequence_summary).reset_index()
    summary = summary.rename(columns={"press_team": "team_name"})
    summary["eligible"] = summary["n_sequences"] >= min_sequences
    return summary.sort_values(["eligible", "press_efficiency"], ascending=[False, False])


def style_fingerprint(summary: pd.DataFrame) -> pd.DataFrame:
    columns = ["mean_pressure", "mean_escape_capacity", "mean_boundary_pressure", "mean_open_passes", "press_efficiency"]
    value = summary[["team_name", *columns]].copy()
    for column in columns:
        sd = value[column].std(ddof=0)
        value[f"z_{column}"] = (value[column] - value[column].mean()) / sd if sd else 0.0
    return value


def team_characteristics(summary: pd.DataFrame) -> pd.DataFrame:
    fields = {
        "mean_pressure": "carrier pressure",
        "mean_boundary_pressure": "boundary trapping",
        "mean_open_passes": "available outlets",
        "press_efficiency": "press efficiency",
    }
    rows = []
    for _, row in summary.iterrows():
        for field, label in fields.items():
            low, high = summary[field].quantile(.25), summary[field].quantile(.75)
            level = "high" if row[field] >= high else "low" if row[field] <= low else "balanced"
            rows.append({"team_name": row["team_name"], "dimension": label, "level": level, "value": row[field]})
    return pd.DataFrame(rows)
