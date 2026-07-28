"""Dataset-driven S5 aggregation helpers; no league or fixed-team assumptions."""
from __future__ import annotations

import pandas as pd


def attach_sequence_change(scored: pd.DataFrame) -> pd.DataFrame:
    scored = scored.sort_values(["match_id", "seq_id", "ev_pos"]).copy()
    scored["d_p_success"] = scored.groupby(["match_id", "seq_id"], sort=False)["p_success"].diff()
    return scored


def team_summary(scored: pd.DataFrame, min_sequences: int = 5) -> pd.DataFrame:
    data = attach_sequence_change(scored)
    grouped = data.groupby("team_name", dropna=True)
    summary = grouped.agg(
        n_frames=("p_success", "size"),
        n_sequences=("seq_id", "nunique"),
        mean_success_probability=("p_success", "mean"),
        press_efficiency=("d_p_success", "mean"),
        mean_pressure=("P_total", "mean"),
        mean_escape_capacity=("escape_capacity", "mean"),
        mean_boundary_pressure=("carrier_boundary_pressure", "mean"),
        mean_open_passes=("n_open_pass", "mean"),
    ).reset_index()
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
