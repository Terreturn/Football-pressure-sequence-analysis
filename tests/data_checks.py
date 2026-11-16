"""Match catalogue validation for the sample workflow."""

import pandas as pd


def validate_match_catalogue(
    matches_table: pd.DataFrame,
    competition_id: int,
    season_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return a check summary, per-match flags, and matches ready for loading."""
    required_columns = [
        "match_id", "competition_id", "season_id", "match_date", "kick_off",
        "home_team_id", "away_team_id", "home_team", "away_team",
        "match_status", "match_status_360",
    ]
    missing_columns = sorted(set(required_columns) - set(matches_table.columns))
    if missing_columns:
        raise ValueError(f"Match catalogue is missing required columns: {missing_columns}")
    if matches_table.empty:
        raise ValueError("The selected competition has no match records to validate.")

    required_values = matches_table[required_columns].replace(r"^\s*$", pd.NA, regex=True)
    match_ids = pd.to_numeric(required_values["match_id"], errors="coerce")
    home_ids = pd.to_numeric(required_values["home_team_id"], errors="coerce")
    away_ids = pd.to_numeric(required_values["away_team_id"], errors="coerce")
    kickoff_times = pd.to_datetime(
        required_values["match_date"].astype("string")
        + "T" + required_values["kick_off"].astype("string"),
        format="ISO8601", errors="coerce",
    )

    match_checks = pd.DataFrame({
        "Required fields populated": required_values.notna().all(axis=1),
        "Positive integer match IDs": match_ids.gt(0) & match_ids.mod(1).eq(0),
        "Unique match IDs": ~match_ids.duplicated(keep=False),
        "Selected competition and season": (
            pd.to_numeric(required_values["competition_id"], errors="coerce").eq(competition_id)
            & pd.to_numeric(required_values["season_id"], errors="coerce").eq(season_id)
        ),
        "Valid date and kickoff time": kickoff_times.notna(),
        "Valid and distinct team IDs": (
            home_ids.gt(0) & home_ids.mod(1).eq(0)
            & away_ids.gt(0) & away_ids.mod(1).eq(0)
            & home_ids.ne(away_ids)
        ),
        "Event data available": required_values["match_status"].eq("available"),
        "360 data available": required_values["match_status_360"].eq("available"),
    }).fillna(False)

    integrity_summary = pd.DataFrame({
        "Check": match_checks.columns,
        "Passed": match_checks.sum().to_numpy(),
        "Flagged": (~match_checks).sum().to_numpy(),
    })
    match_validation = matches_table[["match_id"]].join(match_checks)
    matches_for_processing = matches_table.loc[match_checks.all(axis=1)].copy()
    return integrity_summary, match_validation, matches_for_processing
