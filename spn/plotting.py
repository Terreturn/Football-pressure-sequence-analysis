"""Plot team press values and player positioning for selected SPN match outputs."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
from filelock import FileLock
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import _plot_data as data
from ._plot_style import (
    TEAM_COLOR,
    PLAYER_COLOR,
    paper_style,
    player_panel,
    save_figure,
    team_panel,
)

RANKINGS = ("team-value", "team-positioning", "player-positioning")
COMPARISONS = ("value-xgd", "positioning-points")
PLOTS = RANKINGS + COMPARISONS
VALUE = "expected_press_value"
QUALITY = "position_quality_pp"
TITLES = {
    "team-value": "Expected press value by team",
    "team-positioning": "Presser positioning by team",
    "player-positioning": "Presser positioning by player",
    "value-xgd": "Expected press value and expected-goal difference",
    "positioning-points": "Presser positioning and competition points",
}


def prepare_rankings(cohort, minutes, min_pressures, min_minutes, top_players):
    rankings = {}
    if cohort.sequence_matches == cohort.matches:
        rankings["team-value"] = data.ranking(
            data.team_press_values(cohort), cohort.teams, "team_id", "presses"
        )
    if cohort.position_matches == cohort.matches:
        rankings["team-positioning"] = data.ranking(
            data.positioning_values(cohort, False), cohort.teams, "team_id", "pressures"
        )
        players = data.ranking(
            data.positioning_values(cohort, True), cohort.players, "player_id", "pressures"
        )
        scored = players.pressures.gt(0)
        players.loc[scored & players.pressures.lt(min_pressures), "status"] = "below_min_pressures"
        if minutes is not None:
            players["minutes"] = players.player_id.map(minutes)
            if min_minutes is not None:
                eligible = players.status.eq("ranked")
                players.loc[eligible & players.minutes.isna(), "status"] = "missing_minutes"
                players.loc[eligible & players.minutes.lt(min_minutes), "status"] = (
                    "below_min_minutes"
                )
        rankings["player-positioning"] = players

    for key, table in rankings.items():
        metric = VALUE if key == "team-value" else QUALITY
        eligible = table.status.eq("ranked")
        table["rank"] = (
            table[metric].where(eligible).rank(method="min", ascending=False).astype("Int64")
        )
        id_column = "player_id" if key == "player-positioning" else "team_id"
        table.sort_values(["rank", id_column], inplace=True, ignore_index=True)
        table["displayed"] = table.status.eq("ranked")
        if key == "player-positioning" and top_players is not None:
            table.loc[table.index >= top_players, "displayed"] = False
        same_name = table.name.duplicated(keep=False)
        table.loc[same_name, "name"] += " [" + table.loc[same_name, id_column] + "]"
    return rankings


def comparison(key, rankings, cohort, points, points_note):
    if key == "value-xgd":
        if {row[0] for row in cohort.xg} != cohort.matches:
            return None, "Supply StatsBomb events with shot xG for every selected match."
        teams = rankings["team-value"]
        if not teams.presses.sum():
            return None, "No scored sequence onsets."
        xg = pd.DataFrame(cohort.xg, columns=["match_id", "team_id", "xgd"])
        outcomes = xg.groupby("team_id").xgd.agg(xgd_per_match="mean", matches="size")
        table = teams.loc[teams.status.eq("ranked")].merge(
            outcomes, on="team_id", validate="one_to_one"
        )
        if len(table) != teams.status.eq("ranked").sum():
            raise ValueError("Team IDs in xG events and SPN outputs disagree")
        table["centered_press_value"] = table[VALUE] - np.average(
            table[VALUE], weights=table.presses
        )
    else:
        if points is None:
            return None, points_note
        teams = rankings["team-positioning"]
        table = teams.loc[teams.status.eq("ranked")].merge(
            points, on="team_id", validate="one_to_one"
        )
    if len(table) < 2:
        return None, "At least two teams with scored events and matching outcomes are required."
    return table, None


def render_plot(key, table, stage):
    if key in RANKINGS:
        is_value = key == "team-value"
        count = "presses" if is_value else "pressures"
        height = max(2.5, 1.35 + 0.32 * table.displayed.sum())
        fig, ax = plt.subplots(figsize=(8, height))
    else:
        fig, ax = plt.subplots(figsize=(8, max(4.4, 1.5 + 0.21 * len(table))))
    try:
        ax.set_title(TITLES[key], loc="left", weight="bold", pad=14)
        if key in RANKINGS:
            player_panel(
                ax,
                table,
                VALUE if is_value else QUALITY,
                count,
                xlabel=(
                    "Mean success minus failure probability at onset"
                    if is_value
                    else "Mean position quality (percentage points)"
                ),
                color=PLAYER_COLOR if key == "player-positioning" else TEAM_COLOR,
            )
            missing = table.loc[table.status.eq("no_scored_events"), "name"]
            if len(missing) and table.displayed.any():
                text = ", ".join(missing) if len(missing) <= 3 else f"{len(missing)} (see CSV)"
                ax.annotate(
                    f"No scored events: {text}",
                    (0, 0),
                    xycoords="axes fraction",
                    xytext=(0, -38),
                    textcoords="offset points",
                    fontsize=8,
                    va="top",
                )
        elif key == "value-xgd":
            team_panel(
                ax,
                table,
                "centered_press_value",
                "xgd_per_match",
                xlabel="Expected press value relative to the selected matches",
                ylabel="Expected-goal difference per match",
            )
        else:
            team_panel(
                ax,
                table,
                QUALITY,
                "points",
                xlabel="Mean position quality (percentage points)",
                ylabel="Points in selected matches",
            )
        save_figure(fig, stage / key)
        table.to_csv(stage / f"{key}.csv", index=False)
    finally:
        plt.close(fig)


def publish_report(stage, output, report, overwrite):
    """Publish one consistent set of outputs and restore the previous files if a move fails."""
    manifest = "manifest.json"
    files = [name for entry in report["plots"].values() for name in entry["files"]]
    (stage / manifest).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with FileLock(str(output / ".spn-plots.lock")):
        targets = set([*files, manifest])
        if overwrite and (output / manifest).is_file():
            old = json.loads((output / manifest).read_text(encoding="utf-8"))
            allowed = {f"{key}.{ext}" for key in PLOTS for ext in ("png", "pdf", "csv")}
            targets.update(
                name
                for entry in old.get("plots", {}).values()
                for name in entry.get("files", [])
                if name in allowed
            )
        existing = [name for name in targets if (output / name).exists()]
        for name in existing:
            if (output / name).is_dir() or not overwrite:
                raise FileExistsError(
                    f"{output / name} exists. Use --overwrite or a new output directory."
                )
        backup = stage / "previous"
        backup.mkdir()
        saved, installed = [], []
        try:
            for name in existing:
                (output / name).replace(backup / name)
                saved.append(name)
            for name in [*files, manifest]:
                (stage / name).replace(output / name)
                installed.append(name)
        except OSError:
            for name in installed:
                (output / name).unlink(missing_ok=True)
            for name in saved:
                (backup / name).replace(output / name)
            raise


def generate_plots(
    output_dir,
    *,
    input_source,
    plots=None,
    events_dir=None,
    points=None,
    player_minutes=None,
    min_pressures=1,
    min_minutes=None,
    top_players=10,
    overwrite=False,
):
    """Write PNG, PDF and CSV files for the selected match cohort and return its coverage report."""
    if plots is None:
        selected = list(RANKINGS)
        if events_dir is not None:
            selected.append("value-xgd")
        if points is not None:
            selected.append("positioning-points")
    else:
        selected = list(dict.fromkeys(plots))
    if not selected or any(name not in PLOTS for name in selected):
        raise ValueError(f"Choose plots from {', '.join(PLOTS)}")
    if min_pressures < 1 or (top_players is not None and top_players < 1):
        raise ValueError("Minimum pressures and player limit must be positive")
    if min_minutes is not None and (
        not np.isfinite(min_minutes) or min_minutes < 0 or player_minutes is None
    ):
        raise ValueError("A nonnegative minimum minutes threshold requires a player minutes CSV")
    cohort = data.load_cohort(input_source, events_dir)
    point_values, points_note = (
        data.fixture_points(points, cohort)
        if points is not None
        else (None, "Supply per-match points or a StatsBomb match list.")
    )
    minutes = data.player_minutes(player_minutes, cohort) if player_minutes is not None else None
    rankings = prepare_rankings(cohort, minutes, min_pressures, min_minutes, top_players)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "output": str(output),
        "match_ids": sorted(cohort.matches),
        "plots": {},
        "player_selection": {
            "min_pressures": min_pressures,
            "min_minutes": min_minutes,
            "top_players": top_players,
        },
    }
    manifest = output / "manifest.json"
    if manifest.exists() and not overwrite:
        raise FileExistsError(f"{manifest} exists. Use --overwrite or a new output directory.")
    with TemporaryDirectory(dir=output) as workspace, paper_style():
        stage = Path(workspace)
        for key in selected:
            required = {"value-xgd": "team-value", "positioning-points": "team-positioning"}.get(
                key, key
            )
            reason = None
            table = rankings.get(required)
            if table is None:
                name = "teams.csv" if required == "team-value" else "pressures.csv"
                reason = f"{name} is required for every selected match."
            elif key in COMPARISONS:
                table, reason = comparison(key, rankings, cohort, point_values, points_note)
            if reason:
                report["plots"][key] = {"status": "skipped", "reason": reason, "files": []}
                continue
            render_plot(key, table, stage)
            status = "empty" if key in RANKINGS and not table.displayed.any() else "generated"
            report["plots"][key] = {
                "status": status,
                "files": [f"{key}.{ext}" for ext in ("png", "pdf", "csv")],
            }
        publish_report(stage, output, report, overwrite)
    return report
