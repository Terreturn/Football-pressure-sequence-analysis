"""Read match outputs, aggregate the selected cohort and draw team and player charts."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
from filelock import FileLock
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def generate_plots(
    output_dir,
    *,
    input_source,
    match_ids=None,
    plots=None,
    events_dir=None,
    points=None,
    player_minutes=None,
    min_pressures=1,
    min_minutes=None,
    top_players=10,
    overwrite=False,
):
    """Write PNG, PDF and CSV files for all input matches or the specified match IDs."""
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
    cohort = load_cohort(input_source, events_dir, match_ids)
    point_values, points_note = (
        fixture_points(points, cohort)
        if points is not None
        else (None, "Supply per-match points or a StatsBomb match list.")
    )
    minutes = load_player_minutes(player_minutes, cohort) if player_minutes is not None else None
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


# Match outputs and optional competition metadata.

TEAM_COLUMNS = ["pressing_team_id", "presses", "expected_press_value"]
PRESSURE_COLUMNS = ["match_id", "event_id", "pressing_team_id", "player_id", "position_quality_pp"]


def read_table(path, required, *, matches=None):
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(required) - set(table)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    if matches is not None:
        table = table.loc[table.match_id.isin(matches)].copy()
    if table[required].eq("").any().any():
        raise ValueError(f"{path} contains empty required values")
    return table


def numeric(table, column, *, minimum=None, maximum=None, whole=False):
    values = pd.to_numeric(table[column], errors="raise")
    if (
        not np.isfinite(values).all()
        or (minimum is not None and (values < minimum).any())
        or (maximum is not None and (values > maximum).any())
        or (whole and (values % 1 != 0).any())
    ):
        raise ValueError(f"Invalid {column}")
    return values.astype(int if whole else float)


@dataclass
class Cohort:
    matches: set = field(default_factory=set)
    sequence_matches: set = field(default_factory=set)
    position_matches: set = field(default_factory=set)
    teams: dict = field(default_factory=dict)
    players: dict = field(default_factory=dict)
    team_fixtures: set = field(default_factory=set)
    player_fixtures: set = field(default_factory=set)
    team_rows: list = field(default_factory=list)
    pressure_rows: list = field(default_factory=list)
    xg: list = field(default_factory=list)

    def team(self, match_id, team_id, name=""):
        self.team_fixtures.add((match_id, team_id))
        if name or team_id not in self.teams:
            self.teams[team_id] = name or f"Team {team_id}"

    def player(self, match_id, player_id, name=""):
        self.player_fixtures.add((match_id, player_id))
        if name or player_id not in self.players:
            self.players[player_id] = name or f"Player {player_id}"


def add_events(cohort, match_id, path):
    events = json.loads(path.read_text(encoding="utf-8"))
    scored_teams = {tid for mid, tid in cohort.team_fixtures if mid == match_id}
    totals = {}
    complete = True
    for event in events:
        team = event.get("team", {})
        if team.get("id") is None:
            continue
        tid = str(team["id"])
        cohort.team(match_id, tid, team.get("name"))
        totals.setdefault(tid, 0.0)
        people = [event.get("player", {}), event.get("substitution", {}).get("replacement", {})]
        people.extend(p.get("player", {}) for p in event.get("tactics", {}).get("lineup", []))
        for person in people:
            if person.get("id") is not None:
                cohort.player(match_id, str(person["id"]), person.get("name"))
        if event.get("type", {}).get("name") == "Shot" and event.get("period") != 5:
            xg = event.get("shot", {}).get("statsbomb_xg")
            if xg is None:
                complete = False
            elif not np.isfinite(xg) or not 0 <= xg <= 1:
                raise ValueError(f"Invalid shot xG in {path}")
            else:
                totals[tid] += xg
    if complete and len(totals) == 2:
        if not scored_teams.issubset(totals):
            raise ValueError(f"Team IDs in SPN outputs disagree with {path}")
        a, b = totals
        cohort.xg.extend(
            [(match_id, a, totals[a] - totals[b]), (match_id, b, totals[b] - totals[a])]
        )


def fixture_directories(root):
    if not root.is_dir():
        raise FileNotFoundError(f"Missing match output directory: {root}")
    manifest = root / "batch_result.json"
    if manifest.is_file():
        for item in json.loads(manifest.read_text(encoding="utf-8")):
            mid = str(item["match_id"])
            if mid in {".", ".."} or Path(mid).name != mid:
                raise ValueError(f"Invalid fixture directory in {manifest}")
            folder = root / mid
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing fixture directory: {folder}")
            yield (
                folder,
                mid,
                "prediction_count" in item and (folder / "teams.csv").is_file(),
                "position_count" in item and (folder / "pressures.csv").is_file(),
            )
        return

    def has_output(folder):
        return (folder / "teams.csv").is_file() or (folder / "pressures.csv").is_file()

    folders = (
        [root]
        if has_output(root)
        else sorted(p for p in root.iterdir() if p.is_dir() and has_output(p))
    )
    for folder in folders:
        yield folder, None, (folder / "teams.csv").is_file(), (folder / "pressures.csv").is_file()


def load_cohort(source, events_dir, match_ids=None):
    cohort = Cohort()
    selected = None if match_ids is None else {str(mid) for mid in match_ids}
    if events_dir is not None:
        events_dir = Path(events_dir)
        if not events_dir.is_dir():
            raise FileNotFoundError(f"Missing events directory: {events_dir}")
    for folder, declared_id, sequence, positioning in fixture_directories(Path(source)):
        if selected is not None and declared_id is not None and declared_id not in selected:
            continue
        ids = {declared_id} if declared_id is not None else set()
        names = {}
        if sequence and (folder / "predictions.csv").is_file():
            cols = ["match_id", "pressing_team_id", "press_team"]
            predictions = pd.read_csv(
                folder / "predictions.csv", usecols=cols, dtype=str, keep_default_na=False
            )
            ids.update(predictions.match_id.unique())
            names = dict(
                predictions[["pressing_team_id", "press_team"]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )
        if positioning:
            pressures = read_table(folder / "pressures.csv", PRESSURE_COLUMNS)
            ids.update(pressures.match_id.unique())
            if pressures.event_id.duplicated().any():
                raise ValueError(f"Duplicate pressure event in {folder}")
            pressures["position_quality_pp"] = numeric(
                pressures, "position_quality_pp", minimum=-100, maximum=100
            )
        if not ids and (folder / "result.json").is_file():
            ids.add(
                str(json.loads((folder / "result.json").read_text(encoding="utf-8"))["match_id"])
            )
        if len(ids) > 1:
            raise ValueError(f"Conflicting match IDs in {folder}")
        mid = next(iter(ids), folder.name)
        if selected is not None and mid not in selected:
            continue
        if mid in cohort.matches:
            raise ValueError(f"Duplicate match {mid}")
        cohort.matches.add(mid)
        if sequence:
            teams = read_table(folder / "teams.csv", TEAM_COLUMNS)
            if teams.pressing_team_id.duplicated().any():
                raise ValueError(f"Duplicate team in {folder}")
            teams["presses"] = numeric(teams, "presses", minimum=1, whole=True)
            teams["expected_press_value"] = numeric(
                teams, "expected_press_value", minimum=-1, maximum=1
            )
            for tid in teams.pressing_team_id:
                cohort.team(mid, tid, names.get(tid))
            cohort.team_rows.append(teams[TEAM_COLUMNS])
            cohort.sequence_matches.add(mid)
        if positioning:
            for tid in pressures.pressing_team_id.unique():
                cohort.team(mid, tid)
            for pid in pressures.player_id.unique():
                cohort.player(mid, pid)
            cohort.pressure_rows.append(pressures[PRESSURE_COLUMNS])
            cohort.position_matches.add(mid)
        if events_dir is not None and (events_dir / f"{mid}.json").is_file():
            add_events(cohort, mid, events_dir / f"{mid}.json")
    if selected is not None and selected - cohort.matches:
        raise ValueError(f"Missing SPN outputs for match IDs: {sorted(selected - cohort.matches)}")
    if not cohort.matches:
        raise ValueError(f"No SPN fixture outputs in {source}")
    return cohort


def supplemental_table(path, cohort, id_column, value_column):
    keys = ["match_id", id_column]
    table = read_table(path, [*keys, value_column], matches=cohort.matches)
    if table.duplicated(keys).any():
        raise ValueError(f"Duplicate match and {id_column} in {path}")
    table[value_column] = numeric(table, value_column, minimum=0)
    return table


def team_press_values(cohort):
    rows = pd.concat(cohort.team_rows, ignore_index=True)
    rows["total"] = rows.presses * rows.expected_press_value
    teams = rows.groupby("pressing_team_id")[["presses", "total"]].sum()
    teams["expected_press_value"] = teams.total / teams.presses
    return teams.drop(columns="total")


def positioning_values(cohort, player):
    rows = pd.concat(cohort.pressure_rows, ignore_index=True)
    key = "player_id" if player else "pressing_team_id"
    return rows.groupby(key).position_quality_pp.agg(pressures="size", position_quality_pp="mean")


def ranking(values, names, id_name, count):
    table = values.reindex(pd.Index(sorted(names), name=id_name)).reset_index()
    table["name"] = table[id_name].map(names)
    table[count] = table[count].fillna(0).astype(int)
    table["status"] = np.where(table[count].gt(0), "ranked", "no_scored_events")
    return table


def fixture_points(path, cohort):
    """Accept per-match points or StatsBomb match results for a single league/group stage."""
    path = Path(path)
    if path.suffix.lower() != ".json":
        table = supplemental_table(path, cohort, "team_id", "points")
    else:
        records = []
        stages = set()
        for match in json.loads(path.read_text(encoding="utf-8")):
            mid = str(match["match_id"])
            if mid not in cohort.matches:
                continue
            stage = match["competition_stage"]["name"].casefold()
            stages.add(
                (match["competition"]["competition_id"], match["season"]["season_id"], stage)
            )
            if stage not in {"regular season", "league stage", "group stage"}:
                return None, "Points require league or group-stage fixtures."
            home, away = match["home_score"], match["away_score"]
            if home is None or away is None:
                return None, "A selected match has no final score."
            for side, difference in [("home", home - away), ("away", away - home)]:
                team = match[f"{side}_team"]
                records.append(
                    (
                        mid,
                        str(team[f"{side}_team_id"]),
                        team[f"{side}_team_name"],
                        3 if difference > 0 else int(difference == 0),
                    )
                )
        if len(stages) > 1:
            return None, "Select one competition, season and stage for a points comparison."
        table = pd.DataFrame(records, columns=["match_id", "team_id", "team_name", "points"])
    if table.duplicated(["match_id", "team_id"]).any():
        raise ValueError(f"Duplicate team result in {path}")
    pairs = set(zip(table.match_id, table.team_id))
    if (
        set(table.match_id) != cohort.matches
        or not table.groupby("match_id").size().eq(2).all()
        or not cohort.team_fixtures.issubset(pairs)
    ):
        return None, "Points must cover both teams in every selected match."
    for row in table.itertuples():
        cohort.team(row.match_id, row.team_id, getattr(row, "team_name", ""))
    return table.groupby("team_id").points.agg(points="sum", matches="size"), None


def load_player_minutes(path, cohort):
    table = supplemental_table(path, cohort, "player_id", "minutes")
    supplied = set(zip(table.match_id, table.player_id))
    missing = {pid for mid, pid in cohort.player_fixtures - supplied}
    for row in table.itertuples():
        cohort.player(row.match_id, row.player_id, getattr(row, "player_name", ""))
    return table.groupby("player_id").minutes.sum().drop(list(missing), errors="ignore")


# Figure style and reusable panels.

TEAM_COLOR = "#243F63"
PLAYER_COLOR = "#2A8C82"


@contextmanager
def paper_style():
    with matplotlib.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        yield


def save_figure(fig, stem):
    for suffix in (".pdf", ".png"):
        fig.savefig(
            stem.with_suffix(suffix),
            facecolor="white",
            dpi=425,
            bbox_inches="tight",
            pad_inches=0.12,
        )


def player_panel(ax, table, metric, count, *, xlabel, color):
    """Rank signed means and show the number of observations behind each bar."""
    shown = table.loc[table.displayed]
    if not shown.empty:
        labels = [f"{row['name']}  (n={row[count]})" for _, row in shown.iterrows()]
        bars = ax.barh(np.arange(len(shown)), shown[metric], color=color, height=0.58)
        ax.bar_label(bars, fmt="%.3g", padding=3, fontsize=8)
        ax.margins(x=0.14)
        ax.set_yticks(np.arange(len(shown)), labels=labels)
        ax.invert_yaxis()
        ax.axvline(0, color="#77818A", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E5E8E9", linewidth=0.6)
        ax.set_xlabel(xlabel)
    else:
        message = (
            "No scored events"
            if table[count].sum() == 0
            else "No players meet the selected thresholds"
        )
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()


def labels_without_overlap(ax, points, names, line):
    """Use the collaborator's display-space collision checks, with an outside column for dense plots."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    xy = ax.transData.transform(points)
    line_path = line.get_transform().transform_path(line.get_path()) if line is not None else None
    occupied = []
    labels = []
    for point, name in zip(points, names):
        label = ax.annotate(name, point, xytext=(0, 0), textcoords="offset points", fontsize=8)
        labels.append(label)
        placed = False
        for radius in (5, 10, 16, 24, 36):
            for angle in (20, 160, -20, -160, 90, -90):
                dx, dy = radius * np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
                label.set_position((dx, dy))
                label.set_ha("left" if dx >= 0 else "right")
                label.set_va("bottom" if dy >= 0 else "top")
                box = label.get_window_extent(renderer).padded(1.5)
                if (
                    ax.bbox.contains(*box.min)
                    and ax.bbox.contains(*box.max)
                    and not any(box.overlaps(old) for old in occupied)
                    and not any(box.contains(x, y) for x, y in xy)
                    and (line_path is None or not line_path.intersects_bbox(box, filled=False))
                ):
                    occupied.append(box)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            for artist in labels:
                artist.remove()
            order = np.argsort(points[:, 1], kind="stable")
            for y, index in zip(np.linspace(0.05, 0.95, len(points)), order):
                ax.annotate(
                    names[index],
                    points[index],
                    xytext=(1.04, y),
                    textcoords="axes fraction",
                    va="center",
                    fontsize=8,
                    arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "#89939B"},
                )
            break


def team_panel(ax, table, metric, outcome, *, xlabel, ylabel):
    x, y = table[metric].to_numpy(), table[outcome].to_numpy()
    ax.set_axisbelow(True)
    ax.grid(color="#E5E8E9", linewidth=0.6)
    ax.scatter(x, y, s=33, color=TEAM_COLOR, edgecolors="white", linewidths=0.6, zorder=3)
    line = None
    if len(x) >= 2 and np.ptp(x) > 0:
        fit = np.polynomial.Polynomial.fit(x, y, 1)
        limits = np.array([x.min(), x.max()])
        (line,) = ax.plot(limits, fit(limits), color="#A7AFB5", linewidth=0.85)
    ax.margins(x=0.20, y=0.16)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    labels_without_overlap(ax, np.column_stack((x, y)), table.name.tolist(), line)


# Rankings, comparisons and exports.

matplotlib.use("Agg")


def prepare_rankings(cohort, minutes, min_pressures, min_minutes, top_players):
    rankings = {}
    if cohort.sequence_matches == cohort.matches:
        rankings["team-value"] = ranking(
            team_press_values(cohort), cohort.teams, "team_id", "presses"
        )
    if cohort.position_matches == cohort.matches:
        rankings["team-positioning"] = ranking(
            positioning_values(cohort, False), cohort.teams, "team_id", "pressures"
        )
        players = ranking(
            positioning_values(cohort, True), cohort.players, "player_id", "pressures"
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
