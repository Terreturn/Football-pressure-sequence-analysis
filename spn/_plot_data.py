"""Read one selected set of SPN fixture outputs and the optional comparison data."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


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


def load_cohort(source, events_dir):
    cohort = Cohort()
    if events_dir is not None:
        events_dir = Path(events_dir)
        if not events_dir.is_dir():
            raise FileNotFoundError(f"Missing events directory: {events_dir}")
    for folder, declared_id, sequence, positioning in fixture_directories(Path(source)):
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


def player_minutes(path, cohort):
    table = supplemental_table(path, cohort, "player_id", "minutes")
    supplied = set(zip(table.match_id, table.player_id))
    missing = {pid for mid, pid in cohort.player_fixtures - supplied}
    for row in table.itertuples():
        cohort.player(row.match_id, row.player_id, getattr(row, "player_name", ""))
    return table.groupby("player_id").minutes.sum().drop(list(missing), errors="ignore")
