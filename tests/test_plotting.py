"""Exercise cohort-specific plotting on teams and players outside the paper tables."""

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

pytest.importorskip("matplotlib")
from spn.plotting import generate_plots


def write_csv(path, columns, records):
    pd.DataFrame(records, columns=columns.split()).to_csv(path, index=False)


def shot_event(team, player, xg, period=1):
    return {
        "team": {"id": team, "name": "Club " + team},
        "player": {"id": player, "name": "Athlete " + player},
        "type": {"name": "Shot"},
        "period": period,
        "shot": {"statsbomb_xg": xg},
    }


@pytest.fixture
def cohort(tmp_path):
    source, events = tmp_path / "matches", tmp_path / "events"
    source.mkdir()
    events.mkdir()
    records = [
        ("100", "B", [(1, 0.4), (2, -0.2)], [("e1", "A", "X", 2), ("e2", "B", "Y", 0)]),
        ("101", "C", [(3, 0.0), (1, 0.0)], [("e3", "A", "X", -0.5), ("e4", "A", "Z", 1.5)]),
    ]
    for mid, other, values, pressures in records:
        folder = source / mid
        folder.mkdir()
        write_csv(
            folder / "teams.csv",
            "pressing_team_id presses expected_press_value",
            [(team, n, value) for team, (n, value) in zip(["A", other], values)],
        )
        write_csv(
            folder / "pressures.csv",
            "match_id event_id pressing_team_id player_id position_quality_pp",
            [(mid, *row) for row in pressures],
        )
        ev = [
            shot_event("A", "X", 0.6),
            shot_event(other, "Y" if other == "B" else "Q", 0.2),
            shot_event("A", "X", 0.9, 5),
        ]
        (events / f"{mid}.json").write_text(json.dumps(ev))
    return source, events


def test_batch_rankings_and_matched_comparisons(cohort, tmp_path):
    source, events = cohort
    points = tmp_path / "points.csv"
    write_csv(
        points,
        "match_id team_id points",
        [(100, "A", 3), (100, "B", 0), (101, "A", 1), (101, "C", 1), (999, "D", 3)],
    )
    minutes = tmp_path / "minutes.csv"
    write_csv(
        minutes,
        "match_id player_id minutes",
        [(100, "X", 50), (101, "X", 30), (100, "Y", 90), (101, "Q", 90)],
    )
    out = tmp_path / "plots"
    report = generate_plots(
        out,
        input_source=source,
        events_dir=events,
        points=points,
        player_minutes=minutes,
        min_minutes=60,
        top_players=1,
    )
    assert set(report["match_ids"]) == {"100", "101"}
    assert all(item["status"] == "generated" for item in report["plots"].values())
    for item in report["plots"].values():
        assert all((out / path).is_file() for path in item["files"])
    values = pd.read_csv(out / "team-value.csv").set_index("team_id")
    assert values.loc["A", "expected_press_value"] == pytest.approx(0.1)
    assert values.loc["A", "presses"] == 4
    teams = pd.read_csv(out / "team-positioning.csv").set_index("team_id")
    assert teams.loc["A", "position_quality_pp"] == pytest.approx(1.0)
    assert teams.loc["B", "position_quality_pp"] == 0 and teams.loc["B", "status"] == "ranked"
    assert (
        pd.isna(teams.loc["C", "position_quality_pp"])
        and teams.loc["C", "status"] == "no_scored_events"
    )
    players = pd.read_csv(out / "player-positioning.csv").set_index("player_id")
    assert players.loc["X", "minutes"] == 80 and players.loc["X", "position_quality_pp"] == 0.75
    assert players.loc["X", "name"] == "Athlete X" and players.loc["X", "displayed"]
    assert not players.loc["Y", "displayed"] and players.loc["Z", "status"] == "missing_minutes"
    assert players.loc["Q", "status"] == "no_scored_events"
    xg = pd.read_csv(out / "value-xgd.csv").set_index("team_id")
    assert xg.loc["A", "xgd_per_match"] == pytest.approx(0.4)
    assert xg.loc["A", "matches"] == 2
    assert (xg.centered_press_value * xg.presses).sum() == pytest.approx(0)
    assert pd.read_csv(out / "positioning-points.csv").set_index("team_id").loc["A", "points"] == 4


def test_cli_empty_fixture_and_unavailable_task(tmp_path):
    src = tmp_path / "one"
    src.mkdir()
    write_csv(
        src / "pressures.csv",
        "match_id event_id pressing_team_id player_id position_quality_pp",
        [],
    )
    cli = Path(__file__).resolve().parents[1] / "plot.py"
    out = tmp_path / "charts"
    r = subprocess.run(
        [sys.executable, str(cli), "--input", str(src), "--output", str(out)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    plots = json.loads(r.stdout)["plots"]
    assert plots["team-value"]["status"] == "skipped"
    for key in ["player-positioning", "team-positioning"]:
        assert plots[key]["status"] == "empty"
        assert pd.read_csv(out / (key + ".csv")).empty
        assert (out / (key + ".pdf")).read_bytes().startswith(b"%PDF")


def test_batch_manifest_selects_current_fixtures_and_tasks(cohort, tmp_path):
    source, _ = cohort
    # A moved batch directory can still contain outputs from an earlier scoring run.
    manifest = [{"match_id": "101", "prediction_count": 4, "output": "/original/location/101"}]
    (source / "batch_result.json").write_text(json.dumps(manifest))
    out = tmp_path / "selected"
    report = generate_plots(out, input_source=source)
    assert report["match_ids"] == ["101"]
    assert report["plots"]["player-positioning"]["status"] == "skipped"
    teams = pd.read_csv(out / "team-value.csv").set_index("team_id")
    assert set(teams.index) == {"A", "C"}
    assert teams.loc["A", "presses"] == 3


def test_partial_metadata_and_replacement(cohort, tmp_path):
    source, events = cohort
    output = tmp_path / "plot"
    settings = dict(input_source=source, events_dir=events, plots=["value-xgd"])
    generate_plots(output, **settings)
    with pytest.raises(FileExistsError):
        generate_plots(output, **settings)
    raw = events / "101.json"
    data = json.loads(raw.read_text())
    del data[0]["shot"]["statsbomb_xg"]
    raw.write_text(json.dumps(data))
    r = generate_plots(output, overwrite=True, **settings)
    assert (
        r["plots"]["value-xgd"]["status"] == "skipped" and not (output / "value-xgd.png").exists()
    )
    (source / "101" / "pressures.csv").unlink()
    r = generate_plots(output, input_source=source, plots=["player-positioning"], overwrite=True)
    assert r["plots"]["player-positioning"]["status"] == "skipped"


def test_points_stage_rules_and_minimum_pressures(cohort, tmp_path):
    source, events = cohort
    games = []
    for mid, away in [(100, "B"), (101, "C")]:
        games.append(
            dict(
                match_id=mid,
                competition_stage={"name": "Group Stage"},
                competition={"competition_id": 1},
                season={"season_id": 2},
                home_score=1,
                away_score=0,
                home_team={"home_team_id": "A", "home_team_name": "Club A"},
                away_team={"away_team_id": away, "away_team_name": "Club " + away},
            )
        )
    path = tmp_path / "matches.json"
    path.write_text(json.dumps(games))
    out = tmp_path / "out"
    args = dict(
        input_source=source,
        plots=["positioning-points", "player-positioning"],
        points=path,
        min_pressures=2,
    )
    generate_plots(out, **args)
    assert pd.read_csv(out / "positioning-points.csv").set_index("team_id").loc["A", "points"] == 6
    table = pd.read_csv(out / "player-positioning.csv").set_index("player_id")
    assert table.loc["X", "displayed"] and table.loc["Y", "status"] == "below_min_pressures"
    games[1]["competition_stage"] = {"name": "Final"}
    path.write_text(json.dumps(games))
    r = generate_plots(out, **args, overwrite=True)
    assert r["plots"]["positioning-points"]["status"] == "skipped"
    with pytest.raises(ValueError, match="minutes"):
        generate_plots(out, input_source=source, min_minutes=50)
