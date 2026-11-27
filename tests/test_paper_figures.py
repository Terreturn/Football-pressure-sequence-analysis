"""Exercise the public figure command with only the files available in a checkout."""

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest


def test_paper_figures_from_another_directory(tmp_path):
    pytest.importorskip("matplotlib")
    from PIL import Image

    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "figures"
    completed = subprocess.run(
        [sys.executable, str(root / "plot.py"), "--output", str(output)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)["plots"]
    assert set(report) == {"1", "2", "3", "4"}
    assert sum(len(figure["files"]) for figure in report.values()) == 11
    stems = json.loads((root / "plot_data" / "reference_png_hashes.json").read_text())
    for stem in stems:
        number = stem[len("figure")]
        files = report[number]["files"]
        assert f"{stem}.png" in files and f"{stem}.pdf" in files
        with Image.open(output / f"{stem}.png") as image:
            assert image.info["dpi"] == pytest.approx((425, 425), abs=0.01)
        assert (output / f"{stem}.pdf").read_bytes().startswith(b"%PDF-")

    assert (output / "figure3_working_paper_size.png").is_file()
    caption = "figure4_broad_regain_caption.tex"
    assert (output / caption).read_text() == (root / "plot_data" / caption).read_text()
    with (output / "figure4_broad_regain_top10.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    assert rows[0]["display_name"] == "Jayden Bogle"
    assert all(int(row["pressures"]) >= 15 and float(row["season_minutes"]) >= 1350 for row in rows)
