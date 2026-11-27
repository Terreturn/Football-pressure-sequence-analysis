"""Reproduce the four main figures of the structural pressure network paper."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
from ._paper_figures.figures import draw_figure


PLOT_DATA = Path(__file__).resolve().parents[1] / "plot_data"
FIGURES = (1, 2, 3, 4)


def generate_plots(
    output_dir: str | Path,
    *,
    plots: Sequence[int] = FIGURES,
    tables: str | Path = PLOT_DATA,
    overwrite: bool = False,
) -> dict:
    """Draw the requested paper figures and return their exported filenames by figure number."""
    numbers = list(dict.fromkeys(plots))
    if not numbers or any(number not in FIGURES for number in numbers):
        raise ValueError("Select one or more paper figures numbered 1 through 4")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    with TemporaryDirectory(dir=output) as workspace:
        files = []
        for number in numbers:
            staged = Path(workspace) / str(number)
            draw_figure(number, Path(tables), staged)
            rendered = sorted(path for path in staged.iterdir() if not path.name.startswith("."))
            files.extend(rendered)
            report[str(number)] = {"status": "generated", "files": [p.name for p in rendered]}

        # Complete all selected figures before publishing any of their files.
        for path in files:
            target = output / path.name
            if target.is_dir():
                raise IsADirectoryError(target)
            if target.exists() and not overwrite:
                raise FileExistsError(
                    f"{target} exists. Use --overwrite or another output directory."
                )
        for path in files:
            path.replace(output / path.name)
    return {"output": str(output), "plots": report}
