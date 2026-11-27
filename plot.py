#!/usr/bin/env python3
"""Redraw the four main spn paper figures from the accompanying plot tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--figures",
        "--plots",
        type=int,
        nargs="+",
        choices=(1, 2, 3, 4),
        default=(1, 2, 3, 4),
        help="paper figure numbers (default: all four)",
    )
    parser.add_argument(
        "--tables", type=Path, help="paper table directory (default: bundled plot_data)"
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing figure files")
    args = parser.parse_args()
    try:
        from spn.plotting import PLOT_DATA, generate_plots

        report = generate_plots(
            args.output,
            plots=args.figures,
            tables=args.tables or PLOT_DATA,
            overwrite=args.overwrite,
        )
    except ImportError as error:
        parser.error(f"Missing plotting dependency: {error}. Install requirements-plots.txt")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
