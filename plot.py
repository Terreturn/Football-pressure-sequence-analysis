#!/usr/bin/env python3
"""Plot team and player results from one SPN match or batch output directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="single-match or batch output from run.py"
    )
    parser.add_argument("--output", "--out", type=Path, required=True, help="plot output directory")
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=(
            "team-value",
            "team-positioning",
            "player-positioning",
            "value-xgd",
            "positioning-points",
        ),
        help="default: rankings plus comparisons with supplied data",
    )
    parser.add_argument(
        "--events-dir",
        type=Path,
        help="StatsBomb event JSONs named by match ID (names, player appearances and shot xG)",
    )
    parser.add_argument(
        "--points",
        type=Path,
        help="per-match points CSV or StatsBomb matches JSON for this competition stage",
    )
    parser.add_argument("--player-minutes", type=Path, help="per-player, per-match minutes CSV")
    parser.add_argument(
        "--min-pressures",
        type=int,
        default=1,
        help="minimum scored pressures for a player (default: 1)",
    )
    parser.add_argument(
        "--min-minutes", type=float, help="minimum minutes across the selected matches"
    )
    parser.add_argument(
        "--top-players", type=int, default=10, help="maximum displayed players (default: 10)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the previous report in the output directory",
    )
    args = parser.parse_args()
    try:
        from spn.plotting import generate_plots

        report = generate_plots(
            args.output,
            input_source=args.input,
            plots=args.plots,
            events_dir=args.events_dir,
            points=args.points,
            player_minutes=args.player_minutes,
            min_pressures=args.min_pressures,
            min_minutes=args.min_minutes,
            top_players=args.top_players,
            overwrite=args.overwrite,
        )
    except ImportError as error:
        parser.error(f"Missing plotting dependency: {error}. Install requirements-plots.txt")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
