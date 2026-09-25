#!/usr/bin/env python3
"""Generate public SPN feature-importance, v_t-driver and team plots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="public match output directory, batch directory or predictions CSV")
    parser.add_argument("--features", type=Path, help="optional feature CSV/directory; defaults to features.csv beside each prediction file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plots", nargs="+", choices=("importance", "drivers", "teams"), default=["importance", "drivers", "teams"])
    parser.add_argument("--formats", nargs="+", choices=("png", "jpeg", "pdf", "svg"), default=["png", "jpeg", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--terminal-weight", type=float, default=0.5)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=20260824)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if any(name != "importance" for name in args.plots) and args.input is None:
        parser.error("--input is required for drivers and teams; run inference with --export-features first")
    try:
        from spn.plotting import generate_plots
    except ImportError as error:
        parser.error(f"plot dependencies are unavailable: {error}. Install requirements-plots.txt")
    try:
        result = generate_plots(
            args.output, input_source=args.input, features_source=args.features,
            plots=args.plots, formats=args.formats, dpi=args.dpi,
            terminal_weight=args.terminal_weight, bootstrap_iterations=args.bootstrap_iterations,
            random_state=args.random_state, overwrite=args.overwrite,
        )
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as error:
        parser.error(str(error))
    print(json.dumps({"output": str(args.output.resolve()), "plots": result["plots"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
