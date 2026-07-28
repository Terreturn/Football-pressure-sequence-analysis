# HPN V4 — High-Press Analysis

This is a code-only, notebook-first implementation of the HPN V4 workflow.
It contains no raw data, derived tables, trained models, or season-level results.

## Input

Provide two directories containing StatsBomb-schema JSON files with the same
match identifiers:

    EVENTS_DIR/<match_id>.json
    F360_DIR/<match_id>.json

Set paths before opening the notebooks:

    $env:EVENTS_DIR = "D:\my-data\events"
    $env:F360_DIR = "D:\my-data\three-sixty"
    $env:OUTPUT_DIR = "D:\my-hpn-output"

Run the notebooks in order: S1 detects and labels sequences, S2 selects random
valid networks from that output, S3 builds V4 features, S4 guides selection,
training and robustness, and S5 scores and profiles the teams present in the
user's data.

The supplied V4-top-16 schema is a methodological default. Models, metrics and
team rankings are always rebuilt from the user-provided data.

## Modules

- data_pipeline.py: paired-data loading, detection and labels.
- hpn_network.py: pressure network construction and S2 figures.
- hpn_features.py: S3/V4 feature construction.
- hpn_model.py: grouped modelling, calibration and robustness helpers.
- press_analysis.py: dynamic S5 summaries.

StatsBomb data remains subject to its own licence; the MIT licence in this
repository covers code only.

