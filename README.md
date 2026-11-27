# Structural Pressure Network

This repository contains the compact public release of the Structural
Pressure Network (SPN). It accepts paired StatsBomb-schema event and 360 JSON,
detects high-pressure sequences, assigns realised sequence labels, builds the
26-feature SPN representation specified in `model/model_config.json`, and
scores the anchors of resolved, fully representable sequences with the temperature-calibrated XGBoost model. A separate positioning task scores eligible recorded pressers.

The sample notebook covers loading public data, processing, model inference,
evaluation, team expected press value, and individual positioning. [Sample workflow](tests/Sample_workflow.ipynb)

It also provides optional feature-importance, feature-change/transition-value,
and team pressure-intensity/efficiency plots. It does not contain training,
feature selection, calibration fitting, official league standings,
match results, or raw match data.

## Contributors

This project includes code contributions from:

- [Brennan Klein (@jkbren)](https://github.com/jkbren)
- [Maddalena Torricelli (@maddaleona)](https://github.com/maddaleona)

Their contributions were integrated and uploaded by [Terreturn](https://github.com/Terreturn).

## Files

```text
.
├── run.py
├── plot.py
├── requirements.txt
├── requirements-plots.txt
├── tests/
│   ├── data_checks.py
│   └── Sample_workflow.ipynb
├── model/
│   ├── README.md
│   ├── model.ubj
│   ├── temperature.json
│   ├── regain_0.ubj … regain_4.ubj
│   ├── regain_folds.json
│   └── model_config.json
└── spn/
    ├── MODEL.md
    ├── data_processing.py
    ├── feature_engineering.py
    ├── _network.py
    ├── model.py
    ├── positioning.py
    ├── _core/
    ├── output.py
    ├── analysis.py
    └── plotting.py
```

- `data_processing.py`: JSON loading, pressure-event detection, semantic
  sequence construction, and sequence/anchor labels.
- `feature_engineering.py`: SPN, Voronoi, Delaunay, passing, boundary, incoming
  ball, and temporal features.
- `_network.py` and `_core/`: pressure, passing, Voronoi and shape-graph calculations.
- `positioning.py`: recorded-presser eligibility, fold selection and nearby alternative scoring.
- `model.py`: native XGBoost artifact loading, hash verification, feature
  contract checks, and temperature-calibrated three-class probabilities.
- `output.py`: compact JSON and CSV outputs.
- `analysis.py`: sequence valuation, team aggregation, feature-change
  associations and bootstrap intervals using public outputs.
- `plotting.py`: the three plots and PNG/JPEG/PDF/SVG export; no bottom notes.

## Environment

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS, XGBoost requires the OpenMP runtime, available through `brew install libomp`.

## Sample workflow

The notebook uses Hudl's `statsbombpy` package to load public event and 360
freeze-frame data. The default selection is the 2022 World Cup. Change
`COMPETITION_ID` and `SEASON_ID` in the first code cell to select another
competition and season with event and 360 coverage.
[Sample workflow](tests/Sample_workflow.ipynb),
[StatsBomb Open Data](https://github.com/statsbomb/open-data),
[statsbombpy](https://github.com/hudl/statsbombpy)

Install the notebook dependencies and launch Jupyter from the repository root.

```bash
python -m pip install -r requirements.txt jupyterlab statsbombpy==1.22.0
python -m jupyterlab tests/Sample_workflow.ipynb
```

Run the cells in order. An internet connection is needed to fetch the data.
Responses are cached in memory for the current session, and match objects and
result tables remain available in the notebook kernel.

| Step | Result |
| --- | --- |
| Select and validate matches | `matches_table` stores the catalogue. Checks in `tests/data_checks.py` identify `matches_for_processing`. |
| Process event and 360 data | `sequences_table` and `labels_table` store pressure anchors and sequence outcomes. `processing_log` records per-match processing counts. |
| Build networks and model inputs | `model_input_table` contains four anchor identifiers and the 26 configured features in model order. `feature_log` records sequence retention. |
| Apply SPN Classifier | `model_output_table` stores `p_fail`, `p_neutral`, `p_success`, `predicted_class`, and `confidence` for each modelled anchor. |
| Evaluate predictions | `evaluation_summary` displays overall and per-class metrics against resolved rule-derived labels. Each sequence contributes total weight one, with unweighted anchor accuracy also reported. |
| Summarise teams | `team_summary` averages `p_success - p_fail` at sequence onset, with one vote per sequence. |
| Score positioning | `position_scores` holds eligible recorded pressures for the example fixture. |

## Input

The input is a complete paired StatsBomb-schema match:

```text
events/123456.json
three_sixty/123456.json
```

The event file must be a JSON array. Events require their native `id`, `index`,
`period`, `timestamp`, `type`, team/possession fields, and the usual
type-specific payload such as `pass.end_location` or `carry.end_location`.

The 360 file must be a JSON array containing `event_uuid` and `freeze_frame`.
Each visible player uses StatsBomb `location`, `teammate`, `actor`, and `keeper`
fields. Sequence features that require a valid camera-visible polygon are missing when it is unavailable. Positioning requires a valid visible polygon. Frames normally join on event UUID. Below 95% UUID overlap the paper linker uses the recorded actor location and event order.

This release supports any match, competition, or season that follows those
semantics. A different provider's JSON must first be mapped to the StatsBomb
event/360 contract. This package processes completed matches offline.

## Run one match

```powershell
python run.py `
  --events events\123456.json `
  --three-sixty three_sixty\123456.json `
  --output output\123456
```

`--match-id` is optional. Without it, the event filename stem is used.

Choose `--task sequence` (default), `--task positioning` or `--task both` in single-match or batch mode. Supply the original match ID explicitly when filenames have been renamed. An in-memory match always needs an ID.

```bash
python run.py --events events/3869685.json --three-sixty three_sixty/3869685.json --task both --output output/3869685
```

## Run a directory

Files are paired by their common filename stem.

```powershell
python run.py `
  --events-dir events `
  --three-sixty-dir three_sixty `
  --output output
```

## Python use

```python
from run import run_pipeline

result = run_pipeline(
    events_source="events/123456.json",
    three_sixty_source="three_sixty/123456.json",
    output_dir="output/123456",
)

print(result["status"])
print(result["audit"])
```

Pass `export_features=True` to also write the inputs needed by the data-dependent
plots. The returned result has the same fields with either setting.

The two sources may also be already-decoded Python lists. In that case a
`match_id` must be supplied.

## Output

The sequence task writes these files for each match.

```text
output/123456/
├── result.json
├── predictions.csv
├── sequences.csv
└── teams.csv
```

`predictions.csv` contains one row per modelled anchor, including:

```text
match_id, seq_id, ev_pos, anchor_event_id,
period, timestamp, event_type, pressed_team, press_team,
observed_outcome, censored, evaluation_eligible,
p_fail, p_neutral, p_success, predicted_class, confidence,
model_id, model_version
```

`sequences.csv` contains one compact row per detected pressure sequence.
`result.json` combines model metadata, processing audit counts, sequence rows,
and anchor predictions in a machine-readable envelope. `teams.csv` reports the number of sequence onsets and mean onset `p_success - p_fail` for each pressing team. For a league-season comparison, subtract the mean over all onsets in that league-season.

The positioning task writes `pressures.csv` with match, event, player and team IDs, the observed five-second regain label `e2`, the number of valid alternatives, both predicted probabilities and `position_quality_pp`. Only recorded `Pressure` actors with an identified carrier, qualifying high pressure and a feasible nearby alternative enter the table. The Python equivalent is `spn.score_positions(match)`.

If no high-pressure sequence is found, this is not treated as malformed input:
the pipeline writes a valid result with `status="no_sequences"` and empty
tables. Missing or unusable frames are never fabricated. If one anchor in a
sequence cannot be represented, the complete sequence is excluded and counted
in the audit.

## Generate the three plots

Install the optional plotting dependencies:

```bash
python -m pip install -r requirements-plots.txt
```

Feature importance uses the frozen model alone and does not require match data:

```bash
python plot.py --plots importance --output plots/importance
```

For all three plots, first export features during the normal inference run:

```bash
python run.py --events-dir events --three-sixty-dir three_sixty --output output --export-features
python plot.py --input output --output plots
```

Single-match inference supports the same `--export-features` flag. With this
flag, each match writes `features.csv` and `features_metadata.json` alongside
`result.json`, `predictions.csv`, and `sequences.csv`. Feature CSVs contain the
four anchor keys and exactly the 26 model features. They contain no observed
outcomes or future label evidence. Metadata records model/config hashes and
the feature file hash.

`plot.py` accepts a single-match directory, a batch directory, or a predictions
CSV. It finds `features.csv` beside each predictions file. If that file is
absent, provide `--features path/to/features.csv` or re-run inference with
`--export-features`; probabilities alone cannot recover pressure or geometric
features. An explicit feature CSV must contain the four anchor keys and all
26 model features. CSV identifiers are read as strings.

Select individual plots with `--plots importance drivers teams`, and formats
with `--formats png jpeg pdf svg`. The default is all three plots in all four
formats. Output is grouped into `PNG/`, `JPEG/`, `PDF/`, `SVG/`, and `tables/`.
`manifest.json` records model identity, input coverage, definitions, numerical
audits, output hashes, and any skipped analyses. Use `--overwrite` to replace
existing outputs.

Python entry point:

```python
from spn.plotting import generate_plots

report = generate_plots("plots", input_source="output")
```

### What the plots measure

- **Feature importance:** normalized mean split gain from the frozen sequence
  XGBoost model; it remains the same when only the input matches change.
- **Feature changes and v_t:** Pearson associations between changes in the non-difference
  model features and the next-anchor change in `p_success - p_fail`,
  plus the joint linear-regression R-squared. It is data-dependent descriptive
  association, not a second model-only importance measure. The four temporal
  features are not differenced again. Constant-feature correlations are left
  undefined in the table and omitted from bars.
- **Team intensity and efficiency:** teams are taken from the output's
  `press_team` field, without a fixed team list, season table, or match count.
  Intensity is the sum of `P_total` across eligible anchors divided by matches
  in which that team has eligible sequences. Efficiency is the team mean
  sequence value relative to the sequence-weighted mean of the input dataset,
  multiplied by 100. Bubble area represents evaluated sequences and colour
  represents their observed success rate.

This optional composite-efficiency plot uses a different statistic from the paper’s onset value in `teams.csv`. Its sequence value is the sum of forward probability-value changes plus
`terminal_weight` times the terminal component. That component is the last
`p_success` for success, zero for neutral, and minus the last `p_fail` for fail.
Each eligible sequence has equal weight. Censored and incomplete sequences are
excluded. No additional calibration or training weights are applied.

Defaults: terminal weight 0.5, 2,000 bootstrap iterations, random seed 20260824,
300 DPI. Override these with `--terminal-weight`, `--bootstrap-iterations`,
`--random-state`, and `--dpi`. Intervals resample matches within each team and
recompute the dataset mean. Match counts and the zero reference refer to the
supplied eligible data, not automatically to the entire season. Matches with
no eligible sequence are excluded from this intensity denominator.

Valid empty/censored input and insufficient transitions produce explicit
`skipped` entries instead of invented values. Malformed input, mixed model
identities, mismatched features/probabilities, or incomplete exported sequences
raise errors. When `sequences.csv` is available, its anchor counts are checked
to detect truncated sequence tails. Standalone CSV users must supply complete
sequences; continuity alone cannot prove that the final anchor is present.

## Frozen model contract

The model is named `spn`. Both tasks accept the same 26 feature names in their stored order. The sequence model predicts fail, neutral and success; the positioning model predicts a controlled regain. Training labels and future outcomes are excluded from both sets of predictors.

The reference implementation uses the archived shape graph for sequences and its teamwise correction for positioning. The shape-graph attribution and license are included under `spn/_core/vendor/`.

[Model artifacts](model/README.md) · [Ordered feature definitions](spn/MODEL.md) · [Shape-graph license](spn/_core/vendor/LICENSE)

StatsBomb data remains subject to its own licence. The MIT licence here covers
the released code and model-pipeline packaging only.
