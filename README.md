# Structural Pressure Network

This repository contains the compact public release of the Structural
Pressure Network (SPN). It accepts paired StatsBomb-schema event and 360 JSON,
detects high-pressure sequences, assigns realised sequence labels, builds the
current top-19 SPN representation, and scores every complete anchor with the
frozen sequence-weighted XGBoost model.

It also provides optional feature-importance, feature-change/transition-value,
and team pressure-intensity/efficiency plots. It does not contain training,
feature selection, calibration fitting, notebooks, official league standings,
match results, or raw match data. The World Cup efficiency/win-rate plot is not
included.

## Files

```text
.
├── run.py
├── plot.py
├── requirements.txt
├── requirements-plots.txt
├── model/
│   ├── model.ubj
│   └── model_config.json
└── spn/
    ├── data_processing.py
    ├── feature_engineering.py
    ├── _network.py
    ├── model.py
    ├── output.py
    ├── analysis.py
    └── plotting.py
```

- `data_processing.py`: JSON loading, pressure-event detection, semantic
  sequence construction, and sequence/anchor labels.
- `feature_engineering.py`: SPN, Voronoi, Delaunay, passing, boundary, incoming
  ball, and temporal features.
- `_network.py`: private geometry and network-construction implementation used
  by the two data-processing modules.
- `model.py`: native XGBoost artifact loading, hash verification, feature
  contract checks, and raw three-class probabilities.
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
fields. `visible_area` is optional; pitch-only clipping is used when it cannot
be parsed.

This release supports any match, competition, or season that follows those
semantics. A different provider's JSON must first be mapped to the StatsBomb
event/360 contract. Version 1 is an offline completed-match pipeline, not a
live streaming service.

## Run one match

```powershell
python run.py `
  --events events\123456.json `
  --three-sixty three_sixty\123456.json `
  --output output\123456
```

`--match-id` is optional. Without it, the event filename stem is used.

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
plots. The existing return structure remains unchanged.

The two sources may also be already-decoded Python lists. In that case a
`match_id` must be supplied.

## Output

Each match writes:

```text
output/123456/
├── result.json
├── predictions.csv
└── sequences.csv
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
and anchor predictions in a machine-readable envelope.

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

Single-match inference supports the same `--export-features` flag. Each match
adds `features.csv` and `features_metadata.json`; the original three output
files keep their existing schemas. Feature CSVs contain the four anchor keys
and exactly the 19 model features, without observed outcomes or future label
evidence. Metadata records model/config hashes and the feature file hash.

`plot.py` accepts a single-match directory, a batch directory, or a predictions
CSV. It finds `features.csv` beside each predictions file. For older outputs,
provide `--features path/to/features.csv` or re-run inference with
`--export-features`; probabilities alone cannot recover pressure or geometric
features. An explicit feature CSV must contain the four anchor keys and all
19 model features. CSV identifiers are read as strings.

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

- **Feature importance:** normalized mean split gain from the frozen top-19
  XGBoost model; it remains the same when only the input matches change.
- **Feature changes and v_t:** Pearson associations between changes in the 14
  static model features and the next-anchor change in `p_success - p_fail`,
  plus the joint linear-regression R-squared. It is data-dependent descriptive
  association, not a second model-only importance measure. The five temporal
  features are not differenced again. Constant-feature correlations are left
  undefined in the table and omitted from bars.
- **Team intensity and efficiency:** teams are taken from the output's
  `press_team` field, without a fixed team list, season table, or match count.
  Intensity is the sum of `P_total` across eligible anchors divided by matches
  in which that team has eligible sequences. Efficiency is the team mean
  sequence value relative to the sequence-weighted mean of the input dataset,
  multiplied by 100. Bubble area represents evaluated sequences and colour
  represents their observed success rate.

The sequence value is the sum of forward probability-value changes plus
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

To run the plotting/analysis tests after installing plotting dependencies:

```bash
python -m unittest discover -s tests -v
```

## Frozen model contract

The public model is `current-spn-top19-xgb-raw`:

- M1 sequence-weighted XGBoost;
- raw `fail / neutral / success` probabilities;
- 19 features in the exact order recorded in `model/model_config.json`;
- native XGBoost UBJ artifact with a verified SHA-256 hash.

For model parameters and a categorized description of all 19 inputs, see
[the model and feature guide](spn/MODEL.md).

Raw probabilities support general cross-season inference and ranking; absolute season-level success rates or efficiency estimates should use an optional downstream calibrator fitted only on previously completed matches.

The model configuration also freezes pressure, sequence, label, pitch, passing,
boundary, and edge-threshold parameters. These values are part of the learned
representation. Changing them while retaining the supplied frozen model is not
supported; parameter research requires rebuilding features and retraining.

Observed labels are returned for retrospective analysis but are not passed to
the model. Only the 19 fields named by the model manifest are used to produce
probabilities.

StatsBomb data remains subject to its own licence. The MIT licence here covers
the released code and model-pipeline packaging only.
