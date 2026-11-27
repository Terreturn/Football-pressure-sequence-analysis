# Structural Pressure Network

This repository contains the compact public release of the Structural
Pressure Network (SPN). It accepts paired StatsBomb-schema event and 360 JSON,
detects high-pressure sequences, assigns realised sequence labels, builds the
26-feature SPN representation specified in `model/model_config.json`, and
scores the anchors of resolved, fully representable sequences with the temperature-calibrated XGBoost model. A separate positioning task scores eligible recorded pressers.

The sample notebook covers loading public data, processing, model inference,
evaluation, team expected press value, and individual positioning. [Sample workflow](tests/Sample_workflow.ipynb)

The plotting scripts reproduce Figures 1–4 of the paper from the network schematic
and the published summary tables. [Draw the paper figures](#draw-the-paper-figures)

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
├── plot_data/
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
    ├── _paper_figures/
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
- `plotting.py` and `_paper_figures/` draw the four paper figures and export them as vector PDFs and 425 DPI PNGs.

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

Pass `export_features=True` in Python or `--export-features` on the command line
to save `features.csv` and `features_metadata.json` for each match. The CSV contains
the four anchor identifiers and 26 model inputs. The metadata records their order
and the model, configuration and feature-file checksums.

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

## Draw the paper figures

From the repository root, install the plotting dependencies and render all four figures.

```bash
python -m pip install -r requirements-plots.txt
python plot.py --output plots/paper
```

To render selected figures, pass their paper numbers, for example
`python plot.py --figures 2 4 --output plots/selected`. Use `--overwrite` to replace
existing figure files. In Python, call `spn.plotting.generate_plots` with the same
selection.

```python
from spn.plotting import generate_plots

report = generate_plots("plots/selected", plots=(2, 4), overwrite=True)
```

| Figure | Panels | Inputs |
| --- | --- | --- |
| 1 | A high press, its network layers, the combined structural pressure network and a sequence prediction schematic | The renderer defines the example pitch geometry and constructs the network. |
| 2 | Success prediction AUC, feature contributions from exact TreeSHAP, and the change in log loss from nine to 26 inputs | `figure2_auc.csv`, `figure2_shap.csv`, `figure2_network_gains.csv` |
| 3 | Club expected press value versus expected-goal difference in the 2025/26 Premier League, and rank correlations across 29 league-seasons | `figure3_epl_teams.csv`, `figure3_season_intervals.csv` |
| 4 | A presser's feasible alternative positions, club positioning scores versus season points, and the ten highest eligible player means | `high_clubs.csv`, `high_players.csv`, `club_interval.json`, club labels from `figure3_epl_teams.csv`, and the supplied caption |

Figures 2–4 use the paper's saved estimates in `plot_data/`. A fresh checkout has
all the inputs needed to reproduce the figures. To use prepared tables in the same
format from another directory, pass `--tables PATH`. [Plot tables](plot_data/)

Each figure produces a vector PDF and a 425 DPI PNG in the output directory.
Figure 3 also produces a manuscript-size PNG preview. Figure 4 writes its caption
to a LaTeX file and the ten displayed players to a CSV. Eligibility for the player panel is
at least 15 eligible high presses and 1,350 league minutes.

The paper's reference PNG checksums are in `plot_data/reference_png_hashes.json`.
The original rendering environment used Matplotlib 3.9.4 and Pillow 11.3.0.
Library and font-rendering versions can affect the image bytes.

## Frozen model contract

The model is named `spn`. Both tasks accept the same 26 feature names in their stored order. The sequence model predicts fail, neutral and success; the positioning model predicts a controlled regain. Training labels and future outcomes are excluded from both sets of predictors.

The reference implementation uses the archived shape graph for sequences and its teamwise correction for positioning. The shape-graph attribution and license are included under `spn/_core/vendor/`.

[Model artifacts](model/README.md) · [Ordered feature definitions](spn/MODEL.md) · [Shape-graph license](spn/_core/vendor/LICENSE)

StatsBomb data remains subject to its own licence. The MIT licence here covers
the released code and model-pipeline packaging only.
