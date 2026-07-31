# HPN V4 — High-Press Analysis

This is a code-only, notebook-first implementation of the HPN V4 workflow.
It contains no raw data, derived tables, trained models, or season-level results.

## Quick start

From the repository root, create an environment with Python 3.10+ and install
the public dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q tests
jupyter lab
```

The test suite uses synthetic StatsBomb-schema fixtures only; it does not
download, require, or expose match data. In Jupyter, open and run the notebooks
in the S1-to-S5 order below.

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

The supplied V4-top-15 schema is a methodological default. Models, metrics and
team rankings are always rebuilt from the user-provided data.

## Defaults, calibration, and user choices

The public notebooks expose the final-main defaults while keeping each choice
editable:

- Pressure definition: player distance/spread `6.4 m / 2.2 m`, with boundary
  distance/spread `3.2 m / 1.1 m`; Stage 1 uses a high-pressure threshold of
  `0.65`, own-half limit `x < 60`, and a maximum sequence gap of `5 s`.
- Features: the proposed model table starts from the final `V4-top-15` schema.
  Users may exclude or replace features after inspecting the diagnostics.
- Model: unweighted XGBoost is supplied with the final-main M1 parameters
  (778 trees, depth 7, learning rate 0.02327, and the persisted regularisation
  settings). S4 also compares multinomial logistic regression and a
  class-balanced XGBoost alternative on the user's matches.
- Calibration: `raw` is the default portable probability layer. S4 provides a
  nested, match-grouped OOF comparison of raw and isotonic probabilities. Use
  isotonic only if that comparison improves the chosen probability score for
  the user's labelled data; the fitted bundle records its calibration scope.

When changing pressure parameters, use the same `PRESSURE_PARAMS` values in
S1, S2, and S3. The public workflow deliberately exposes these objects rather
than hard-coding a competition, season, or research choice.

## Data and parameter flexibility

The default network definitions and parameter values reproduce the main
implementation. Paths, match identifiers, event identifiers, and competition
or season names are never hard-coded, so the workflow can be run on any paired
StatsBomb-schema event and 360 directories.

For S2, `event["location"]` is the authoritative carrier position. If a custom
event feed omits that field, the freeze-frame actor location is used as a
fallback. Voronoi cells are clipped to the supplied StatsBomb `visible_area`;
if a custom 360 feed omits or cannot parse that optional polygon, pitch-only
clipping is used instead.

For strict reproduction of the main V4 table, the S3 feature
`carrier_x_norm` is the one deliberate legacy exception: it is read from the
attack-normalised freeze-frame actor and clipped to the pitch range. A feature
row without a freeze-frame actor is therefore not retained. Event files are
sorted by their native `index`, and incoming-ball lookup stops at the first
possession boundary.

Pressure parameters may be supplied explicitly:

```python
from src.hpn_network import PressureParams, build_hpn_network

params = PressureParams(
    player_distance=6.4,
    player_sd=2.2,
    boundary_distance=3.2,
    boundary_sd=1.1,
)
network = build_hpn_network(event, frame, params=params)
```

The displayed values are the main-pipeline defaults. Public users may change
them for their own data or sensitivity analysis. When strict reproduction of
the main specification is required, the boundary distance and spread should
remain one half of their corresponding player values.

Stage-1 labels use the same anchor-level outcome state machine as the main
implementation. Later terminal events may provide evidence for assigning an
anchor's realised outcome, but they are neither labelled anchors nor prediction
targets. High-pressure `Miscontrol` events remain part of a sequence but are not
labelled anchors; labels are assigned to `Pass`, `Carry`, and `Dribble`
candidates (and to `Shot` if it is admitted by a custom detector).
The state machine follows the native StatsBomb possession transition and
records `state`, `outcome_tag`, `terminal`, direction fields, resolution fields,
and the `censored` indicator. The model modules retain only
`success`/`neutral`/`fail`, so censored anchors are not used for fitting.

The published defaults reproduce the main label specification, while numerical
cutoffs remain configurable:

```python
from src.data_pipeline import LabelConfig, SequenceConfig, build_sequences

labels_config = LabelConfig(
    min_displacement_m=3.0,
    long_switch_m=25.0,
    clearance_high_pass_m=35.0,
    receiver_relief_threshold=0.65,
)
sequences, labels = build_sequences(
    EVENTS_DIR,
    F360_DIR,
    SequenceConfig(),
    label_config=labels_config,
)
```

## Modules

- data_pipeline.py: paired-data loading, detection and labels.
- hpn_network.py: pressure network construction and S2 figures.
- hpn_features.py: S3/V4 feature construction.
- hpn_model.py: grouped modelling, calibration and robustness helpers.
- press_analysis.py: dynamic S5 summaries.

S5 treats the team in possession at each labelled anchor as the pressed team
and attributes the pressure to its opponent. Sequence identifiers are scoped
by both `match_id` and `seq_id`, so repeated sequence numbers across matches do
not collapse team totals. The public bundle defaults to raw probabilities;
isotonic should be enabled only after grouped validation on the user's own
labelled data, and its fit-scope metadata should be retained with the bundle.

StatsBomb data remains subject to its own licence; the MIT licence in this
repository covers code only.
