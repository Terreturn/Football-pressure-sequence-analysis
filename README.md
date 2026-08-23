# Structural Pressure Network: public inference pipeline

This repository contains the compact inference-only release of the Structural
Pressure Network (SPN). It accepts paired StatsBomb-schema event and 360 JSON,
detects high-pressure sequences, assigns realised sequence labels, builds the
current top-19 SPN representation, and scores every complete anchor with the
frozen sequence-weighted XGBoost model.

It does not contain training, feature selection, calibration fitting,
ablation, robustness, notebooks, season rankings, or raw match data.

## Files

```text
.
├── run.py
├── requirements.txt
├── model/
│   ├── model.ubj
│   └── model_config.json
└── spn/
    ├── data_processing.py
    ├── feature_engineering.py
    ├── _network.py
    ├── model.py
    └── output.py
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

## Frozen model contract

The public model is `current-spn-top19-xgb-raw`:

- M1 sequence-weighted XGBoost;
- raw `fail / neutral / success` probabilities;
- 19 features in the exact order recorded in `model/model_config.json`;
- native XGBoost UBJ artifact with a verified SHA-256 hash.

The model configuration also freezes pressure, sequence, label, pitch, passing,
boundary, and edge-threshold parameters. These values are part of the learned
representation. Changing them while retaining the supplied frozen model is not
supported; parameter research requires rebuilding features and retraining.

Observed labels are returned for retrospective analysis but are not passed to
the model. Only the 19 fields named by the model manifest are used to produce
probabilities.

StatsBomb data remains subject to its own licence. The MIT licence here covers
the released code and model-pipeline packaging only.
