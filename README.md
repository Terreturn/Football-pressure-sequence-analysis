# HPN — High-Press Analysis

A four-stage pipeline that detects **high-press sequences** from StatsBomb 360 data, turns
them into a tabular ML problem, trains a calibrated outcome model, and runs season-level
pressing analysis (VAEP-style step valuation + team profiling).

## Pipeline

| Stage | What | File | Type |
|---|---|---|---|
| **1** | raw events + 360 JSON → high-press sequences + labels | `s1_build_sequences.py` | script |
| **2** | HPN construction & illustration figures | `s2_hpn_construction.ipynb` | notebook |
| **3** | sequences → carrier-centric ML feature table | `s3_build_features.py` | script |
| **4–5** | train ML (model comparison) + season analysis (VAEP, team profiling) | `s45_ml_pipeline.ipynb` | notebook |

Scripts do the batch work; notebooks are where results are **shown and explained**
(HPN figures, model comparison, calibration, VAEP plots — saved inline). The detection
algorithm is specified in [`high_press_detection_spec.md`](high_press_detection_spec.md).

Every command below uses a Python env with the packages in `requirements.txt`. On Windows
prefix `$env:PYTHONIOENCODING="utf-8"` so non-ASCII log output prints cleanly.

---

## Running each stage

### Stage 1 — `s1_build_sequences.py`
**What.** High-press detection (event ∈ {Pass, Carry, Miscontrol, Dribble}, not a set piece,
carrier in own half `x<60`, `P_total>0.65`, ≥1 defender ≤6.4 m; sequences = same period /
same carrier team / consecutive gap ≤5 s) + a terminal-first label state machine. Coordinates
are possession-attack-normalised, so **no matches.json is needed**. Events ↔ 360 are paired by
`match_id`; unpaired / non-overlapping files are reported and skipped.

**Usage.**
```bash
export EVENTS_DIR=/data/events      # folder of event *.json
export F360_DIR=/data/360           # folder of 360 *.json   (independent of EVENTS_DIR)
python s1_build_sequences.py
# quick test on the first N matches:  S1_LIMIT=5 python s1_build_sequences.py
```
```powershell
# Windows PowerShell
$env:EVENTS_DIR="C:\data\events"; $env:F360_DIR="C:\data\360"
python s1_build_sequences.py
```
(Or set one root `STATSBOMB_DIR` whose subfolders are `events/` and `360/`.)

**Output** (written next to the script):
- `sequences.csv` — one row per **high-press frame** (type, P_total, nearest_def_m, …).
- `labels.csv` — one row per **anchor** (Pass/Carry/Dribble): `outcome_tag` ∈
  {success, fail, neutral, censored}, `terminal`, ball-direction `dir_*`, …
  (rename with `S1_OUT_SEQ` / `S1_OUT_LAB`).

### Stage 2 — `s2_hpn_construction.ipynb`
**What.** Builds and draws the HPN (pressure network) for an example freeze frame — the
spatial illustration of how pressure, passing lanes and boundary traps are modelled.

**Usage.** Open in Jupyter / VSCode **from the repo folder** and run the cells (paths come from
`HPN_DIR` / `STATSBOMB_DIR`, see below).

**Output.** HPN figures rendered **inline in the notebook**.

### Stage 3 — `s3_build_features.py`
**What.** For every labelled anchor, reads its 360 freeze frame and computes a **19-dim
carrier-centric feature vector** (pressure on carrier, nearest defender, pass openness,
teammate pressure, boundary trap, local densities, …). Keeps only `success/fail/neutral`
anchors; one row per anchor.

**Usage.**
```bash
export LABELS_CSV=/path/to/labels.csv        # e.g. a Stage-1 output
export EVENTS_DIR=/data/events
export F360_DIR=/data/360
export FEAT_PARQUET=/path/to/features.parquet
python s3_build_features.py
```

**Output.** `features.parquet` (`FEAT_PARQUET`, default `hpn_carrier_features.parquet`):
**24 columns** = 19 features + `match_id, seq_id, ev_pos, outcome_tag, terminal`.

### Stage 4–5 — `s45_ml_pipeline.ipynb`
**What.** Loads the feature parquet, **adds 7 temporal first-difference features (Δt=1) + 4
incoming-ball (`ball_in`) features → 30 features**, trains and compares models (Logistic vs
XGBoost balanced vs XGBoost tuned), validates (GroupKFold CV + calibration / reliability),
then runs the VAEP-style step valuation and the season team analysis (intensity landscape,
efficiency ranking, pressing-style fingerprint).

**Usage.** Open in Jupyter / VSCode **from the repo folder** and **Run All**, or headless:
```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 s45_ml_pipeline.ipynb
```

**Output.** A trained model `hpn_xgb_outcome_ballin.joblib` + metrics, calibration/reliability
plots, VAEP and team figures **embedded inline** in the notebook.

---

## Data
This repository is **code only** — bring your own StatsBomb event + 360 JSON (e.g. the public
[StatsBomb Open Data](https://github.com/statsbomb/open-data)) and point `EVENTS_DIR` /
`F360_DIR` at them. The pipeline then produces sequences/labels (s1) → features (s3) →
trained model + analysis (s45). The committed notebooks keep the **saved outputs** of a prior
run so the results are visible without re-running.

## Setup & paths
No absolute paths are hard-coded; locations resolve from environment variables.

| Env var | Meaning | Default |
|---|---|---|
| `EVENTS_DIR` | events JSON folder | `STATSBOMB_DIR/events` (s1) · `…/events/events` (s3) |
| `F360_DIR` | 360 freeze-frame JSON folder | `STATSBOMB_DIR/360` (s1) · `…/three_sixty/three_sixty` (s3) |
| `STATSBOMB_DIR` | data root (only used to build the two defaults above) | the repo's **parent** folder |
| `LABELS_CSV` / `FEAT_PARQUET` | s3 input labels / output parquet | `labels.csv` (repo) / `hpn_carrier_features.parquet` |
| `S1_OUT_SEQ` / `S1_OUT_LAB` / `S1_LIMIT` | s1 output names / debug match limit | `sequences.csv` / `labels.csv` / all |
| `HPN_DIR` | repo root (notebooks) | the notebook's working directory |

`EVENTS_DIR` and `F360_DIR` are **independent** — set them to any two folders (they need not
share a parent). Run notebooks from the repo folder (so `os.getcwd()` resolves the repo) or set
`HPN_DIR`. Scripts find the repo and `src/` via their own location.

## Layout
```
hpn_high_press/
├─ s1_build_sequences.py        stage 1   (JSON → sequences/labels)
├─ s2_hpn_construction.ipynb    stage 2   (HPN illustration)
├─ s3_build_features.py         stage 3   (sequences → 19-feature table)
├─ s45_ml_pipeline.ipynb        stage 4-5 (train + season analysis)
├─ src/
│   ├─ pressure_distance_v2.py  pressure model (logistic kernels, total pressure)
│   └─ voronoi_pitch.py         pitch / Voronoi helpers (stage 2)
├─ high_press_detection_spec.md detection algorithm spec
├─ requirements.txt
├─ .gitignore                   ignores raw data, generated CSV/parquet/models/figures
└─ README.md
```

## Environment
Python 3.10+ with: `pandas, scikit-learn, xgboost, pyarrow, scipy, shapely, matplotlib,
numpy`. Install with:
```bash
pip install -r requirements.txt
```
