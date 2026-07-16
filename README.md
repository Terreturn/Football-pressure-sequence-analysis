# HPN — High-Press Analysis

A pipeline that detects **high-press sequences** from StatsBomb event + 360 data, turns
them into a tabular ML problem, trains a **calibrated outcome model**, and runs
season-level pressing analysis (VAEP-style step valuation + team profiling).

Works with **any StatsBomb events + 360 dataset** (e.g. the public
[StatsBomb Open Data](https://github.com/statsbomb/open-data)) — no competition,
season, or local folder layout is assumed; all locations come from environment
variables.

## Pipeline

| Stage | What | File | Type |
|---|---|---|---|
| **1** | raw events + 360 JSON → high-press sequences + labels | `scripts/s1_build_sequences.py` | script |
| **2** | HPN construction & illustration figures | `notebooks/s2_hpn_construction.ipynb` | notebook |
| **3** | sequences → carrier-centric ML feature table | `scripts/s3_build_features.py` | script |
| **4** | model training: comparison, tuning, de-collinearisation (30→17), calibration → deployed bundle | `notebooks/s4_model_training.ipynb` | notebook |
| **5** | season analysis: efficiency + style, figures & tables for ANY held-out season | `notebooks/s5_press_analysis.ipynb` | notebook |
| 4′ | CLI mirror of Stage 4's selected model (headless retrain) | `scripts/train_calibrated.py` | script |
| — | fit a per-season isotonic calibration layer for a NEW season | `scripts/calibrate_season.py` | script |
| 5′ | CLI mirror of Stage 5 (headless season analysis) | `scripts/apply_season.py` | script |
| — | calibration diagnostics (deployed model vs baselines, reliability curves) | `scripts/compare_calibration.py` | script |

Notebooks carry the **narrative** (why this model, how to read each figure); the CLI
mirrors run the **same functions** from `src/` headlessly, so the two can never drift.
The detection algorithm is specified in
[`docs/high_press_detection_spec.md`](docs/high_press_detection_spec.md).

## Workflow

The end-to-end order is **S1 → S3 → S4 → (per-season calibration) → S5**; the CLI
scripts are alternatives to S4/S5, not later stages.

**A. Model a new dataset from scratch**
```
scripts/s1_build_sequences.py → scripts/s3_build_features.py → notebooks/s4_model_training.ipynb
                                                └→ hpn_xgb_outcome_calibrated.joblib
```

**B. Analyse a season (the routine loop)**
```
run S1+S3 on that season's raw JSON
→ scripts/calibrate_season.py    fit the season's isotonic layer (frozen ranker)
→ notebooks/s5_press_analysis.ipynb   point MODEL / FEAT_PARQUET / LABELS_CSV /
  (Run All)                          EVENTS_DIR / SEASON_LABEL — zero code changes
  └→ analysis_out/: efficiency + intensity + step_valuation + drivers (csv),
     4 figures, lineage.txt
```
The **S5 notebook is the pipeline's terminus** — the tables and figures it writes are
the deliverables. `apply_season.py` produces the identical outputs headlessly (cron/CI).

**C. Guard rail.** Scoring matches the model saw in training triggers a lineage
**warning** (run still completes); the training season's trustworthy analysis lives in
S4's OOF sections instead.

Every command below uses a Python env with the packages in `requirements.txt`. On Windows
prefix `$env:PYTHONIOENCODING="utf-8"` so non-ASCII log output prints cleanly.

## The model in one paragraph

Each **anchor** (a Pass/Carry/Dribble under high press) becomes a carrier-centric
feature vector. The builder (`src/hpn_features.py`) emits a **30-column matrix**:
19 static carrier features (Stage 3) + 7 temporal first differences (Δt = 1 within
sequence) + 4 incoming-ball features. The **deployed model** is a tuned, unweighted
XGBoost on the **17-feature de-collinearised subset** (`PRUNED_17`: one representative
per collinear cluster; max VIF 12.3 → 2.3, max |r| 0.95 → 0.61, at ΔAUC −0.0035),
trained on 80% of the training season's matches and isotonically calibrated on the
other 20% (group-aware split). Ranking transfers across seasons, but class base rates
drift — so the ranking model is **frozen** and each new season gets its **own isotonic
layer** (`calibrate_season.py`), fitted on the season's first ~20% of matches.
Downstream analysis consumes **calibrated probabilities**: the per-step press value
v_t = Δp(success) − Δp(fail) (VAEP-style) feeds team efficiency, drivers and
style-fingerprint outputs.

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
python scripts/s1_build_sequences.py
# quick test on the first N matches:  S1_LIMIT=5 python scripts/s1_build_sequences.py
```
```powershell
# Windows PowerShell
$env:EVENTS_DIR="C:\data\events"; $env:F360_DIR="C:\data\360"
python scripts/s1_build_sequences.py
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
python scripts/s3_build_features.py
```

**Output.** `features.parquet` (`FEAT_PARQUET`, default `hpn_carrier_features.parquet`):
**24 columns** = 19 features + `match_id, seq_id, ev_pos, outcome_tag, terminal`.

### Stage 4 — `s4_model_training.ipynb`
**What.** The modelling narrative, end to end: loads the feature parquet, **adds 7
temporal first-difference features (Δt=1) + 4 incoming-ball (`ball_in`) features → 30
features**, trains and compares models (Logistic vs XGBoost balanced vs XGBoost tuned),
validates (GroupKFold CV + OOF calibration / reliability), **de-collinearises 30 → 17**
and selects the deployed model, then trains + isotonically calibrates it (80/20
group-aware split) and saves the bundle **with train/calib lineage**. Contains no team
analysis — that is Stage 5.

**Usage.** Open in Jupyter / VSCode **from the repo folder** and **Run All**
(`RUN_SEARCH=1` re-runs the ~11-min hyper-parameter search; default uses the pinned
result), or headless:
```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/s4_model_training.ipynb
```

**Output.** `hpn_xgb_outcome_calibrated.joblib` (deployed bundle, incl.
`train_matches`/`calib_matches`) + metrics and reliability plots inline.

### Stage 5 — `s5_press_analysis.ipynb`
**What.** The analysis narrative for **any held-out season**: scores every anchor with a
trained bundle, derives the step press-value $v_t$ and step labels, then the efficiency
analysis (regain table + intensity-vs-efficiency quadrant map) and the style analysis
(pressing-style fingerprint + drivers of $v_t$). All computation is imported from
`src/press_analysis.py` — the notebook is the narrated shell.

**Usage.** Season selection is pure environment variables (see the notebook's config
cell): set `MODEL` (the season's bundle), `FEAT_PARQUET`, `LABELS_CSV`, `EVENTS_DIR`,
`SEASON_LABEL` and **Run All**. A **lineage check** warns — without blocking — if any
scored match was in the bundle's training/calibration sets (the training season itself
belongs in Stage 4's OOF sections).

**Output.** Tables + figures inline **and** written to `OUTDIR`
(default `analysis_out/`): `intensity.csv`, `efficiency.csv`, `step_valuation.csv`,
`drivers.csv`, `lineage.txt` + 4 figures.

### CLI mirror of Stage 4 — `train_calibrated.py`
**What.** Headless retrain of the selected model: tuned unweighted XGBoost on the
17 de-collinearised features, 80/20 **group-aware** split (by match), isotonic
calibration on the held-out 20%. Prints before/after reliability. Saves the same
lineage fields as the notebook.

**Usage.**
```bash
export FEAT_PARQUET=/path/to/features.parquet   # Stage-3 output
export LABELS_CSV=/path/to/labels.csv           # Stage-1 output
export EVENTS_DIR=/data/events                  # raw events (ball_in features)
python scripts/train_calibrated.py                      # -> hpn_xgb_outcome_calibrated.joblib
```
The pinned `XGB_PARAMS` come from a grouped randomised search on the original training
season; re-run the notebook's search (cell 14) when training on a very different dataset.

### Per-season calibration — `calibrate_season.py`
**What.** Keeps the deployed ranking model **frozen** and fits a new isotonic layer on a
NEW season's first `N_CALIB` matches (chronological if you provide `MATCHES_JSON`), then
evaluates uncalibrated vs old layer vs new layer on the remaining matches. Use when class
base rates drift across seasons. The base bundle's `train_matches` lineage is carried
over into the per-season bundle.

**Usage.**
```bash
export FEAT_PARQUET=/path/to/new_season_features.parquet
export LABELS_CSV=/path/to/new_season_labels.csv
export EVENTS_DIR=/data/new_season/events
export MATCHES_JSON=/data/new_season/matches.json   # optional, for chronological order
export SEASON_LABEL="2025-26"
python scripts/calibrate_season.py    # -> hpn_xgb_outcome_calibrated_2025-26.joblib
```

### CLI mirror of Stage 5 — `apply_season.py`
**What.** Headless version of the S5 notebook — same functions from
`src/press_analysis.py`, same outputs, same lineage check.

**Usage.**
```bash
export FEAT_PARQUET=/path/to/season_features.parquet
export LABELS_CSV=/path/to/season_labels.csv
export EVENTS_DIR=/data/season/events
export MODEL=hpn_xgb_outcome_calibrated_2025-26.joblib   # the per-season bundle
export SEASON_LABEL="2025/26"
# optional: fingerprint row order (e.g. final league table), one team per line
# export TEAM_ORDER=/path/to/league_table.txt
python scripts/apply_season.py   # -> analysis_out/{intensity,efficiency,step_valuation,drivers}.csv
                         #    + lineage.txt + 4 figures
```

### Lineage check (all Stage-5 entry points)
Every trained bundle records which matches its ranker was **trained** on and which
fitted its **isotonic layer**. When a season is scored, `press_analysis.check_lineage`
compares the scored matches against both sets and **warns without blocking**:

- overlap with `train_matches` → **warning** (scores are in-sample; use S4's OOF
  analysis for the training season);
- overlap with `calib_matches` → **note** (mild — only the monotone calibration map
  saw them);
- no overlap → a green-light confirmation line.

The verdict is persisted to `OUTDIR/lineage.txt`, so any saved analysis remains
auditable later. Old bundles without lineage fields skip the check with a notice.

### Diagnostics — `compare_calibration.py`
**What.** Reliability comparison of the deployed bundle against a Logistic baseline and a
class-balanced XGBoost rival, on (A) the training-season calibration holdout and (B)
optionally a fully held-out second season (set `FEAT_PARQUET_NEW` / `LABELS_NEW` /
`EVENTS_DIR_NEW`). Writes `calibration_compare.csv` + reliability curves.

---

## Data
This repository is **code only** — bring your own StatsBomb event + 360 JSON (e.g. the public
[StatsBomb Open Data](https://github.com/statsbomb/open-data)) and point `EVENTS_DIR` /
`F360_DIR` at them. The pipeline then produces sequences/labels (s1) → features (s3) →
trained model (s4) → per-season calibration (`calibrate_season.py`) → season analysis (s5).
The committed notebooks keep the **saved outputs** of a prior run so the results are visible
without re-running.

## Setup & paths
No absolute paths are hard-coded; locations resolve from environment variables.

| Env var | Used by | Meaning | Default |
|---|---|---|---|
| `STATSBOMB_DIR` | s1, s3, 4′, s5, diag | data root (builds the two defaults below) | the repo's **parent** folder |
| `EVENTS_DIR` | s1, s3, s5, 4′, 5′, calib, diag | events JSON folder | `$STATSBOMB_DIR/events` |
| `F360_DIR` | s1, s3 | 360 freeze-frame JSON folder | `$STATSBOMB_DIR/360` (s1) |
| `LABELS_CSV` | s3, s5, 4′, 5′, calib, diag | labels csv (Stage-1 output) | `labels.csv` / `all_sequence_labels_v2.csv` |
| `FEAT_PARQUET` | s3, s5, 4′, 5′, calib, diag | features parquet (Stage-3 output) | `hpn_carrier_features.parquet` |
| `MODEL` / `BASE_MODEL` / `OUT_MODEL` | s5, 4′, 5′, calib, diag | trained bundle paths | `hpn_xgb_outcome_calibrated*.joblib` |
| `MATCHES_JSON` / `N_CALIB` / `SEASON_LABEL` | s5, 5′, calib | chronological order · calib size · display label | — / 20% / "season" |
| `TEAM_ORDER` / `OUTDIR` | s5, 5′, diag | fingerprint row order file · output folder | regain-rate order / `analysis_out/` |
| `RUN_SEARCH` | s4 | `1` re-runs the hyper-parameter search (else pinned result) | `0` |
| `FEAT_PARQUET_NEW` / `LABELS_NEW` / `EVENTS_DIR_NEW` | diag | optional held-out second season | unset (section B skipped) |
| `S1_OUT_SEQ` / `S1_OUT_LAB` / `S1_LIMIT` | s1 | output names / debug match limit | `sequences.csv` / `labels.csv` / all |
| `HPN_DIR` | notebooks | repo root | the notebook's working directory |

(4′/5′ = the CLI mirrors `train_calibrated.py` / `apply_season.py`; calib =
`calibrate_season.py`; diag = `compare_calibration.py`.)

`EVENTS_DIR` and `F360_DIR` are **independent** — set them to any two folders (they need not
share a parent). Notebooks resolve the repo root automatically whether run from the repo
folder or from `notebooks/`; anything else, set `HPN_DIR`. Scripts anchor on the repo root
via their own location (they live in `scripts/`), so they run from anywhere. Scripts that
require an input **fail fast with a clear message** naming the missing env var.

## Layout
```
Football-pressure-sequence-analysis/
├─ notebooks/                       the narrative pipeline
│   ├─ s2_hpn_construction.ipynb      stage 2  (HPN illustration)
│   ├─ s4_model_training.ipynb        stage 4  (model comparison → 17-feat calibrated bundle)
│   └─ s5_press_analysis.ipynb        stage 5  (season analysis: efficiency + style, any season)
├─ scripts/                         batch entry points
│   ├─ s1_build_sequences.py          stage 1  (JSON → sequences/labels)
│   ├─ s3_build_features.py           stage 3  (sequences → 19-feature table)
│   ├─ train_calibrated.py            CLI mirror of stage 4 (headless retrain)
│   ├─ calibrate_season.py            per-season isotonic layer for a new season
│   ├─ apply_season.py                CLI mirror of stage 5 (headless season analysis)
│   └─ compare_calibration.py         diagnostics (reliability vs baselines)
├─ src/                             shared libraries (imported by notebooks AND scripts)
│   ├─ hpn_features.py                feature builder (30-col matrix; PRUNED_17 deployed subset)
│   ├─ press_analysis.py              analysis library (lineage check, v_t, tables, figures)
│   ├─ pressure_distance_v2.py        pressure model (logistic kernels, total pressure)
│   └─ voronoi_pitch.py               pitch / Voronoi helpers (stage 2)
├─ docs/
│   └─ high_press_detection_spec.md   detection algorithm spec
├─ requirements.txt
├─ .gitignore                       ignores raw data, generated CSV/parquet/models/figures
└─ README.md
```
Generated data (sequences/labels csv, feature parquet, model joblib, `analysis_out/`)
lives in the repo **root**, git-ignored — the same defaults as before the reorganisation.

## Environment
Python 3.10+ with: `pandas, scikit-learn>=1.6, xgboost, pyarrow, scipy, shapely, matplotlib,
numpy, joblib` (+ optional `adjustText` for nicer figure labels). Install with:
```bash
pip install -r requirements.txt
```
