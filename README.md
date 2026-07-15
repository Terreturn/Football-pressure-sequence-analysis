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
| **1** | raw events + 360 JSON → high-press sequences + labels | `s1_build_sequences.py` | script |
| **2** | HPN construction & illustration figures | `s2_hpn_construction.ipynb` | notebook |
| **3** | sequences → carrier-centric ML feature table | `s3_build_features.py` | script |
| **4–5** | model comparison, tuning, de-collinearisation, calibration + season analysis | `s45_ml_pipeline.ipynb` | notebook |
| **6** | train + calibrate the deployed model (CLI mirror of the notebook's selected model) | `train_calibrated.py` | script |
| **7** | fit a per-season isotonic calibration layer for a NEW season | `calibrate_season.py` | script |
| **8** | score any (held-out) season + full team pressing analysis | `apply_season.py` | script |
| — | calibration diagnostics (deployed model vs baselines, reliability curves) | `compare_calibration.py` | script |

Scripts do the batch work; notebooks are where results are **shown and explained**
(HPN figures, model comparison, calibration, VAEP plots — saved inline). The detection
algorithm is specified in [`high_press_detection_spec.md`](high_press_detection_spec.md).

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
**de-collinearises 30 → 17** (cell 25e) and selects the deployed model, then runs the
VAEP-style step valuation and the season team analysis (intensity vs. efficiency,
efficiency ranking, pressing-style fingerprint, drivers).

**Usage.** Open in Jupyter / VSCode **from the repo folder** and **Run All**, or headless:
```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 s45_ml_pipeline.ipynb
```

**Output.** Trained bundles (`hpn_xgb_outcome*.joblib`) + metrics, calibration/reliability
plots, VAEP and team figures **embedded inline** in the notebook.

### Stage 6 — `train_calibrated.py`
**What.** CLI mirror of the notebook's selected model: tuned unweighted XGBoost on the
17 de-collinearised features, 80/20 **group-aware** split (by match), isotonic
calibration on the held-out 20%. Prints before/after reliability.

**Usage.**
```bash
export FEAT_PARQUET=/path/to/features.parquet   # Stage-3 output
export LABELS_CSV=/path/to/labels.csv           # Stage-1 output
export EVENTS_DIR=/data/events                  # raw events (ball_in features)
python train_calibrated.py                      # -> hpn_xgb_outcome_calibrated.joblib
```
The pinned `XGB_PARAMS` come from a grouped randomised search on the original training
season; re-run the notebook's search (cell 14) when training on a very different dataset.

### Stage 7 — `calibrate_season.py`
**What.** Per-season calibration: keeps the deployed ranking model **frozen** and fits a
new isotonic layer on a NEW season's first `N_CALIB` matches (chronological if you provide
`MATCHES_JSON`), then evaluates uncalibrated vs old layer vs new layer on the remaining
matches. Use when class base rates drift across seasons.

**Usage.**
```bash
export FEAT_PARQUET=/path/to/new_season_features.parquet
export LABELS_CSV=/path/to/new_season_labels.csv
export EVENTS_DIR=/data/new_season/events
export MATCHES_JSON=/data/new_season/matches.json   # optional, for chronological order
export SEASON_LABEL="2025-26"
python calibrate_season.py    # -> hpn_xgb_outcome_calibrated_2025-26.joblib
```

### Stage 8 — `apply_season.py`
**What.** Scores a (held-out) season with a trained bundle and reproduces the full team
pressing analysis: per-step valuation v_t + step labels (Effective / Beaten / Risky / …),
team **intensity vs. regain-efficiency** quadrant map, efficiency ranking (regain rate,
value/seq, residual, E/(E+B)), pressing-style fingerprint (10 z-scored descriptors), and
the pressure-channel **drivers** of v_t.

**Usage.**
```bash
export FEAT_PARQUET=/path/to/season_features.parquet
export LABELS_CSV=/path/to/season_labels.csv
export EVENTS_DIR=/data/season/events
export MODEL=hpn_xgb_outcome_calibrated_2025-26.joblib   # the per-season bundle
export SEASON_LABEL="2025/26"
# optional: fingerprint row order (e.g. final league table), one team per line
# export TEAM_ORDER=/path/to/league_table.txt
python apply_season.py        # -> analysis_out/{intensity,efficiency,step_valuation,drivers}.csv + 4 figures
```

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
trained model (s45 / stage 6) → per-season calibration (stage 7) → season analysis (stage 8).
The committed notebooks keep the **saved outputs** of a prior run so the results are visible
without re-running.

## Setup & paths
No absolute paths are hard-coded; locations resolve from environment variables.

| Env var | Used by | Meaning | Default |
|---|---|---|---|
| `STATSBOMB_DIR` | s1, s3, 6, diag | data root (builds the two defaults below) | the repo's **parent** folder |
| `EVENTS_DIR` | s1, s3, 6, 7, 8, diag | events JSON folder | `$STATSBOMB_DIR/events` |
| `F360_DIR` | s1, s3 | 360 freeze-frame JSON folder | `$STATSBOMB_DIR/360` (s1) |
| `LABELS_CSV` | s3, 6, 7, 8, diag | labels csv (Stage-1 output) | `labels.csv` / `all_sequence_labels_v2.csv` |
| `FEAT_PARQUET` | s3, 6, 7, 8, diag | features parquet (Stage-3 output) | `hpn_carrier_features.parquet` |
| `MODEL` / `BASE_MODEL` / `OUT_MODEL` | 6, 7, 8, diag | trained bundle paths | `hpn_xgb_outcome_calibrated*.joblib` |
| `MATCHES_JSON` / `N_CALIB` / `SEASON_LABEL` | 7, 8 | chronological order · calib size · display label | — / 20% / "season" |
| `TEAM_ORDER` / `OUTDIR` | 8, diag | fingerprint row order file · output folder | regain-rate order / `analysis_out/` |
| `FEAT_PARQUET_NEW` / `LABELS_NEW` / `EVENTS_DIR_NEW` | diag | optional held-out second season | unset (section B skipped) |
| `S1_OUT_SEQ` / `S1_OUT_LAB` / `S1_LIMIT` | s1 | output names / debug match limit | `sequences.csv` / `labels.csv` / all |
| `HPN_DIR` | notebooks | repo root | the notebook's working directory |

`EVENTS_DIR` and `F360_DIR` are **independent** — set them to any two folders (they need not
share a parent). Run notebooks from the repo folder (so `os.getcwd()` resolves the repo) or set
`HPN_DIR`. Scripts find the repo and `src/` via their own location. Scripts that require an
input **fail fast with a clear message** naming the missing env var.

## Layout
```
Football-pressure-sequence-analysis/
├─ s1_build_sequences.py        stage 1   (JSON → sequences/labels)
├─ s2_hpn_construction.ipynb    stage 2   (HPN illustration)
├─ s3_build_features.py         stage 3   (sequences → 19-feature table)
├─ s45_ml_pipeline.ipynb        stage 4-5 (model comparison → 17-feat deployed model + analysis)
├─ train_calibrated.py          stage 6   (CLI: train + calibrate the deployed model)
├─ calibrate_season.py          stage 7   (CLI: per-season isotonic layer for a new season)
├─ apply_season.py              stage 8   (CLI: score a season + team pressing analysis)
├─ compare_calibration.py       diagnostics (reliability vs baselines)
├─ src/
│   ├─ hpn_features.py          shared feature builder (30-col matrix; PRUNED_17 deployed subset)
│   ├─ pressure_distance_v2.py  pressure model (logistic kernels, total pressure)
│   └─ voronoi_pitch.py         pitch / Voronoi helpers (stage 2)
├─ high_press_detection_spec.md detection algorithm spec
├─ requirements.txt
├─ .gitignore                   ignores raw data, generated CSV/parquet/models/figures
└─ README.md
```

## Environment
Python 3.10+ with: `pandas, scikit-learn>=1.6, xgboost, pyarrow, scipy, shapely, matplotlib,
numpy, joblib` (+ optional `adjustText` for nicer figure labels). Install with:
```bash
pip install -r requirements.txt
```
