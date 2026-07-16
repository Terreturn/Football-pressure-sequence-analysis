# HPN — High-Press Analysis

Detects **high-press sequences** from StatsBomb event + 360 data, trains a **calibrated
outcome model** (17-feature XGBoost, isotonic calibration, per-season recalibration),
and produces season-level pressing analysis: VAEP-style step values $v_t$, team
**efficiency** (regain rate, value/seq, intensity-vs-efficiency quadrant map) and
**style** (pressing fingerprint, drivers).

Works with **any StatsBomb events + 360 dataset** (e.g. the public
[StatsBomb Open Data](https://github.com/statsbomb/open-data)). The repo is **code
only** — you bring the raw JSON; every location is set by environment variables, and
no competition, season, or folder layout is assumed. The detection algorithm is
specified in [`docs/high_press_detection_spec.md`](docs/high_press_detection_spec.md).

## Structure

```
├─ notebooks/                       the narrative pipeline (run in order)
│   ├─ s2_hpn_construction.ipynb      HPN pressure-network illustration
│   ├─ s4_model_training.ipynb        model comparison → 17-feat calibrated bundle
│   └─ s5_press_analysis.ipynb        season analysis (efficiency + style, ANY season)
├─ scripts/                         batch entry points
│   ├─ s1_build_sequences.py          stage 1: raw JSON → sequences.csv + labels.csv
│   ├─ s3_build_features.py           stage 3: labels + 360 → features.parquet
│   ├─ train_calibrated.py            headless mirror of s4 (train + calibrate)
│   ├─ calibrate_season.py            per-season isotonic layer (frozen ranker)
│   ├─ apply_season.py                headless mirror of s5 (season analysis)
│   └─ compare_calibration.py         calibration diagnostics vs baselines
├─ src/                             shared libraries (used by notebooks AND scripts)
│   ├─ hpn_features.py                feature builder (30-col matrix; deployed PRUNED_17)
│   ├─ press_analysis.py              analysis library (lineage check, v_t, tables, figures)
│   ├─ pressure_distance_v2.py        pressure model (logistic kernels)
│   └─ voronoi_pitch.py               pitch / Voronoi helpers
└─ docs/high_press_detection_spec.md  detection algorithm spec
```

Notebooks carry the narrative (why this model, how to read each figure); the CLI
mirrors run the **same `src/` functions** headlessly. Generated data (csv/parquet/
joblib, `analysis_out/`) lands in the repo root and is git-ignored.

## Workflow

Pipeline order: **s1 → s3 → s4 → (calibrate_season) → s5**. The S5 notebook is the
terminus — its tables and figures are the deliverables.

**Using your own dataset.** You need two folders of StatsBomb JSON — events
(`<match_id>.json`) and 360 freeze-frames (`<match_id>.json`) — paired by filename.
Point the env vars at them; everything downstream is derived. (Windows PowerShell:
`$env:EVENTS_DIR="C:\data\events"` instead of `export`.)

**1. Build sequences + features** (once per dataset/season):
```bash
export EVENTS_DIR=/data/events  F360_DIR=/data/360
python scripts/s1_build_sequences.py            # -> sequences.csv + labels.csv
export LABELS_CSV=labels.csv  FEAT_PARQUET=features.parquet
python scripts/s3_build_features.py             # -> features.parquet
```

**2. Train the model** (once per training season): open
`notebooks/s4_model_training.ipynb` and **Run All** (set `RUN_SEARCH=1` to re-run the
hyper-parameter search; default uses the pinned result), or headlessly
`python scripts/train_calibrated.py` with the same env vars.
→ `hpn_xgb_outcome_calibrated.joblib` (bundle incl. train/calib lineage).

**3. Analyse a season** (the routine loop). Repeat step 1 on that season's JSON, then:
```bash
export FEAT_PARQUET=season_features.parquet  LABELS_CSV=season_labels.csv \
       EVENTS_DIR=/data/season/events        SEASON_LABEL="2025/26"
python scripts/calibrate_season.py    # fit the season's isotonic layer (frozen ranker)
                                      # optional: MATCHES_JSON=... for chronological order
export MODEL=hpn_xgb_outcome_calibrated_2025_26.joblib
```
then open `notebooks/s5_press_analysis.ipynb` and **Run All** (or
`python scripts/apply_season.py`) → `analysis_out/`: `efficiency.csv`,
`intensity.csv`, `step_valuation.csv`, `drivers.csv`, `lineage.txt` + 4 figures.

Switching to another season = changing the env vars — zero code changes. Optional:
`TEAM_ORDER=<file>` fixes the fingerprint row order (one team per line, e.g. the
final league table); `OUTDIR=<dir>` redirects the outputs.

**Guard rails.** Required inputs fail fast with a message naming the missing env var.
A **lineage check** warns — without blocking — if scored matches were in the model's
training/calibration sets (verdict persisted to `lineage.txt`); the training season's
trustworthy analysis is s4's OOF sections. Full env-var documentation lives in each
script's docstring.

## Environment

Python 3.10+; install with:
```bash
pip install -r requirements.txt
```
`numpy · pandas · scipy · scikit-learn ≥ 1.6` (FrozenEstimator) `· xgboost · pyarrow ·
shapely · matplotlib · joblib`, plus optional `adjustText` (nicer figure labels — the
plots degrade gracefully without it). On Windows, prefix commands with
`$env:PYTHONIOENCODING="utf-8"` so non-ASCII log output prints cleanly.

## License

[MIT](LICENSE). Note the StatsBomb data itself is licensed separately by StatsBomb —
this license covers the code only.
