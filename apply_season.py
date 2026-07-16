"""
Apply a trained HPN outcome bundle to a (held-out) season and reproduce the
Stage-5 team pressing analysis. Thin CLI shell over src/press_analysis.py — the
S5 notebook (s5_press_analysis.ipynb) calls the exact same functions, so the two
can never drift apart.

The scored season should be fully held out from the bundle's training season
(anchors are scored directly — no out-of-fold; the training season itself
belongs in the S4 notebook's OOF analysis). A lineage check warns — but never
blocks — when scored matches overlap the bundle's train/calib sets.

Inputs  (env-overridable):
  FEAT_PARQUET  the season's carrier features (Stage 3 output)   (required)
  LABELS_CSV    the season's labels csv       (Stage 1 output)   (required)
  EVENTS_DIR    the season's event JSON folder (for ball_in reconstruction) (required)
  MODEL         trained bundle (default: hpn_xgb_outcome_calibrated.joblib;
                use the calibrate_season.py bundle when scoring a newer season)
  SEASON_LABEL  free text used in figure titles (default "season")
  TEAM_ORDER    optional path to a text file, one team name per line, that fixes
                the fingerprint row order (e.g. final league table); default:
                teams ordered by regain rate
  OUTDIR        output folder (default: analysis_out/)

Outputs -> OUTDIR/
  intensity.csv, efficiency.csv, step_valuation.csv, drivers.csv, lineage.txt
  fig_intensity_landscape.png, fig_efficiency.png, fig_style_fingerprint.png,
  fig_drivers.png
"""
import os, sys, warnings
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from hpn_features import build_feature_matrix, FEATURES
import press_analysis as pa

def _req(name):
    v = os.environ.get(name)
    if not v:
        sys.exit(f"error: set {name} (see module docstring)")
    return v

FEAT_PARQUET = _req("FEAT_PARQUET")
LABELS_CSV   = _req("LABELS_CSV")
EVENTS_DIR   = _req("EVENTS_DIR")
MODEL_PATH   = os.environ.get("MODEL", os.path.join(HERE, "hpn_xgb_outcome_calibrated.joblib"))
SEASON_LABEL = os.environ.get("SEASON_LABEL", "season")
TEAM_ORDER   = os.environ.get("TEAM_ORDER")            # optional file: one team per line
OUTDIR       = os.environ.get("OUTDIR", os.path.join(HERE, "analysis_out"))

# ── features + model ──────────────────────────────────────────────────────────
dft = build_feature_matrix(FEAT_PARQUET, LABELS_CSV, EVENTS_DIR)
lab = pd.read_csv(LABELS_CSV); lab["match_id"] = lab["match_id"].astype(str)
print(f"features: {dft.shape} | matches {dft['match_id'].nunique()} | "
      f"{len(FEATURES)}-col matrix; model scores on its trained feature subset")

bundle = joblib.load(MODEL_PATH)
print(f"model: {os.path.basename(MODEL_PATH)} | {len(bundle['features'])} feats | "
      f"classes {list(bundle['label_encoder'].classes_)} | calib={bundle.get('calib_method')}")
_, _, lineage_msgs = pa.check_lineage(bundle, dft["match_id"].unique())

# ── score + analyse (all logic in src/press_analysis.py) ─────────────────────
p_s, p_f = pa.score(dft, bundle)
vp = pa.build_step_table(dft, lab, p_s, p_f)
print(vp.loc[vp["vt"].notna(), "label"].value_counts().to_string())

intensity, team = pa.team_tables(vp)
print("\n=== EFFICIENCY (sorted by regain_rate) ===")
print(team.round(4).to_string())

Zt, row_title = pa.fingerprint_matrix(vp, intensity, team, TEAM_ORDER)
drivers, r2 = pa.drivers_table(vp)
print(f"\n=== DRIVERS of v_t (joint linear R^2 = {r2:.3f}; channels collinear) ===")
print(drivers.round(4).to_string())

figs = {
    "fig_intensity_landscape.png": pa.fig_intensity_efficiency(intensity, team, SEASON_LABEL),
    "fig_efficiency.png":          pa.fig_efficiency(team, SEASON_LABEL),
    "fig_style_fingerprint.png":   pa.fig_fingerprint(Zt, row_title, SEASON_LABEL),
    "fig_drivers.png":             pa.fig_drivers(drivers, SEASON_LABEL),
}
pa.write_outputs(OUTDIR, intensity, team, vp, drivers, figs, lineage_msgs)
