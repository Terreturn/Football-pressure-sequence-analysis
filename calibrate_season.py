"""
Fit a season-specific isotonic calibration layer on a frozen ranking model
(per-season calibration scheme).

Scheme: ONE frozen ranking model (the deployed bundle from train_calibrated.py) +
one isotonic layer PER SEASON. This script fits a new season's layer on that
season's FIRST N_CALIB matches (chronological — simulating "lock calibration early
in the season"), evaluates on the remaining matches, and saves a per-season bundle.

Why: tree-ensemble ranking typically transfers across seasons (small macro-AUC
drop), but the class base rates can drift (e.g. 2024/25 -> 2025/26 success rate
doubled 6.0% -> 11.6%), which makes probabilities calibrated on the training
season systematically off-scale. A season-specific isotonic layer fixes the scale
without touching the model.

Inputs (env-overridable):
  BASE_MODEL    deployed bundle from train_calibrated.py
                (default: hpn_xgb_outcome_calibrated.joblib next to this script)
  FEAT_PARQUET  the new season's Stage-3 feature parquet          (required)
  LABELS_CSV    the new season's Stage-1 labels csv               (required)
  EVENTS_DIR    the new season's raw event JSON folder            (required;
                needed to rebuild the incoming-ball features)
  MATCHES_JSON  StatsBomb matches.json for the season (optional — used to order
                matches chronologically; without it matches are ordered by
                match_id and a warning is printed)
  N_CALIB       matches used to fit the layer (default: 20% of the season)
  SEASON_LABEL  free-text tag stored in the bundle (default "new-season")
  OUT_MODEL     output bundle path (default: hpn_xgb_outcome_calibrated_<label>.joblib)

Output: joblib bundle {model(calibrated), base_model, label_encoder, features,
params, calib_*} + an evaluation table (late-season matches):
uncalibrated vs training-season layer vs new-season layer.
"""
import os, sys, json, warnings
import numpy as np, pandas as pd, joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "src"))
from hpn_features import build_feature_matrix

def _req(name):
    v = os.environ.get(name)
    if not v:
        sys.exit(f"error: set {name} (see module docstring)")
    return v

BASE_MODEL   = os.environ.get("BASE_MODEL", os.path.join(HERE, "hpn_xgb_outcome_calibrated.joblib"))
FEAT_PARQUET = _req("FEAT_PARQUET")
LABELS_CSV   = _req("LABELS_CSV")
EVENTS_DIR   = _req("EVENTS_DIR")
MATCHES_JSON = os.environ.get("MATCHES_JSON")               # optional
SEASON_LABEL = os.environ.get("SEASON_LABEL", "new-season")
OUT_MODEL    = os.environ.get("OUT_MODEL", os.path.join(
    HERE, f"hpn_xgb_outcome_calibrated_{SEASON_LABEL.replace('/', '_').replace(' ', '_')}.joblib"))

B = joblib.load(BASE_MODEL)
base, cal_train, le = B["base_model"], B["model"], B["label_encoder"]
FEATS = list(B["features"]); CLS = list(le.classes_); i_s = CLS.index("success")
print(f"base model: {os.path.basename(BASE_MODEL)} | {len(FEATS)} feats | classes {CLS}")

# ── new season's feature matrix ───────────────────────────────────────────────
dft = build_feature_matrix(FEAT_PARQUET, LABELS_CSV, EVENTS_DIR)
dft = dft[dft.outcome_tag.isin(CLS)].reset_index(drop=True)

# ── temporal split: first N_CALIB matches by date -> calibrate; rest -> evaluate ──
if MATCHES_JSON and os.path.exists(MATCHES_JSON):
    dates = {str(m["match_id"]): m["match_date"]
             for m in json.load(open(MATCHES_JSON, encoding="utf-8"))}
else:
    print("warning: MATCHES_JSON not provided — ordering matches by match_id, "
          "which is NOT guaranteed chronological")
    dates = {}
mids = sorted(dft["match_id"].unique(), key=lambda m: (dates.get(m, "9999"), m))
N_CALIB = int(os.environ.get("N_CALIB", max(1, round(0.2 * len(mids)))))
calib_mids, eval_mids = set(mids[:N_CALIB]), set(mids[N_CALIB:])
if not eval_mids:
    sys.exit(f"error: N_CALIB={N_CALIB} leaves no evaluation matches ({len(mids)} total)")
mc, me = dft.match_id.isin(calib_mids), dft.match_id.isin(eval_mids)
span = (f"{dates[mids[0]]} -> {dates[mids[N_CALIB - 1]]}" if dates else "by match_id")
print(f"calib: first {len(calib_mids)} matches ({span}), {int(mc.sum())} anchors | "
      f"eval: {len(eval_mids)} matches, {int(me.sum())} anchors")

Xc, yc = dft.loc[mc, FEATS].astype(float), le.transform(dft.loc[mc, "outcome_tag"].values)
Xe, ye = dft.loc[me, FEATS].astype(float), le.transform(dft.loc[me, "outcome_tag"].values)

# ── fit the season's isotonic layer on the frozen base model ─────────────────
cal_new = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic").fit(Xc, yc)

# ── evaluate on the late-season matches (out-of-sample for BOTH layers) ────────
def ece_cls(yk, pk, nb=10):
    ed = np.linspace(0, 1, nb + 1); idx = np.digitize(pk, ed[1:-1]); s = 0.
    for b in range(nb):
        m = idx == b
        if m.any(): s += m.sum() / len(pk) * abs(pk[m].mean() - yk[m].mean())
    return s

Yb = label_binarize(ye, classes=[0, 1, 2])
rows = []
for tag, m in [("uncalibrated", base), ("calibrated-train-season", cal_train),
               (f"calibrated-{SEASON_LABEL}", cal_new)]:
    P = m.predict_proba(Xe)
    rec = {"model": tag, "log_loss": log_loss(ye, P, labels=[0, 1, 2]),
           "accuracy": accuracy_score(ye, P.argmax(1)),
           "macro_AUC": roc_auc_score(Yb, P, average="macro"),
           "mean_pred_succ": P[:, i_s].mean(), "obs_succ": (ye == i_s).mean()}
    eces = []
    for k, c in enumerate(CLS):
        yk = (ye == k).astype(int)
        rec[f"ece_{c}"] = ece_cls(yk, P[:, k]); rec[f"brier_{c}"] = brier_score_loss(yk, P[:, k])
        eces.append(rec[f"ece_{c}"])
    rec["ece_mean"] = float(np.mean(eces)); rows.append(rec)
comp = pd.DataFrame(rows).set_index("model")
cols = ["log_loss", "accuracy", "macro_AUC", "ece_mean", "ece_success",
        "brier_success", "mean_pred_succ", "obs_succ"]
pd.set_option("display.width", 200)
print(f"\n=== {SEASON_LABEL} late-season eval ({len(eval_mids)} matches, out-of-sample for both layers) ===")
print(comp[cols].round(4).to_string())

# ── save the per-season bundle (train_matches carried over = lineage survives) ─
joblib.dump({"model": cal_new, "base_model": base, "label_encoder": le,
             "features": FEATS, "params": B.get("params"), "weighting": "unweighted",
             "calib_method": "isotonic", "calib_scheme": "per-season",
             "calib_fit": f"{SEASON_LABEL} first {N_CALIB} matches ({span})",
             "train_matches": B.get("train_matches", []),
             "calib_matches": sorted(calib_mids)}, OUT_MODEL)
print(f"\nsaved -> {os.path.basename(OUT_MODEL)}")
