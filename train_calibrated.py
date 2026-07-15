"""
Train + calibrate the SELECTED HPN outcome model on a training season (CLI
mirror of notebook s45 cells 25a/25b — the deployed model M1).

Model = tuned, UNWEIGHTED XGBoost (BEST_PARAMS from the grouped search) on the
17 de-collinearised features (src/hpn_features.py PRUNED_17 — a subset of the 30-column
matrix; first-frame diffs left NaN). No sample weights: it optimises probability quality.

Split (by match_id, group-aware, test_size=0.20):
  80% of matches -> train the XGBoost outcome model
  20% of matches -> fit an isotonic probability calibrator (CalibratedClassifierCV)

Inputs (env-overridable):
  FEAT_PARQUET  Stage-3 carrier-feature parquet (default: hpn_carrier_features.parquet)
  LABELS_CSV    Stage-1 labels csv              (default: all_sequence_labels_v2.csv)
  EVENTS_DIR    raw event JSON folder — needed to rebuild the incoming-ball features
                (default: $STATSBOMB_DIR/events, STATSBOMB_DIR defaulting to the
                repo's parent folder)
  OUT_MODEL     output bundle path              (default: hpn_xgb_outcome_calibrated.joblib)
  CALIB_METHOD  isotonic | sigmoid              (default: isotonic)

Outputs:
  OUT_MODEL   {model(calibrated), base_model, label_encoder, features, params}
  reliability before/after printed to stdout.

NOTE: XGB_PARAMS below were tuned on the original 2024/25 EPL training season via
a grouped randomised search (notebook cell 14). They are sensible defaults for a
similar-sized dataset, but re-run the notebook's search when training on a
substantially different competition.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import log_loss, brier_score_loss
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from hpn_features import build_feature_matrix, PRUNED_17

FEAT_PARQUET = os.environ.get("FEAT_PARQUET", os.path.join(HERE, "hpn_carrier_features.parquet"))
LABELS_CSV = os.environ.get("LABELS_CSV", os.path.join(HERE, "all_sequence_labels_v2.csv"))
STATSBOMB_DIR = os.environ.get("STATSBOMB_DIR", os.path.dirname(HERE))
EVENTS_DIR = os.environ.get(                                   # ball_in needs the raw event JSON
    "EVENTS_DIR", os.path.join(STATSBOMB_DIR, "events"))
OUT_MODEL = os.environ.get("OUT_MODEL", os.path.join(HERE, "hpn_xgb_outcome_calibrated.joblib"))
CALIB_METHOD = os.environ.get("CALIB_METHOD", "isotonic")   # isotonic | sigmoid
SEED = 42

# tuned params from the grouped randomized search (notebook #14 BEST_PARAMS)
XGB_PARAMS = dict(
    n_estimators=754, max_depth=6, learning_rate=0.0238,
    subsample=0.666, colsample_bytree=0.6527, min_child_weight=15,
    reg_lambda=2.6194, gamma=2.149, objective="multi:softprob", num_class=3,
    eval_metric="mlogloss", tree_method="hist", n_jobs=-1, random_state=SEED,
)

# ── build the 30-column matrix, then train on the 17 deployed features ─────────
print("building training-season features ...")
dft = build_feature_matrix(FEAT_PARQUET, LABELS_CSV, EVENTS_DIR)
dft = dft[dft["outcome_tag"].isin(["success", "fail", "neutral"])].reset_index(drop=True)
X = dft[PRUNED_17].astype(float)
y = dft["outcome_tag"].values
groups = dft["match_id"].values
print(f"rows {len(dft)} | matches {dft['match_id'].nunique()} | feats {len(PRUNED_17)}")
print("class balance:", pd.Series(y).value_counts(normalize=True).round(3).to_dict())

# ── group-aware 80/20 split by match: train vs calibrate ─────────────────────
tr, ca = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
              .split(X, y, groups))
Xtr, ytr = X.iloc[tr], y[tr]
Xca, yca = X.iloc[ca], y[ca]
print(f"train {Xtr.shape} ({pd.Series(groups[tr]).nunique()} matches) | "
      f"calib {Xca.shape} ({pd.Series(groups[ca]).nunique()} matches)")

le = LabelEncoder().fit(["fail", "neutral", "success"])   # stable class order
ytr_i = le.transform(ytr)
CLS = list(le.classes_)                                    # ['fail','neutral','success']
i_s = CLS.index("success")

# ── train XGBoost on 80% (unweighted — optimises probability quality) ─────────
base = XGBClassifier(**XGB_PARAMS)
base.fit(Xtr, ytr_i)

# ── calibrate on held-out 20% (isotonic, one-vs-rest, prefit via FrozenEstimator)
cal = CalibratedClassifierCV(FrozenEstimator(base), method=CALIB_METHOD)
cal.fit(Xca, le.transform(yca))

# ── reliability on the calibration set: uncalibrated vs calibrated ────────────
yca_i = le.transform(yca)
p_un = base.predict_proba(Xca)
p_ca = cal.predict_proba(Xca)


def report(tag, P):
    ll = log_loss(yca_i, P, labels=[0, 1, 2])
    print(f"\n[{tag}] log_loss={ll:.4f}")
    print(f"{'class':9s} {'pred_mean':>9s} {'true_rate':>9s} {'brier':>7s}")
    for k, c in enumerate(CLS):
        yk = (yca_i == k).astype(int)
        print(f"{c:9s} {P[:, k].mean():9.4f} {yk.mean():9.4f} {brier_score_loss(yk, P[:, k]):7.4f}")


report("uncalibrated", p_un)
report("calibrated", p_ca)

# ── save bundle ───────────────────────────────────────────────────────────────
joblib.dump({"model": cal, "base_model": base, "label_encoder": le,
             "features": PRUNED_17, "params": XGB_PARAMS,
             "weighting": "unweighted", "calib_method": CALIB_METHOD}, OUT_MODEL)
print(f"\nsaved -> {OUT_MODEL}")
