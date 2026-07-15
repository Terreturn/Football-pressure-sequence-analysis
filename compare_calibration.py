"""
Before/after calibration comparison for the deployed bundle, with two reference
baselines (Logistic, balanced-tuned XGBoost).

Evaluates on:
  (A) the training season's calibration holdout (the calibrator's fit set — in-sample
      for the calibrated models, out-of-sample for the uncalibrated / logistic)
  (B) optionally, a full second season (held out from BOTH training and calibration —
      real deployment); enabled when the *_NEW env vars are set

Models compared:
  uncalibrated   the deployed base model (tuned unweighted XGB, 17 feats)
  calibrated     the deployed model (isotonic on the training-season holdout)
  logistic       Logistic baseline (median-impute + scale, class_weight=balanced) — notebook recipe
  balanced-cal   tuned balanced XGB (notebook 25c M3, BEST_BAL) + isotonic — the class-weighted rival

Inputs (env-overridable):
  MODEL             deployed bundle (default: hpn_xgb_outcome_calibrated.joblib)
  FEAT_PARQUET      training-season Stage-3 parquet (default: hpn_carrier_features.parquet)
  LABELS_CSV        training-season Stage-1 labels  (default: all_sequence_labels_v2.csv)
  EVENTS_DIR        training-season event JSON      (default: $STATSBOMB_DIR/events)
  FEAT_PARQUET_NEW / LABELS_NEW / EVENTS_DIR_NEW
                    a second, fully held-out season (all three set -> section B runs)
  OUTDIR            output folder (default: analysis_out/)

Metrics: log-loss, accuracy, per-class Brier, per-class ECE (10-bin), multiclass ECE.
Writes calibration_compare.csv + reliability curves (success class) -> OUTDIR/fig_calibration.png

Columns fed to the models come from bundle["features"] (the 17 de-collinearised features for the
current deployed model), never from hpn_features.FEATURES — so this stays correct if the model's
feature set changes. The train/calibrate split mirrors training exactly:
GroupShuffleSplit(test_size=0.20, random_state=42).
"""
import os, sys, warnings
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "src"))
from hpn_features import build_feature_matrix

STATSBOMB_DIR = os.environ.get("STATSBOMB_DIR", os.path.dirname(HERE))
FEAT_PARQUET = os.environ.get("FEAT_PARQUET", os.path.join(HERE, "hpn_carrier_features.parquet"))
LABELS_CSV   = os.environ.get("LABELS_CSV", os.path.join(HERE, "all_sequence_labels_v2.csv"))
EVENTS_DIR   = os.environ.get("EVENTS_DIR", os.path.join(STATSBOMB_DIR, "events"))
FEAT_NEW   = os.environ.get("FEAT_PARQUET_NEW")
LABELS_NEW = os.environ.get("LABELS_NEW")
EVENTS_NEW = os.environ.get("EVENTS_DIR_NEW")
HAS_NEW = all([FEAT_NEW, LABELS_NEW, EVENTS_NEW])
MODEL_PATH = os.environ.get("MODEL", os.path.join(HERE, "hpn_xgb_outcome_calibrated.joblib"))
OUTDIR = os.environ.get("OUTDIR", os.path.join(HERE, "analysis_out"))
os.makedirs(OUTDIR, exist_ok=True)

B = joblib.load(MODEL_PATH)
base, cal, le = B["base_model"], B["model"], B["label_encoder"]
FEATS = list(B["features"])          # follow the loaded model (17), not hpn_features.FEATURES (30)
CLS = list(le.classes_); i_s = CLS.index("success")
print(f"model: {len(FEATS)} feats | classes {CLS} | calib {B.get('calib_method')}")

# notebook 25c M3 pinned params (balanced-tuned rival)
BEST_BAL = {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05, "min_child_weight": 10,
            "subsample": 0.6, "colsample_bytree": 0.8, "gamma": 0.0, "reg_lambda": 2.0}
_FIX = dict(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            tree_method="hist", n_jobs=-1, random_state=42)


def reliability(y_onehot_k, p_k, n_bins=10):
    """Return (bin_mid, bin_pred, bin_true, bin_n) and ECE for one class."""
    edges = np.linspace(0, 1, n_bins + 1); idx = np.digitize(p_k, edges[1:-1])
    mid, bp, bt, bn = [], [], [], []
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any(): continue
        pr, tr, n = p_k[m].mean(), y_onehot_k[m].mean(), m.sum()
        mid.append((edges[b] + edges[b + 1]) / 2); bp.append(pr); bt.append(tr); bn.append(n)
        ece += n / len(p_k) * abs(pr - tr)
    return np.array(mid), np.array(bp), np.array(bt), np.array(bn), ece


def evaluate(name, models, X, y):
    """models: {tag: fitted estimator with predict_proba over encoded [0,1,2]}."""
    yi = le.transform(y)
    rows, probas = [], {}
    for tag, m in models.items():
        P = m.predict_proba(X); probas[tag] = P
        rec = {"set": name, "model": tag,
               "log_loss": log_loss(yi, P, labels=[0, 1, 2]),
               "accuracy": accuracy_score(yi, P.argmax(1))}
        eces = []
        for k, c in enumerate(CLS):
            yk = (yi == k).astype(int)
            rec[f"brier_{c}"] = brier_score_loss(yk, P[:, k])
            _, _, _, _, e = reliability(yk, P[:, k]); rec[f"ece_{c}"] = e; eces.append(e)
        rec["ece_mean"] = float(np.mean(eces)); rows.append(rec)
    return pd.DataFrame(rows), probas, yi


# (A) training season: reproduce the training split exactly (same seed / test_size as cell 25a)
dfA = build_feature_matrix(FEAT_PARQUET, LABELS_CSV, EVENTS_DIR)
dfA = dfA[dfA.outcome_tag.isin(["success", "fail", "neutral"])].reset_index(drop=True)
tr, ca = next(GroupShuffleSplit(1, test_size=0.20, random_state=42).split(dfA[FEATS], dfA.outcome_tag, dfA.match_id))
Xtr, ytr_i = dfA[FEATS].iloc[tr].astype(float), le.transform(dfA.outcome_tag.values[tr])
Xca, yca_i = dfA[FEATS].iloc[ca].astype(float), le.transform(dfA.outcome_tag.values[ca])
print(f"training-season split: train {pd.Series(dfA.match_id.values[tr]).nunique()} matches | "
      f"calib {pd.Series(dfA.match_id.values[ca]).nunique()} matches")

# baselines trained on the SAME train matches (notebook recipes)
print("fitting baselines (logistic + balanced-tuned XGB) ...")
logit = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0)
                      ).fit(Xtr, ytr_i)
bal = XGBClassifier(**BEST_BAL, **_FIX)
bal.fit(Xtr, ytr_i, sample_weight=compute_sample_weight("balanced", ytr_i))
bal_cal = CalibratedClassifierCV(FrozenEstimator(bal), method="isotonic").fit(Xca, yca_i)

MODELS = {"uncalibrated": base, "calibrated": cal, "logistic": logit, "balanced-cal": bal_cal}

A = dfA.iloc[ca]
tA, prA, yA = evaluate("train-season calib-holdout", MODELS, A[FEATS].astype(float), A.outcome_tag.values)

# (B) optional second season (held out from train + calib for every model above)
parts, panels = [tA], [("train-season calib-holdout", prA, yA)]
if HAS_NEW:
    dfB = build_feature_matrix(FEAT_NEW, LABELS_NEW, EVENTS_NEW)
    dfB = dfB[dfB.outcome_tag.isin(["success", "fail", "neutral"])].reset_index(drop=True)
    tB, prB, yB = evaluate("new-season held-out", MODELS, dfB[FEATS].astype(float), dfB.outcome_tag.values)
    parts.append(tB); panels.append(("new-season held-out", prB, yB))
else:
    print("note: FEAT_PARQUET_NEW / LABELS_NEW / EVENTS_DIR_NEW not all set — skipping section (B)")

comp = pd.concat(parts, ignore_index=True)
cols = ["set", "model", "log_loss", "accuracy", "ece_mean",
        "ece_success", "ece_fail", "ece_neutral", "brier_success"]
pd.set_option("display.width", 200)
print(comp[cols].round(4).to_string(index=False))
comp.round(5).to_csv(os.path.join(OUTDIR, "calibration_compare.csv"), index=False)

# reliability figure: success class, both seasons, all four models
COLORS = {"uncalibrated": "#d62728", "calibrated": "#2ca02c",
          "logistic": "#9467bd", "balanced-cal": "#1f77b4"}
fig, ax = plt.subplots(1, len(panels), figsize=(6.7 * len(panels), 5.4), squeeze=False)
ax = ax[0]
for j, (title, probas, yv) in enumerate(panels):
    yk = (yv == i_s).astype(int)
    ax[j].plot([0, 1], [0, 1], "--", color="#999", lw=1, label="perfect")
    for tag, P in probas.items():
        mid, bp, bt, bn, e = reliability(yk, P[:, i_s])
        ax[j].plot(bp, bt, "o-", color=COLORS[tag], label=f"{tag} (ECE={e:.3f})")
    ax[j].set_title(f"success reliability — {title}")
    ax[j].set_xlabel("mean predicted P(success)"); ax[j].set_ylabel("observed success rate")
    ax[j].legend(); ax[j].grid(alpha=.25); ax[j].set_xlim(0, 1); ax[j].set_ylim(0, 1)
plt.tight_layout()
fig_path = os.path.join(OUTDIR, "fig_calibration.png")
plt.savefig(fig_path, dpi=150); plt.close()
print(f"\nsaved -> calibration_compare.csv + {os.path.basename(fig_path)}")
