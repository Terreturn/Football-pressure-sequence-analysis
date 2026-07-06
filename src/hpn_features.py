"""
Shared HPN feature builder — used by scoring (apply_2025_2026.py) / comparison
scripts so the 30-feature matrix is constructed identically to the notebook's
selected model (no train/serve skew).

Feature set (30) — matches the notebook's feature selection (ALL_FEATURES_BI):
  19 base carrier features (from the Stage-3 parquet)
+  7 temporal first-difference features (Δt=1 within sequence)
+  4 incoming-ball (ball_in) features

The 7 temporal diffs are NaN on the first frame of every sequence (no t-1) and are
LEFT as NaN — XGBoost handles them natively, exactly as the deployed model
(hpn_xgb_outcome_calibrated.joblib: tuned, unweighted, isotonic-calibrated) was
trained. Do not fill them, or you reintroduce a train/serve skew.
"""
import os
import json
import numpy as np
import pandas as pd

BASE_FEATURES = [
    "P_total", "carrier_enemy_density", "carrier_friendly_density",
    "carrier_dist_boundary", "carrier_x_norm",
    "n_pressers_on_carrier", "nearest_def_dist", "max_press_on_carrier",
    "best_pass_w", "best_forward_pass_w", "n_open_pass", "mean_lane_openness",
    "mean_recv_freedom", "min_recv_freedom", "n_press_on_attackers",
    "total_press_on_attackers", "max_press_on_attacker", "frac_attackers_pressed",
    "carrier_trap_w",
]

DIFF_SRC = {
    "d_P_total_dt": "P_total", "d_nearest_def_dist_dt": "nearest_def_dist",
    "d_carrier_x_norm_dt": "carrier_x_norm", "d_best_pass_w_dt": "best_pass_w",
    "d_carrier_trap_w_dt": "carrier_trap_w",
    "d_frac_attackers_pressed_dt": "frac_attackers_pressed",
}
TEMPORAL = list(DIFF_SRC) + ["d_press_redistribution_dt"]
BALLIN = ["ball_in_dx", "ball_in_dy", "ball_in_dist", "ball_in_angle"]

# canonical model input order (30) — the notebook's ALL_FEATURES_BI
FEATURES = BASE_FEATURES + TEMPORAL + BALLIN

GK = ["match_id", "seq_id"]


def _build_temporal(dft):
    """7 temporal first-difference features; first-of-sequence left as NaN (native to XGBoost)."""
    for nf, base in DIFF_SRC.items():
        dft[nf] = dft.groupby(GK)[base].diff()
    dft["_redist"] = dft["P_total"] - dft["max_press_on_attacker"]
    dft["d_press_redistribution_dt"] = dft.groupby(GK)["_redist"].diff()
    dft.drop(columns="_redist", inplace=True)
    return dft


def _build_ballin(dft, labels_csv, events_dir):
    """Incoming-ball direction (attack-normalised, leak-free)."""
    lab = pd.read_csv(labels_csv)
    lab["match_id"] = lab["match_id"].astype(str)
    aid_map = lab.set_index(["match_id", "seq_id", "ev_pos"])["anchor_event_id"].to_dict()
    dft["aid"] = [aid_map.get((m, s, e))
                  for m, s, e in zip(dft.match_id, dft.seq_id, dft.ev_pos)]
    bin_ = {}
    for mid in dft["match_id"].unique():
        with open(os.path.join(events_dir, f"{mid}.json"), encoding="utf-8") as f:
            events = json.load(f)
        EV = sorted(events, key=lambda e: e["index"])
        pos = {e["index"]: i for i, e in enumerate(EV)}
        by = {e["id"]: e for e in EV}
        for a in dft.loc[dft.match_id == mid, "aid"].dropna().unique():
            ev = by.get(a)
            if ev is None or not ev.get("location"):
                continue
            isd = ev["team"]["id"] != ev["possession_team"]["id"]
            flip = (lambda x, y: (120 - x, 80 - y)) if isd else (lambda x, y: (x, y))
            cx, cy = flip(ev["location"][0], ev["location"][1])
            i0 = pos[ev["index"]]
            pv = None
            for k in range(i0 - 1, -1, -1):
                pe = EV[k]
                if pe.get("possession") != ev.get("possession"):
                    break
                if pe.get("location"):
                    pv = pe
                    break
            if pv is None:
                continue
            px, py = flip(pv["location"][0], pv["location"][1])
            dx = (cx - px) * 105 / 120
            dy = (cy - py) * 68 / 80
            bin_[a] = (dx, dy, float(np.hypot(dx, dy)), float(np.arctan2(dy, dx)))
    bidf = pd.DataFrame([(a, *v) for a, v in bin_.items()], columns=["aid"] + BALLIN)
    return dft.merge(bidf, on="aid", how="left")


def build_feature_matrix(parquet_path, labels_csv, events_dir):
    """Return the full frame with the 31 FEATURES + keys/labels, ordered by seq."""
    df = pd.read_parquet(parquet_path)
    df["match_id"] = df["match_id"].astype(str)
    dft = df.sort_values(["match_id", "seq_id", "ev_pos"]).reset_index(drop=True)
    dft = _build_temporal(dft)
    dft = _build_ballin(dft, labels_csv, events_dir)
    return dft
