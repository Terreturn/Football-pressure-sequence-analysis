"""
Stage 3 — high-press anchors  ->  carrier-centric ML feature table
==================================================================
Turns each labelled high-press anchor (all_sequence_labels_v2.csv) into a
19-dim carrier-centric feature vector by reading its StatsBomb freeze frame,
and caches the result to hpn_carrier_features.parquet (the table consumed by the
stage 4/5 ML notebook). Same recipe as the HPN builder; leak-free (current frame
only). Extracted from the ML notebook so the pipeline stage is a standalone script.

Run:  python s3_build_features.py        # builds hpn_carrier_features.parquet
"""
import os, sys, json, math, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)                       # repo root (scripts/ lives one level down)
sys.path.insert(0, os.path.join(_REPO, "src"))
from pressure_distance_v2 import sigmoid_pressure, PressureParams

# Paths are configurable via environment variables (portable / GitHub-friendly).
HPN          = _REPO
DATA_DIR     = os.environ.get("STATSBOMB_DIR", os.path.dirname(HPN))   # parent holds the raw JSON
EVENTS_DIR   = os.environ.get("EVENTS_DIR",   os.path.join(DATA_DIR, "events", "events"))
F360_DIR     = os.environ.get("F360_DIR",     os.path.join(DATA_DIR, "three_sixty", "three_sixty"))
LABELS_CSV   = os.environ.get("LABELS_CSV",   os.path.join(HPN, "all_sequence_labels_v2.csv"))
FEAT_PARQUET = os.environ.get("FEAT_PARQUET", os.path.join(HPN, "hpn_carrier_features.parquet"))

P = PressureParams()
K_PLAYER, K_LINE = P.k_player, P.k_line
D_PLAYER, D_LINE = P.D_star_player, P.D_star_line
D_LANE = 1.2
K_LANE = math.pi / (math.sqrt(3) * 0.6)          # ≈3.02  (narrow pass corridor)
R_SOFT, P_EPS = 2.0, 0.05
TAU = 0.5
SB_L, SB_W, M_L, M_W = 120.0, 80.0, 105.0, 68.0
OUTCOME_CLASSES = ["success", "fail", "neutral"]

FEATURE_NAMES = [
    "P_total", "carrier_enemy_density", "carrier_friendly_density",
    "carrier_dist_boundary", "carrier_x_norm",
    "n_pressers_on_carrier", "nearest_def_dist", "max_press_on_carrier",
    "best_pass_w", "best_forward_pass_w", "n_open_pass", "mean_lane_openness",
    "mean_recv_freedom", "min_recv_freedom", "n_press_on_attackers",
    "total_press_on_attackers", "max_press_on_attacker", "frac_attackers_pressed",
    "carrier_trap_w",
]

def sb_to_m(x, y):
    return x * M_L / SB_L, y * M_W / SB_W

def _perp(p, a, b):
    a, b, p = np.asarray(a), np.asarray(b), np.asarray(p)
    ab = b - a; L2 = float(ab @ ab)
    if L2 == 0:
        return float(np.linalg.norm(p - a)), 0.0
    t = float((p - a) @ ab) / L2
    return float(np.linalg.norm(p - (a + t * ab))), t

def _nearest_boundary(ref):
    cx, cy = float(ref[0]), float(ref[1])
    ds = [cx, M_L - cx, cy, M_W - cy]
    prod = 1.0
    for d in ds:
        prod *= (1.0 - sigmoid_pressure(K_LINE, D_LINE, d))
    return min(ds), 1.0 - prod

def carrier_features(players):
    """players: list of metric dicts {x,y,team,keeper,actor} (attack→+x). 19-feature dict or None."""
    pos = np.array([[p["x"], p["y"]] for p in players], float)
    n = len(players)
    ci = next((i for i, p in enumerate(players) if p["actor"]), None)
    if ci is None:
        return None
    c = pos[ci]
    atk = [i for i in range(n) if players[i]["team"] == "attacker"]
    dfd = [i for i in range(n) if players[i]["team"] == "defender"]
    def_of = [i for i in dfd if not players[i]["keeper"]]
    atk_of = [i for i in atk if not players[i]["keeper"] and not players[i]["actor"]]
    atk_recv = [i for i in atk if not players[i]["actor"]]
    press_targets = [ci] + atk_of
    pr = lambda d: sigmoid_pressure(K_PLAYER, D_PLAYER, d)

    press_on = {}
    for j in def_of:
        for k in sorted(press_targets, key=lambda k: np.linalg.norm(pos[j] - pos[k]))[:3]:
            p = pr(float(np.linalg.norm(pos[j] - pos[k])))
            if p > P_EPS:
                press_on.setdefault(k, []).append(p)
    carrier_press = press_on.get(ci, [])
    att_targets = [k for k in press_on if k in atk_of]

    lanes, recvs, ws, fwd_ws = [], [], [], []
    for r in atk_recv:
        block = 1.0
        for j in def_of:
            e_d, t = _perp(pos[j], c, pos[r])
            if 0.0 <= t <= 1.0:
                block *= (1.0 - sigmoid_pressure(K_LANE, D_LANE, e_d))
            else:
                d_recv = float(np.linalg.norm(pos[j] - pos[r]))
                if d_recv <= R_SOFT:
                    block *= (1.0 - sigmoid_pressure(K_LANE, D_LANE, d_recv))
        free = 1.0
        for j in def_of:
            free *= (1.0 - pr(float(np.linalg.norm(pos[j] - pos[r]))))
        w = 0.5 * block + 0.5 * free
        lanes.append(block); recvs.append(free); ws.append(w)
        if pos[r, 0] - c[0] > 0:
            fwd_ws.append(w)

    def density(idx, R=10.0):
        return int(sum(1 for k in idx if k != ci and np.linalg.norm(c - pos[k]) <= R))

    P_total = (1.0 - float(np.prod([1.0 - pr(float(np.linalg.norm(pos[j] - c))) for j in def_of]))
               if def_of else 0.0)
    dist_b, trap_w = _nearest_boundary(c)
    nearest_def = min((float(np.linalg.norm(c - pos[j])) for j in dfd), default=np.nan)
    att_max_press = [max(press_on.get(k, [0.0])) for k in atk_of]

    return {
        "P_total": float(P_total),
        "carrier_enemy_density": density(dfd),
        "carrier_friendly_density": density(atk),
        "carrier_dist_boundary": float(dist_b),
        "carrier_x_norm": float(c[0] / M_L),
        "n_pressers_on_carrier": int(len(carrier_press)),
        "nearest_def_dist": float(nearest_def),
        "max_press_on_carrier": float(max(carrier_press) if carrier_press else 0.0),
        "best_pass_w": float(max(ws) if ws else 0.0),
        "best_forward_pass_w": float(max(fwd_ws) if fwd_ws else 0.0),
        "n_open_pass": int(sum(1 for w in ws if w > TAU)),
        "mean_lane_openness": float(np.mean(lanes) if lanes else 0.0),
        "mean_recv_freedom": float(np.mean(recvs) if recvs else 0.0),
        "min_recv_freedom": float(min(recvs) if recvs else 0.0),
        "n_press_on_attackers": int(sum(len(press_on[k]) for k in att_targets)),
        "total_press_on_attackers": float(sum(sum(press_on[k]) for k in att_targets)),
        "max_press_on_attacker": float(max((max(press_on[k]) for k in att_targets), default=0.0)),
        "frac_attackers_pressed": float(np.mean([1.0 if m > TAU else 0.0 for m in att_max_press])
                                        if att_max_press else 0.0),
        "carrier_trap_w": float(trap_w),
    }

def players_from_frame(ev, freeze):
    is_def = ev["team"]["id"] != ev["possession_team"]["id"]
    def flip(x, y):
        return (SB_L - x, SB_W - y) if is_def else (x, y)
    players = []
    for p in freeze:
        loc = p.get("location")
        if loc is None:
            continue
        x, y = flip(loc[0], loc[1])
        xm, ym = sb_to_m(min(SB_L, max(0, x)), min(SB_W, max(0, y)))
        tm = p.get("teammate", False)
        team = ("defender" if tm else "attacker") if is_def else ("attacker" if tm else "defender")
        players.append({"x": xm, "y": ym, "team": team,
                        "keeper": p.get("keeper", False), "actor": p.get("actor", False)})
    return players

def build_feature_table():
    lab = pd.read_csv(LABELS_CSV)
    lab = lab[lab["outcome_tag"].isin(OUTCOME_CLASSES)].copy()
    mids = sorted(lab["match_id"].astype(str).unique())
    rows, t0, kept = [], time.perf_counter(), 0
    for ki, mid in enumerate(mids, 1):
        ef = os.path.join(EVENTS_DIR, f"{mid}.json"); ff = os.path.join(F360_DIR, f"{mid}.json")
        if not (os.path.exists(ef) and os.path.exists(ff)):
            continue
        with open(ef, encoding="utf-8") as f: events = json.load(f)
        with open(ff, encoding="utf-8") as f: raw = json.load(f)
        by_id  = {e["id"]: e for e in events}
        frames = {fr["event_uuid"]: fr["freeze_frame"] for fr in raw if fr.get("freeze_frame")}
        for r in lab[lab["match_id"].astype(str) == mid].to_dict("records"):
            eid = r["anchor_event_id"]; ev = by_id.get(eid)
            if ev is None or eid not in frames:
                continue
            players = players_from_frame(ev, frames[eid])
            if not any(p["actor"] for p in players):
                continue
            feats = carrier_features(players)
            if feats is None:
                continue
            feats.update({"match_id": str(mid), "seq_id": int(r["seq_id"]),
                          "ev_pos": int(r["ev_pos"]), "outcome_tag": r["outcome_tag"],
                          "terminal": bool(r["terminal"])})
            rows.append(feats); kept += 1
        if ki % 50 == 0 or ki == len(mids):
            print(f"[{ki:3d}/{len(mids)}] match {mid}  rows={kept}  ({time.perf_counter()-t0:.0f}s)")
    return pd.DataFrame(rows)

def main():
    print(f"feature extractor: {len(FEATURE_NAMES)} features")
    df = build_feature_table()
    df.to_parquet(FEAT_PARQUET, index=False)
    print("built & saved:", df.shape, "->", FEAT_PARQUET)
    print("\noutcome_tag balance:")
    print(df["outcome_tag"].value_counts(normalize=True).round(3).to_string())

if __name__ == "__main__":
    main()
