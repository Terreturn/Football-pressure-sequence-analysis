"""
Stage 1 — raw StatsBomb (events + 360) JSON  ->  high-press sequence frames + labels
====================================================================================
Clean, self-contained re-implementation of the high-press detection + labelling
pipeline, faithful to the `high-press-detection` skill (SKILL.md §1–§5) for the
DETECTION half and to build_all_matches_v2.py for the LABEL state machine.

NOTE: the original v1 builder (build_all_matches.py) was lost; it produced a richer
cached schema (pre_context / synthetic frames, source/pressure_level, etc.). This
script does NOT reproduce `all_*_v2.csv` byte-for-byte — those remain frozen as the
canonical data behind the trained model. This is a spec-faithful builder for NEW raw
data (e.g. the 2026 World Cup set) with a lean schema.

Pipeline (SKILL.md):
  §3 high-press EVENT  = type∈{Pass,Carry,Miscontrol,Dribble}, not a set piece,
       carrier in OWN HALF (x<60; data is possession-attack-normalised), P_total>0.65, ≥1 defender ≤6.4 m.
  §4 high-press SEQUENCE = maximal chain of high-press events, same period, same
       carrier team, consecutive gap ≤ 5 s.
  Labels = v2 terminal-first state machine (success/fail/neutral/censored).

Outputs (join key = match_id, seq_id, ev_pos):
  OUT_SEQ  — one row per high-press frame  (lean schema)
  OUT_LAB  — one row per labelled anchor   (v2 LAB_COLS schema)
"""
import os, sys, csv, json, glob, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)                       # repo root (scripts/ lives one level down)
for _p in (os.path.join(_REPO, "src"), _REPO):       # pressure model lives in src/ (or root)
    if os.path.exists(os.path.join(_p, "pressure_distance_v2.py")):
        sys.path.insert(0, _p); break
from pressure_distance_v2 import PressureParams, total_pressure

# ── config (point these at any StatsBomb dataset) ─────────────────────────────
# ── DATA PATHS — two INDEPENDENT folders (they need NOT share a parent) ──────
# Point one at the events folder, one at the 360 folder. Set them directly below
# OR via env vars EVENTS_DIR / F360_DIR. Files are paired across the
# two folders by match_id (the *.json filename); unpaired files are skipped (main()).
EVENTS_DIR   = os.environ.get("EVENTS_DIR")    # folder containing event *.json
F360_DIR     = os.environ.get("F360_DIR")      # folder containing 360 freeze-frame *.json

# Convenience defaults for any path left unset above: look under one root
# STATSBOMB_DIR, expecting subfolders  events/  and  360/ .
_ROOT        = os.environ.get("STATSBOMB_DIR", os.path.dirname(_REPO))
EVENTS_DIR   = EVENTS_DIR or os.path.join(_ROOT, "events")
F360_DIR     = F360_DIR   or os.path.join(_ROOT, "360")
OUT_SEQ      = os.environ.get("S1_OUT_SEQ", os.path.join(_REPO, "sequences.csv"))
OUT_LAB      = os.environ.get("S1_OUT_LAB", os.path.join(_REPO, "labels.csv"))
LIMIT        = int(os.environ.get("S1_LIMIT", "0"))   # >0 = only first N matches (debug)

params = PressureParams()
P_THRESH, D_STAR, T_GAP = 0.65, params.D_star_player, 5.0
HP_EVENT_TYPES = {"Pass", "Carry", "Miscontrol", "Dribble"}     # SKILL §3 cond 1
SETPIECE_PP    = {"From Free Kick", "From Corner", "From Throw In", "From Kick Off"}

# ── v2 label state machine (copied verbatim; the only intact half of the old code) ──
ANCHOR_TYPES = {"Pass", "Carry", "Dribble", "Shot"}
ACTION_TYPES = {"Pass", "Carry", "Dribble", "Shot"}
GK_COLLECT   = {"Collected", "Smother", "Keeper Sweeper", "Claim", "Punch"}
D_MIN, D_SWITCH, CLEAR_HIGH_LEN, RELIEF_THRESH = 3.0, 25.0, 35.0, 0.65
OUTCOME_TAG = {
    "shot": "fail", "foul_conceded": "fail", "keeper_collect": "success",
    "clearance_lost": "success", "regain_live": "success", "forced_out": "success",
    "self_out": "fail", "throw_def": "success", "throw_atk": "neutral",
    "freekick_def": "success", "forward_progress": "fail", "long_switch": "fail",
    "contained": "neutral", "censored": "censored"}
TERMINAL_STATE = {
    "shot": True, "foul_conceded": True, "keeper_collect": True, "clearance_lost": True,
    "regain_live": True, "forced_out": True, "self_out": True, "throw_def": True,
    "throw_atk": True, "freekick_def": True, "forward_progress": False,
    "long_switch": False, "contained": False, "censored": False}
RETAINED_STATES = {"forward_progress", "long_switch", "contained"}

# ── geometry / coordinate helpers ─────────────────────────────────────────────
def sb_to_m(x, y):
    return x * 105.0 / 120.0, y * 68.0 / 80.0

def _attnorm(e, x, y):
    """Possession-attack-normalised point. StatsBomb event/360 coordinates are already
    given with the in-possession team attacking toward x=120, so the carrier's own half
    is simply x<60. A defending-team on-ball event is point-reflected into the same frame.
    (No matches.json / home-team needed — see the event 'attacking_direction' field.)"""
    if e["team"]["id"] != e["possession_team"]["id"]:
        return 120.0 - x, 80.0 - y
    return x, y

def ts_seconds(ts):
    h, m, s = ts.split(":"); return int(h) * 3600 + int(m) * 60 + float(s)

def build_f360(raw_360):
    """{event_uuid: [{x,y,keeper}, ...]}  — opponents (teammate=False) of the actor."""
    out = {}
    for fr in raw_360:
        ff = fr.get("freeze_frame")
        if not ff:
            continue
        opps = [{"x": p["location"][0], "y": p["location"][1], "keeper": bool(p.get("keeper"))}
                for p in ff if p.get("location") is not None and not p.get("teammate")]
        out[fr["event_uuid"]] = opps
    return out

def pj_of_event(e, f360):
    """Post-hoc P_total at an event's freeze frame (actor=event location, GK excluded)."""
    loc = e.get("location"); opps = f360.get(e.get("id"))
    if not loc or not opps:
        return None
    actor = sb_to_m(loc[0], loc[1])
    defs = [sb_to_m(o["x"], o["y"]) for o in opps if not o["keeper"]]
    if not defs:
        return None
    return total_pressure(np.array(actor), defs, params, include_boundary=True,
                          use_all_lines=False)["P_total"]

def _direction(e):
    """Attack-normalised ball displacement (metric, forward=+x): (None, dx, dy, mag, theta_deg)."""
    loc = e.get("location"); t = e["type"]["name"]
    end = ((e.get("pass") or {}).get("end_location") if t == "Pass" else
           (e.get("carry") or {}).get("end_location") if t == "Carry" else
           (e.get("shot") or {}).get("end_location") if t == "Shot" else None)
    if not loc or not end:
        return None
    x0, y0 = _attnorm(e, loc[0], loc[1])
    x1, y1 = _attnorm(e, end[0], end[1])
    dx = (x1 - x0) * 105.0 / 120.0
    dy = (y1 - y0) * 68.0 / 80.0
    return (None, dx, dy, float(np.hypot(dx, dy)), float(np.degrees(np.arctan2(dy, dx))))

# ── §3 high-press event test ──────────────────────────────────────────────────
def high_press_metrics(e, f360):
    if e["type"]["name"] not in HP_EVENT_TYPES:
        return None
    if ((e.get("play_pattern") or {}).get("name")) in SETPIECE_PP:
        return None
    loc = e.get("location")
    if not loc:
        return None
    xn, yn = _attnorm(e, loc[0], loc[1])
    if xn >= 60.0:                                      # carrier not in own half
        return None
    opps = f360.get(e.get("id"))
    if not opps:
        return None
    actor = np.array(sb_to_m(loc[0], loc[1]))           # pressure is distance-based → flip-invariant
    defs = [sb_to_m(o["x"], o["y"]) for o in opps if not o["keeper"]]
    if not defs:
        return None
    P = total_pressure(actor, defs, params, include_boundary=True, use_all_lines=False)["P_total"]
    dists = [float(np.hypot(actor[0] - d[0], actor[1] - d[1])) for d in defs]
    n_in = sum(1 for d in dists if d <= D_STAR)
    if P <= P_THRESH or n_in < 1:                       # cond 4 + cond 5
        return None
    return {"P_total": round(P, 4), "nearest_def_m": round(min(dists), 2), "n_in_D": n_in,
            "actor_x_norm": round(xn, 2), "actor_y": round(yn, 1)}

# ── §4 sequence grouping ──────────────────────────────────────────────────────
def detect_sequences(match_id, EV, f360):
    frames, seq_id, ev_pos, prev = [], 0, 0, None
    for e in EV:                                         # EV sorted by index = chronological
        m = high_press_metrics(e, f360)
        if m is None:
            continue
        t, period, team = ts_seconds(e["timestamp"]), e["period"], e["team"]["id"]
        new_seq = (prev is None or period != prev[0] or team != prev[1] or (t - prev[2]) > T_GAP)
        seq_id, ev_pos = (seq_id + 1, 0) if new_seq else (seq_id, ev_pos + 1)
        frames.append({
            "match_id": match_id, "seq_id": seq_id, "ev_pos": ev_pos, "frame_role": "high_press",
            "id": e["id"], "period": period, "timestamp": e["timestamp"],
            "minute": e.get("minute"), "second": e.get("second"), "type": e["type"]["name"],
            "play_pattern": (e.get("play_pattern") or {}).get("name"),
            "team_name": (e.get("team") or {}).get("name"),
            "player_name": (e.get("player") or {}).get("name"), **m})
        prev = (period, team, t)
    return frames

# ── label state machine (verbatim v2, B.* replaced by local helpers) ──────────
def _mk(state, direction, resolution=None, is_penalty=False, is_clearance=False, kickoff_seen=False):
    dx, dy, mag, th = (direction[1:] if direction else (None, None, None, None))
    return {"state": state, "outcome_tag": OUTCOME_TAG[state], "terminal": TERMINAL_STATE[state],
            "dir_dx": dx, "dir_dy": dy, "dir_mag": mag, "dir_theta_deg": th,
            "dir_mask": 1 if state in RETAINED_STATES else 0, "censored": state == "censored",
            "resolution_id": resolution["id"] if resolution else "",
            "resolution_pp": resolution["play_pattern"]["name"] if resolution else "",
            "is_penalty": bool(is_penalty), "is_clearance": bool(is_clearance),
            "kickoff_seen": bool(kickoff_seen)}

def _is_clearance_like(e):
    t = e["type"]["name"]
    if t == "Clearance":
        return True
    if t == "Pass":
        p = e.get("pass", {}) or {}
        if (p.get("height") or {}).get("name") == "High Pass" and (p.get("length") or 0) >= CLEAR_HIGH_LEN:
            return True
    return False

def _receiver_relieved(e, EV, POS, f360):
    if e["type"]["name"] != "Pass":
        return None
    recip = ((e.get("pass") or {}).get("recipient") or {}).get("id")
    if not recip:
        return None
    poss = e["possession"]
    for nx in EV[POS[e["index"]] + 1:]:
        if nx["possession"] != poss:
            break
        if (nx.get("player") or {}).get("id") == recip:
            pj = pj_of_event(nx, f360)
            if pj is not None:
                return pj < RELIEF_THRESH
    return None

def _retained_state(e, EV, POS, f360, direction):
    if direction is None:
        return _mk("censored", direction)
    _, dx, dy, mag, th = direction
    if mag < D_MIN:
        return _mk("contained", direction)
    if abs(th) <= 60:
        return _mk("forward_progress", direction)
    if abs(th) >= 120:
        return _mk("contained", direction)
    if mag >= D_SWITCH:
        if _receiver_relieved(e, EV, POS, f360) is False:
            return _mk("contained", direction)
        return _mk("long_switch", direction)
    return _mk("contained", direction)

def classify_anchor(e, EV, POS, TEAM_IDS, f360):
    atk = e["team"]["id"]; deff = next(t for t in TEAM_IDS if t != atk); poss = e["possession"]
    direction = _direction(e)
    if e["type"]["name"] == "Shot":
        return _mk("shot", direction)
    chg = None; foul_def = foul_pen = keeper_collected = False
    for nx in EV[POS[e["index"]] + 1:]:
        if nx["possession"] != poss:
            chg = nx; break
        tn = nx["type"]["name"]
        if tn == "Foul Committed" and nx["team"]["id"] == deff:
            foul_def = True; foul_pen = bool((nx.get("foul_committed") or {}).get("penalty"))
        if tn == "Goalkeeper" and nx["team"]["id"] == deff:
            if ((nx.get("goalkeeper") or {}).get("type") or {}).get("name") in GK_COLLECT:
                keeper_collected = True
        if nx["team"]["id"] == atk and tn in ACTION_TYPES:
            return _retained_state(e, EV, POS, f360, direction)
    if chg is None:
        return _mk("censored", direction)
    pp = chg["play_pattern"]["name"]; restart_team = chg["possession_team"]["id"]
    if pp == "From Kick Off":
        return _mk("censored", direction, kickoff_seen=True)
    if foul_def or (pp == "From Free Kick" and restart_team == atk):
        return _mk("foul_conceded", direction, resolution=chg, is_penalty=foul_pen)
    if keeper_collected or pp == "From Keeper":
        return _mk("keeper_collect", direction, resolution=chg)
    if pp in ("From Corner", "From Goal Kick"):
        return _mk("forced_out" if restart_team == deff else "self_out", direction, resolution=chg)
    if pp == "From Throw In":
        return _mk("throw_def" if restart_team == deff else "throw_atk", direction, resolution=chg)
    if pp == "From Free Kick":
        return _mk("freekick_def", direction, resolution=chg)
    if restart_team == deff:
        if _is_clearance_like(e):
            return _mk("clearance_lost", direction, resolution=chg, is_clearance=True)
        return _mk("regain_live", direction, resolution=chg)
    return _retained_state(e, EV, POS, f360, direction)

def label_frames(match_id, EV, POS, TEAM_IDS, f360, frames):
    ev_by_id = {e["id"]: e for e in EV}; out = []
    for a in frames:
        if a["frame_role"] != "high_press" or a["type"] not in ANCHOR_TYPES:
            continue
        e = ev_by_id.get(a["id"])
        if e is None:
            continue
        out.append({"match_id": match_id, "seq_id": a["seq_id"], "ev_pos": a["ev_pos"],
                    "anchor_event_id": a["id"], "type": a["type"], "team_name": a["team_name"],
                    "player_name": a["player_name"], "minute": a["minute"], "second": a["second"],
                    **classify_anchor(e, EV, POS, TEAM_IDS, f360)})
    return out

SEQ_COLS = ["match_id", "seq_id", "ev_pos", "frame_role", "id", "period", "timestamp",
            "minute", "second", "type", "play_pattern", "team_name", "player_name",
            "actor_x_norm", "actor_y", "P_total", "nearest_def_m", "n_in_D"]
LAB_COLS = ["match_id", "seq_id", "ev_pos", "anchor_event_id", "type", "team_name",
            "player_name", "minute", "second", "state", "outcome_tag", "terminal",
            "dir_dx", "dir_dy", "dir_mag", "dir_theta_deg", "dir_mask", "censored",
            "resolution_id", "resolution_pp", "is_penalty", "is_clearance", "kickoff_seen"]

def _match_ids(folder):
    return {os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(os.path.join(folder, "*.json"))}

def main():
    # ── validate the two data folders ──
    for label, path in [("EVENTS_DIR", EVENTS_DIR), ("F360_DIR", F360_DIR)]:
        if not os.path.isdir(path):
            raise SystemExit(f"[s1] {label} is not a folder:\n      {path}\n"
                             f"      set env var {label} (or edit s1_build_sequences.py).")

    # ── pair events ↔ 360 by match_id; guard against mismatch ──
    ev_ids, f3_ids = _match_ids(EVENTS_DIR), _match_ids(F360_DIR)
    matched = sorted(ev_ids & f3_ids)                 # need events AND 360
    only_ev, only_f3 = ev_ids - f3_ids, f3_ids - ev_ids

    print(f"EVENTS_DIR: {EVENTS_DIR}  ({len(ev_ids)} files)")
    print(f"F360_DIR  : {F360_DIR}  ({len(f3_ids)} files)")
    print(f"usable (events ∩ 360): {len(matched)}")
    if only_ev: print(f"  ⚠ {len(only_ev)} event file(s) have NO matching 360 → skipped  e.g. {sorted(only_ev)[:3]}")
    if only_f3: print(f"  ⚠ {len(only_f3)} 360 file(s) have NO matching events → skipped  e.g. {sorted(only_f3)[:3]}")
    if not matched:
        raise SystemExit("[s1] no match has BOTH events and 360.\n"
                         "      EVENTS_DIR / F360_DIR likely point at non-corresponding folders.")
    if LIMIT:
        matched = matched[:LIMIT]
    print(f"\nprocessing {len(matched)} matches | P>{P_THRESH}, own-half, gap≤{T_GAP}s\n")

    sf = open(OUT_SEQ, "w", newline="", encoding="utf-8")
    lf = open(OUT_LAB, "w", newline="", encoding="utf-8")
    sw = csv.DictWriter(sf, fieldnames=SEQ_COLS, extrasaction="ignore", restval="")
    lw = csv.DictWriter(lf, fieldnames=LAB_COLS, extrasaction="ignore", restval="")
    sw.writeheader(); lw.writeheader()

    from collections import defaultdict
    n_ok = n_skip = tot_frames = tot_anchor = tot_seq = 0
    tag_dist = defaultdict(int); t0 = time.perf_counter()
    for k, mid in enumerate(matched, 1):
        ef = os.path.join(EVENTS_DIR, mid + ".json")
        f360p = os.path.join(F360_DIR, mid + ".json")
        try:
            events = json.load(open(ef, encoding="utf-8"))
            f360 = build_f360(json.load(open(f360p, encoding="utf-8")))
            EV = sorted(events, key=lambda e: e["index"]); POS = {e["index"]: i for i, e in enumerate(EV)}
            # mismatch guard: the 360 file must actually be FOR this match (event_uuids overlap)
            if not any(e.get("id") in f360 for e in EV):
                n_skip += 1
                print(f"[{k:3d}/{len(matched)}] {mid}  ⚠ 360 has no event_uuid overlap with events → skipped")
                continue
            TEAM_IDS = sorted({e["possession_team"]["id"] for e in EV})
            frames = detect_sequences(mid, EV, f360)
            labs = label_frames(mid, EV, POS, TEAM_IDS, f360, frames)
            for r in frames: sw.writerow(r)
            for r in labs: lw.writerow(r); tag_dist[r["outcome_tag"]] += 1
            n_seq = len({r["seq_id"] for r in frames})
            tot_frames += len(frames); tot_anchor += len(labs); tot_seq += n_seq; n_ok += 1
            print(f"[{k:3d}/{len(matched)}] {mid}  seqs={n_seq:3d}  frames={len(frames):4d}  anchors={len(labs):4d}")
        except Exception as ex:
            n_skip += 1; print(f"[{k:3d}/{len(matched)}] {mid}  ERROR: {ex}")
    sf.close(); lf.close()
    print("\n" + "=" * 56)
    print(f"ok {n_ok} | skip {n_skip} | {(time.perf_counter()-t0)/60:.1f} min")
    print(f"sequences {tot_seq} | frames {tot_frames} -> {OUT_SEQ}")
    print(f"labelled anchors {tot_anchor} -> {OUT_LAB}")
    if tot_anchor:
        print("outcome_tag:", {t: f"{tag_dist[t]} ({tag_dist[t]/tot_anchor*100:.1f}%)"
                               for t in ["success", "neutral", "fail", "censored"]})

if __name__ == "__main__":
    main()
