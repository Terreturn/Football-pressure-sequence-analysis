"""
Apply a trained HPN outcome bundle to a (held-out) season and reproduce the
Stage 4-5 team pressing analysis: intensity-vs-efficiency map, efficiency
ranking, pressing-style fingerprint, VAEP-style step valuation and its drivers.

The scored season should be fully held out from the bundle's training season, so
anchors are scored DIRECTLY with the loaded model — no out-of-fold needed (the
notebook uses OOF only because it trains and analyses on the same season).

Per-season calibration: the ranking model inside the bundle is frozen; the
isotonic layer should be season-specific when base rates drift (fit it with
calibrate_season.py). Any bundle with {model, label_encoder, features} works.

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
  intensity.csv, efficiency.csv, step_valuation.csv, drivers.csv
  fig_intensity_landscape.png, fig_efficiency.png, fig_style_fingerprint.png,
  fig_drivers.png
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from hpn_features import build_feature_matrix, FEATURES, TEMPORAL, GK

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
os.makedirs(OUTDIR, exist_ok=True)

OUTCOME_CLASSES = ["success", "fail", "neutral"]

# ── build feature matrix via the shared builder (NaN diffs; model subsets to its 17 feats) ─
dft = build_feature_matrix(FEAT_PARQUET, LABELS_CSV, EVENTS_DIR)
lab = pd.read_csv(LABELS_CSV); lab["match_id"] = lab["match_id"].astype(str)
print(f"features: {dft.shape} | matches {dft['match_id'].nunique()} | "
      f"{len(FEATURES)}-col matrix; model scores on its trained feature subset (NaN diffs native to XGBoost)")

# ── load calibrated model + score directly (season held out from training) ────
bundle = joblib.load(MODEL_PATH)
model, le, feats = bundle["model"], bundle["label_encoder"], bundle["features"]
CLS = list(le.classes_)                              # ['fail','neutral','success']
i_s, i_f, i_n = CLS.index("success"), CLS.index("fail"), CLS.index("neutral")
proba = model.predict_proba(dft[feats].astype(float))
print(f"model: {os.path.basename(MODEL_PATH)} | {len(feats)} feats | classes {CLS} | "
      f"calib={bundle.get('calib_method')}")

# ── (cell 26) full-season step table: pressed/press team + vt + labels ────────
labV = lab[lab["outcome_tag"].isin(OUTCOME_CLASSES)]
pressed = labV.set_index(["match_id", "seq_id", "ev_pos"])["team_name"].to_dict()
mteams  = {m: list(v) for m, v in labV.groupby("match_id")["team_name"].unique().items()}

vp = dft[["match_id", "seq_id", "ev_pos", "outcome_tag"]].copy().reset_index(drop=True)
vp["pressed_team"] = [pressed.get((m, s, e)) for m, s, e in zip(vp.match_id, vp.seq_id, vp.ev_pos)]
vp["press_team"]   = [next((t for t in mteams.get(m, []) if t != p), None)
                      for m, p in zip(vp.match_id, vp.pressed_team)]
vp["p_s"] = proba[:, i_s]; vp["p_f"] = proba[:, i_f]

CH = {"d_Ptot": "P_total", "d_maxcar": "max_press_on_carrier", "d_totatt": "total_press_on_attackers",
      "d_maxatt": "max_press_on_attacker", "d_fracpr": "frac_attackers_pressed", "d_trap": "carrier_trap_w",
      "d_neardef": "nearest_def_dist", "d_bestpass": "best_pass_w"}
for c in CH.values(): vp[c] = dft[c].values
for nf, base in CH.items(): vp[nf] = vp.groupby(GK)[base].diff()
vp["dps"] = vp.groupby(GK)["p_s"].diff(); vp["dpf"] = vp.groupby(GK)["p_f"].diff()
vp["vt"]  = vp["dps"] - vp["dpf"]

msk = vp["vt"].notna()
allabs = np.concatenate([vp.loc[msk, "dps"].abs(), vp.loc[msk, "dpf"].abs()])
DELTA_G = float(np.percentile(allabs, 25))
def classify_g(ds, df_, vt_, dd):
    if np.isnan(ds): return None
    s = 1 if ds > dd else (-1 if ds < -dd else 0)
    f = 1 if df_ > dd else (-1 if df_ < -dd else 0)
    if s == 0 and f == 0: return "Negligible"
    if s == 1 and f == 1: return "Risky-fav" if vt_ > 0 else "Risky-adv"
    if s == 1 and f <= 0: return "Effective"
    if s <= 0 and f == 1: return "Beaten"
    return "De-escalation"
vp["label"] = [classify_g(a, b, c, DELTA_G) for a, b, c in zip(vp.dps, vp.dpf, vp.vt)]
print(f"global delta = {DELTA_G:.4f} | step transitions {int(msk.sum())} | teams {vp['press_team'].nunique()}")
print(vp.loc[msk, "label"].value_counts().to_string())

# ── (cell 28) intensity + efficiency tables ───────────────────────────────────
for c in ["carrier_x_norm", "n_pressers_on_carrier"]:
    if c not in vp.columns: vp[c] = dft[c].values
vp["seqkey"] = vp["match_id"] + "_" + vp["seq_id"].astype(str)

gA = vp.groupby("press_team"); nm = gA["match_id"].nunique()
intensity = pd.DataFrame({
    "n_match": nm, "frames_per_match": gA.size() / nm,
    "seq_per_match": gA["seqkey"].nunique() / nm, "steps_per_seq": gA.size() / gA["seqkey"].nunique(),
    "mean_Ptot": gA["P_total"].mean(),
})
# ── team regain-efficiency: 4 complementary metrics (per-step mean_vt dropped: dilutes long seqs)
#    1 regain_rate (outcome, model-free) · 2 mean_p_s/p_f (calibrated expected prob)
#    3a vt_per_seq = sum(v_t)/n_seq (length-neutral value) · 3b resid_vt (length-adjusted residual)
#    4 eff_over_EB = Effective/(Effective+Beaten) = sigmoid(log(E/B)), process quality
sv = vp.dropna(subset=["vt"]).copy()
sv["pos"] = sv["ev_pos"].clip(upper=8)
_league_pos = sv.groupby("pos")["vt"].mean()
sv["exp_vt"] = sv["pos"].map(_league_pos)
gV = sv.groupby("press_team")
lab_eb = sv.groupby("press_team")["label"].value_counts().unstack(fill_value=0)
seqA = vp.groupby(["press_team", "seqkey"]).agg(
    endtag=("outcome_tag", "last"), sum_vt=("vt", "sum")).reset_index()
gS = seqA.groupby("press_team")
team = pd.DataFrame({
    "n_seq":       gS.size(),
    "regain_rate": gS["endtag"].apply(lambda s: (s == "success").mean()),
    "mean_p_s":    gA["p_s"].mean(), "mean_p_f": gA["p_f"].mean(),
    "vt_per_seq":  gS["sum_vt"].mean(),
    "resid_vt":    gV["vt"].mean() - gV["exp_vt"].mean(),
    "eff_over_EB": lab_eb["Effective"] / (lab_eb["Effective"] + lab_eb["Beaten"]),
}).sort_values("regain_rate", ascending=False)

intensity.round(4).to_csv(os.path.join(OUTDIR, "intensity.csv"))
team.round(4).to_csv(os.path.join(OUTDIR, "efficiency.csv"))
vp.to_csv(os.path.join(OUTDIR, "step_valuation.csv"), index=False)
print("\n=== EFFICIENCY (sorted by regain_rate) ===")
print(team.round(4).to_string())

# ── figure (i): intensity × regain-efficiency quadrant map ────────────────────
# x = per-match total pressure on the carrier I_T = (anchors/match) x mean P_total;
# y = regain rate; bubble size = value/seq; colour = process cleanliness E/(E+B).
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
SURF, INK, SEC, MUT, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CMAP = LinearSegmentedColormap.from_list(  # dataviz blue sequential ramp (steps 200->700)
    "blues_seq", ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
ie = intensity.join(team[["regain_rate", "vt_per_seq", "eff_over_EB"]])
ie["I_T"] = ie["frames_per_match"] * ie["mean_Ptot"]
xv, yv = ie["I_T"].values, ie["regain_rate"].values
val, clean = ie["vt_per_seq"].values, ie["eff_over_EB"].values
xmed, ymed = np.median(xv), np.median(yv)
amin, amax = 110, 620
sz = amin + (val - val.min()) / (val.max() - val.min()) * (amax - amin)

fig, ax = plt.subplots(figsize=(10, 7.6), dpi=200)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
ax.axvline(xmed, color=BASE, ls=(0, (5, 4)), lw=1.1, zorder=1)
ax.axhline(ymed, color=BASE, ls=(0, (5, 4)), lw=1.1, zorder=1)
ax.grid(True, color=GRID, lw=.7, zorder=0); ax.set_axisbelow(True)
xpad = (xv.max() - xv.min()) * .10; ypad = (yv.max() - yv.min()) * .12
ax.set_xlim(xv.min() - xpad, xv.max() + xpad); ax.set_ylim(yv.min() - ypad, yv.max() + ypad)
x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
qs = dict(fontsize=10.5, color=MUT, style="italic", zorder=1)
ax.text(x0 + (xmed - x0) * .5, y1 - .004, "efficient minimalists", ha="center", va="top", **qs)
ax.text(x1 - (x1 - xmed) * .5, y1 - .004, "elite press", ha="center", va="top", **qs)
ax.text(x1 - (x1 - xmed) * .5, y0 + .004, "high grind, low return", ha="center", va="bottom", **qs)
ax.text(x0 + (xmed - x0) * .5, y0 + .004, "passive · ineffective", ha="center", va="bottom", **qs)
sc = ax.scatter(xv, yv, s=sz, c=clean, cmap=CMAP, vmin=clean.min(), vmax=clean.max(),
                edgecolors="white", linewidths=1.1, zorder=4, alpha=.96)
texts = [ax.text(xv[k], yv[k], ie.index[k], fontsize=8, color=INK, zorder=6) for k in range(len(ie))]
try:
    from adjustText import adjust_text
    adjust_text(texts, x=xv, y=yv, ax=ax, expand=(1.2, 1.5), force_text=(.4, .6),
                max_move=40, iter_lim=400,
                arrowprops=dict(arrowstyle="-", color=MUT, lw=.6, alpha=.8))
except Exception:
    for k in range(len(ie)):
        texts[k].set_position((xv[k], yv[k])); texts[k].set_ha("left")
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]: ax.spines[sp].set_color(BASE)
ax.tick_params(colors=MUT, labelsize=9)
ax.set_xlabel("pressing intensity  —  per-match total pressure on the carrier  $I_T$",
              fontsize=11, color=SEC, labelpad=8)
ax.set_ylabel("regain efficiency  —  share of high-press sequences won back",
              fontsize=11, color=SEC, labelpad=8)
ax.set_title(f"{SEASON_LABEL} pressing: intensity vs. regain efficiency",
             fontsize=14, color=INK, fontweight="semibold", pad=30, loc="left")
ax.annotate("bubble size = value per sequence ($\\Sigma v_t/n_{seq}$)   ·   dashed = league median",
            xy=(0, 1.012), xycoords="axes fraction", fontsize=9, color=MUT, va="bottom")
cb = fig.colorbar(sc, ax=ax, pad=.015, fraction=.045)
cb.set_label("process cleanliness  $E/(E{+}B)$", fontsize=9.5, color=SEC)
cb.ax.tick_params(colors=MUT, labelsize=8); cb.outline.set_edgecolor(BASE)
lv = [val.min(), np.median(val), val.max()]
handles = [Line2D([0], [0], marker="o", ls="", markerfacecolor="#bcbcb6", markeredgecolor="white",
           markersize=np.sqrt(amin + (v - val.min()) / (val.max() - val.min()) * (amax - amin)) / 1.6)
           for v in lv]
leg = ax.legend(handles, [f"{v:.3f}" for v in lv], title="value/seq", loc="lower right",
                frameon=False, labelspacing=1.3, handletextpad=1.1, borderpad=1.0,
                fontsize=8.5, title_fontsize=9)
leg.get_title().set_color(SEC); [t.set_color(SEC) for t in leg.get_texts()]
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig_intensity_landscape.png"), dpi=200,
            facecolor=SURF, bbox_inches="tight"); plt.close()

# ── figure (ii): efficiency ───────────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
vps = team["vt_per_seq"].sort_values()
vps.plot.barh(ax=ax[0], color=["#d62728" if v < 0 else "#2ca02c" for v in vps])
ax[0].set_title("Value per sequence ($\\Sigma v_t$/seq) — 2025/26")
team["regain_rate"].sort_values().plot.barh(ax=ax[1], color="#ff7f0e")
ax[1].set_title(f"Actual regain rate — {SEASON_LABEL}")
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "fig_efficiency.png"), dpi=150); plt.close()

# ── figure (iii): pressing-style fingerprint ──────────────────────────────────
STYLE = [("Carrier total press", "P_total", "frame", +1),
         ("Swarm (#pressers)", "n_pressers_on_carrier", "frame", +1),
         ("Tightness (-near-def)", "nearest_def_dist", "frame", -1),
         ("Press height (-carrier x)", "carrier_x_norm", "frame", -1),
         ("Outlet-deny breadth", "frac_attackers_pressed", "frame", +1),
         ("Outlet-deny (-best pass)", "best_pass_w", "frame", -1),
         ("Boundary trap", "carrier_trap_w", "frame", +1),
         ("Press frequency", "seq_per_match", "team", +1),
         ("Press build-up (dcarrier)", "d_Ptot", "frame", +1),
         ("Seq length (steps/seq)", "steps_per_seq", "team", +1)]
cols = {}
for name, src, scope, sgn in STYLE:
    v = intensity[src] if scope == "team" else vp.groupby("press_team")[src].mean()
    cols[name] = sgn * v
# row order: TEAM_ORDER file (one team per line, e.g. final league table) if given,
# else teams sorted by regain rate — works for any competition / season
if TEAM_ORDER and os.path.exists(TEAM_ORDER):
    ROW_ORDER = [ln.strip() for ln in open(TEAM_ORDER, encoding="utf-8") if ln.strip()]
    ROW_TITLE = "rows by TEAM_ORDER file"
else:
    ROW_ORDER = list(team.index)                       # efficiency table order (regain rate)
    ROW_TITLE = "rows by regain rate"
style = pd.DataFrame(cols).reindex([t for t in ROW_ORDER if t in cols[next(iter(cols))].index])
Zt = (style - style.mean()) / style.std()
fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(Zt.values, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
ax.set_xticks(range(len(Zt.columns))); ax.set_xticklabels(Zt.columns, rotation=40, ha="right", fontsize=8)
ax.set_yticks(range(len(Zt.index))); ax.set_yticklabels([f"{i+1}. {t}" for i, t in enumerate(Zt.index)], fontsize=8)
fig.colorbar(im, label="z across teams (red = more aggressive)")
ax.set_title(f"{SEASON_LABEL} pressing-style fingerprint — {ROW_TITLE} (scalar levels + Δ, oriented)")
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "fig_style_fingerprint.png"), dpi=150); plt.close()

# ── (cell 29) drivers: which pressure-channel Δ raise press value v_t? ─────────
import scipy.stats as st
from sklearn.linear_model import LinearRegression
CHNAME = {"d_Ptot": "Carrier total press Δ", "d_maxcar": "Max press on carrier Δ",
          "d_totatt": "Total press on attackers Δ", "d_maxatt": "Max press on attacker Δ",
          "d_fracpr": "Frac attackers pressed Δ", "d_trap": "Boundary trap Δ",
          "d_neardef": "Nearest-def dist Δ", "d_bestpass": "Best pass openness Δ"}
CHCOLS = list(CHNAME)
_sub = vp.dropna(subset=["vt"] + CHCOLS)
drivers = pd.DataFrame({
    "pearson_r_vt":   [st.pearsonr(_sub[c], _sub["vt"])[0] for c in CHCOLS],
    "mean_Effective": [_sub.loc[_sub.label == "Effective", c].mean() for c in CHCOLS],
    "mean_Beaten":    [_sub.loc[_sub.label == "Beaten", c].mean() for c in CHCOLS],
}, index=[CHNAME[c] for c in CHCOLS])
drivers["Eff_minus_Beaten"] = drivers["mean_Effective"] - drivers["mean_Beaten"]
_Z = (_sub[CHCOLS] - _sub[CHCOLS].mean()) / _sub[CHCOLS].std()
_reg = LinearRegression().fit(_Z, _sub["vt"]); drivers["std_reg_beta"] = _reg.coef_
_R2 = _reg.score(_Z, _sub["vt"])
drivers = drivers.sort_values("pearson_r_vt", ascending=False)
drivers.round(4).to_csv(os.path.join(OUTDIR, "drivers.csv"))
print(f"\n=== DRIVERS of v_t (joint linear R^2 = {_R2:.3f}; channels collinear) ===")
print(drivers.round(4).to_string())

fig, ax = plt.subplots(figsize=(8, 5))
drivers["pearson_r_vt"].sort_values().plot.barh(ax=ax, color="#2F6FD0")
ax.axvline(0, color="k", lw=.7)
ax.set_title(f"{SEASON_LABEL} — Pearson r of pressure-channel Δ with press value $v_t$")
ax.set_xlabel("Pearson r with $v_t$")
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "fig_drivers.png"), dpi=150); plt.close()

print(f"\nsaved tables + 4 figures -> {OUTDIR}")
