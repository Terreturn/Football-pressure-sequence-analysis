"""
Shared season-analysis library — the single source of truth for the Stage-5
pressing analysis. Both the S5 notebook (s5_press_analysis.ipynb) and the CLI
(apply_season.py) call these functions, so the notebook narrative and the batch
script can never drift apart.

Contents
  check_lineage(bundle, match_ids)   warn-and-continue lineage check (train/calib overlap)
  score(dft, bundle)                 calibrated probabilities for every anchor
  build_step_table(dft, lab, p_s, p_f)
                                     per-step table: press/pressed team, channel Δ,
                                     v_t = Δp_s − Δp_f, step labels (Effective/Beaten/…)
  team_tables(vp)                    (intensity, efficiency) per-team tables
  fingerprint_matrix(vp, intensity, team, team_order_file=None)
                                     z-scored 10-descriptor style matrix (+ row-order title)
  drivers_table(vp)                  pressure-channel Δ vs v_t (Pearson r, E/B means, std β) + joint R²
  fig_intensity_efficiency / fig_efficiency / fig_fingerprint / fig_drivers
                                     matplotlib Figures (caller decides show/save)
  write_outputs(outdir, ...)         4 csv + figures + lineage.txt

All functions are side-effect-free except write_outputs. No matplotlib backend
is forced here — the CLI sets Agg itself; notebooks keep their inline backend.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from hpn_features import GK

OUTCOME_CLASSES = ["success", "fail", "neutral"]

# pressure channels: step-table column -> source feature
CHANNELS = {"d_Ptot": "P_total", "d_maxcar": "max_press_on_carrier",
            "d_totatt": "total_press_on_attackers", "d_maxatt": "max_press_on_attacker",
            "d_fracpr": "frac_attackers_pressed", "d_trap": "carrier_trap_w",
            "d_neardef": "nearest_def_dist", "d_bestpass": "best_pass_w"}
CHANNEL_NAMES = {"d_Ptot": "Carrier total press Δ", "d_maxcar": "Max press on carrier Δ",
                 "d_totatt": "Total press on attackers Δ", "d_maxatt": "Max press on attacker Δ",
                 "d_fracpr": "Frac attackers pressed Δ", "d_trap": "Boundary trap Δ",
                 "d_neardef": "Nearest-def dist Δ", "d_bestpass": "Best pass openness Δ"}

# 10 style descriptors: (label, source column, frame|team scope, orientation sign)
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

# dataviz palette (light surface)
_SURF, _INK, _SEC, _MUT = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
_GRID, _BASE = "#e1e0d9", "#c3c2b7"
_CMAP = LinearSegmentedColormap.from_list(   # blue sequential ramp (steps 200->700)
    "blues_seq", ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


# ── lineage ────────────────────────────────────────────────────────────────────
def check_lineage(bundle, match_ids):
    """Warn (never block) when scored matches overlap the bundle's train/calib sets.

    Returns (seen_train, seen_calib, messages) — the overlaps as sets plus the
    human-readable lines (also written to lineage.txt by write_outputs).
    """
    mids = {str(m) for m in match_ids}
    seen_tr = mids & {str(m) for m in bundle.get("train_matches", [])}
    seen_ca = mids & {str(m) for m in bundle.get("calib_matches", [])}
    msgs = []
    if "train_matches" not in bundle:
        msgs.append("[lineage] bundle has no train_matches field (older bundle) — "
                    "training-overlap check skipped")
    if seen_tr:
        m = (f"[lineage] WARNING: {len(seen_tr)}/{len(mids)} matches were in the model's "
             f"TRAINING set — their scores are in-sample and team aggregates may be "
             f"optimistic. For the training season use the OOF analysis (S4).")
        msgs.append(m); warnings.warn(m)
    if seen_ca:
        msgs.append(f"[lineage] note: {len(seen_ca)}/{len(mids)} matches were used to fit "
                    f"the isotonic calibration layer (mild, monotone-map-only effect).")
    if not seen_tr and not seen_ca and "train_matches" in bundle:
        msgs.append(f"[lineage] OK: 0/{len(mids)} matches seen in training/calibration — "
                    f"fully held-out.")
    for m in msgs:
        print(m)
    return seen_tr, seen_ca, msgs


# ── scoring ────────────────────────────────────────────────────────────────────
def score(dft, bundle):
    """Calibrated class probabilities for every anchor row of dft.

    Returns (p_s, p_f) arrays aligned with dft. Columns come from
    bundle["features"] so scoring always follows the loaded model.
    """
    model, le, feats = bundle["model"], bundle["label_encoder"], bundle["features"]
    CLS = list(le.classes_)
    proba = model.predict_proba(dft[list(feats)].astype(float))
    return proba[:, CLS.index("success")], proba[:, CLS.index("fail")]


# ── step table ─────────────────────────────────────────────────────────────────
def _classify(ds, df_, vt_, dd):
    if np.isnan(ds): return None
    s = 1 if ds > dd else (-1 if ds < -dd else 0)
    f = 1 if df_ > dd else (-1 if df_ < -dd else 0)
    if s == 0 and f == 0: return "Negligible"
    if s == 1 and f == 1: return "Risky-fav" if vt_ > 0 else "Risky-adv"
    if s == 1 and f <= 0: return "Effective"
    if s <= 0 and f == 1: return "Beaten"
    return "De-escalation"


def build_step_table(dft, lab, p_s, p_f):
    """Per-step valuation table.

    dft: feature matrix (build_feature_matrix output, season-ordered)
    lab: the season's Stage-1 labels csv as a DataFrame (match_id as str)
    p_s/p_f: calibrated probabilities aligned with dft (from score())

    Returns vp with press/pressed team, channel levels + within-sequence Δ,
    v_t = Δp_s − Δp_f and the step label (Effective/Beaten/Risky-fav/Risky-adv/
    De-escalation/Negligible; None on each sequence's first frame).
    """
    labV = lab[lab["outcome_tag"].isin(OUTCOME_CLASSES)]
    pressed = labV.set_index(["match_id", "seq_id", "ev_pos"])["team_name"].to_dict()
    mteams = {m: list(v) for m, v in labV.groupby("match_id")["team_name"].unique().items()}

    vp = dft[["match_id", "seq_id", "ev_pos", "outcome_tag"]].copy().reset_index(drop=True)
    vp["pressed_team"] = [pressed.get((m, s, e)) for m, s, e in zip(vp.match_id, vp.seq_id, vp.ev_pos)]
    vp["press_team"] = [next((t for t in mteams.get(m, []) if t != p), None)
                        for m, p in zip(vp.match_id, vp.pressed_team)]
    vp["p_s"] = np.asarray(p_s); vp["p_f"] = np.asarray(p_f)

    for c in CHANNELS.values():
        vp[c] = dft[c].values
    for nf, base in CHANNELS.items():
        vp[nf] = vp.groupby(GK)[base].diff()
    vp["dps"] = vp.groupby(GK)["p_s"].diff(); vp["dpf"] = vp.groupby(GK)["p_f"].diff()
    vp["vt"] = vp["dps"] - vp["dpf"]

    msk = vp["vt"].notna()
    allabs = np.concatenate([vp.loc[msk, "dps"].abs(), vp.loc[msk, "dpf"].abs()])
    delta_g = float(np.percentile(allabs, 25))
    vp["label"] = [_classify(a, b, c, delta_g) for a, b, c in zip(vp.dps, vp.dpf, vp.vt)]

    for c in ["carrier_x_norm", "n_pressers_on_carrier"]:
        if c not in vp.columns:
            vp[c] = dft[c].values
    vp["seqkey"] = vp["match_id"] + "_" + vp["seq_id"].astype(str)
    print(f"global delta = {delta_g:.4f} | step transitions {int(msk.sum())} | "
          f"teams {vp['press_team'].nunique()}")
    return vp


# ── team tables ────────────────────────────────────────────────────────────────
def team_tables(vp):
    """(intensity, efficiency) per pressing team.

    intensity: volume/frequency/length/per-frame pressure.
    efficiency: 4 complementary regain metrics (per-step mean_vt deliberately
    dropped — dividing by step count penalises teams that run longer sequences):
      1 regain_rate (outcome, model-free) · 2 mean_p_s/p_f (calibrated expected prob)
      3a vt_per_seq = sum(v_t)/n_seq · 3b resid_vt (length-adjusted residual)
      4 eff_over_EB = Effective/(Effective+Beaten), process quality
    """
    gA = vp.groupby("press_team"); nm = gA["match_id"].nunique()
    intensity = pd.DataFrame({
        "n_match": nm, "frames_per_match": gA.size() / nm,
        "seq_per_match": gA["seqkey"].nunique() / nm,
        "steps_per_seq": gA.size() / gA["seqkey"].nunique(),
        "mean_Ptot": gA["P_total"].mean(),
    })

    sv = vp.dropna(subset=["vt"]).copy()
    sv["pos"] = sv["ev_pos"].clip(upper=8)
    sv["exp_vt"] = sv["pos"].map(sv.groupby("pos")["vt"].mean())
    gV = sv.groupby("press_team")
    lab_eb = sv.groupby("press_team")["label"].value_counts().unstack(fill_value=0)
    seqA = vp.groupby(["press_team", "seqkey"]).agg(
        endtag=("outcome_tag", "last"), sum_vt=("vt", "sum")).reset_index()
    gS = seqA.groupby("press_team")
    team = pd.DataFrame({
        "n_seq": gS.size(),
        "regain_rate": gS["endtag"].apply(lambda s: (s == "success").mean()),
        "mean_p_s": gA["p_s"].mean(), "mean_p_f": gA["p_f"].mean(),
        "vt_per_seq": gS["sum_vt"].mean(),
        "resid_vt": gV["vt"].mean() - gV["exp_vt"].mean(),
        "eff_over_EB": lab_eb["Effective"] / (lab_eb["Effective"] + lab_eb["Beaten"]),
    }).sort_values("regain_rate", ascending=False)
    return intensity, team


# ── style fingerprint ──────────────────────────────────────────────────────────
def fingerprint_matrix(vp, intensity, team, team_order_file=None):
    """Z-scored 10-descriptor style matrix (rows = teams, oriented so higher =
    more aggressive). Row order: TEAM_ORDER file (one team per line, e.g. final
    league table) if given, else regain-rate order — works for any competition."""
    cols = {}
    for name, src, scope, sgn in STYLE:
        v = intensity[src] if scope == "team" else vp.groupby("press_team")[src].mean()
        cols[name] = sgn * v
    if team_order_file and os.path.exists(team_order_file):
        row_order = [ln.strip() for ln in open(team_order_file, encoding="utf-8") if ln.strip()]
        row_title = "rows by TEAM_ORDER file"
    else:
        row_order = list(team.index)
        row_title = "rows by regain rate"
    style = pd.DataFrame(cols).reindex([t for t in row_order if t in cols[next(iter(cols))].index])
    Zt = (style - style.mean()) / style.std()
    return Zt, row_title


# ── drivers ────────────────────────────────────────────────────────────────────
def drivers_table(vp):
    """Pressure-channel Δ vs v_t: Pearson r, Effective/Beaten means, standardized
    joint-regression β. Returns (drivers, joint_R2)."""
    import scipy.stats as st
    from sklearn.linear_model import LinearRegression
    chcols = list(CHANNEL_NAMES)
    sub = vp.dropna(subset=["vt"] + chcols)
    drivers = pd.DataFrame({
        "pearson_r_vt": [st.pearsonr(sub[c], sub["vt"])[0] for c in chcols],
        "mean_Effective": [sub.loc[sub.label == "Effective", c].mean() for c in chcols],
        "mean_Beaten": [sub.loc[sub.label == "Beaten", c].mean() for c in chcols],
    }, index=[CHANNEL_NAMES[c] for c in chcols])
    drivers["Eff_minus_Beaten"] = drivers["mean_Effective"] - drivers["mean_Beaten"]
    Z = (sub[chcols] - sub[chcols].mean()) / sub[chcols].std()
    reg = LinearRegression().fit(Z, sub["vt"])
    drivers["std_reg_beta"] = reg.coef_
    r2 = reg.score(Z, sub["vt"])
    return drivers.sort_values("pearson_r_vt", ascending=False), r2


# ── figures ────────────────────────────────────────────────────────────────────
def fig_intensity_efficiency(intensity, team, season_label="season"):
    """Quadrant map: intensity I_T (per-match total carrier pressure) vs regain
    rate; bubble size = value/seq, colour = process cleanliness E/(E+B)."""
    ie = intensity.join(team[["regain_rate", "vt_per_seq", "eff_over_EB"]])
    ie["I_T"] = ie["frames_per_match"] * ie["mean_Ptot"]
    xv, yv = ie["I_T"].values, ie["regain_rate"].values
    val, clean = ie["vt_per_seq"].values, ie["eff_over_EB"].values
    xmed, ymed = np.median(xv), np.median(yv)
    amin, amax = 110, 620
    sz = amin + (val - val.min()) / (val.max() - val.min()) * (amax - amin)

    fig, ax = plt.subplots(figsize=(10, 7.6), dpi=200)
    fig.patch.set_facecolor(_SURF); ax.set_facecolor(_SURF)
    ax.axvline(xmed, color=_BASE, ls=(0, (5, 4)), lw=1.1, zorder=1)
    ax.axhline(ymed, color=_BASE, ls=(0, (5, 4)), lw=1.1, zorder=1)
    ax.grid(True, color=_GRID, lw=.7, zorder=0); ax.set_axisbelow(True)
    xpad = (xv.max() - xv.min()) * .10; ypad = (yv.max() - yv.min()) * .12
    ax.set_xlim(xv.min() - xpad, xv.max() + xpad); ax.set_ylim(yv.min() - ypad, yv.max() + ypad)
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    qs = dict(fontsize=10.5, color=_MUT, style="italic", zorder=1)
    ax.text(x0 + (xmed - x0) * .5, y1 - .004, "efficient minimalists", ha="center", va="top", **qs)
    ax.text(x1 - (x1 - xmed) * .5, y1 - .004, "elite press", ha="center", va="top", **qs)
    ax.text(x1 - (x1 - xmed) * .5, y0 + .004, "high grind, low return", ha="center", va="bottom", **qs)
    ax.text(x0 + (xmed - x0) * .5, y0 + .004, "passive · ineffective", ha="center", va="bottom", **qs)
    sc = ax.scatter(xv, yv, s=sz, c=clean, cmap=_CMAP, vmin=clean.min(), vmax=clean.max(),
                    edgecolors="white", linewidths=1.1, zorder=4, alpha=.96)
    texts = [ax.text(xv[k], yv[k], ie.index[k], fontsize=8, color=_INK, zorder=6)
             for k in range(len(ie))]
    try:
        from adjustText import adjust_text
        adjust_text(texts, x=xv, y=yv, ax=ax, expand=(1.2, 1.5), force_text=(.4, .6),
                    max_move=40, iter_lim=400,
                    arrowprops=dict(arrowstyle="-", color=_MUT, lw=.6, alpha=.8))
    except Exception:
        pass
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]: ax.spines[sp].set_color(_BASE)
    ax.tick_params(colors=_MUT, labelsize=9)
    ax.set_xlabel("pressing intensity  —  per-match total pressure on the carrier  $I_T$",
                  fontsize=11, color=_SEC, labelpad=8)
    ax.set_ylabel("regain efficiency  —  share of high-press sequences won back",
                  fontsize=11, color=_SEC, labelpad=8)
    ax.set_title(f"{season_label} pressing: intensity vs. regain efficiency",
                 fontsize=14, color=_INK, fontweight="semibold", pad=30, loc="left")
    ax.annotate("bubble size = value per sequence ($\\Sigma v_t/n_{seq}$)   ·   dashed = league median",
                xy=(0, 1.012), xycoords="axes fraction", fontsize=9, color=_MUT, va="bottom")
    cb = fig.colorbar(sc, ax=ax, pad=.015, fraction=.045)
    cb.set_label("process cleanliness  $E/(E{+}B)$", fontsize=9.5, color=_SEC)
    cb.ax.tick_params(colors=_MUT, labelsize=8); cb.outline.set_edgecolor(_BASE)
    lv = [val.min(), np.median(val), val.max()]
    handles = [Line2D([0], [0], marker="o", ls="", markerfacecolor="#bcbcb6",
               markeredgecolor="white",
               markersize=np.sqrt(amin + (v - val.min()) / (val.max() - val.min()) * (amax - amin)) / 1.6)
               for v in lv]
    leg = ax.legend(handles, [f"{v:.3f}" for v in lv], title="value/seq", loc="lower right",
                    frameon=False, labelspacing=1.3, handletextpad=1.1, borderpad=1.0,
                    fontsize=8.5, title_fontsize=9)
    leg.get_title().set_color(_SEC)
    for t in leg.get_texts():
        t.set_color(_SEC)
    fig.tight_layout()
    return fig


def fig_efficiency(team, season_label="season"):
    """Two panels: value per sequence + actual regain rate, per pressing team."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    vps = team["vt_per_seq"].sort_values()
    vps.plot.barh(ax=ax[0], color=["#d62728" if v < 0 else "#2ca02c" for v in vps])
    ax[0].set_title(f"Value per sequence ($\\Sigma v_t$/seq) — {season_label}")
    team["regain_rate"].sort_values().plot.barh(ax=ax[1], color="#ff7f0e")
    ax[1].set_title(f"Actual regain rate — {season_label}")
    fig.tight_layout()
    return fig


def fig_fingerprint(Zt, row_title, season_label="season"):
    """Heatmap of the z-scored style matrix (red = more aggressive)."""
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(Zt.values, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(len(Zt.columns)))
    ax.set_xticklabels(Zt.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(Zt.index)))
    ax.set_yticklabels([f"{i+1}. {t}" for i, t in enumerate(Zt.index)], fontsize=8)
    fig.colorbar(im, label="z across teams (red = more aggressive)")
    ax.set_title(f"{season_label} pressing-style fingerprint — {row_title} (scalar levels + Δ, oriented)")
    fig.tight_layout()
    return fig


def fig_drivers(drivers, season_label="season"):
    """Bar chart: Pearson r of each pressure-channel Δ with v_t."""
    fig, ax = plt.subplots(figsize=(8, 5))
    drivers["pearson_r_vt"].sort_values().plot.barh(ax=ax, color="#2F6FD0")
    ax.axvline(0, color="k", lw=.7)
    ax.set_title(f"{season_label} — Pearson r of pressure-channel Δ with press value $v_t$")
    ax.set_xlabel("Pearson r with $v_t$")
    fig.tight_layout()
    return fig


# ── outputs ────────────────────────────────────────────────────────────────────
def write_outputs(outdir, intensity, team, vp, drivers, figs, lineage_msgs=()):
    """Persist 4 csv + the given figures + lineage.txt.

    figs: {filename: Figure}, e.g. {"fig_intensity_landscape.png": fig1, ...}
    """
    os.makedirs(outdir, exist_ok=True)
    intensity.round(4).to_csv(os.path.join(outdir, "intensity.csv"))
    team.round(4).to_csv(os.path.join(outdir, "efficiency.csv"))
    vp.to_csv(os.path.join(outdir, "step_valuation.csv"), index=False)
    drivers.round(4).to_csv(os.path.join(outdir, "drivers.csv"))
    for name, fig in figs.items():
        fig.savefig(os.path.join(outdir, name), dpi=200 if "landscape" in name else 150,
                    facecolor=fig.get_facecolor(), bbox_inches="tight")
    with open(os.path.join(outdir, "lineage.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lineage_msgs) + "\n" if lineage_msgs
                else "no lineage information recorded\n")
    print(f"saved tables + {len(figs)} figures + lineage.txt -> {outdir}")
