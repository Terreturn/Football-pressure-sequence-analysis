"""
Off-Ball Pressure Network — Distance-Based Formulation v2
==========================================================

Updates from v1:
  1. k is now scaled proportionally to D* (σ_d = 2.2m for players)
  2. Boundary pressure added (sideline + byline) with separate params:
       D*_line = D*_player / 2 = 3.2m
       σ_d_line = σ_d_player / 2 = 1.1m  →  k_line = 2 × k_player
  3. Only nearest boundary line is used (min distance across 4 lines)
  4. Boundary pressure enters total pressure formula identically to
     a single additional "defender" (same 1 - ∏(1-p) structure)

Pitch convention (StatsBomb):
  x ∈ [0, 120] yards  or  [0, 105] metres  (left → right)
  y ∈ [0, 80]  yards  or  [0, 68]  metres  (bottom → top)

Formula reference:
  pi,j  = sigmoid(k       · (D*       - Di,j  ))   defender
  p_bnd = sigmoid(k_line  · (D*_line  - d_line))   boundary
  Pj    = 1 - (1 - p_bnd) · ∏_i (1 - pi,j)
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ── 1. PARAMETERS ─────────────────────────────────────────────────────────────

class PressureParams:
    """
    Unified parameter set for player pressure + boundary pressure.

    Player pressure
    ---------------
    D*_player  = 6.4 m   (effective interception radius, local high-press)
    σ_d        = 2.2 m   (proportionally scaled from original 3.6m × 6.4/10.4)
    k_player   = π / (√3 · σ_d)  ≈ 0.821

    Boundary pressure  (sideline / byline)
    ----------------------------------------
    D*_line    = D*_player / 2 = 3.2 m
    σ_d_line   = σ_d / 2      = 1.1 m
    k_line     = π / (√3 · σ_d_line) ≈ 1.643
    """

    # Player
    D_star_player: float = 6.4
    sigma_d_player: float = 2.2

    # Boundary (half of player params)
    @property
    def D_star_line(self) -> float:
        return self.D_star_player / 2          # 3.2 m

    @property
    def sigma_d_line(self) -> float:
        return self.sigma_d_player / 2         # 1.1 m

    @property
    def k_player(self) -> float:
        return np.pi / (np.sqrt(3) * self.sigma_d_player)   # ≈ 0.821

    @property
    def k_line(self) -> float:
        return np.pi / (np.sqrt(3) * self.sigma_d_line)     # ≈ 1.643

    # Pitch dimensions (metres, StatsBomb)
    pitch_length: float = 105.0   # x-axis
    pitch_width:  float = 68.0    # y-axis

    def summary(self):
        print("=== Pressure Parameters v2 ===")
        print(f"\n  [Player]")
        print(f"    D*_player  : {self.D_star_player} m")
        print(f"    σ_d        : {self.sigma_d_player} m")
        print(f"    k_player   : {self.k_player:.4f}")
        print(f"\n  [Boundary]")
        print(f"    D*_line    : {self.D_star_line} m")
        print(f"    σ_d_line   : {self.sigma_d_line} m")
        print(f"    k_line     : {self.k_line:.4f}")
        print(f"\n  [Pitch]  {self.pitch_length} × {self.pitch_width} m")


PARAMS = PressureParams()


# ── 2. CORE FUNCTIONS ─────────────────────────────────────────────────────────

def sigmoid_pressure(k: float, D_star: float, dist: float) -> float:
    """
    Unified logistic pressure function.

    p = [1 + exp(-k · (D* - dist))]^(-1)

    Works for both player pressure (k_player, D*_player)
    and boundary pressure (k_line, D*_line).
    """
    return float(1.0 / (1.0 + np.exp(-k * (D_star - dist))))


def player_pressure(ri: np.ndarray, rj: np.ndarray,
                    params: PressureParams = PARAMS) -> float:
    """
    Pressure from one defender i on target j.

    p_i,j = sigmoid(k_player · (D*_player - ||rj - ri||))
    """
    dist = float(np.linalg.norm(np.array(rj) - np.array(ri)))
    return sigmoid_pressure(params.k_player, params.D_star_player, dist)


def boundary_distance(rj: np.ndarray,
                      params: PressureParams = PARAMS) -> dict:
    """
    Distances from position rj to each of the four pitch boundaries.

    Returns dict with individual distances and the minimum.

    Note: Using min(four lines) models the single most threatening
    boundary. For corner situations, consider using all four lines
    in the total pressure product instead.
    """
    x, y = float(rj[0]), float(rj[1])
    d = {
        "left":   x,
        "right":  params.pitch_length - x,
        "bottom": y,
        "top":    params.pitch_width - y,
    }
    d["nearest"] = min(d["left"], d["right"], d["bottom"], d["top"])
    d["nearest_side"] = min(d.values(), key=lambda v: v)  # for reference
    d["which"] = min(d, key=lambda k: d[k] if k not in ("nearest","nearest_side") else np.inf)
    return d


def boundary_pressure(rj: np.ndarray,
                      params: PressureParams = PARAMS,
                      use_all_lines: bool = False) -> float:
    """
    Pressure from pitch boundaries on target j.

    Default (use_all_lines=False):
        Only the nearest boundary line is used.
        p_bnd = sigmoid(k_line · (D*_line - d_nearest))

    Optional (use_all_lines=True):
        All four lines contribute independently.
        p_bnd = 1 - ∏_b (1 - p_b)
        Recommended for corner / wide area analysis.

    Same sigmoid parameters as player pressure but halved:
        D*_line  = D*_player / 2 = 3.2 m
        k_line   = 2 × k_player  ≈ 1.643
    """
    bd = boundary_distance(rj, params)

    if not use_all_lines:
        return sigmoid_pressure(params.k_line, params.D_star_line, bd["nearest"])
    else:
        sides = ["left", "right", "bottom", "top"]
        probs = [sigmoid_pressure(params.k_line, params.D_star_line, bd[s])
                 for s in sides]
        return float(1.0 - np.prod([1.0 - p for p in probs]))


def total_pressure(rj: np.ndarray,
                   defender_positions: list,
                   params: PressureParams = PARAMS,
                   include_boundary: bool = True,
                   use_all_lines: bool = False) -> dict:
    """
    Total pressure on target j combining player + boundary pressure.

    Formula:
        Pj = 1 - (1 - p_bnd) · ∏_i (1 - p_i,j)

    Boundary enters the product identically to a single defender,
    preserving the independence assumption of the original framework.

    Returns dict with breakdown for interpretability.
    """
    rj = np.array(rj)

    # Individual player pressures
    p_players = []
    for ri in defender_positions:
        p_players.append(player_pressure(ri, rj, params))

    # Boundary pressure
    p_bnd = boundary_pressure(rj, params, use_all_lines) if include_boundary else 0.0

    # Total: boundary acts as one additional "defender" in the product
    terms = [1.0 - p for p in p_players] + [1.0 - p_bnd]
    P_total = float(1.0 - np.prod(terms))

    # Player-only (for comparison)
    P_players_only = float(1.0 - np.prod([1.0 - p for p in p_players]))

    return {
        "P_total":        round(P_total, 4),
        "P_players_only": round(P_players_only, 4),
        "P_boundary":     round(p_bnd, 4),
        "p_individual":   [round(p, 4) for p in p_players],
        "boundary_dist":  boundary_distance(rj, params),
    }


# ── 3. FULL FREEZE FRAME ANALYSIS ─────────────────────────────────────────────

def analyze_freeze_frame(
    freeze_frame: list,
    ball_position: list,
    params: PressureParams = PARAMS,
    include_boundary: bool = True,
    use_all_lines: bool = False,
    edge_threshold: float = 0.25,
    high_pressure_threshold: float = 0.65,
) -> dict:
    """
    Full pipeline: StatsBomb 360 freeze frame → pressure analysis.

    Parameters
    ----------
    freeze_frame            : StatsBomb 360 freeze frame list
    ball_position           : [x, y] of the ball
    include_boundary        : whether to add boundary pressure
    use_all_lines           : use all 4 lines (True) or nearest only (False)
    edge_threshold          : min p_i,j to include edge in network
    high_pressure_threshold : Pj threshold for high-pressure zone label
    """
    attackers, defenders = [], []
    for p in freeze_frame:
        loc = p.get("location")
        if loc is None:
            continue
        (attackers if p.get("teammate") else defenders).append(loc)

    records = []
    for idx_j, rj in enumerate(attackers):
        rj_arr = np.array(rj)
        result = total_pressure(rj_arr, defenders, params,
                                include_boundary, use_all_lines)

        dists_to_defs = [float(np.linalg.norm(rj_arr - np.array(ri)))
                         for ri in defenders]
        nearest_def = min(dists_to_defs) if dists_to_defs else np.nan
        n_in_range  = sum(1 for d in dists_to_defs if d <= params.D_star_player)
        bd          = result["boundary_dist"]

        Pj = result["P_total"]
        if   Pj >= 0.85: level = "critical"
        elif Pj >= 0.65: level = "high"
        elif Pj >= 0.35: level = "medium"
        else:            level = "low"

        records.append({
            "attacker_id":          idx_j,
            "x":                    round(rj[0], 2),
            "y":                    round(rj[1], 2),
            "P_total":              result["P_total"],
            "P_players_only":       result["P_players_only"],
            "P_boundary":           result["P_boundary"],
            "nearest_def_m":        round(nearest_def, 2),
            "nearest_line_m":       round(bd["nearest"], 2),
            "nearest_line_side":    bd["which"],
            "n_defenders_in_range": n_in_range,
            "pressure_level":       level,
            "boundary_contribution": round(
                result["P_total"] - result["P_players_only"], 4),
        })

    df = pd.DataFrame(records)

    # Ball pressure
    ball_result = total_pressure(
        np.array(ball_position), defenders, params,
        include_boundary, use_all_lines
    )

    # Pressure network (edges between defenders and attackers)
    edges = []
    for idx_i, ri in enumerate(defenders):
        for idx_j, rj in enumerate(attackers):
            p_ij = player_pressure(ri, rj, params)
            if p_ij >= edge_threshold:
                edges.append({
                    "source":   f"D{idx_i}",
                    "target":   f"A{idx_j}",
                    "distance": round(float(np.linalg.norm(
                        np.array(rj) - np.array(ri))), 2),
                    "p_ij":     round(p_ij, 4),
                })

    return {
        "pressure_df":    df,
        "high_pressure":  df[df["P_total"] >= high_pressure_threshold],
        "ball_pressure":  ball_result,
        "network_edges":  edges,
        "params": {
            "D_star_player": params.D_star_player,
            "D_star_line":   params.D_star_line,
            "k_player":      round(params.k_player, 4),
            "k_line":        round(params.k_line, 4),
            "include_boundary": include_boundary,
            "use_all_lines":    use_all_lines,
        }
    }


# ── 4. DEMO ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    PARAMS.summary()
    print()

    freeze_frame = [
        # Attackers (teammate=True)
        {"location": [98.0, 4.0],  "teammate": True},   # A0: near right byline + bottom sideline (corner)
        {"location": [90.0, 34.0], "teammate": True},   # A1: central, open
        {"location": [85.0, 64.0], "teammate": True},   # A2: near top sideline
        {"location": [78.0, 34.0], "teammate": True},   # A3: deeper, more space
        # Defenders (teammate=False)
        {"location": [96.0, 6.0],  "teammate": False},  # D0: tight on A0
        {"location": [91.0, 33.0], "teammate": False},  # D1: tight on A1
        {"location": [84.0, 62.0], "teammate": False},  # D2: close to A2
        {"location": [80.0, 36.0], "teammate": False},  # D3: near A3
    ]
    ball_position = [90.0, 34.0]

    print("─" * 60)
    print("[ With boundary pressure — nearest line only ]")
    result = analyze_freeze_frame(
        freeze_frame, ball_position,
        include_boundary=True, use_all_lines=False
    )
    cols = ["attacker_id", "x", "y",
            "P_total", "P_players_only", "P_boundary",
            "nearest_def_m", "nearest_line_m", "nearest_line_side",
            "pressure_level"]
    print(result["pressure_df"][cols].to_string(index=False))

    print()
    print("[ Ball pressure ]")
    bp = result["ball_pressure"]
    print(f"  P_total        : {bp['P_total']}")
    print(f"  P_players_only : {bp['P_players_only']}")
    print(f"  P_boundary     : {bp['P_boundary']}")
    print(f"  Nearest line   : {bp['boundary_dist']['which']} "
          f"({bp['boundary_dist']['nearest']:.1f} m)")

    print()
    print("─" * 60)
    print("[ Boundary contribution — how much does the line add? ]")
    df = result["pressure_df"]
    for _, row in df.iterrows():
        bar = "█" * int(row["boundary_contribution"] * 100)
        print(f"  A{int(row['attacker_id'])} "
              f"(nearest: {row['nearest_line_side']:6s} {row['nearest_line_m']:4.1f}m) "
              f"Δ={row['boundary_contribution']:+.3f}  {bar}")

    print()
    print("─" * 60)
    print("[ Corner case: use_all_lines=True ]")
    result_all = analyze_freeze_frame(
        freeze_frame, ball_position,
        include_boundary=True, use_all_lines=True
    )
    df_all = result_all["pressure_df"][["attacker_id","P_total","P_boundary"]]
    df_cmp = df.merge(df_all, on="attacker_id", suffixes=("_nearest","_all"))
    print(df_cmp[["attacker_id",
                  "P_total_nearest","P_boundary_nearest",
                  "P_total_all","P_boundary_all"]].to_string(index=False))
