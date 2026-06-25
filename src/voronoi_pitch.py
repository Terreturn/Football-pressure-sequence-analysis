"""
Voronoi pitch-control construction for StatsBomb 360 freeze frames.
==================================================================
Works entirely in METRES (105 x 68) so areas and aspect ratio are correct.

Pipeline (see the project spec §C / §D):
  C. Base Voronoi tessellation (scipy + shapely), clipped to the pitch.
     Boundary handled as a defence-biased control subject (Method A: the
     3.2 m edge strip — where the line is the nearest control subject — is
     carved out of player cells and counted as defensive "boundary band").
  D. Incomplete-data handling:
     D1 confidence stats, D2 convex-hull uncertainty mask,
     D3 reachable-radius cap (disk = D*_player), D4 honest per-cell features.

Coordinate convention (after sb_to_m): origin bottom-left, x→right, y→up,
attacking team normalised to attack left→right by StatsBomb.
"""

import numpy as np
from scipy.spatial import Voronoi, ConvexHull
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union

# ── Constants ─────────────────────────────────────────────────────────────────
PITCH_L, PITCH_W = 105.0, 68.0          # metric pitch
SB_L,    SB_W    = 120.0, 80.0          # StatsBomb units
D_PLAYER         = 6.4                   # effective interception radius (m)
D_LINE           = 3.2                   # boundary band half-width (m)
K_LINE           = np.pi / (np.sqrt(3) * 1.1)   # ≈ 1.6489

PITCH_BOX = box(0.0, 0.0, PITCH_L, PITCH_W)


def sb_to_m(x, y):
    """StatsBomb (120x80) → metres (105x68)."""
    return x * PITCH_L / SB_L, y * PITCH_W / SB_W


# ── Voronoi reconstruction (finite polygons for every site) ───────────────────
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite Voronoi regions into finite polygons.
    Canonical implementation; returns (regions, vertices) where regions[i]
    corresponds to input point i.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("2D input required")

    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = float(np.ptp(vor.points, axis=0).max()) * 2

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def _safe(poly):
    """Repair invalid polygons."""
    return poly if poly.is_valid else poly.buffer(0)


def player_cells(points):
    """
    Voronoi cells for player sites, clipped to the pitch rectangle.
    Returns list of shapely geometries (index matches points).
    Degenerate (<2 sites) handled by giving the single site the whole pitch.
    """
    points = np.asarray(points, dtype=float)
    if len(points) == 1:
        return [PITCH_BOX]
    regions, vertices = voronoi_finite_polygons_2d(Voronoi(points))
    cells = []
    for region in regions:
        poly = _safe(Polygon(vertices[region]))
        cells.append(_safe(poly).intersection(PITCH_BOX))
    return cells


# ── Boundary band (Method A: 3.2 m defence-biased edge strip) ─────────────────
def boundary_band():
    """The 3.2 m strip just inside all four touch/goal lines."""
    inner = PITCH_BOX.buffer(-D_LINE)
    return _safe(PITCH_BOX.difference(inner))


def boundary_pressure_grid(res=0.4):
    """
    Combined four-line boundary pressure field p_bnd over the pitch
    (corner formula: 1 - ∏(1 - sigmoid)), for the gradient overlay.
    Returns (xs, ys, P) with P shape (len(ys), len(xs)).
    """
    xs = np.arange(0, PITCH_L + res, res)
    ys = np.arange(0, PITCH_W + res, res)
    gx, gy = np.meshgrid(xs, ys)
    dl, dr, db, dt = gx, PITCH_L - gx, gy, PITCH_W - gy

    def sig(d):
        return 1.0 / (1.0 + np.exp(-K_LINE * (D_LINE - d)))

    P = 1.0 - (1 - sig(dl)) * (1 - sig(dr)) * (1 - sig(db)) * (1 - sig(dt))
    return xs, ys, P


# ── Convex-hull uncertainty mask (D2) ─────────────────────────────────────────
def hull_mask(points, buffer=D_PLAYER):
    """Convex hull of visible players, buffered outward. None if < 3 points."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return None
    try:
        hull = ConvexHull(points)
    except Exception:
        return None
    poly = Polygon(points[hull.vertices])
    return _safe(poly).buffer(buffer)


# ── Full construction ─────────────────────────────────────────────────────────
def build_voronoi(players, cap_reachable=False, radius=D_PLAYER,
                  include_keeper=True):
    """
    players : list of dicts with metric coords:
        {x, y, team ('attacker'|'defender'), keeper(bool), actor(bool)}

    Returns dict:
        cells      : list of per-player records (D4 fields + shapely 'cell')
        band       : boundary-band geometry (defensive territory)
        hull       : hull+buffer geometry (None if <3 players)
        uncontested: geometry not controlled by any player or band
        areas      : dict of territory areas (m²)
    """
    pl = [p for p in players if include_keeper or not p["keeper"]]
    pts = np.array([[p["x"], p["y"]] for p in pl], dtype=float)

    raw_cells = player_cells(pts)
    band      = boundary_band()
    hull      = hull_mask(pts)

    records, capped_geoms = [], []
    for i, p in enumerate(pl):
        cell0 = _safe(raw_cells[i])                 # full Voronoi cell
        cell  = _safe(cell0.difference(band))       # carve out boundary band

        if cap_reachable:
            disk = Point(p["x"], p["y"]).buffer(radius)
            cell = _safe(cell.intersection(disk))

        # fraction of this player's territory pinned against the boundary band
        frac_band = (cell0.intersection(band).area / cell0.area) if cell0.area else 0.0
        inside_hull = bool(hull.contains(Point(p["x"], p["y"]))) if hull else True

        records.append({
            **p,
            "cell":        cell,
            "area_m2":     round(cell.area, 2),
            "frac_in_band": round(frac_band, 3),
            "inside_hull": inside_hull,
            "capped":      cap_reachable,
        })
        capped_geoms.append(cell)

    # Uncontested = pitch minus (all player cells ∪ band)
    controlled = unary_union(capped_geoms + [band])
    uncontested = _safe(PITCH_BOX.difference(controlled))

    atk = sum(r["area_m2"] for r in records if r["team"] == "attacker")
    dfd = sum(r["area_m2"] for r in records if r["team"] == "defender")
    areas = {
        "attacker_m2":    round(atk, 1),
        "defender_m2":    round(dfd, 1),
        "boundary_m2":    round(band.area, 1),
        "uncontested_m2": round(uncontested.area, 1),
    }

    return {"cells": records, "band": band, "hull": hull,
            "uncontested": uncontested, "areas": areas}
