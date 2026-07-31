"""Pressure-network construction, random frame sampling, and S2 visualisations."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Polygon as MplPolygon, Rectangle
from scipy.spatial import Delaunay, Voronoi
from shapely.geometry import Polygon, box


PITCH_L, PITCH_W, SB_L, SB_W = 105.0, 68.0, 120.0, 80.0
P_EPS = 0.05
ATT_C, DEF_C, GK_C = "#2a78d6", "#eb6834", "#4a3aa7"
WEB, BND_C, TURF, LINE_C, MUTED = "#b6b6b0", "#747a82", "#f2f4f0", "#ffffff", "#8b8b86"


@dataclass(frozen=True)
class PressureParams:
    player_distance: float = 6.4
    player_sd: float = 2.2
    boundary_distance: float = 3.2
    boundary_sd: float = 1.1

    @property
    def k_player(self) -> float:
        return math.pi / (math.sqrt(3.0) * self.player_sd)

    @property
    def k_boundary(self) -> float:
        return math.pi / (math.sqrt(3.0) * self.boundary_sd)


def sigmoid_pressure(k: float, threshold: float, distance: float) -> float:
    return float(1.0 / (1.0 + math.exp(-k * (threshold - float(distance)))))


def total_player_pressure(point: np.ndarray, defenders: list[np.ndarray], params: PressureParams) -> float:
    weights = [sigmoid_pressure(params.k_player, params.player_distance, np.linalg.norm(point - defender)) for defender in defenders]
    return float(1.0 - np.prod([1.0 - weight for weight in weights])) if weights else 0.0


def boundary_distances(point: np.ndarray) -> dict[str, float | str]:
    """Return metric distances from a point to all four pitch boundaries."""
    x, y = map(float, point)
    distances = {
        "left": x,
        "right": PITCH_L - x,
        "bottom": y,
        "top": PITCH_W - y,
    }
    nearest_side = min(distances, key=distances.__getitem__)
    return {**distances, "nearest": distances[nearest_side], "which": nearest_side}


def boundary_pressure(
    point: np.ndarray,
    params: PressureParams,
    *,
    use_all_lines: bool = False,
) -> float:
    """Return boundary pressure, using the nearest line by default."""
    distances = boundary_distances(point)
    if not use_all_lines:
        return sigmoid_pressure(
            params.k_boundary,
            params.boundary_distance,
            float(distances["nearest"]),
        )
    weights = [
        sigmoid_pressure(params.k_boundary, params.boundary_distance, float(distances[side]))
        for side in ("left", "right", "bottom", "top")
    ]
    return float(1.0 - np.prod([1.0 - weight for weight in weights]))


def total_pressure(
    point: np.ndarray,
    defenders: list[np.ndarray],
    params: PressureParams,
    *,
    include_boundary: bool = True,
    use_all_lines: bool = False,
) -> dict:
    """Combine player and boundary pressure using the Stage-1 formulation."""
    player_weights = [
        sigmoid_pressure(params.k_player, params.player_distance, np.linalg.norm(point - defender))
        for defender in defenders
    ]
    player_total = (
        float(1.0 - np.prod([1.0 - weight for weight in player_weights]))
        if player_weights
        else 0.0
    )
    boundary = (
        boundary_pressure(point, params, use_all_lines=use_all_lines)
        if include_boundary
        else 0.0
    )
    combined = float(
        1.0 - np.prod([*[1.0 - weight for weight in player_weights], 1.0 - boundary])
    )
    return {
        "P_total": round(combined, 4),
        "P_players_only": round(player_total, 4),
        "P_boundary": round(boundary, 4),
        "p_individual": [round(weight, 4) for weight in player_weights],
        "boundary_dist": boundary_distances(point),
    }


def metric_xy(x: float, y: float, flip: bool = False) -> np.ndarray:
    if flip:
        x, y = SB_L - x, SB_W - y
    return np.array([float(x) * PITCH_L / SB_L, float(y) * PITCH_W / SB_W], dtype=float)


def frame_players(event: dict, frame: dict) -> tuple[list[dict], int | None]:
    """Convert a StatsBomb frame into attack-normalised player nodes.

    The event location is authoritative for the carrier, matching the main
    pipeline.  For user data without an event location, the freeze-frame actor
    location remains a supported fallback.
    """
    defending_event = event.get("team", {}).get("id") != event.get("possession_team", {}).get("id")
    event_location = event.get("location")
    event_carrier = (
        metric_xy(*event_location, flip=defending_event)
        if event_location is not None
        else None
    )
    nodes: list[dict] = []
    for player in frame.get("freeze_frame") or []:
        location = player.get("location")
        if location is None:
            continue
        actor = bool(player.get("actor"))
        teammate = bool(player.get("teammate"))
        if actor:
            role = "carrier"
        elif defending_event:
            role = "defender" if teammate else "attacker"
        else:
            role = "attacker" if teammate else "defender"
        nodes.append(
            {
                "xy": (
                    event_carrier.copy()
                    if actor and event_carrier is not None
                    else metric_xy(location[0], location[1], flip=defending_event)
                ),
                "role": role,
                "keeper": bool(player.get("keeper")),
                "actor": actor,
            }
        )
    carrier = next((index for index, player in enumerate(nodes) if player["actor"]), None)
    if carrier is None and event_carrier is not None:
        nodes.insert(
            0,
            {
                "xy": event_carrier,
                "role": "carrier",
                "keeper": False,
                "actor": True,
            },
        )
        carrier = 0
    return nodes, carrier


def _finite_voronoi(voronoi: Voronoi, radius: float = 200.0) -> tuple[list[list[int]], np.ndarray]:
    center = voronoi.points.mean(axis=0)
    vertices = voronoi.vertices.tolist()
    ridges: dict[int, list[tuple[int, int, int]]] = {}
    for (left, right), (first, second) in zip(voronoi.ridge_points, voronoi.ridge_vertices):
        ridges.setdefault(left, []).append((right, first, second))
        ridges.setdefault(right, []).append((left, first, second))
    regions: list[list[int]] = []
    for point, region_index in enumerate(voronoi.point_region):
        region = voronoi.regions[region_index]
        if region and all(vertex >= 0 for vertex in region):
            regions.append(region)
            continue
        new_region = [vertex for vertex in region if vertex >= 0]
        for other, first, second in ridges.get(point, []):
            if first >= 0 and second >= 0:
                continue
            first, second = second, first
            tangent = voronoi.points[other] - voronoi.points[point]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = voronoi.points[[point, other]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far = voronoi.vertices[first] + direction * radius
            new_region.append(len(vertices))
            vertices.append(far.tolist())
        ordered = np.asarray([vertices[vertex] for vertex in new_region])
        centroid = ordered.mean(axis=0)
        angles = np.arctan2(ordered[:, 1] - centroid[1], ordered[:, 0] - centroid[0])
        regions.append([vertex for _, vertex in sorted(zip(angles, new_region))])
    return regions, np.asarray(vertices)


def _safe_polygon(points: np.ndarray):
    polygon = Polygon(points)
    return polygon.buffer(0) if not polygon.is_valid else polygon


def _visible_pitch_area(frame: dict, *, flip: bool):
    """Return the valid camera-visible pitch polygon when supplied.

    StatsBomb 360 normally supplies ``visible_area``.  User-provided data may
    omit it, in which case callers deliberately fall back to pitch-only
    clipping rather than rejecting an otherwise usable freeze frame.
    """
    raw = frame.get("visible_area") or []
    if len(raw) < 6 or len(raw) % 2:
        return None
    try:
        coordinates = np.asarray(raw, dtype=float).reshape(-1, 2)
        metric = np.vstack(
            [metric_xy(x, y, flip=flip) for x, y in coordinates]
        )
        visible = _safe_polygon(metric).intersection(
            box(0.0, 0.0, PITCH_L, PITCH_W)
        )
    except (TypeError, ValueError):
        return None
    return None if visible.is_empty else visible


def _boundary_edges(point: np.ndarray, params: PressureParams, allowed: set[str] | None = None) -> list[dict]:
    x, y = map(float, point)
    candidates = [
        ("own goal", np.array([0.0, y]), x),
        ("opp goal", np.array([PITCH_L, y]), PITCH_L - x),
        ("touchline low", np.array([x, 0.0]), y),
        ("touchline high", np.array([x, PITCH_W]), PITCH_W - y),
    ]
    edges = []
    for side, foot, distance in candidates:
        if allowed is not None and side not in allowed:
            continue
        pressure = sigmoid_pressure(params.k_boundary, params.boundary_distance, distance)
        if pressure >= P_EPS:
            edges.append({"side": side, "foot": foot, "p": pressure})
    return edges


def _perpendicular(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[float, float]:
    vector = end - start
    length2 = float(vector @ vector)
    if length2 == 0:
        return float(np.linalg.norm(point - start)), -1.0
    t = float((point - start) @ vector / length2)
    return float(np.linalg.norm(point - (start + t * vector))), t


def build_spn_network(event: dict, frame: dict, params: PressureParams = PressureParams()) -> dict:
    """Build all node, area, and edge families for a single freeze frame."""
    players, carrier_index = frame_players(event, frame)
    if carrier_index is None or len(players) < 4:
        raise ValueError("A network requires a carrier and at least four visible players")
    xy = np.vstack([player["xy"] for player in players])
    attackers = [i for i, player in enumerate(players) if player["role"] == "attacker" and i != carrier_index]
    defenders = [i for i, player in enumerate(players) if player["role"] == "defender" and not player["keeper"]]
    carrier = xy[carrier_index]

    topology: set[tuple[int, int]] = set()
    try:
        triangulation = Delaunay(xy)
        for simplex in triangulation.simplices:
            for left in range(3):
                for right in range(left + 1, 3):
                    topology.add(tuple(sorted((int(simplex[left]), int(simplex[right])))))
    except Exception:
        triangulation = None

    pitch = box(0.0, 0.0, PITCH_L, PITCH_W)
    defending_event = event.get("team", {}).get("id") != event.get("possession_team", {}).get("id")
    visible_area = _visible_pitch_area(frame, flip=defending_event)
    area_clip = visible_area if visible_area is not None else pitch
    areas = []
    try:
        regions, vertices = _finite_voronoi(Voronoi(xy))
        for region in regions:
            areas.append(
                _safe_polygon(vertices[region])
                .intersection(area_clip)
                .intersection(pitch)
            )
    except Exception:
        areas = [None] * len(players)

    pressure_edges = []
    targets = [carrier_index] + attackers
    for defender in defenders:
        active = []
        for target in targets:
            p = sigmoid_pressure(params.k_player, params.player_distance, np.linalg.norm(xy[defender] - xy[target]))
            if p > P_EPS:
                active.append((target, p))
        total = sum(p for _, p in active)
        pressure_edges.extend(
            {
                "src": defender,
                "dst": target,
                "p": p,
                "w": p / total,
            }
            for target, p in active
        )

    lane_k = math.pi / (math.sqrt(3.0) * 0.6)
    pass_edges = []
    for receiver in attackers:
        blockers = []
        for defender in defenders:
            distance, t = _perpendicular(xy[defender], carrier, xy[receiver])
            if 0.0 <= t <= 1.0:
                p = sigmoid_pressure(lane_k, 1.2, distance)
                if p > P_EPS:
                    blockers.append((defender, p))
        block = max((item[1] for item in blockers), default=0.0)
        distance = float(np.linalg.norm(xy[receiver] - carrier))
        pass_edges.append(
            {
                "dst": receiver,
                "dist": distance,
                "block": block,
                "blocker": max(blockers, key=lambda item: item[1])[0] if blockers else None,
                "w": float(math.exp(-distance / 18.0) * (1.0 - block)),
            }
        )

    carrier_boundaries = _boundary_edges(carrier, params)
    active_sides = {edge["side"] for edge in carrier_boundaries}
    boundary_edges = [{**edge, "dst": carrier_index, "on_carrier": True} for edge in carrier_boundaries]
    for receiver in attackers:
        boundary_edges.extend(
            {**edge, "dst": receiver, "on_carrier": False}
            for edge in _boundary_edges(xy[receiver], params, active_sides)
        )
    return {
        "event": event,
        "players": players,
        "xy": xy,
        "carrier_index": carrier_index,
        "attackers": attackers,
        "defenders": defenders,
        "topology": topology,
        "areas": areas,
        "visible_area": visible_area,
        "pressure_edges": pressure_edges,
        "pass_edges": pass_edges,
        "boundary_edges": boundary_edges,
    }


def _match_data(events_dir: str | Path, three_sixty_dir: str | Path, match_id: str) -> tuple[list[dict], dict[str, dict]]:
    event_path = Path(events_dir) / f"{match_id}.json"
    frame_path = Path(three_sixty_dir) / f"{match_id}.json"
    with event_path.open(encoding="utf-8") as handle:
        events = json.load(handle)
    with frame_path.open(encoding="utf-8") as handle:
        frames = {item["event_uuid"]: item for item in json.load(handle) if item.get("freeze_frame")}
    return events, frames


def _network_for_anchor(
    row: pd.Series,
    events_dir: str | Path,
    three_sixty_dir: str | Path,
    cache: dict,
    params: PressureParams,
) -> dict | None:
    match_id = str(row["match_id"])
    if match_id not in cache:
        cache[match_id] = _match_data(events_dir, three_sixty_dir, match_id)
    events, frames = cache[match_id]
    event_id = row["anchor_event_id"]
    event = next((item for item in events if item.get("id") == event_id), None)
    frame = frames.get(event_id)
    if event is None or frame is None:
        return None
    try:
        return build_spn_network(event, frame, params=params)
    except ValueError:
        return None


def sample_network_frame(
    labels: pd.DataFrame,
    events_dir: str | Path,
    three_sixty_dir: str | Path,
    random_state: int = 42,
    params: PressureParams = PressureParams(),
) -> dict:
    """Choose a reproducible random informative network from the user's own labels."""
    if labels.empty:
        raise ValueError("labels is empty")
    rng, cache, candidates = np.random.default_rng(random_state), {}, []
    for index in rng.permutation(len(labels)):
        network = _network_for_anchor(
            labels.iloc[int(index)], events_dir, three_sixty_dir, cache, params
        )
        if network and len(network["players"]) >= 10 and network["pressure_edges"] and network["pass_edges"]:
            candidates.append(network)
    if not candidates:
        raise ValueError("No complete informative freeze frame was available for network visualisation")
    return candidates[int(rng.integers(len(candidates)))]


def sample_network_sequence(
    labels: pd.DataFrame,
    events_dir: str | Path,
    three_sixty_dir: str | Path,
    max_frames: int = 5,
    min_frames: int = 3,
    random_state: int = 42,
    params: PressureParams = PressureParams(),
) -> list[dict]:
    """Choose a reproducible random valid sequence; fall back to the longest one."""
    cache, valid = {}, []
    for _, group in labels.sort_values(["match_id", "seq_id", "ev_pos"]).groupby(["match_id", "seq_id"]):
        networks = [
            network
            for _, row in group.iterrows()
            if (
                network := _network_for_anchor(
                    row, events_dir, three_sixty_dir, cache, params
                )
            )
        ]
        if networks:
            valid.append(networks)
    eligible = [item for item in valid if len(item) >= min_frames] or valid
    if not eligible:
        raise ValueError("No valid network sequence was available")
    rng = np.random.default_rng(random_state)
    selected = eligible[int(rng.integers(len(eligible)))]
    indices = np.linspace(0, len(selected) - 1, min(max_frames, len(selected)), dtype=int)
    return [selected[index] for index in indices]


def draw_pitch(ax) -> None:
    ax.add_patch(Rectangle((0, 0), PITCH_L, PITCH_W, fc=TURF, ec=LINE_C, lw=1.2, zorder=0))
    ax.plot([PITCH_L / 2, PITCH_L / 2], [0, PITCH_W], color=LINE_C, lw=1.2, zorder=0)
    ax.add_patch(Circle((PITCH_L / 2, PITCH_W / 2), 9.15, fc="none", ec=LINE_C, lw=1.2, zorder=0))
    for x, sign in ((0, 1), (PITCH_L, -1)):
        ax.add_patch(Rectangle((x if sign > 0 else x - 16.5, 13.85), 16.5, 40.3, fc="none", ec=LINE_C, lw=1.1))
        ax.add_patch(Rectangle((x if sign > 0 else x - 5.5, 24.84), 5.5, 18.32, fc="none", ec=LINE_C, lw=1.1))
        ax.add_patch(Arc((x + sign * 11, PITCH_W / 2), 18.3, 18.3, theta1=-53 if sign > 0 else 127, theta2=53 if sign > 0 else 233, color=LINE_C, lw=1.1))


def draw_network(ax, network: dict, *, areas: bool = False, topology: bool = False, pressure: bool = False, passing: bool = False, boundary: bool = False) -> None:
    """Draw a chosen layer combination. Nodes are always rendered last."""
    draw_pitch(ax)
    xy = network["xy"]
    if areas:
        for index, area in enumerate(network["areas"]):
            if area is None or area.is_empty:
                continue
            colour = GK_C if network["players"][index]["keeper"] else ATT_C if network["players"][index]["role"] == "attacker" else DEF_C
            geometries = [area] if area.geom_type == "Polygon" else list(area.geoms)
            for geometry in geometries:
                ax.add_patch(MplPolygon(np.asarray(geometry.exterior.coords), closed=True, fc=colour, ec=colour, alpha=.13, lw=.35, zorder=1))
    if topology:
        for left, right in network["topology"]:
            ax.plot(xy[[left, right], 0], xy[[left, right], 1], color=WEB, lw=.7, alpha=.85, zorder=2)
    if passing:
        carrier = network["carrier_index"]
        for edge in network["pass_edges"]:
            blocked = edge["block"] > P_EPS
            ax.plot([xy[carrier, 0], xy[edge["dst"], 0]], [xy[carrier, 1], xy[edge["dst"], 1]], color=ATT_C if not blocked else "#1a4f91", lw=1.0 + 3.2 * edge["w"] if not blocked else 1.3, ls="-" if not blocked else (0, (3.4, 2.4)), zorder=4)
    if pressure:
        carrier = network["carrier_index"]
        for edge in network["pressure_edges"]:
            on_carrier = edge["dst"] == carrier
            ax.plot(xy[[edge["src"], edge["dst"]], 0], xy[[edge["src"], edge["dst"]], 1], color=DEF_C, lw=(1.0 + 3.2 * edge["p"]) if on_carrier else (.7 + 1.4 * edge["p"]), alpha=.95 if on_carrier else .5, ls="-" if on_carrier else (0, (5, 2)), zorder=5 if on_carrier else 3)
    if boundary:
        for edge in network["boundary_edges"]:
            target, foot, p = edge["dst"], edge["foot"], edge["p"]
            ax.plot([xy[target, 0], foot[0]], [xy[target, 1], foot[1]], color=BND_C, lw=(1.0 + 3.2 * p) if edge["on_carrier"] else (.7 + 1.3 * p), alpha=.95 if edge["on_carrier"] else .5, ls="-" if edge["on_carrier"] else (0, (3.2, 2.2)), zorder=5)
            ax.plot(*foot, "s", ms=7 if edge["on_carrier"] else 4, color=BND_C, mec="white", mew=.8, zorder=7)
    for index, player in enumerate(network["players"]):
        if index == network["carrier_index"]:
            continue
        colour = GK_C if player["keeper"] else ATT_C if player["role"] == "attacker" else DEF_C
        ax.plot(*player["xy"], "o", ms=8, mfc=colour, mec="white", mew=1.2, zorder=8)
    ax.plot(*xy[network["carrier_index"]], "*", ms=15, mfc=ATT_C, mec="white", mew=1.3, zorder=9)
    ax.set(xlim=(-2, PITCH_L + 2), ylim=(-2, PITCH_W + 2), aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_network_layers(network: dict):
    panels = [
        ("A. Nodes", {}),
        ("B. Nodes + Delaunay", {"topology": True}),
        ("C. Nodes + control area", {"areas": True}),
        ("D. Nodes + pressure edges", {"pressure": True}),
        ("E. Nodes + pass and boundary edges", {"passing": True, "boundary": True}),
        ("F. Composite network", {"areas": True, "topology": True, "pressure": True, "passing": True, "boundary": True}),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, (title, layers) in zip(axes.flat, panels):
        draw_network(axis, network, **layers)
        axis.set_title(title, loc="left", fontsize=10, fontweight="bold")
    fig.legend(handles=[
        Line2D([], [], marker="*", ls="none", mfc=ATT_C, mec="white", ms=11, label="carrier"),
        Line2D([], [], marker="o", ls="none", mfc=ATT_C, mec="white", ms=7, label="attacker"),
        Line2D([], [], marker="o", ls="none", mfc=DEF_C, mec="white", ms=7, label="defender"),
        Line2D([], [], color=DEF_C, lw=2, label="pressure"),
        Line2D([], [], color=ATT_C, lw=2, label="pass"),
        Line2D([], [], color=BND_C, lw=2, label="boundary"),
    ], ncol=6, loc="lower center", frameon=False)
    fig.tight_layout(rect=(0, .06, 1, 1))
    return fig


def plot_network_sequence(networks: list[dict]):
    fig, axes = plt.subplots(1, len(networks), figsize=(4.5 * len(networks), 5.2), squeeze=False)
    for index, (axis, network) in enumerate(zip(axes.flat, networks), start=1):
        draw_network(axis, network, areas=False, topology=True, pressure=True, passing=True, boundary=True)
        axis.set_title(f"Frame {index}", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig
