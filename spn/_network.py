"""Private pressure-network construction used by the public data pipeline."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay, Voronoi
from shapely.geometry import Polygon, box


PITCH_L, PITCH_W, SB_L, SB_W = 105.0, 68.0, 120.0, 80.0
P_EPS = 0.05


@dataclass(frozen=True)
class PressureParams:
    player_distance: float = 6.4
    player_sd: float = 2.2
    boundary_distance: float = 3.2
    boundary_sd: float = 1.1
    lane_distance: float = 1.2
    lane_sd: float = 0.6
    pass_decay_m: float = 18.0
    pass_open_threshold: float = 0.5
    edge_epsilon: float = P_EPS

    def __post_init__(self) -> None:
        positive = {
            "player_distance": self.player_distance,
            "player_sd": self.player_sd,
            "boundary_distance": self.boundary_distance,
            "boundary_sd": self.boundary_sd,
            "lane_distance": self.lane_distance,
            "lane_sd": self.lane_sd,
            "pass_decay_m": self.pass_decay_m,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError(f"pressure/network parameters must be positive: {invalid}")
        if not 0.0 < self.pass_open_threshold < 1.0:
            raise ValueError("pass_open_threshold must be between zero and one")
        if not 0.0 < self.edge_epsilon < 1.0:
            raise ValueError("edge_epsilon must be between zero and one")

    @property
    def k_player(self) -> float:
        return math.pi / (math.sqrt(3.0) * self.player_sd)

    @property
    def k_boundary(self) -> float:
        return math.pi / (math.sqrt(3.0) * self.boundary_sd)

    @property
    def k_lane(self) -> float:
        return math.pi / (math.sqrt(3.0) * self.lane_sd)


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
        if pressure >= params.edge_epsilon:
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
            if p > params.edge_epsilon:
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

    pass_edges = []
    for receiver in attackers:
        blockers = []
        for defender in defenders:
            distance, t = _perpendicular(xy[defender], carrier, xy[receiver])
            if 0.0 <= t <= 1.0:
                p = sigmoid_pressure(params.k_lane, params.lane_distance, distance)
                if p > params.edge_epsilon:
                    blockers.append((defender, p))
        block = max((item[1] for item in blockers), default=0.0)
        distance = float(np.linalg.norm(xy[receiver] - carrier))
        pass_edges.append(
            {
                "dst": receiver,
                "dist": distance,
                "block": block,
                "blocker": max(blockers, key=lambda item: item[1])[0] if blockers else None,
                "w": float(math.exp(-distance / params.pass_decay_m) * (1.0 - block)),
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
        "params": params,
    }
