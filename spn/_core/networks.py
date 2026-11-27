"""Construct structural pressure networks from visible players and observed ball positions.

Provider coordinates are rescaled independently to a 105 by 68 meter pitch.
Only the explicitly authorized shape-graph routines are reused; all other
geometry, edge construction and feature calculations are independent."""

from __future__ import annotations

from dataclasses import dataclass

import math

import numpy as np

from scipy.spatial import Delaunay, Voronoi

from shapely.geometry import Polygon, box

from spn._core.pressure_events import logistic, team

from spn._core.vendor import shape_corrected, shape_legacy


@dataclass(frozen=True)
class Parameters:
    """Logistic midpoints/spreads and decay in meters; thresholds are unitless."""

    player_mid: float = 6.4
    player_sd: float = 2.2
    line_mid: float = 3.2
    line_sd: float = 1.1
    lane_mid: float = 1.2
    lane_sd: float = 0.6
    decay: float = 18.0
    epsilon: float = 0.05
    open_threshold: float = 0.5

    def __post_init__(self):
        """Validate finite positive meter parameters and probabilities; raise on invalid inputs."""
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid network parameter: " + name)
        if self.epsilon >= 1 or self.open_threshold >= 1:
            raise ValueError("Network thresholds must be less than one")


P = Parameters()


def xy_m(location, flip=False):
    """Rescale provider x/y to meters, optionally reversing both pitch axes."""
    x, y = location[:2]
    if flip:
        x, y = 120.0 - x, 80.0 - y
    return np.array([float(x) * 105.0 / 120.0, float(y) * 68.0 / 80.0])


def player_nodes(event, frame, flip=None):
    """Return node dictionaries and carrier index from provider records; xy is in meters."""
    defensive = team(event) != team(event, "possession_team")
    orientation = defensive if flip is None else bool(flip)
    ball = event.get("location")
    ball_m = xy_m(ball, orientation) if ball is not None else None
    nodes = []
    for player in frame.get("freeze_frame") or []:
        if player.get("location") is None:
            continue
        actor = bool(player.get("actor"))
        same = bool(player.get("teammate"))
        role = "carrier" if actor else ("defender" if same == defensive else "attacker")
        nodes.append(
            {
                "xy": ball_m.copy()
                if actor and ball_m is not None
                else xy_m(player["location"], orientation),
                "role": role,
                "keeper": bool(player.get("keeper")),
                "actor": actor,
            }
        )
    carrier = next((i for i, node in enumerate(nodes) if node["actor"]), None)
    if carrier is None and ball_m is not None:
        nodes.insert(
            0, {"xy": ball_m, "role": "carrier", "keeper": False, "actor": True}
        )
        carrier = 0
    return nodes, carrier


def repaired_polygon(vertices):
    """Return a polygon from meter coordinates, repairing self-intersections with buffer(0)."""
    polygon = Polygon(vertices)
    return polygon if polygon.is_valid else polygon.buffer(0)


def visible_polygon(frame, flip=False):
    """Return camera-visible pitch geometry in square meters, or None for unavailable geometry."""
    raw = frame.get("visible_area") or []
    if len(raw) < 6 or len(raw) % 2:
        return None
    try:
        points = np.asarray(raw, dtype=float).reshape(-1, 2)
        polygon = repaired_polygon([xy_m(p, flip) for p in points]).intersection(
            box(0, 0, 105, 68)
        )
    except (ValueError, TypeError):
        return None
    return None if polygon.is_empty else polygon


def finite_cells(diagram, radius=200.0):
    """Close unbounded SciPy Voronoi cells with 200 m rays; return meter vertex arrays per node.

    This deliberately reproduces the released finite-ray construction. It is
    not a claim that a fixed ray length works for every possible point set.
    """
    vertices = list(diagram.vertices)
    center = diagram.points.mean(axis=0)
    incident = {i: [] for i in range(len(diagram.points))}
    for pair, ends in zip(diagram.ridge_points, diagram.ridge_vertices):
        a, b = map(int, pair)
        incident[a].append((b, ends))
        incident[b].append((a, ends))
    cells = []
    for i, region_id in enumerate(diagram.point_region):
        region = diagram.regions[region_id]
        if region and min(region) >= 0:
            cells.append(np.asarray([vertices[j] for j in region]))
            continue
        ids = [j for j in region if j >= 0]
        for other, ends in incident[i]:
            if min(ends) >= 0:
                continue
            # SciPy orders the infinite endpoint first; retain the release's
            # finite endpoint and its signed outward ridge normal.
            tangent = diagram.points[other] - diagram.points[i]
            tangent = tangent / np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            middle = diagram.points[[i, other]].mean(axis=0)
            far = (
                diagram.vertices[ends[1]]
                + radius * np.sign(np.dot(middle - center, normal)) * normal
            )
            ids.append(len(vertices))
            vertices.append(far)
        coordinates = np.asarray([vertices[j] for j in ids])
        centroid = coordinates.mean(axis=0)
        angles = np.arctan2(
            coordinates[:, 1] - centroid[1], coordinates[:, 0] - centroid[0]
        )
        cells.append(np.asarray([vertices[j] for _, j in sorted(zip(angles, ids))]))
    return cells


def spatial_geometry(xy, visible):
    """Return Delaunay pairs, clipped Voronoi polygons and failure-class audit; coordinates are meters."""
    edges, failures = set(), {}
    try:
        triangles = Delaunay(xy).simplices
        for a, b, c in triangles:
            edges.update(
                tuple(sorted(map(int, pair))) for pair in ((a, b), (a, c), (b, c))
            )
    except Exception as error:
        failures["delaunay"] = type(error).__name__
    try:
        pitch = box(0, 0, 105, 68)
        clip = visible if visible is not None else pitch
        areas = [
            repaired_polygon(cell).intersection(clip).intersection(pitch)
            for cell in finite_cells(Voronoi(xy))
        ]
    except Exception as error:
        failures["voronoi"] = type(error).__name__
        areas = [None] * len(xy)
    return edges, areas, failures


def projection(point, start, end, zero_t=-1.0):
    """Return perpendicular distance in meters and unbounded segment fraction; define zero-length t explicitly."""
    vector = end - start
    length2 = float(vector @ vector)
    t = float((point - start) @ vector / length2) if length2 else zero_t
    return float(np.linalg.norm(point - (start + t * vector))), t


def line_edges(point, parameters=P, allowed=None):
    """Return active pitch-line edges with meter feet and unitless weights, optionally restricting sides."""
    x, y = map(float, point)
    sides = [
        ("own goal", (0.0, y), x),
        ("opp goal", (105.0, y), 105.0 - x),
        ("touchline low", (x, 0.0), y),
        ("touchline high", (x, 68.0), 68.0 - y),
    ]
    edges = []
    for side, foot, distance in sides:
        weight = logistic(distance, parameters.line_mid, parameters.line_sd)
        if weight >= parameters.epsilon and (allowed is None or side in allowed):
            edges.append({"side": side, "foot": foot, "p": weight})
    return edges


def team_shapes(network, corrected=False):
    """Return team edge sets and exception classes from meter nodes; legacy couples failures, corrected isolates them."""
    function = (
        shape_corrected.shape_graph_edges
        if corrected
        else shape_legacy.shape_graph_edges
    )
    pressed = [network["carrier"]] + [
        i for i in network["receivers"] if not network["nodes"][i]["keeper"]
    ]
    result, failures = {}, {}
    for side, indices in [("pressed", pressed), ("pressing", network["defenders"])]:
        try:
            result[side] = function({i: network["xy"][i] for i in indices})
        except Exception as error:
            result[side] = None
            failures[side] = type(error).__name__
            if not corrected:
                return {"pressed": None, "pressing": None}, failures
    return result, failures


def build_network(event, frame, parameters=P, flip=None):
    """Build physical, pressure, pass and boundary structure from provider records; metric geometry and unitless weights."""
    nodes, carrier = player_nodes(event, frame, flip)
    if carrier is None:
        raise ValueError("missing_carrier")
    if len(nodes) < 4:
        raise ValueError("fewer_than_four_nodes")
    xy = np.vstack([node["xy"] for node in nodes])
    receivers = [
        i for i, node in enumerate(nodes) if node["role"] == "attacker" and i != carrier
    ]
    defenders = [
        i
        for i, node in enumerate(nodes)
        if node["role"] == "defender" and not node["keeper"]
    ]
    orientation = (
        team(event) != team(event, "possession_team") if flip is None else bool(flip)
    )
    visible = visible_polygon(frame, orientation)
    topology, areas, failures = spatial_geometry(xy, visible)
    targets = [carrier] + receivers
    weights = np.array(
        [
            [
                logistic(
                    np.linalg.norm(xy[d] - xy[t]),
                    parameters.player_mid,
                    parameters.player_sd,
                )
                for t in targets
            ]
            for d in defenders
        ]
    ).reshape(len(defenders), len(targets))
    edges = []
    for row, defender in zip(weights, defenders):
        active = [(t, p) for t, p in zip(targets, row) if p > parameters.epsilon]
        denominator = sum(p for _, p in active)
        edges.extend(
            {"src": defender, "dst": target, "p": p, "w": p / denominator}
            for target, p in active
        )
    passes = []
    for receiver in receivers:
        blockers = []
        for defender in defenders:
            distance, fraction = projection(xy[defender], xy[carrier], xy[receiver])
            if 0 <= fraction <= 1:
                weight = logistic(distance, parameters.lane_mid, parameters.lane_sd)
                if weight > parameters.epsilon:
                    blockers.append((defender, weight))
        blocker, block = (
            max(blockers, key=lambda pair: pair[1]) if blockers else (None, 0.0)
        )
        distance = float(np.linalg.norm(xy[receiver] - xy[carrier]))
        passes.append(
            {
                "dst": receiver,
                "dist": distance,
                "block": block,
                "blocker": blocker,
                "w": math.exp(-distance / parameters.decay) * (1 - block),
            }
        )
    carrier_lines = line_edges(xy[carrier], parameters)
    allowed = {edge["side"] for edge in carrier_lines}
    boundaries = [
        {**edge, "dst": carrier, "on_carrier": True} for edge in carrier_lines
    ]
    for receiver in receivers:
        boundaries.extend(
            {**edge, "dst": receiver, "on_carrier": False}
            for edge in line_edges(xy[receiver], parameters, allowed)
        )
    return {
        "nodes": nodes,
        "carrier": carrier,
        "xy": xy,
        "receivers": receivers,
        "defenders": defenders,
        "topology": topology,
        "areas": areas,
        "visible": visible,
        "geometry_failures": failures,
        "pressure_weights": weights,
        "pressure_edges": edges,
        "passes": passes,
        "boundaries": boundaries,
        "parameters": parameters,
    }
