"""Match visible players between nearby event frames."""

from __future__ import annotations

from collections import Counter

import math

import networkx as nx

import numpy as np

from scipy.optimize import linear_sum_assignment

from spn._core.named_pressure import actor_index

from spn._core.pressure_events import seconds, team

from spn._core.frame_coordinates import frame_points

from spn._core.event_clock import stamp


def cardinality(allowed):
    """Return the maximum count of allowed pairs in a Boolean bipartite matrix; no units."""
    if not allowed.size:
        return 0
    rows, cols = linear_sum_assignment((~allowed).astype(int))
    return int(allowed[rows, cols].sum())


def unique_required(allowed, actor):
    """Return the actor's forced target in every maximum matching, or None; indices are unitless."""
    count = cardinality(allowed)
    if cardinality(np.delete(allowed, actor, axis=0)) == count:
        return None
    possible = [
        int(j)
        for j in np.flatnonzero(allowed[actor])
        if 1 + cardinality(np.delete(np.delete(allowed, actor, axis=0), j, axis=1))
        == count
    ]
    return possible[0] if len(possible) == 1 else None


def partial_match(reference, rf, target, tf, owner, gap):
    """Return an unshifted partial carrier match with meter/second diagnostics, or an explicit rejection."""
    out = {"reason": "partial_rejected", "carrier_index": -1, "method": "partial"}
    dt = abs(seconds(reference.get("timestamp")) - seconds(target.get("timestamp")))
    if (
        not math.isfinite(dt)
        or dt > gap + 1e-10
        or reference.get("period") != target.get("period")
        or team(reference) != owner
    ):
        return out
    actor = actor_index(rf)
    if actor is None or tf is None:
        return out
    try:
        _, x, xf = frame_points(rf, team(reference) != owner)
        _, y, yf = frame_points(tf, team(target) != owner)
    except (ValueError, TypeError):
        return out
    distance = np.linalg.norm(x[:, None] - y[None, :], axis=2)
    allowed = (distance <= 0.5 + 12 * dt) & np.all(xf[:, None] == yf[None, :], axis=2)
    count = cardinality(allowed)
    if count < max(4, math.ceil(0.8 * min(len(x), len(y)))):
        return out
    # Team roles partition the graph, so a maximum solution matches each team
    # whenever that team's subgraph has at least one feasible edge.
    if not all(allowed[xf[:, 0] == side].any() for side in (False, True)):
        return out
    index = unique_required(allowed, actor)
    if index is None:
        return {**out, "reason": "partial_carrier_not_forced"}
    return {
        **out,
        "reason": "matched",
        "carrier_index": index,
        "gap_seconds": dt,
        "bound_m": 0.5 + 12 * dt,
        "distance_m": float(distance[actor, index]),
        "matched_players": count,
    }


def independent_match(reference, rf, target, tf, owner, gap, partial=False):
    """Reconstruct geometry and use NetworkX to prove a forced carrier; return an index or None."""
    dt = abs(stamp(reference) - stamp(target))
    if (
        not math.isfinite(dt)
        or dt > gap + 1e-10
        or reference["period"] != target["period"]
        or reference["team"]["id"] != owner
    ):
        return None
    if rf is None or tf is None:
        return None
    actors = [
        i
        for i, p in enumerate(rf.get("freeze_frame", []))
        if p.get("actor") and p.get("teammate") and p.get("location") is not None
    ]
    if len(actors) != 1:
        return None
    points, flags = [], []
    for event, frame in ((reference, rf), (target, tf)):
        rows = frame.get("freeze_frame") or []
        if not rows or any(p.get("location") is None for p in rows):
            return None
        xy = np.asarray([p["location"][:2] for p in rows], float)
        reverse = event["team"]["id"] != owner
        if reverse:
            xy = [120.0, 80.0] - xy
        xy = xy * [0.875, 0.85]
        if xy.shape != (len(rows), 2) or not np.isfinite(xy).all():
            return None
        points.append(xy)
        flags.append(
            [(bool(p.get("teammate")) != reverse, bool(p.get("keeper"))) for p in rows]
        )
    offset = np.zeros(2)
    if not partial:
        if len(points[0]) != len(points[1]) or Counter(flags[0]) != Counter(flags[1]):
            return None
        offset = points[1].mean(0) - points[0].mean(0)
        offsets = []
        for side in (False, True):
            groups = [
                np.array([p for p, flag in zip(xy, fs) if flag[0] == side])
                for xy, fs in zip(points, flags)
            ]
            if not all(len(g) for g in groups):
                return None
            offsets.append(groups[1].mean(0) - groups[0].mean(0))
        if np.linalg.norm(offset) > 3 or np.linalg.norm(offsets[0] - offsets[1]) > 0.5:
            return None
    left = [("a", i) for i in range(len(points[0]))]
    right = [("b", j) for j in range(len(points[1]))]
    graph = nx.Graph()
    graph.add_nodes_from(left + right)
    for i, x in enumerate(points[0]):
        for j, y in enumerate(points[1]):
            if (
                flags[0][i] == flags[1][j]
                and math.hypot(*(x + offset - y)) <= 0.5 + 12 * dt
            ):
                graph.add_edge(left[i], right[j])

    def size(g, nodes):
        """Return the edge count of a maximum bipartite matching; nodes carry no units."""
        return len(nx.bipartite.maximum_matching(g, top_nodes=nodes)) // 2

    count = size(graph, left)
    if partial:
        if count < max(4, math.ceil(0.8 * min(len(left), len(right)))):
            return None
        if not all(
            any(graph.degree(left[i]) for i, f in enumerate(flags[0]) if f[0] == side)
            for side in (False, True)
        ):
            return None
    elif count != len(left):
        return None
    actor = left[actors[0]]
    without = graph.copy()
    without.remove_node(actor)
    remainder = [x for x in left if x != actor]
    if size(without, remainder) == count:
        return None
    possible = []
    for target_node in graph.neighbors(actor):
        residual = without.copy()
        residual.remove_node(target_node)
        if size(residual, remainder) + 1 == count:
            possible.append(target_node[1])
    return possible[0] if len(possible) == 1 else None
