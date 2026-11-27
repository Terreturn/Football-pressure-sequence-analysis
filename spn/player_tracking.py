"""Identify the recorded presser and infer the ball carrier across adjacent 360 frames."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import math
import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

from .pressure_events import ACTIONS, HARD_STOPS, RESTARTS, end_seconds, name, seconds, team


def frame_points(frame, flip=False):
    """Return raw indices, meter coordinates and team/keeper flags; reject missing or nonfinite positions."""
    rows = frame.get("freeze_frame") or []
    if not rows or any(p.get("location") is None for p in rows):
        raise ValueError("missing_player_position")
    xy = np.asarray([p["location"][:2] for p in rows], float)
    if xy.shape != (len(rows), 2) or not np.isfinite(xy).all():
        raise ValueError("invalid_player_position")
    if flip:
        xy = np.array([120.0, 80.0]) - xy
    xy = xy * np.array([105.0, 68.0]) / np.array([120.0, 80.0])
    flags = np.asarray(
        [(bool(p.get("teammate")) != flip, bool(p.get("keeper"))) for p in rows]
    )
    return np.arange(len(rows)), xy, flags


def stamp(event):
    """Return period-relative seconds parsed independently from a recorded event timestamp."""
    parts = event.get("timestamp", "").split(":")
    return (
        float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        if len(parts) == 3
        else math.nan
    )


# Identify the actor in a frame and orient play to the team in possession.

def actor_index(frame):
    """Return the unique raw actor row, or None; identifiers are unitless and missing coordinates are rejected."""
    if frame is None:
        return None
    found = [
        i
        for i, p in enumerate(frame.get("freeze_frame") or [])
        if p.get("actor") and p.get("location") is not None and p.get("teammate")
    ]
    return found[0] if len(found) == 1 else None


def carrier_match(
    reference, reference_frame, target, target_frame, owner, maximum_gap=0.100
):
    """Return a unique carrier row and meter/second diagnostics under the prespecified motion allowance; no target actor flag used."""
    result = {
        "reason": "",
        "carrier_index": -1,
        "gap_seconds": math.nan,
        "distance_m": math.nan,
        "bound_m": math.nan,
        "candidates": 0,
    }
    if reference.get("period") != target.get("period") or team(reference) != owner:
        return {**result, "reason": "period_or_owner_mismatch"}
    gap = abs(seconds(reference.get("timestamp")) - seconds(target.get("timestamp")))
    result["gap_seconds"] = gap
    if not math.isfinite(gap) or gap > maximum_gap + 1e-10:
        return {**result, "reason": "outside_time_window"}
    actor = actor_index(reference_frame)
    if actor is None or target_frame is None:
        return {**result, "reason": "missing_reference_actor_or_frame"}
    try:
        _, ref, _ = frame_points(reference_frame, team(reference) != owner)
        _, dst, flags = frame_points(target_frame, team(target) != owner)
    except (ValueError, TypeError):
        return {**result, "reason": "invalid_frame"}
    candidate_rows = np.flatnonzero(flags[:, 0])
    distances = np.linalg.norm(dst[candidate_rows] - ref[actor], axis=1)
    bound = 0.5 + 12 * gap
    result.update(
        bound_m=bound,
        distance_m=float(np.min(distances)) if len(distances) else math.nan,
    )
    inside = candidate_rows[distances <= bound]
    result["candidates"] = len(inside)
    if len(inside) != 1:
        return {
            **result,
            "reason": "ambiguous_carrier" if len(inside) else "no_carrier_in_bound",
        }
    return {**result, "reason": "matched", "carrier_index": int(inside[0])}


def carrier_match_aligned(
    reference, reference_frame, target, target_frame, owner, maximum_gap=0.100
):
    """Return a carrier row unique across all feasible whole-frame matchings after an audited meter-valued offset."""
    result = carrier_match(
        reference, reference_frame, target, target_frame, owner, maximum_gap
    )
    if result["reason"] in (
        "period_or_owner_mismatch",
        "outside_time_window",
        "missing_reference_actor_or_frame",
        "invalid_frame",
    ):
        return result
    result.update(carrier_index=-1, reason="frame_incompatible")
    _, ref, rf = frame_points(reference_frame, team(reference) != owner)
    _, dst, tf = frame_points(target_frame, team(target) != owner)
    if len(ref) != len(dst) or Counter(map(tuple, rf)) != Counter(map(tuple, tf)):
        return {**result, "reason": "different_player_composition"}
    offsets = []
    for side in (False, True):
        if not np.any(rf[:, 0] == side):
            return {**result, "reason": "missing_team_for_alignment"}
        offsets.append(
            dst[tf[:, 0] == side].mean(axis=0) - ref[rf[:, 0] == side].mean(axis=0)
        )
    offset = dst.mean(axis=0) - ref.mean(axis=0)
    result.update(
        alignment_x_m=float(offset[0]),
        alignment_y_m=float(offset[1]),
        alignment_norm_m=float(np.linalg.norm(offset)),
        alignment_team_disagreement_m=float(np.linalg.norm(offsets[0] - offsets[1])),
    )
    if result["alignment_norm_m"] > 3 or result["alignment_team_disagreement_m"] > 0.5:
        return {**result, "reason": "inconsistent_frame_offset"}
    residual = np.linalg.norm(ref[:, None, :] + offset - dst[None, :, :], axis=2)
    allowed = (residual <= result["bound_m"]) & np.all(
        rf[:, None, :] == tf[None, :, :], axis=2
    )
    rr, cc = linear_sum_assignment((~allowed).astype(int))
    if not allowed[rr, cc].all():
        return {**result, "reason": "no_feasible_frame_matching"}
    source_actor = actor_index(reference_frame)
    target_actor = int(cc[np.flatnonzero(rr == source_actor)[0]])
    alternate = allowed.copy()
    alternate[source_actor, target_actor] = False
    ar, ac = linear_sum_assignment((~alternate).astype(int))
    if alternate[ar, ac].all():
        return {**result, "reason": "ambiguous_global_carrier"}
    return {
        **result,
        "reason": "matched",
        "carrier_index": target_actor,
        "distance_m": float(residual[source_actor, target_actor]),
        "unaligned_distance_m": float(
            np.linalg.norm(ref[source_actor] - dst[target_actor])
        ),
        "candidates": int(allowed[source_actor].sum()),
    }


def pressure_view(event, frame, owner, carrier, carrier_player):
    """Return a possession-oriented scoring event/frame, preserving raw row order and all physical positions in meters."""
    if team(event) == owner or carrier == actor_index(frame):
        raise ValueError("Pressure actor cannot be assigned as carrier")
    rows = []
    for i, player in enumerate(frame["freeze_frame"]):
        location = player.get("location")
        if location is None:
            raise ValueError("Missing position")
        rows.append(
            {
                **player,
                "location": [120.0 - location[0], 80.0 - location[1]],
                "teammate": not bool(player.get("teammate")),
                "actor": i == carrier,
            }
        )
    if not rows[carrier]["teammate"]:
        raise ValueError("Carrier is not on possession team")
    visible = np.asarray(frame.get("visible_area") or [], dtype=float).reshape(-1, 2)
    changed = {
        **frame,
        "freeze_frame": rows,
        "visible_area": (np.array([120.0, 80.0]) - visible).ravel().tolist(),
    }
    view = {
        **event,
        "type": {"name": "Pressure-frame scoring view"},
        "team": {"id": owner},
        "possession_team": {"id": owner},
        "player": carrier_player,
        "location": rows[carrier]["location"],
    }
    return view, changed


# Match player rows across nearby frames.

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


# Use the event timeline to decide which carrier matches have supporting evidence.

EPS = 1e-9

CONTROL = ACTIONS | {"Ball Receipt*"}

BREAK = HARD_STOPS | {
    "Clearance",
    "Miscontrol",
    "Dispossessed",
    "Shot",
    "Duel",
    "50/50",
    "Interception",
    "Ball Recovery",
    "Goal Keeper",
    "Goalkeeper",
    "Error",
}


def start(e):
    """Return an event's start time in period seconds."""
    return e["_start"] if "_start" in e else seconds(e.get("timestamp"))


def end(e):
    """Return an event's end time in period seconds."""
    return e["_end"] if "_end" in e else end_seconds(e)


class Timeline:
    """Index each period in seconds, retaining intervals that can span a query time."""

    def __init__(self, events):
        """Index raw match events and maximum period action duration in seconds."""
        self.periods = defaultdict(list)
        for e in events:
            t = seconds(e.get("timestamp"))
            if math.isfinite(t):
                self.periods[e["period"]].append(
                    {**e, "_start": t, "_end": end_seconds(e)}
                )
        self.times = {}
        self.spans = {}
        for period, rows in self.periods.items():
            rows.sort(key=lambda e: (start(e), e["index"], e["id"]))
            self.times[period] = [start(e) for e in rows]
            self.spans[period] = max(
                (end(e) - start(e) for e in rows if name(e) in ("Pass", "Carry")),
                default=0.0,
            )

    def evidence(self, target):
        """Return other period events sufficient for a half-second window and any overlapping pass/carry."""
        period = target["period"]
        t = start(target)
        lower = t - 0.5 - self.spans[period] - EPS
        times = self.times[period]
        return [
            e
            for e in self.periods[period][
                bisect_left(times, lower) : bisect_right(times, t + 0.5 + EPS)
            ]
            if e["id"] != target["id"]
        ]


def boundary(e):
    """Return whether an action interrupts corroborated control, using the recorded outcome and restart fields."""
    return (
        name(e) in BREAK
        or bool(e.get("out"))
        or (name(e) in ("Pass", "Shot") and name(e, "play_pattern") in RESTARTS)
        or (
            name(e) == "Ball Receipt*"
            and bool((e.get("ball_receipt") or {}).get("outcome"))
        )
        or (
            name(e) == "Dribble"
            and ((e.get("dribble") or {}).get("outcome") or {}).get("name")
            == "Incomplete"
        )
    )


def continuous_between(ref, target, evidence, owner, pid):
    """Test for a clear path from a controlling observation to the target, including endpoint timestamps in seconds."""
    t0, t1 = start(ref), start(target)
    if name(ref) == "Pass" and (
        t0 < t1 - EPS or (abs(t0 - t1) <= EPS and ref["index"] < target["index"])
    ):
        return False
    if (
        team(ref, "possession_team") not in (None, owner)
        or boundary(ref)
        or (
            target.get("possession") is not None
            and ref.get("possession") is not None
            and ref["possession"] != target["possession"]
        )
    ):
        return False
    lo, hi = sorted((t0, t1))
    for e in evidence:
        if e["id"] == ref["id"] or not lo - EPS <= start(e) <= hi + EPS:
            continue
        if boundary(e):
            return False
        if name(e) in CONTROL and (team(e) != owner or team(e, "player") != pid):
            return False
        if name(e) == "Pass" and (start(e) < hi - EPS or e["index"] < target["index"]):
            return False
        if (
            name(e) in CONTROL
            and ref.get("possession") is not None
            and e.get("possession") is not None
            and e["possession"] != ref["possession"]
        ):
            return False
    return True


def supported_owner(target, evidence, owner, gap):
    """Return the corroborated carrier ID, evidence type and source events at target time, or an abstention."""
    t = start(target)
    same = [e for e in evidence if abs(start(e) - t) <= EPS]
    if any(boundary(e) for e in same):
        return None, "control_boundary", []
    if any(
        name(e) == "Pass" and start(e) < t - EPS and t < end(e) - EPS for e in evidence
    ):
        return None, "ball_in_flight", []
    exact = [e for e in same if name(e) in CONTROL]
    identities = {(team(e), team(e, "player")) for e in exact}
    if len(identities) > 1 or any(tid != owner or not pid for tid, pid in identities):
        return None, "contradictory_tied_control", []
    if any(name(e) == "Pass" and e["index"] < target["index"] for e in exact):
        return None, "released_at_target", []
    spanning = [
        e
        for e in evidence
        if name(e) == "Carry" and start(e) < t - EPS and end(e) >= t - EPS
    ]
    usable = [e for e in exact if e["index"] < target["index"]] + spanning
    if usable:
        ids = {(team(e), team(e, "player")) for e in usable + exact}
        if len(ids) != 1 or next(iter(ids))[0] != owner or not next(iter(ids))[1]:
            return None, "conflicting_control_evidence", []
        pid = next(iter(ids))[1]
        if any(not continuous_between(e, target, evidence, owner, pid) for e in usable):
            return None, "intervening_control_or_release", []
        return (
            pid,
            "spanning_carry" if spanning else "observed_same_time_control",
            usable,
        )
    nearby = [
        e for e in evidence if name(e) in CONTROL and abs(start(e) - t) <= gap + EPS
    ]
    before = [e for e in nearby if start(e) < t - EPS]
    after = [e for e in nearby if start(e) > t + EPS]
    if before and after:
        left = max(map(start, before))
        right = min(map(start, after))
        bracket = [
            e
            for e in before + after
            if abs(start(e) - left) <= EPS or abs(start(e) - right) <= EPS
        ]
        ids = {(team(e), team(e, "player")) for e in bracket + exact}
        if len(ids) == 1:
            tid, pid = next(iter(ids))
            if (
                tid == owner
                and pid
                and all(
                    continuous_between(e, target, evidence, owner, pid) for e in bracket
                )
            ):
                return pid, "bracketed_same_controller", bracket
    return None, "no_supported_continuity", []


def eligible_refs(target, evidence, frames, owner, gap):
    """Return framed owner-team reference candidates using only observed times, types, roles and team."""
    return [
        e
        for e in evidence
        if name(e) in ACTIONS
        and team(e) == owner
        and e["id"] in frames
        and team(e, "player")
        and abs(start(e) - start(target)) <= gap + EPS
        and actor_index(frames[e["id"]]) is not None
    ]


def infer(target, tf, evidence, frames, owner, gap, mode, cache):
    """Infer an identity and a raw frame row, verifying each accepted geometric assignment independently."""
    pid, reason, sources = supported_owner(target, evidence, owner, gap)
    row = {
        "reason": reason,
        "continuity": reason,
        "carrier_id": pid,
        "carrier_index": None,
        "evidence_ids": "|".join(sorted(e["id"] for e in sources)),
        "reference_id": "",
        "method": "",
        "signed_reference_seconds": None,
    }
    if pid is None:
        return row
    refs = [
        e
        for e in eligible_refs(target, evidence, frames, owner, gap)
        if team(e, "player") == pid
        and continuous_between(e, target, evidence, owner, pid)
    ]
    for method in ["aligned", "partial"] if mode == "partial_fallback" else ["aligned"]:
        matches = []
        for ref in refs:
            key = (ref["id"], target["id"], owner, method)
            if key not in cache:
                matcher = (
                    carrier_match_aligned if method == "aligned" else partial_match
                )
                result = matcher(ref, frames[ref["id"]], target, tf, owner, 0.5)
                index = (
                    result["carrier_index"] if result["reason"] == "matched" else None
                )
                if index is not None:
                    check = independent_match(
                        ref,
                        frames[ref["id"]],
                        target,
                        tf,
                        owner,
                        0.5,
                        method == "partial",
                    )
                    if check != index:
                        raise ValueError("Independent geometric replay differs")
                cache[key] = index
            if cache[key] is not None:
                matches.append((ref, cache[key]))
        if matches:
            if len({i for _, i in matches}) != 1:
                return {**row, "reason": "conflicting_carrier_rows"}
            ref, index = min(
                matches,
                key=lambda x: (
                    abs(start(x[0]) - start(target)),
                    x[0]["index"],
                    x[0]["id"],
                ),
            )
            return {
                **row,
                "reason": "accepted",
                "carrier_index": int(index),
                "reference_id": ref["id"],
                "method": method,
                "signed_reference_seconds": start(ref) - start(target),
            }
    return {**row, "reason": "no_forced_reference_row"}
