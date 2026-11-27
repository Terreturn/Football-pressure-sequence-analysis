"""Infer the ball carrier from nearby control events and visible players."""

from __future__ import annotations

from bisect import bisect_left, bisect_right

from collections import defaultdict

import math

import numpy as np

from spn._core.named_pressure import actor_index, carrier_match_aligned

from spn._core.player_matching import partial_match, independent_match

from spn._core.positioning import move_reason, stencil, strict_visible

from spn._core.pressure_events import (
    ACTIONS,
    HARD_STOPS,
    RESTARTS,
    end_seconds,
    name,
    seconds,
    team,
)

from spn._core.frame_coordinates import frame_points

from spn._core.pitch_bounds import canonical_meters, contains_closed

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


def masks(frame, actor, carrier):
    """Return independently checked eight-direction validity masks for the 1.25 m and 2 m rings."""
    _, xy, _ = frame_points(frame)
    origin = xy[actor]
    ball = xy[carrier]
    if np.linalg.norm(ball - origin) == 0:
        return {1.25: 0, 2.0: 0}
    visible = strict_visible(frame, False)
    polygon = canonical_meters(np.asarray(frame["visible_area"]).reshape(-1, 2))
    # Use the product's physical-coordinate operation order for strict collision comparisons.
    physical = canonical_meters([p["location"][:2] for p in frame["freeze_frame"]])
    origin, ball = physical[actor], physical[carrier]
    results = {1.25: 0, 2.0: 0}
    for radius in results:
        targets = []
        for r, d, target in stencil(origin, ball):
            if r == 1.25:
                targets.append(
                    target
                    if radius == 1.25
                    else origin + (target - origin) * (radius / 1.25)
                )
        angle = math.atan2(*(ball - origin)[::-1]) + np.arange(8) * math.pi / 4
        independent = origin + radius * np.column_stack([np.cos(angle), np.sin(angle)])
        np.testing.assert_allclose(targets, independent, rtol=0, atol=1e-6)
        covered = contains_closed(polygon, np.asarray(targets))
        for d, target in enumerate(targets):
            actual = move_reason(target, visible, physical, actor) == "valid"
            expected = (
                0 <= target[0] <= 105
                and 0 <= target[1] <= 68
                and covered[d]
                and all(
                    math.hypot(*(target - p)) >= 0.5
                    for j, p in enumerate(physical)
                    if j != actor
                )
            )
            if actual != expected:
                raise ValueError("Direction-mask replay differs")
            if actual:
                results[radius] |= 1 << d
    return results
