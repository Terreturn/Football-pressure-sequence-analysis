"""Score a recorded presser's observed location against feasible alternatives in the same frame."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import qmc
from shapely.geometry import Point, Polygon

from .feature_engineering import (
    INPUTS,
    after_opponent,
    context_features,
    frame_features,
    sequence_incoming,
    static_features,
)
from .model import MODEL_DIR, load_booster, probabilities, verified_path
from .networks import build_network, xy_m
from .player_tracking import Timeline, actor_index, frame_points, infer, pressure_view
from .pressure_events import Rules, name, pre_context, pressure, seconds, team


def score_positions(match, model=None):
    """Score eligible recorded pressers in one MatchData object, in percentage points."""
    return (model or PositioningModel()).predict(match)


class PositioningModel:
    """Use the fixture's held-out regain model, or all five folds on a new match."""

    def __init__(self, config_path=None):
        config_path = Path(config_path or MODEL_DIR / "model_config.json")
        config = json.loads(config_path.read_text())
        self.model_id = config["model_id"]
        self.directory = config_path.parent
        self.checksums = config["artifact_checksums"]
        settings = config["positioning"]
        self.artifacts = settings["fold_artifacts"]
        fold_file = settings["fold_map"]
        fold_path = verified_path(self.directory / fold_file, self.checksums[fold_file])
        self.fold_map = json.loads(fold_path.read_text())
        self.models = {}

    def predict(self, match):
        match_id = str(match.match_id)
        folds = (
            [self.fold_map[match_id]] if match_id in self.fold_map else range(len(self.artifacts))
        )
        models = []
        for fold in folds:
            if fold not in self.models:
                name = self.artifacts[fold]
                self.models[fold] = load_booster(
                    verified_path(self.directory / name, self.checksums[name])
                )
            models.append(self.models[fold])
        rows = []
        for info, vectors in pressure_positions(match.events, match.frames, match.match_id):
            if not info["high"] or len(vectors) < 2:
                continue
            table = pd.DataFrame(vectors, columns=INPUTS)
            p = np.mean([probabilities(model, table).astype(float) for model in models], axis=0)
            rows.append(
                dict(
                    info,
                    valid_alternatives=len(p) - 1,
                    observed_probability=p[0],
                    alternative_probability=p[1:].mean(),
                    position_quality_pp=position_quality(p[0], p[1:]),
                )
            )
        return pd.DataFrame(
            rows,
            columns=[
                "match_id",
                "event_id",
                "player_id",
                "pressing_team_id",
                "high",
                "e2",
                "valid_alternatives",
                "observed_probability",
                "alternative_probability",
                "position_quality_pp",
            ],
        )


def pressure_positions(events, frames, match_id, *, move_players=True):
    """Yield each eligible recorded pressure and its ordered list of 26-feature vectors.

    A pressure in the broader regain cohort contributes its observed vector. For a
    high press, move_players also returns vectors for feasible alternatives in the
    same frame. A player's location changes in meters; every other player, the
    identity of the carrier, the event outcome and preceding play remain fixed.
    """
    teams = {team(e) for e in events if team(e) is not None}
    if len(teams) != 2:
        raise ValueError("Expected two teams in the event file")
    timeline = Timeline(events)
    rules = Rules()
    for index, event in enumerate(events):
        if event["type"]["name"] != "Pressure" or not team(event, "player"):
            continue
        uid = event["id"]
        frame = frames.get(uid)
        actor = actor_index(frame)
        if actor is None:
            continue
        player = frame["freeze_frame"][actor]
        if player.get("keeper") or not event.get("location"):
            continue
        # Compare on the StatsBomb grid; retain the original 360 coordinates for scoring.
        if not np.array_equal(np.round(player["location"], 1), np.round(event["location"], 1)):
            continue
        pressing = team(event)
        (owner,) = teams - {pressing}
        outcome = label_onset(events, index, pressing, owner)["e2"]
        if outcome not in (0, 1):
            continue
        # A Pressure frame identifies the presser. Locate the carrier using control
        # evidence within half a second and a unique correspondence between frames.
        ref = infer(
            event,
            frame,
            timeline.evidence(event),
            frames,
            owner,
            0.5,
            "partial_fallback",
            {},
        )
        if ref["reason"] != "accepted" or len(frame["freeze_frame"]) < 4:
            continue
        carrier = ref["carrier_index"]
        view, oriented = pressure_view(event, frame, owner, carrier, {"id": ref["carrier_id"]})
        try:
            visible = strict_visible(oriented, False)
        except ValueError as error:
            if str(error) in (
                "missing_or_nonfinite_visible_polygon",
                "invalid_visible_polygon",
            ):
                continue
            raise
        # Eligibility requires four feasible directions on each of the 1.25 m and
        # 2 m rings. The 64 area-uniform alternatives below are a separate sample.
        rings = masks(oriented, actor, carrier)
        if any(bin(mask).count("1") < 4 for mask in rings.values()):
            continue
        context = onset_context(events, event, view, frames)
        measured = pressure(view["location"], oriented, rules)
        high = bool(
            view["location"][0] < rules.own_half_x
            and measured
            and measured["n_in_D"] >= 1
            and measured["P_total"] > rules.pressure_threshold
        )
        _, xy = physical_players(oriented, False)
        # Preserve the meter-to-provider-grid round trip used for the observed position in the fits.
        original, _ = move_named(oriented, actor, xy[actor])
        vectors = [regain_inputs(view, original, context)]
        if high and move_players:
            sampled = candidates(str(match_id), uid, xy[actor], xy[carrier])
            for target in sampled[["target_x", "target_y"]].to_numpy():
                if move_reason(target, visible, xy, actor) == "valid":
                    moved, _ = move_named(oriented, actor, target)
                    vectors.append(regain_inputs(view, moved, context))
        yield (
            {
                "match_id": str(match_id),
                "event_id": uid,
                "player_id": team(event, "player"),
                "pressing_team_id": pressing,
                "high": high,
                "e2": int(outcome),
            },
            vectors,
        )


def regain_inputs(event, frame, context):
    """Measure the 26 binary-model inputs at one position; preceding play stays fixed across moves."""
    network = build_network(event, frame)
    current = static_features(network, event, frame)
    row = frame_features(network, current, context["previous"], corrected_shape=True)
    row.update(context["constants"])
    return {name: row[name] for name in INPUTS}


def position_quality(observed, alternatives):
    """Return observed minus mean alternative regain probability, in percentage points."""
    values = np.asarray(alternatives, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.isfinite(values).all()
        or not np.all((values >= 0) & (values <= 1))
        or not 0 <= observed <= 1
    ):
        raise ValueError("Expected one observed probability and at least one valid alternative")
    return 100 * (float(observed) - values.mean())


# Resolve five-second controlled regains from the event timeline.

WON = frozenset({"Won", "Success", "Success In Play"})


def detail(event, field, subfield):
    """Return a nested event descriptor string, without touching possession fields."""
    return ((event.get(field) or {}).get(subfield) or {}).get("name", "")


def regain_candidate(event):
    """Identify recoveries, won interceptions and won tackle duels requiring a confirming contact; output is Boolean."""
    kind = name(event)
    if kind == "Ball Recovery":
        return not (event.get("ball_recovery") or {}).get("recovery_failure", False)
    if kind == "Interception":
        return detail(event, "interception", "outcome") in WON
    return (
        kind == "Duel"
        and detail(event, "duel", "type") == "Tackle"
        and detail(event, "duel", "outcome") in WON
    )


CONTACTS = frozenset(
    {
        "Pass",
        "Carry",
        "Dribble",
        "Ball Receipt*",
        "Ball Recovery",
        "Interception",
        "Miscontrol",
        "Dispossessed",
        "Duel",
        "50/50",
        "Shot",
        "Goal Keeper",
        "Goalkeeper",
        "Clearance",
        "Shield",
        "Block",
        "Error",
    }
)

OUTCOMES = {
    "pass": {"Out", "Pass Offside"},
    "shot": {"Off T", "Goal"},
    "duel": {"Lost Out", "Success Out"},
    "interception": {"Lost Out", "Success Out"},
    "goalkeeper": {
        "Touched Out",
        "Punched Out",
        "Punched out",
        "Lost Out",
        "Success Out",
    },
}


def clock_ms(stamp):
    """Return period-relative integer milliseconds, truncating fractional digits as final D6 does; None stays missing."""
    if not stamp:
        return None
    h, m, s = str(stamp).split(":")
    whole, _, fraction = s.partition(".")
    return (int(h) * 3600 + int(m) * 60 + int(whole)) * 1000 + int(
        (fraction + "000")[:3]
    )


def set_piece(event):
    """Return the enumerated restart name or an empty string, without using possession or play-pattern fields."""
    kind = name(event)
    if kind == "Referee Ball-Drop":
        return "Drop Ball"
    block = "pass" if kind == "Pass" else "shot" if kind == "Shot" else None
    if block is None:
        return ""
    value = detail(event, block, "type")
    allowed = (
        {"Throw-in", "Free Kick", "Corner", "Goal Kick", "Kick Off"}
        if block == "pass"
        else {"Free Kick", "Penalty", "Corner", "Kick Off"}
    )
    return value if value in allowed else ""


def stop_reason(event):
    """Return final D6's stopping category or an empty string; administrative player entries are not stoppages."""
    kind = name(event)
    if kind in {"Foul Committed", "Foul Won"}:
        key = "foul_committed" if kind == "Foul Committed" else "foul_won"
        return "" if (event.get(key) or {}).get("advantage") else "foul"
    if kind == "Offside" or detail(event, "pass", "outcome") == "Pass Offside":
        return "offside"
    if kind in {"Own Goal Against", "Own Goal For"} or (
        kind == "Shot" and detail(event, "shot", "outcome") == "Goal"
    ):
        return "goal"
    if (
        event.get("out") is True
        or kind == "Shield"
        or any(
            detail(event, b, "outcome") in outcomes for b, outcomes in OUTCOMES.items()
        )
    ):
        return "out"
    if kind == "Injury Stoppage":
        return "injury"
    if kind == "Substitution":
        return "substitution"
    return "restart" if set_piece(event) else ""


def control_reason(event):
    """Return a direct-control reason under final D6, before this same event's terminal consequence."""
    kind = name(event)
    if kind == "Pass" and not detail(event, "pass", "outcome") and not set_piece(event):
        return "press_pass"
    if kind == "Carry":
        return "press_carry"
    if kind == "Shot" and not set_piece(event):
        return "press_shot"
    if kind in {"Goal Keeper", "Goalkeeper"}:
        subtype, outcome = (
            detail(event, "goalkeeper", "type"),
            detail(event, "goalkeeper", "outcome"),
        )
        if (
            (subtype == "Collected" and outcome != "Fail")
            or (subtype == "Keeper Sweeper" and outcome == "Claim")
            or (
                subtype == "Smother"
                and outcome in {"Won", "Success", "Success In Play"}
            )
        ):
            return "press_keeper"
    return ""


def confirms(events, position, pressing, pressed):
    """Return whether either of the next two contacts confirms the regain within 2,000 ms, allowing that contact to end play."""
    origin = events[position]
    start = clock_ms(origin.get("timestamp"))
    seen = 0
    for i in range(position + 1, len(events)):
        event = events[i]
        kind = name(event)
        if event.get("period") != origin.get("period") or kind == "Half End":
            return False
        if kind not in CONTACTS:
            if stop_reason(event):
                return False
            continue
        moment = clock_ms(event.get("timestamp"))
        if moment is None or start is None or moment > start + 2000:
            return False
        seen += 1
        if team(event) == pressed:
            return False
        if (
            team(event) == pressing
            and kind in {"Pass", "Carry", "Shot"}
            and not set_piece(event)
        ):
            return True
        if stop_reason(event) or seen >= 2:
            return False
    return False


def awarded_restart(events, position, period, pressing):
    """Return restart credit 0/1 and its event ID for the first subsequent restart in this period."""
    for i in range(position, len(events)):
        event = events[i]
        if event.get("period") != period or name(event) == "Half End":
            break
        kind = set_piece(event)
        if kind:
            return int(
                team(event) == pressing
                and kind in {"Free Kick", "Penalty", "Throw-in", "Corner"}
            ), event.get("id")
    return 0, None


def label_onset(events, position, pressing, pressed):
    """Return the controlled-regain label, restart credit and supporting events for a pressure."""
    onset = events[position]
    period = onset.get("period")
    start = clock_ms(onset.get("timestamp"))
    last = start

    def answer(value, reason, event=None, secondary=None, restart_id=None):
        """Build dimensionless endpoint labels and event identifiers, preserving censored outcomes as None."""
        return {
            "e2": value,
            "e2r": value if secondary is None else secondary,
            "reason": reason,
            "evidence_id": event.get("id") if event else None,
            "restart_id": restart_id,
        }

    own_stop = stop_reason(onset)
    if own_stop in {"out", "offside"}:
        credit, uid = awarded_restart(events, position + 1, period, pressing)
        return answer(0, "stoppage_onset_" + own_stop, onset, credit, uid)
    for i in range(position + 1, len(events)):
        event = events[i]
        moment = clock_ms(event.get("timestamp"))
        if event.get("period") != period:
            break
        if moment is not None and moment > start + 5000:
            return answer(0, "window_closed", event)
        if name(event) == "Half End":
            return answer(None, "period_end", event)
        if moment is not None:
            last = moment
        if team(event) == pressing:
            control = control_reason(event)
            if control:
                return answer(1, control, event)
            if (
                regain_candidate(event)
                and not stop_reason(event)
                and confirms(events, i, pressing, pressed)
            ):
                reason = {
                    "Ball Recovery": "ball_recovery",
                    "Interception": "interception",
                    "Duel": "duel",
                }[name(event)]
                return answer(1, "recovery_confirmed_" + reason, event)
        stop = stop_reason(event)
        if stop:
            credit, uid = awarded_restart(events, i, period, pressing)
            return answer(0, "stoppage_" + stop, event, credit, uid)
    return (
        answer(None, "data_end") if last <= start + 5000 else answer(0, "window_closed")
    )


# Check where one player can move in the observed frame.

def canonical_meters(provider, flip=False):
    """Independently rescale provider points to meters in the frozen builder's multiply-then-divide operation order.

    Collapsing the scale to .875/.85 is algebraically equal but can move a strict
    half-meter collision comparison by a floating-point rounding unit. The coordinate
    representation is an input convention; polygon and stencil checks remain separate.
    """
    points = np.asarray(provider, dtype=float)
    if flip:
        points = np.array([120.0, 80.0]) - points
    return np.divide(np.multiply(points, [105.0, 68.0]), [120.0, 80.0])


def contains_closed(vertices, points):
    """Return closed-polygon containment using ray crossings and segment membership for points in meters."""
    vertices, points = np.asarray(vertices, float), np.atleast_2d(points).astype(float)
    inside = np.zeros(len(points), bool)
    boundary = np.zeros(len(points), bool)
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        edge = b - a
        rel = points - a
        cross = edge[0] * rel[:, 1] - edge[1] * rel[:, 0]
        dot = rel @ edge
        if np.dot(edge, edge) == 0:
            boundary |= np.linalg.norm(rel, axis=1) <= 1e-12
        else:
            boundary |= (
                (np.abs(cross) <= 1e-12) & (dot >= 0) & (dot <= np.dot(edge, edge))
            )
        crosses = (a[1] > points[:, 1]) != (b[1] > points[:, 1])
        if b[1] != a[1]:
            intersection_x = a[0] + (points[:, 1] - a[1]) * (b[0] - a[0]) / (
                b[1] - a[1]
            )
            inside ^= crosses & (points[:, 0] < intersection_x)
    return inside | boundary


RADII = (1.25, 2.5)


def strict_visible(frame, flip):
    """Return the recorded valid Polygon in meters; raise on unavailable/invalid geometry, without repair."""
    raw = np.asarray(frame.get("visible_area") or [], dtype=float)
    if raw.size < 6 or raw.size % 2 or not np.isfinite(raw).all():
        raise ValueError("missing_or_nonfinite_visible_polygon")
    polygon = Polygon([xy_m(p, flip) for p in raw.reshape(-1, 2)])
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise ValueError("invalid_visible_polygon")
    return polygon


def physical_players(frame, flip):
    """Return located raw-row indices and physical metric positions, retaining keepers and the frame actor."""
    indices = [
        i
        for i, p in enumerate(frame.get("freeze_frame") or [])
        if p.get("location") is not None
    ]
    xy = np.array([xy_m(frame["freeze_frame"][i]["location"], flip) for i in indices])
    if not len(xy) or xy.shape != (len(indices), 2) or not np.isfinite(xy).all():
        raise ValueError("invalid_physical_positions")
    return indices, xy


def stencil(origin, carrier):
    """Return sixteen (radius m, CCW direction index, target xy m) tuples; reject undefined zero-distance bearings."""
    delta = np.asarray(carrier, float) - origin
    distance = float(np.linalg.norm(delta))
    if not np.isfinite(distance) or distance == 0:
        raise ValueError("undefined_carrier_bearing")
    toward = delta / distance
    perpendicular = np.array([-toward[1], toward[0]])
    points = [
        (
            radius,
            direction,
            np.asarray(origin)
            + radius
            * (
                math.cos(direction * math.pi / 4) * toward
                + math.sin(direction * math.pi / 4) * perpendicular
            ),
        )
        for radius in RADII
        for direction in range(8)
    ]
    if len({tuple(p) for _, _, p in points}) != 16:
        raise ValueError("non_distinct_stencil")
    for radius, direction, target in points:
        if abs(np.linalg.norm(target - origin) - radius) > 1e-6:
            raise ValueError("wrong_stencil_radius")
        if (
            direction == 0
            and np.linalg.norm((target - origin) / radius - toward) > 1e-6
        ):
            raise ValueError("wrong_stencil_bearing")
    return points


def move_reason(target, visible, physical, node_index):
    """Return first invalidity reason, or valid, for meter positions; collision is strictly less than .5 m."""
    x, y = target
    if not (0 <= x <= 105 and 0 <= y <= 68):
        return "pitch"
    if not visible.covers(Point(float(x), float(y))):
        return "visibility"
    others = np.delete(physical, node_index, axis=0)
    if np.any(np.linalg.norm(others - target, axis=1) < 0.5):
        return "collision"
    return "valid"


def moved_frame(frame, raw_index, target, flip):
    """Copy only the moved raw player; return the frame and metric round-trip error, leaving all other records identical."""
    provider = np.asarray(target, float) / np.array([0.875, 0.85])
    if flip:
        provider = np.array([120.0, 80.0]) - provider
    players = list(frame["freeze_frame"])
    players[raw_index] = {**players[raw_index], "location": provider.tolist()}
    altered = {**frame, "freeze_frame": players}
    error = float(np.linalg.norm(xy_m(provider, flip) - target))
    if error > 1e-6:
        raise ValueError("movement_roundtrip_failed")
    return altered, error


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


def move_named(frame, index, target):
    """Return a possession-oriented frame with only one raw row moved to target meters; zero movement is exact identity."""
    indices, xy = physical_players(frame, False)
    origin = xy[indices.index(index)]
    if np.array_equal(origin, np.asarray(target)):
        return deepcopy(frame), 0.0
    return moved_frame(frame, index, target, False)


def event_seed(match_id, event_id):
    """Map two unitless stable IDs to the specified 32-bit Sobol scramble seed, independent of outcomes and row order."""
    return int(
        sha256(f"epl-sobol-area-v1|{match_id}|{event_id}".encode()).hexdigest()[:8], 16
    )


def transform(unit, origin, carrier):
    """Map unit-square fractions to radius/CCW angle/target-meter arrays; theta zero points from origin to carrier."""
    unit, origin, carrier = (
        np.asarray(unit, float),
        np.asarray(origin, float),
        np.asarray(carrier, float),
    )
    if (
        unit.ndim != 2
        or unit.shape[1] != 2
        or origin.shape != (2,)
        or carrier.shape != (2,)
        or not np.isfinite(unit).all()
        or not np.isfinite(origin).all()
        or not np.isfinite(carrier).all()
        or np.any((unit < 0) | (unit > 1))
    ):
        raise ValueError("Invalid sampling coordinates")
    delta = carrier - origin
    distance = np.linalg.norm(delta)
    if distance == 0:
        raise ValueError("undefined_carrier_bearing")
    toward = delta / distance
    across = np.array([-toward[1], toward[0]])
    radius = np.sqrt(0.25 + 3.75 * unit[:, 0])
    theta = 2 * np.pi * unit[:, 1]
    target = origin + radius[:, None] * (
        np.cos(theta)[:, None] * toward + np.sin(theta)[:, None] * across
    )
    return radius, theta, target


def candidates(match_id, event_id, origin, carrier, diagnostic=False):
    """Return 64 primary or 128 nested candidate records with meter coordinates and radian angles; no geometry filtering."""
    seed = event_seed(match_id, event_id)
    unit = qmc.Sobol(d=2, scramble=True, seed=seed).random_base2(7 if diagnostic else 6)
    radius, theta, target = transform(unit, origin, carrier)
    return pd.DataFrame(
        {
            "sample_id": np.arange(len(unit)),
            "seed": seed,
            "u": unit[:, 0],
            "v": unit[:, 1],
            "radius_m": radius,
            "theta_rad": theta,
            "target_x": target[:, 0],
            "target_y": target[:, 1],
            "dx_m": target[:, 0] - origin[0],
            "dy_m": target[:, 1] - origin[1],
            "primary": np.arange(len(unit)) < 64,
            "octant": np.minimum(7, np.floor(theta / (np.pi / 4)).astype(int)),
            "radial_band": np.where(
                radius < 1, "0.5–1", np.where(radius < 1.5, "1–1.5", "1.5–2")
            ),
        }
    )


# Keep preceding play fixed when comparing candidate locations.

def available_events(events, observation):
    """Return events available at the observation's period, elapsed seconds and provider order."""
    period, stamp, index = (
        observation["period"],
        seconds(observation["timestamp"]),
        observation["index"],
    )
    return [
        e
        for e in events
        if e["index"] <= index
        and (
            e["period"] < period
            or (e["period"] == period and seconds(e["timestamp"]) <= stamp)
        )
    ]


def onset_context(events, raw, view, frames):
    """Construct the onset features from the carrier position and strictly prior source events, in period-relative seconds."""
    known = available_events(events, raw)
    if known[-1]["id"] != raw["id"]:
        raise ValueError("Invalid event order for onset")
    prefix = known[:-1] + [view]
    history = pre_context(
        prefix, len(prefix) - 1, view["location"], frames, set(), Rules()
    )
    by_id = {e["id"]: e for e in prefix}
    uid = history.get("pre_context_event_id", "")
    if uid and seconds(by_id[uid]["timestamp"]) >= seconds(raw["timestamp"]):
        history = {
            "pre_context_status": "tied_or_later_context",
            "pre_context_direction_valid": False,
        }
    anchor = {
        **history,
        "anchor_event_id": view["id"],
        "actor_x_norm": round(view["location"][0], 2),
        "actor_y": round(view["location"][1], 1),
    }
    previous, _ = context_features(anchor, view, by_id, frames)
    incoming = sequence_incoming([anchor], by_id)[view["id"]]
    return {
        "previous": previous,
        "constants": {**incoming, "seq_after_opponent": after_opponent(anchor)},
    }
