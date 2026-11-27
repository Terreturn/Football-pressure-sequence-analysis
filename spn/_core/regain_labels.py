"""Resolve controlled regains from the event timeline, starting at a recorded pressure. The e2 field counts control in play within five seconds."""

from __future__ import annotations

from spn._core.controlled_regain import (
    detail,
    regain_candidate,
    label_onset as first_pass_label,
)

from spn._core.pressure_events import name, team

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


def label_onset(events, position, pressing, pressed, mode="historical"):
    """Return E2/E2r 0/1/None and evidence for index-ordered events, or explicitly select untouched first-pass behavior."""
    if mode == "first_pass":
        return first_pass_label(events, position, pressing, pressed)
    if mode != "historical":
        raise ValueError("Unknown controlled-regain rule mode")
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
