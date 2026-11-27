"""Label controlled regains within five seconds without possession bookkeeping.

The primary label requires control in play; the secondary label also credits an
awarded restart. These are sensitivities to the existing broad success outcome."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

from spn._core.pressure_events import BALL_POINTS, name, team

WON = frozenset({"Won", "Success", "Success In Play"})

ENDS = frozenset({"Half End", "Match End"})

STOP_TYPES = frozenset(
    {
        "Ball Out",
        "Offside",
        "Injury Stoppage",
        "Substitution",
        "Shield",
        "Player On",
        "Player Off",
        "Own Goal Against",
        "Own Goal For",
    }
)


def milliseconds(timestamp):
    """Parse a period-relative HH:MM:SS timestamp into integer milliseconds; malformed or negative input returns None."""
    try:
        h, m, s = str(timestamp).split(":")
        result = (Decimal(h) * 3600 + Decimal(m) * 60 + Decimal(s)) * 1000
        return (
            int(result.to_integral_value(rounding=ROUND_HALF_UP))
            if result.is_finite() and result >= 0
            else None
        )
    except (ValueError, TypeError, InvalidOperation):
        return None


def detail(event, field, subfield):
    """Return a nested event descriptor string, without touching possession fields."""
    return ((event.get(field) or {}).get(subfield) or {}).get("name", "")


def restart(event):
    """Return restart kind and awarded event-team ID; a drop ball has no owner, and absence returns (None, None)."""
    kind = name(event)
    if kind == "Referee Ball-Drop":
        return "drop_ball", None
    raw = (
        detail(event, "pass", "type")
        if kind == "Pass"
        else detail(event, "shot", "type")
        if kind == "Shot"
        else ""
    )
    normalized = raw.lower().replace(" ", "").replace("-", "")
    mapping = {
        "freekick": "free_kick",
        "throwin": "throw_in",
        "corner": "corner",
        "penalty": "free_kick",
        "goalkick": "goal_kick",
        "kickoff": "kick_off",
    }
    return (mapping[normalized], team(event)) if normalized in mapping else (None, None)


def stoppage(event):
    """Return a plain stoppage reason or None; advantage fouls and injury-clearance passes without out remain in play."""
    kind = name(event)
    if kind in STOP_TYPES:
        return kind.lower().replace(" ", "_")
    if event.get("out") is True:
        return "out"
    if kind in {"Foul Committed", "Foul Won"}:
        field = "foul_committed" if kind == "Foul Committed" else "foul_won"
        return None if (event.get(field) or {}).get("advantage") else "foul"
    if kind == "Shot" and detail(event, "shot", "outcome") == "Goal":
        return "goal"
    outcomes = [
        detail(event, field, "outcome")
        for field in ("pass", "shot", "duel", "interception", "goalkeeper")
    ]
    if any(
        value
        in {
            "Out",
            "Off T",
            "Wayward",
            "Lost Out",
            "Won Out",
            "Success Out",
            "Pass Offside",
        }
        for value in outcomes
    ):
        return "out_or_offside"
    return None


def keeper_collection(event):
    """Return whether a goalkeeper event establishes in-play control according to D6; no possession field is consulted."""
    if name(event) not in {"Goal Keeper", "Goalkeeper"} or stoppage(event):
        return False
    kind, outcome = (
        detail(event, "goalkeeper", "type"),
        detail(event, "goalkeeper", "outcome"),
    )
    failed = any(
        word in outcome.lower() for word in ("fail", "lost", "no touch", "unsuccess")
    )
    return (
        (kind == "Collected" and not failed)
        or (kind == "Keeper Sweeper" and outcome == "Claim")
        or (kind == "Smother" and outcome in WON)
    )


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


def direct_control(event):
    """Return the direct-control reason for a completed pass, carry, shot or qualifying keeper collection; otherwise None."""
    kind = name(event)
    if (
        kind == "Pass"
        and detail(event, "pass", "outcome") in {"", "Complete", "Success"}
        and not event.get("out")
    ):
        return "completed_pass"
    if kind in {"Carry", "Shot"}:
        return kind.lower()
    return "keeper_collection" if keeper_collection(event) else None


def confirmation(events, position, pressing, pressed):
    """Return a confirming event ID among the next two ball contacts within 2,000 ms; the primary horizon may be exceeded."""
    origin = events[position]
    start = milliseconds(origin.get("timestamp"))
    if start is None:
        return None
    contacts = 0
    for event in events[position + 1 :]:
        moment = milliseconds(event.get("timestamp"))
        if (
            event.get("period") != origin.get("period")
            or moment is None
            or moment > start + 2000
        ):
            break
        if moment < start:
            continue
        if name(event) in ENDS or stoppage(event) or restart(event)[0]:
            break
        if name(event) not in BALL_POINTS:
            continue
        contacts += 1
        if team(event) == pressed:
            break
        if (
            contacts <= 2
            and team(event) == pressing
            and name(event) in {"Pass", "Carry", "Shot"}
        ):
            return event.get("id")
        if contacts >= 2:
            break
    return None


def next_restart(events, position):
    """Find the first set piece/drop ball at or after a stoppage in its period; return kind, team ID, and evidence ID."""
    period = events[position].get("period")
    for event in events[position:]:
        if event.get("period") != period or name(event) in ENDS:
            break
        kind, owner = restart(event)
        if kind is not None:
            return kind, owner, event.get("id")
    return None, None, None


def label_onset(events, position, pressing, pressed):
    """Return E2/E2r (0/1 or None if unobservable), reasons and evidence IDs for one onset; time windows use milliseconds."""
    if (
        position < 0
        or position >= len(events)
        or pressing is None
        or pressed is None
        or pressing == pressed
    ):
        raise ValueError("Invalid onset position or team identity")
    onset = events[position]
    start = milliseconds(onset.get("timestamp"))
    if start is None:
        return {
            "e2": None,
            "e2r": None,
            "reason": "invalid_onset_time",
            "evidence_id": None,
            "restart_id": None,
        }

    def result(value, reason, event=None, restart_id=None, restart_value=None):
        """Build one label record with unitless outcomes and event identifiers, keeping unknown outcomes distinct."""
        return {
            "e2": value,
            "e2r": value if restart_value is None else restart_value,
            "reason": reason,
            "evidence_id": event.get("id") if event else None,
            "restart_id": restart_id,
        }

    for i in range(position, len(events)):
        event = events[i]
        moment = milliseconds(event.get("timestamp"))
        if event.get("period") != onset.get("period"):
            return result(None, "period_changed_before_observed_window", event)
        if moment is None:
            return result(None, "invalid_following_time", event)
        if moment > start + 5000:
            return result(0, "window_elapsed", event)
        if moment < start:
            continue
        if name(event) in ENDS:
            return result(None, "period_end_within_window", event)
        restarting = restart(event)[0] if i > position else None
        # Direct control is established by a pressing-team shot even if that shot
        # ends in a goal or out. A set piece cannot establish control in play.
        if i > position and not restarting and team(event) == pressing:
            control = direct_control(event)
            if control:
                return result(1, control, event)
            if regain_candidate(event) and not stoppage(event):
                confirmed = confirmation(events, i, pressing, pressed)
                if confirmed is not None:
                    answer = result(
                        1, "confirmed_" + name(event).lower().replace(" ", "_"), event
                    )
                    answer["confirmation_id"] = confirmed
                    return answer
        reason = stoppage(event) or ("restart" if restarting else None)
        if reason:
            kind, owner, evidence = next_restart(events, i)
            credit = int(
                kind in {"free_kick", "throw_in", "corner"} and owner == pressing
            )
            return result(0, "stoppage_" + reason, event, evidence, credit)
    return result(None, "unobserved_window_end")
