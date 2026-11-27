"""Select high-pressure events, group them into sequences and resolve their outcomes.

The feature stage checks whether every event in a resolved sequence has a usable frame.
"""

from __future__ import annotations

import math

from .pressure_events import (
    ACTIONS,
    ADMIN,
    RESTARTS,
    TOUCHES,
    Rules,
    ball_point,
    close_time,
    crosses_half,
    direction,
    distance,
    dribble_point,
    end_seconds,
    endpoint,
    high_pressure,
    keeper_control,
    name,
    pass_outcome,
    point,
    pre_context,
    pressure,
    provisional_regain,
    seconds,
    stopped,
    team,
)

SUCCESS = {
    "keeper_collect",
    "clearance_lost",
    "regain_control",
    "forced_out",
    "pressed_team_foul",
}

FAILURE = {
    "crossed_half",
    "goal_conceded",
    "press_foul",
    "forward_escape",
    "long_switch_escape",
}

NEITHER = {"retained_out", "pressed_keeper_control", "contained_release"}

VALUES = {"success": 1, "neutral": 0, "fail": -1}


def sequence_anchors(events: list[dict], frames: dict, rules: Rules = Rules()) -> list[dict]:
    """Select, group and label pressure events in one ordered match."""
    ids, indices = [e["id"] for e in events], [e["index"] for e in events]
    if (
        len(ids) != len(set(ids))
        or len(indices) != len(set(indices))
        or indices != sorted(indices)
    ):
        raise ValueError("Unique ordered events required")
    selected = detect(events, frames, rules)
    groups, reasons = group(events, frames, selected, rules)
    anchors, sequences = attach_targets(events, frames, groups, reasons, selected, rules)
    validate(anchors, sequences, {events[index]["id"] for index in selected})
    return anchors


def opponent_touch(events: list[dict], left: int, right: int) -> bool:
    """Check strict intervening raw positions for an opponent deliberate touch."""
    owner = team(events[left])
    return any(
        team(e) is not None and team(e) != owner and name(e) in TOUCHES
        for e in events[left + 1 : right]
    )


def same_context(events: list[dict], left: int, right: int, rules: Rules) -> bool:
    """Test period/team, <=5 s inactive gap and intervening touches; return boolean."""
    a, b = events[left], events[right]
    end, start = end_seconds(a), seconds(b.get("timestamp"))
    return (
        a.get("period") == b.get("period")
        and team(a) == team(b)
        and math.isfinite(end)
        and math.isfinite(start)
        and max(0, start - end) <= rules.gap_seconds
        and not opponent_touch(events, left, right)
    )


def semantic_split(
    events: list[dict],
    left: int,
    right: int,
    frames: dict,
    rules: Rules,
    selected: dict[int, dict],
) -> str:
    """Return a semantic grouping boundary reason, or empty string; no units."""
    anchor = events[left]
    if stopped(anchor):
        return "prior_anchor_stoppage"
    if crosses_half(anchor, rules):
        return "prior_anchor_crossed_half"
    for i in range(left + 1, right):
        event = events[i]
        kind = name(event)
        if event.get("period") != anchor.get("period"):
            return "period_boundary"
        if kind in ADMIN | {"Half End", "Match End", "Foul Committed", "Shot"}:
            return "terminal_or_administrative_event"
        if stopped(event):
            return "intervening_stoppage"
        if name(event, "play_pattern") in RESTARTS and event.get(
            "possession"
        ) != anchor.get("possession"):
            return "new_possession_restart_pattern"
        if keeper_control(event):
            return "keeper_control"
        if team(event) != team(anchor) and kind in TOUCHES and team(event) is not None:
            return "opponent_touch"
        if team(event) == team(anchor) and kind in ACTIONS:
            if crosses_half(event, rules):
                return "intervening_crossed_half"
            if (
                ball_point(events, i, rules) is not None
                and event["id"] in frames
                and i not in selected
            ):
                return "observable_action_not_high_pressure"
    return ""


def detect(events: list[dict], frames: dict, rules: Rules) -> dict[int, dict]:
    """Select high-pressure anchors by raw event position."""
    selected = {}
    for i, event in enumerate(events):
        if name(event) not in ACTIONS | {"Miscontrol"}:
            continue
        repair = dribble_point(events, i, rules) if name(event) == "Dribble" else None
        xy = repair["location"] if repair else point(event.get("location"))
        metrics, _ = high_pressure(event, frames.get(event["id"]), xy, rules)
        if metrics is not None:
            if repair:
                metrics.update(
                    {
                        "dribble_" + key: value
                        for key, value in repair.items()
                        if key != "location"
                    }
                )
            selected[i] = metrics
    return selected


def group(
    events: list[dict], frames: dict, selected: dict[int, dict], rules: Rules
) -> tuple[list[list[int]], list[str]]:
    """Partition selected raw positions in order; return groups and their start reasons."""
    groups, reasons = [], []
    for position in selected:
        if not groups:
            reason = "first_anchor"
        else:
            prior = groups[-1][-1]
            reason = (
                "period_team_gap_or_opponent_touch"
                if not same_context(events, prior, position, rules)
                else semantic_split(events, prior, position, frames, rules, selected)
            )
        if reason:
            groups.append([])
            reasons.append(reason)
        groups[-1].append(position)
    return groups, reasons


def restart_owner(event: dict) -> int | None:
    """Return restart team ID, rejecting conflicting event/possession ownership."""
    a, b = team(event), team(event, "possession_team")
    return (
        None
        if a is not None and b is not None and a != b
        else a
        if a is not None
        else b
    )


def confirmed_restart(event: dict, possession: int | None) -> bool:
    """Require a restart-capable event in a changed known possession; boolean output."""
    return (
        name(event, "play_pattern") in RESTARTS
        and name(event) in {"Pass", "Shot", "Goal Keeper", "Goalkeeper"}
        and possession is not None
        and event.get("possession") is not None
        and event["possession"] != possession
    )


def escape_state(anchor: dict, rules: Rules) -> str:
    """Classify release from the last anchor's displacement in metres and degrees."""
    d = direction(anchor)
    if d is None:
        return "pressure_released_unknown"
    _, _, magnitude, angle = d
    if magnitude < rules.min_displacement_m:
        return "contained_release"
    if abs(angle) <= rules.forward_degrees:
        return "forward_escape"
    if abs(angle) < rules.backward_degrees and magnitude >= rules.long_switch_m:
        return "long_switch_escape"
    return "contained_release"


class Resolution:
    """Maintain evidence while scanning forward from a sequence's final anchor."""

    def __init__(
        self,
        events: list[dict],
        frames: dict,
        positions: list[int],
        rules: Rules,
        team_ids: set[int] | None = None,
    ):
        """Accept ordered raw events, linked frames and member positions; times are seconds."""
        self.events, self.frames, self.positions, self.rules = (
            events,
            frames,
            positions,
            rules,
        )
        self.last_position = positions[-1]
        self.last = events[self.last_position]
        self.pressed = team(self.last)
        other = (
            team_ids
            if team_ids is not None
            else {team(e) for e in events if team(e) is not None}
        ) - {self.pressed}
        if len(other) > 1:
            raise ValueError("Ambiguous pressing team")
        self.pressing = next(iter(other), None)
        self.context_count = self.unstable_count = self.unobserved_count = 0
        self.pending = stopped(self.last)
        self.out_event = self.last if self.pending else None
        self.clearance = self.unstable_control = self.shot = False
        self.recovery = self.unobserved_event = None

    def unresolved(self) -> str:
        """Choose the unresolved state by evidence priority; return a unitless label."""
        for condition, state in (
            (self.pending, "unresolved_restart"),
            (self.unobserved_count, "unobservable_continuation"),
            (self.recovery is not None, "unconfirmed_regain"),
            (self.clearance, "unresolved_clearance"),
            (self.unstable_control, "unresolved_control"),
            (self.shot, "exceptional_shot"),
        ):
            if condition:
                return state
        return "unresolved"

    def finish(
        self,
        state: str,
        event: dict | None = None,
        *,
        control: dict | None = None,
        xy: list[float] | None = None,
        p: float | None = None,
        confidence: str = "high",
        reason: str = "",
        evidence: str = "",
        restart: str = "",
        penalty: bool = False,
        post_shot: bool = False,
        consistency: str = "",
    ) -> dict:
        """Package targets; displacement is m, angle degrees, control x provider units, delay s."""
        output = (
            "success"
            if state in SUCCESS
            else "fail"
            if state in FAILURE
            else "neutral"
            if state in NEITHER
            else "censored"
        )
        d = direction(self.last) or (None, None, None, None)
        resolved = event or {}
        control = control or {}
        delta = seconds(resolved.get("timestamp")) - end_seconds(self.last)
        return {
            "pressed_team_id": self.pressed,
            "press_team_id": self.pressing,
            "anchor_count": len(self.positions),
            "first_anchor_event_id": self.events[self.positions[0]]["id"],
            "last_anchor_event_id": self.last["id"],
            "last_anchor_type": name(self.last),
            "sequence_start_timestamp": self.events[self.positions[0]].get(
                "timestamp", ""
            ),
            "sequence_end_timestamp": self.last.get("timestamp", ""),
            "sequence_end_state": state,
            "sequence_output": output,
            "sequence_terminal": output != "censored",
            "resolution_event_id": resolved.get("id", ""),
            "resolution_event_type": name(resolved),
            "resolution_team_id": team(resolved),
            "resolution_timestamp": resolved.get("timestamp", ""),
            "resolution_delay_seconds": max(0, delta) if math.isfinite(delta) else None,
            "next_control_event_id": control.get("id", ""),
            "next_control_team_id": team(control),
            "next_control_x": xy[0] if xy is not None else None,
            "next_control_P_total": p,
            "context_event_count": self.context_count,
            "unstable_event_count": self.unstable_count,
            "skipped_unobservable_control_count": self.unobserved_count,
            "pending_restart_seen": self.pending,
            "out_signal_event_id": (self.out_event or {}).get("id", ""),
            "first_unobservable_control_event_id": (self.unobserved_event or {}).get(
                "id", ""
            ),
            "resolution_confidence": confidence,
            "resolution_evidence": evidence,
            "restart_evidence": restart,
            "censored_reason": reason,
            "sequence_consistency_error": consistency,
            "post_regain_shot": post_shot,
            "regain_candidate_event_id": (self.recovery or {}).get("id", ""),
            "dir_dx": d[0],
            "dir_dy": d[1],
            "dir_mag": d[2],
            "dir_theta_deg": d[3],
            "dir_mask": int(
                state in {"forward_escape", "long_switch_escape", "contained_release"}
            ),
            "is_penalty": penalty,
            "is_clearance": self.clearance,
            "resolution_pp": name(resolved, "play_pattern"),
            "kickoff_seen": name(resolved, "play_pattern") == "From Kick Off",
        }

    def ownership(self, event: dict, position: int, explicit: bool) -> dict:
        """Resolve restart ownership, returning final target; coordinates use provider units."""
        owner = restart_owner(event)
        pattern = name(event, "play_pattern")
        if explicit:
            restart = "explicit_out" if self.pending else "implicit_restart"
            confidence = "high" if self.pending else "medium"
            if pattern == "From Kick Off":
                return self.finish(
                    "unresolved_goal_restart",
                    event,
                    confidence="low",
                    reason="kickoff_without_observed_goal",
                    restart=restart,
                )
            if pattern == "From Free Kick" and owner in {self.pressed, self.pressing}:
                state = "press_foul" if owner == self.pressed else "pressed_team_foul"
                return self.finish(
                    state,
                    event,
                    confidence=confidence,
                    evidence="free_kick_restart",
                    restart=restart,
                )
            evidence, kwargs = "restart_owner", {}
        else:
            restart, confidence, evidence = (
                "inferred_new_possession_control",
                "medium",
                "restart_owner_from_live_control",
            )
            kwargs = {
                "control": event,
                "xy": ball_point(self.events, position, self.rules),
            }
        if owner == self.pressing:
            state = "forced_out"
        elif owner == self.pressed:
            state = "retained_out"
        else:
            return self.finish(
                "unresolved_restart",
                event,
                confidence="low",
                restart=restart,
                reason="unknown_restart_team"
                if explicit
                else "inconsistent_live_control_after_out",
            )
        return self.finish(
            state,
            event,
            confidence=confidence,
            evidence=evidence,
            restart=restart,
            **kwargs,
        )

    def stable_action(self, event: dict, position: int) -> dict | None:
        """Resolve a deliberate action, or record missing observation; pressure is unitless."""
        xy = ball_point(self.events, position, self.rules)
        if team(event) == self.pressing:
            if self.unobserved_count:
                return self.finish(
                    "ambiguous_turnover_after_unobservable_control",
                    event,
                    control=event,
                    xy=xy,
                    confidence="low",
                    reason="unobservable_control_before_turnover",
                    evidence="press_team_stable_control",
                )
            return self.finish(
                "clearance_lost" if self.clearance else "regain_control",
                event,
                control=event,
                xy=xy,
                evidence="press_team_stable_control",
            )
        if team(event) != self.pressed:
            return None
        frame = self.frames.get(event["id"])
        metrics, _ = high_pressure(event, frame, xy, self.rules)
        measured = pressure(xy, frame, self.rules)
        p = measured["P_total"] if measured else None
        if metrics is not None:
            if same_context(self.events, self.last_position, position, self.rules):
                state, reason, consistency = (
                    "continued_pressure",
                    "missing_anchor_inside_sequence",
                    "missing_anchor_inside_sequence",
                )
            elif opponent_touch(self.events, self.last_position, position):
                state, reason, consistency = (
                    "pressure_restarted_after_opponent_touch",
                    "opponent_touch_split_sequence",
                    "",
                )
            else:
                state, reason, consistency = (
                    "new_pressure_spell",
                    "inactive_gap_before_new_pressure",
                    "",
                )
            return self.finish(
                state,
                event,
                control=event,
                xy=xy,
                p=p,
                confidence="low",
                reason=reason,
                consistency=consistency,
            )
        if crosses_half(event, self.rules):
            end = endpoint(event)
            return self.finish(
                "crossed_half",
                event,
                control=event,
                xy=end if end is not None and end[0] >= self.rules.own_half_x else xy,
                evidence="stable_action_crossed_half",
            )
        if xy is None or frame is None or p is None:
            self.unobserved_count += 1
            if self.unobserved_event is None:
                self.unobserved_event = event
            return None
        state = escape_state(self.last, self.rules)
        censored = state == "pressure_released_unknown"
        return self.finish(
            state,
            event,
            control=event,
            xy=xy,
            p=p,
            confidence="low" if censored else "medium",
            reason="final_anchor_has_no_endpoint" if censored else "",
            evidence="pressure_released"
            if p <= self.rules.pressure_threshold
            else "not_high_by_full_anchor_rule",
        )

    def run(self) -> dict:
        """Scan same-period raw context with released priority; return one final outcome."""
        if not math.isfinite(end_seconds(self.last)):
            return self.finish(
                "unresolved", confidence="low", reason="invalid_last_anchor_timestamp"
            )
        if pass_outcome(self.last) == "Injury Clearance":
            return self.finish(
                "administrative_stoppage",
                self.last,
                confidence="low",
                reason="injury_clearance",
                evidence="injury_clearance",
            )
        if not self.pending and crosses_half(self.last, self.rules):
            return self.finish(
                "crossed_half",
                self.last,
                control=self.last,
                xy=endpoint(self.last),
                evidence="final_anchor_endpoint_crossed_half",
            )
        for position in range(self.last_position + 1, len(self.events)):
            event = self.events[position]
            if event.get("period") != self.last.get("period"):
                return self.finish(
                    self.unresolved(), confidence="low", reason="period_end"
                )
            self.context_count += 1
            kind, owner = name(event), team(event)
            if pass_outcome(event) == "Injury Clearance":
                return self.finish(
                    "administrative_stoppage",
                    event,
                    confidence="low",
                    reason="injury_clearance",
                    evidence="injury_clearance",
                )
            if kind in {"Half End", "Match End"}:
                return self.finish(
                    "unresolved_restart" if self.pending else "unresolved",
                    event,
                    confidence="low",
                    reason="period_end",
                )
            if not self.pending and kind in ADMIN:
                return self.finish(
                    "administrative_stoppage",
                    event,
                    confidence="low",
                    reason="administrative_stoppage",
                )
            stopped_possession = (
                self.out_event if self.out_event is not None else self.last
            ).get("possession")
            if confirmed_restart(event, stopped_possession):
                return self.ownership(event, position, explicit=True)
            if self.pending:
                if (
                    stopped_possession is not None
                    and event.get("possession") is not None
                    and event["possession"] != stopped_possession
                    and (kind in ACTIONS or keeper_control(event))
                ):
                    return self.ownership(event, position, explicit=False)
                continue
            if (
                kind == "Shot"
                and owner == self.pressed
                and ((event.get("shot") or {}).get("outcome") or {}).get("name")
                == "Goal"
            ):
                return self.finish("goal_conceded", event, evidence="goal")
            if kind == "Foul Committed" and owner in {self.pressed, self.pressing}:
                return self.finish(
                    "press_foul" if owner == self.pressing else "pressed_team_foul",
                    event,
                    penalty=bool((event.get("foul_committed") or {}).get("penalty")),
                    evidence="foul",
                )
            if kind == "Shot" and owner == self.pressing:
                return self.finish(
                    "regain_control", event, post_shot=True, evidence="press_team_shot"
                )
            if stopped(event):
                self.pending, self.out_event = True, event
                self.shot |= kind == "Shot"
                continue
            if keeper_control(event):
                if owner == self.pressing:
                    return self.finish(
                        "keeper_collect", event, evidence="keeper_control"
                    )
                if owner == self.pressed:
                    return self.finish(
                        "pressed_keeper_control",
                        event,
                        control=event,
                        xy=ball_point(self.events, position, self.rules),
                        evidence="keeper_control",
                    )
            if kind == "Shot":
                self.shot = True
                self.unstable_count += 1
            elif provisional_regain(event):
                self.recovery = event
                self.unstable_count += 1
            elif kind in ACTIONS:
                result = self.stable_action(event, position)
                if result is not None:
                    return result
            elif kind in {
                "Miscontrol",
                "Dispossessed",
                "Duel",
                "50/50",
                "Block",
                "Clearance",
                "Shield",
                "Pressure",
            }:
                self.unstable_count += 1
                self.clearance |= kind == "Clearance"
                self.unstable_control |= kind in {"Miscontrol", "Dispossessed"}
        return self.finish(self.unresolved(), confidence="low", reason="data_end")


def transition(
    current: dict, row: dict, later: dict | None, next_row: dict | None, rules: Rules
) -> dict:
    """Describe next-anchor evidence; pressure is unitless, gap seconds, endpoint error provider units."""
    result = {
        "next_anchor_event_id": "",
        "direct_transition": False,
        "transition_scope": "sequence_terminal",
        "P_start": row["P_total"],
        "P_end": None,
        "delta_P": None,
    }
    if later is None:
        return result
    xy = [next_row["actor_x_norm"], next_row["actor_y"]]
    end = endpoint(current)
    error = distance(end, xy) if end is not None else None  # provider-grid units
    gap = seconds(later.get("timestamp")) - end_seconds(current)
    direct = bool(
        end is not None
        and close_time(
            seconds(later.get("timestamp")), end_seconds(current), rules.time_tolerance
        )
        and error <= rules.location_tolerance
    )
    result.update(
        next_anchor_event_id=later["id"],
        next_anchor_type=name(later),
        next_anchor_P_total=next_row["P_total"],
        transition_time_gap_seconds=gap if math.isfinite(gap) else None,
        endpoint_distance_error=error,
        direct_transition=direct,
        transition_scope="direct_next_anchor"
        if direct
        else "next_anchor_not_direct_endpoint",
        carry_to_dribble=name(later) == "Dribble"
        and current["id"] == next_row.get("dribble_prior_carry_id"),
    )
    if direct:
        result.update(
            P_end=next_row["P_total"], delta_P=next_row["P_total"] - row["P_total"]
        )
    return result


def attach_targets(
    events: list[dict],
    frames: dict,
    groups: list[list[int]],
    reasons: list[str],
    selected: dict[int, dict],
    rules: Rules,
) -> tuple[list[dict], list[dict]]:
    """Return anchor and sequence tables with common final targets and unit sequence weights."""
    anchors, sequences = [], []
    all_ids = {events[i]["id"] for i in selected}
    team_ids = {team(e) for e in events if team(e) is not None}
    for seq_id, (positions, reason) in enumerate(zip(groups, reasons), 1):
        result = Resolution(events, frames, positions, rules, team_ids).run()
        onset = events[positions[0]]["id"]
        sequence = {
            "seq_id": seq_id,
            "onset_event_id": onset,
            "start_reason": reason,
            **result,
        }
        sequences.append(sequence)
        context = pre_context(
            events,
            positions[0],
            [selected[positions[0]]["actor_x_norm"], selected[positions[0]]["actor_y"]],
            frames,
            all_ids,
            rules,
        )
        for ev_pos, position in enumerate(positions):
            event = events[position]
            last = ev_pos == len(positions) - 1
            following = events[positions[ev_pos + 1]] if not last else None
            output = result["sequence_output"]
            row = {
                **sequence,
                "anchor_event_id": event["id"],
                "raw_event_position": position,
                "ev_pos": ev_pos,
                "is_onset": ev_pos == 0,
                "is_sequence_last": last,
                "event_type": name(event),
                "period": event.get("period"),
                "timestamp": event.get("timestamp"),
                **selected[position],
                **(
                    context
                    if ev_pos == 0
                    else {"pre_context_status": "not_sequence_start"}
                ),
                "outcome_tag": output,
                "future_output_target": output,
                "event_value": VALUES.get(output),
                "training_weight": 1 / len(positions),
                "value_model_eligible": output in VALUES,
                "representation_status": "not_yet_evaluated",
                "event_state": result["sequence_end_state"]
                if last
                else "continued_pressure",
                "state": result["sequence_end_state"] if last else "continued_pressure",
                "transition_outcome_tag": output if last else "neutral",
                "terminal": result["sequence_terminal"] if last else False,
                "censored": output == "censored" if last else False,
                "event_label_source": "sequence_output"
                if last
                else "next_sequence_anchor",
                "pass_outcome": pass_outcome(event) if name(event) == "Pass" else "",
                "dribble_outcome": (
                    (event.get("dribble") or {}).get("outcome") or {}
                ).get("name", ""),
            }
            if not last:
                row.update(
                    resolution_event_id=following["id"],
                    resolution_event_type=name(following),
                    resolution_team_id=team(following),
                    resolution_pp=name(following, "play_pattern"),
                )
            row.update(
                transition(
                    event,
                    selected[position],
                    following,
                    selected[positions[ev_pos + 1]] if not last else None,
                    rules,
                )
            )
            anchors.append(row)
    return anchors, sequences


def validate(anchors: list[dict], sequences: list[dict], selected_ids: set) -> None:
    """Assert membership, onset, weight and target conservation; counts are rows."""
    ids = [a["anchor_event_id"] for a in anchors]
    if len(ids) != len(set(ids)) or set(ids) != selected_ids:
        raise ValueError("Anchor membership conservation failed")
    grouped = {}
    for anchor in anchors:
        grouped.setdefault(anchor["seq_id"], []).append(anchor)
    if set(grouped) != {s["seq_id"] for s in sequences}:
        raise ValueError("Sequence conservation failed")
    for seq in sequences:
        rows = grouped[seq["seq_id"]]
        if (
            [a["ev_pos"] for a in rows] != list(range(len(rows)))
            or [a["raw_event_position"] for a in rows]
            != sorted(a["raw_event_position"] for a in rows)
            or sum(a["is_onset"] for a in rows) != 1
            or not rows[0]["is_onset"]
            or rows[0]["anchor_event_id"] != seq["onset_event_id"]
            or len(rows) != seq["anchor_count"]
            or len({a["pressed_team_id"] for a in rows}) != 1
            or len({a["period"] for a in rows}) != 1
            or any(a["outcome_tag"] != seq["sequence_output"] for a in rows)
            or any(
                a["value_model_eligible"] != (seq["sequence_output"] in VALUES)
                for a in rows
            )
            or not math.isclose(
                sum(a["training_weight"] for a in rows), 1, abs_tol=1e-12
            )
        ):
            raise ValueError("Sequence invariant failed")
        if seq["sequence_output"] == "censored" and not seq["censored_reason"]:
            raise ValueError("Censor reason missing")
