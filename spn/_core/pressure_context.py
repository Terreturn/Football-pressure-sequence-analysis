"""Find the preceding event context for a recorded presser."""

from __future__ import annotations

from spn._core.sequence_context import convention

from spn._core.positioning import fixed_context

from spn._core.pressure_events import Rules, pre_context, seconds

from spn._core.frame_events import available_events


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
        uid = ""
    anchor = {
        **history,
        "anchor_event_id": view["id"],
        "actor_x_norm": round(view["location"][0], 2),
        "actor_y": round(view["location"][1], 1),
    }
    return fixed_context(anchor, view, by_id, frames, convention(known)[0]), uid
