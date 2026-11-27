"""Find eligible events carrying a visible frame."""

from __future__ import annotations

from spn._core.pressure_events import seconds


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
