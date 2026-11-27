"""Read the within-period time of a StatsBomb event, in seconds."""

from __future__ import annotations

import math


def stamp(event):
    """Return period-relative seconds parsed independently from a recorded event timestamp."""
    parts = event.get("timestamp", "").split(":")
    return (
        float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        if len(parts) == 3
        else math.nan
    )
