"""Read visible players in frame order and rescale their coordinates to meters."""

from __future__ import annotations

import numpy as np


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
