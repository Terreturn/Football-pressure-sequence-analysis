"""Move one identified player to a feasible location and preserve the other players in the frame."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from spn._core.positioning import moved_frame, physical_players


def move_named(frame, index, target):
    """Return a possession-oriented frame with only one raw row moved to target meters; zero movement is exact identity."""
    indices, xy = physical_players(frame, False)
    origin = xy[indices.index(index)]
    if np.array_equal(origin, np.asarray(target)):
        return deepcopy(frame), 0.0
    return moved_frame(frame, index, target, False)
