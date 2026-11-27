"""Use familiar names when labeling the Premier League players in Figure 4."""

from __future__ import annotations

from .club_labels import NAMES

DISPLAY_NAMES = dict(
    NAMES,
    **{
        "Dango Aboubacar Faissal Ouattara": "Dango Ouattara",
        "João Victor Gomes da Silva": "João Gomes",
        "João Pedro Junqueira de Jesus": "João Pedro",
        "Igor Jesus Maciel da Cruz": "Igor Jesus",
        "Antoine Semenyo": "Semenyo",
    },
)
