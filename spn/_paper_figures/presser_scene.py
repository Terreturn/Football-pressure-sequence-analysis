"""Draw alternative positions around one observed pressing player."""

from __future__ import annotations

import math

POSSESSION = "#35546F"

PRESSING = "#CD8059"

INK = "#242A30"

GRAY = "#747B82"

LIGHT = "#D5DADF"

PLAYER_MARKERS = {
    "carrier": "*",
    "possession": "o",
    "pressing": "^",
    "alternative": "^",
}

ATTACKERS = ((24.0, 34.0), (18.0, 43.0), (34.0, 44.0), (38.0, 29.0), (19.0, 25.0))

PRESSERS = ((27.4, 33.0), (29.5, 41.0), (36.0, 36.0), (28.0, 26.0))

VISIBLE = (
    (14.0, 25.0),
    (16.0, 44.0),
    (23.0, 48.0),
    (42.0, 46.0),
    (43.0, 25.0),
    (26.0, 21.0),
)

RADII = (1.25, 2.5)


def candidate_positions(origin, carrier, radii=RADII):
    """Return (x, y) alternatives in meters, eight per supplied radius, toward-carrier first."""
    dx, dy = carrier[0] - origin[0], carrier[1] - origin[1]
    if math.hypot(dx, dy) == 0:
        raise ValueError("The moved player must differ from the carrier.")
    direction = math.atan2(dy, dx)
    return [
        (
            origin[0] + radius * math.cos(direction + k * math.pi / 4),
            origin[1] + radius * math.sin(direction + k * math.pi / 4),
        )
        for radius in radii
        for k in range(8)
    ]


def draw_scene(ax, index, title, carrier_label=False, radii=RADII, alternatives=None):
    """Draw invented meter positions; optional alternatives replace the ring stencil while radii mark its boundaries."""
    from matplotlib.patches import Circle, Polygon, Rectangle
    import matplotlib.patheffects as path_effects

    ax.set(xlim=(13, 45), ylim=(22, 47), aspect="equal", xticks=[], yticks=[])
    ax.axis("off")
    ax.add_patch(
        Rectangle(
            (13, 22), 32, 25, facecolor="#F5F6F5", edgecolor=LIGHT, lw=1, zorder=0
        )
    )
    # This is a local pitch view. The penalty-area line and colors match Figure 1.
    ax.plot([16.5, 16.5], [22, 47], color="#CAD1CC", lw=1, zorder=1)
    ax.add_patch(
        Polygon(
            VISIBLE,
            facecolor="#E4ECE7",
            edgecolor="#9AA89E",
            lw=1.1,
            linestyle=(0, (4, 3)),
            alpha=0.7,
            zorder=2,
        )
    )
    ax.set_title(title, fontsize=16, pad=8, color=INK)
    # All positions are identical across views. Only the candidate stencil changes.
    ax.scatter(
        *zip(*ATTACKERS[1:]),
        s=125,
        marker=PLAYER_MARKERS["possession"],
        facecolor=POSSESSION,
        edgecolor="white",
        lw=1.2,
        zorder=5,
    )
    ax.scatter(
        *zip(*PRESSERS),
        s=160,
        marker=PLAYER_MARKERS["pressing"],
        facecolor=PRESSING,
        edgecolor="white",
        lw=1.2,
        zorder=5,
    )
    ax.scatter(
        *ATTACKERS[0],
        s=360,
        marker=PLAYER_MARKERS["carrier"],
        facecolor=POSSESSION,
        edgecolor="white",
        lw=1.2,
        zorder=6,
    )
    origin = PRESSERS[index]
    for radius in radii:
        ax.add_patch(
            Circle(
                origin,
                radius,
                fill=False,
                edgecolor="#C7CDD1",
                lw=0.9,
                linestyle=(0, (2, 3)),
                zorder=2,
            )
        )
    candidates = (
        candidate_positions(origin, ATTACKERS[0], radii)
        if alternatives is None
        else alternatives
    )
    ax.scatter(
        *zip(*candidates),
        s=38 if alternatives is None else 13,
        marker=PLAYER_MARKERS["alternative"],
        facecolor="white",
        edgecolor=GRAY,
        lw=1.05 if alternatives is None else 0.6,
        zorder=4,
    )
    # The highlighted arrow shows one alternative; it encodes no model response.
    # Examples are chosen geometrically, in different directions; neither is a recommended move.
    endpoint = (
        candidates[8 * (len(radii) - 1) + (1, 3)[index]]
        if alternatives is None
        else candidates[(1, 3)[index]]
    )
    ax.annotate(
        "",
        xy=endpoint,
        xytext=origin,
        arrowprops={
            "arrowstyle": "-|>",
            "color": INK,
            "lw": 1.5,
            "mutation_scale": 12,
            "shrinkA": 7,
            "shrinkB": 3,
        },
        zorder=7,
    )
    ax.scatter(
        *origin,
        s=175,
        marker=PLAYER_MARKERS["pressing"],
        facecolor=PRESSING,
        edgecolor=INK,
        lw=1.25,
        zorder=6,
    )
    ax.scatter(
        *endpoint,
        s=38,
        marker=PLAYER_MARKERS["alternative"],
        facecolor="white",
        edgecolor=INK,
        lw=1.4,
        zorder=8,
    )
    if carrier_label:
        ax.annotate(
            "Ball carrier",
            xy=ATTACKERS[0],
            xytext=(14.8, 36.2),
            fontsize=13.5,
            color=INK,
            ha="left",
            va="center",
            arrowprops={
                "arrowstyle": "-|>",
                "color": GRAY,
                "lw": 1,
                "shrinkA": 3,
                "shrinkB": 10,
                "mutation_scale": 10,
            },
        )
    # A literal distance scale reinforces that the local view is measured in meters.
    scale_halo = [
        path_effects.Stroke(linewidth=4, foreground="white"),
        path_effects.Normal(),
    ]
    ax.plot([37, 42], [23, 23], color=GRAY, lw=1.2, path_effects=scale_halo, zorder=9)
    ax.plot([37, 37], [22.7, 23.3], color=GRAY, lw=1, path_effects=scale_halo, zorder=9)
    ax.plot([42, 42], [22.7, 23.3], color=GRAY, lw=1, path_effects=scale_halo, zorder=9)
    ax.text(
        39.5,
        23.7,
        "5 m",
        ha="center",
        fontsize=13,
        color=GRAY,
        zorder=9,
        path_effects=[
            path_effects.Stroke(linewidth=3, foreground="white"),
            path_effects.Normal(),
        ],
    )
