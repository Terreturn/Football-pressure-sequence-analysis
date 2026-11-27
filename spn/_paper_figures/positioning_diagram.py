"""Arrange the pitch, the sampled positions and the model comparison in Figure 4."""

from __future__ import annotations

import numpy as np

from shapely.geometry import Polygon

from .presser_scene import (
    ATTACKERS,
    PRESSERS,
    VISIBLE,
    INK,
    GRAY,
    POSSESSION,
    PRESSING,
    draw_scene,
)

from .._core.position_sampler import candidates

from .._core.positioning import move_reason


def sample_geometry():
    """Return one invented frame's 64 absolute and relative meter positions plus a validity/provenance record."""
    origin = np.asarray(PRESSERS[0])
    rows = candidates("invented-figure4", "invented-presser-0", origin, ATTACKERS[0])
    xy = rows[["target_x", "target_y"]].to_numpy()
    reasons = [
        move_reason(
            p, Polygon(VISIBLE), np.asarray(ATTACKERS + PRESSERS), len(ATTACKERS)
        )
        for p in xy
    ]
    assert len(xy) == 64 and set(reasons) == {"valid"}
    relative = xy - origin
    return (
        xy,
        relative,
        {
            "synthetic_only": True,
            "model_scored": False,
            "moved_player": "named Pressure-event actor",
            "origin_m": origin.tolist(),
            "sample_positions_m": xy.tolist(),
            "inset_offsets_m": relative.tolist(),
            "proposals": 64,
            "valid": 64,
            "seed": int(rows.seed.iloc[0]),
            "radius_bounds_m": [0.5, 2.0],
            "other_players_fixed": True,
        },
    )


def heading(fig, x, y, letter, title):
    """Draw a compact panel heading at inch coordinates; return no numerical result."""
    w, h = fig.get_size_inches()
    fig.text(x / w, y / h, letter, weight="bold", fontsize=16, va="center")
    fig.text((x + 0.48) / w, y / h, title, fontsize=13.5, va="center")


def draw_schematic(
    fig,
    xy,
    relative,
    lower_inset=False,
    tight=False,
    editorial=False,
    refined=False,
    polished=False,
):
    """Draw one meter-coordinate frame and a connected enlargement of exactly its 64 alternative positions."""
    from matplotlib.patches import (
        Circle,
        Rectangle,
        ConnectionPatch,
        FancyBboxPatch,
        FancyArrowPatch,
    )
    from matplotlib.path import Path as PlotPath
    from matplotlib.lines import Line2D

    common_arrow = {
        "arrowstyle": "-|>",
        "mutation_scale": 6,
        "lw": 0.8,
        "color": GRAY,
        "capstyle": "butt",
        "joinstyle": "miter",
    }
    matching_arrows = []
    w, h = fig.get_size_inches()
    heading(fig, 0.35, 8.05, "(a)", "Counterfactual positioning experiment")
    scene = fig.add_axes([0.38 / w, 4.93 / h, 3.85 / w, 2.97 / h])
    draw_scene(
        scene, 0, "", carrier_label=not editorial, radii=(0.5, 2.0), alternatives=xy
    )
    # Remove the old arbitrary example-move arrow and its extra endpoint marker.
    from matplotlib.text import Annotation

    for artist in list(scene.texts):
        if isinstance(artist, Annotation) and not artist.get_text():
            artist.remove()
    scene.collections[-1].remove()
    if refined:
        for text in scene.texts:
            if text.get_text() == "5 m":
                text.set_text("5m")
    if polished:
        scale_lines = [line for line in scene.lines if line.get_path_effects()]
        assert len(scale_lines) == 3
        for line in scale_lines:
            line.set_path_effects([])
            line.set_linewidth(1.0)
            line.set_solid_capstyle("butt")
        assert next(t for t in scene.texts if t.get_text() == "5m").get_path_effects()
    if editorial:
        carrier_props = (
            dict(common_arrow, shrinkA=4, shrinkB=10)
            if polished
            else {
                "arrowstyle": "-|>",
                "color": GRAY,
                "lw": 0.8,
                "shrinkA": 4,
                "shrinkB": 10,
                "mutation_scale": 6,
            }
        )
        presser_props = (
            carrier_props.copy()
            if polished
            else {
                "arrowstyle": "-",
                "color": GRAY,
                "lw": 0.8,
                "shrinkA": 3,
                "shrinkB": 2,
            }
        )
        carrier = scene.annotate(
            "Ball carrier",
            xy=ATTACKERS[0],
            xytext=(14.8, 38.0),
            fontsize=12,
            color=INK,
            ha="left",
            va="center",
            arrowprops=carrier_props,
        )
        presser = scene.annotate(
            "Presser",
            xy=PRESSERS[0] if polished else (PRESSERS[0][0], 31.8),
            xytext=(PRESSERS[0][0], 28.7),
            fontsize=12 if refined else 11,
            ha="center",
            va="center",
            color=INK,
            arrowprops=presser_props,
        )
        if polished:
            matching_arrows.extend([carrier.arrow_patch, presser.arrow_patch])
    else:
        scene.text(27.4, 29.6, "Named presser", fontsize=11, ha="center", color=INK)
    origin = np.asarray(PRESSERS[0])
    box = Rectangle(origin - 2.35, 4.7, 4.7, fill=False, ec=GRAY, lw=0.9)
    scene.add_patch(box)
    zoom_box = (4.20, 4.84, 2.10, 2.10) if lower_inset else (4.75, 5.04, 2.72, 2.72)
    if refined:
        zoom_box = (4.55, 4.84, 2.10, 2.10)
    zoom = fig.add_axes(
        [zoom_box[0] / w, zoom_box[1] / h, zoom_box[2] / w, zoom_box[3] / h]
    )
    zoom.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), aspect="equal", xticks=[], yticks=[])
    zoom.axis("off")
    zoom.add_patch(Circle((0, 0), 2.0, fc="#F5F6F5", ec=GRAY, lw=0.8))
    zoom.add_patch(Circle((0, 0), 0.5, fc="white", ec=GRAY, lw=0.8, ls="--"))
    points = zoom.scatter(
        relative[:, 0],
        relative[:, 1],
        marker="^",
        s=18 if lower_inset else 29,
        fc="white",
        ec=GRAY,
        lw=0.7 if lower_inset else 0.8,
        zorder=3,
    )
    points.set_gid("sobol-inset-64")
    zoom.scatter(
        0,
        0,
        marker="^",
        s=75 if lower_inset else 105,
        fc=PRESSING,
        ec=INK,
        lw=1.0,
        zorder=4,
    )
    zoom.text(
        0,
        2.34,
        "Sample alternative\npressing locations"
        if editorial
        else "64 Sobol alternatives",
        ha="center",
        fontsize=11 if lower_inset else 12,
        linespacing=1.15,
    )
    zoom.annotate(
        "0.5m" if refined else "0.5 m",
        xy=(0.36, -0.36),
        xytext=(0.78, -0.55),
        fontsize=9 if lower_inset else 10,
        ha="left",
        arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.7},
    )
    if editorial:
        zoom.annotate(
            "2m" if refined else "2 m",
            xy=(1.4142, -1.4142),
            xytext=(1.65, -2.30),
            fontsize=9,
            ha="center",
            arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.7},
        )
    else:
        zoom.text(
            0,
            -2.36,
            "2 m outer radius",
            ha="center",
            fontsize=10 if lower_inset else 11,
            color=GRAY,
        )
    for dy in (-2.35, 2.35):
        fig.add_artist(
            ConnectionPatch(
                origin + [2.35, dy],
                (-2.15, np.sign(dy) * 1.25),
                coordsA=scene.transData,
                coordsB=zoom.transData,
                color="#A8B0B6",
                lw=0.8,
                zorder=2,
                linestyle=(0, (3, 3)),
            )
        )
    flow_x = 7.0 if editorial else (6.55 if tight else 8.40)
    flow_y, flow_width, flow_height = (
        (4.85, 3.27, 2.95) if editorial else (5.08, 3.72, 2.6)
    )
    if refined:
        flow_y += 0.15
    flow = fig.add_axes([flow_x / w, flow_y / h, flow_width / w, flow_height / h])
    flow.set(xlim=(0, 1), ylim=(0, 1))
    flow.axis("off")
    flow.text(
        0.5, 1.0, "For each valid alternative", ha="center", va="top", fontsize=12
    )
    box_height = 0.17 if editorial else 0.20
    bottoms = (0.66, 0.34, 0.02) if editorial else (0.64, 0.34, 0.04)
    for y, label in zip(
        bottoms,
        ("Rebuild network", "Apply held-out model", "Compare regain probabilities"),
    ):
        model_box = FancyBboxPatch(
            (0.02, y),
            0.96,
            box_height,
            boxstyle="round,pad=.015,rounding_size=.02",
            fc="#F3F5F4",
            ec="#7F8D91",
            lw=0.9,
        )
        model_box.set_gid("model-step-box")
        flow.add_patch(model_box)
        flow.text(0.5, y + box_height / 2, label, ha="center", va="center", fontsize=12)
    for upper, lower in zip(bottoms, bottoms[1:]):
        connector = flow.annotate(
            "",
            xy=(0.5, lower + box_height + 0.018),
            xytext=(0.5, upper - 0.018),
            arrowprops=common_arrow.copy()
            if polished
            else {"arrowstyle": "-|>", "color": GRAY},
        )
        if polished:
            matching_arrows.append(connector.arrow_patch)
    if lower_inset:
        # Leave the upper-right pitch edge, clear the inset, and enter the first model block.
        turn_x = flow_x - 0.15 if tight else flow_x - 0.42
        if polished:
            turn_x = flow_x - 0.50
        entry_y = flow_y + flow_height * (bottoms[0] + box_height / 2)
        route = np.array(
            [(4.23, 7.42), (turn_x, 7.42), (turn_x, entry_y), (flow_x, entry_y)]
        )
        path = PlotPath(
            route / np.array([w, h]),
            [PlotPath.MOVETO, PlotPath.LINETO, PlotPath.LINETO, PlotPath.LINETO],
        )
        incoming_props = (
            common_arrow.copy()
            if polished
            else {"arrowstyle": "-|>", "mutation_scale": 12, "color": GRAY, "lw": 1.1}
        )
        arrow = FancyArrowPatch(
            path=path, transform=fig.transFigure, zorder=3, **incoming_props
        )
        if polished:
            matching_arrows.append(arrow)
        arrow.set_gid("pitch-to-rebuild-above-inset")
        fig.add_artist(arrow)
        assert route[0, 1] > zoom_box[1] + zoom_box[3] + 0.2
        assert route[2, 0] > zoom_box[0] + zoom_box[2] or (
            polished and route[2, 1] > zoom_box[1] + zoom_box[3] + 0.2
        )
    else:
        fig.add_artist(
            ConnectionPatch(
                (2.5, 0),
                (-0.04, 0.54),
                coordsA=zoom.transData,
                coordsB=flow.transAxes,
                arrowstyle="-|>",
                color=GRAY,
                lw=1.0,
            )
        )
    if not editorial:
        fig.text(
            0.5,
            4.72 / h,
            "Move the named presser only; keep every other player and prior context fixed.",
            ha="center",
            fontsize=11,
            color=GRAY,
        )
    handles = [
        Line2D(
            [],
            [],
            ls="",
            marker="*",
            ms=12,
            mfc=POSSESSION,
            mec="white",
            label="Carrier",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=8,
            mfc=POSSESSION,
            mec="white",
            label="Team in possession",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="^",
            ms=9,
            mfc=PRESSING,
            mec="white",
            label="Pressing team",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="^",
            ms=7,
            mfc="white",
            mec=GRAY,
            label="Alternative position",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.5, (4.46 if editorial else 4.37) / h),
        ncol=4,
        frameon=False,
        fontsize=12 if editorial else 11,
        handleheight=1.8 if editorial else 0.7,
        handletextpad=0.3,
        columnspacing=1.2,
    )
    np.testing.assert_allclose(points.get_offsets(), xy - origin, rtol=0, atol=0)
    if refined:
        assert {
            t.get_text(): t.get_fontsize()
            for t in scene.texts
            if t.get_text() in ("Ball carrier", "Presser")
        } == {"Ball carrier": 12.0, "Presser": 12.0}
        assert "5m" in {t.get_text() for t in scene.texts}
        assert {"0.5m", "2m"} <= {t.get_text() for t in zoom.texts}
    if polished:
        from matplotlib.colors import to_rgba

        assert len(matching_arrows) == 5
        for arrow in matching_arrows:
            assert arrow.get_linewidth() == 0.8 and arrow.get_mutation_scale() == 6
            assert arrow.get_edgecolor() == to_rgba(
                GRAY
            ) and arrow.get_facecolor() == to_rgba(GRAY)
            assert type(arrow.get_arrowstyle()) is type(
                matching_arrows[0].get_arrowstyle()
            )
        assert all(
            not line.get_path_effects() and line.get_linewidth() == 1.0
            for line in scale_lines
        )
