"""Draw the four-panel structural pressure network schematic from invented player positions."""

from __future__ import annotations

from .pitch_scene import ATTACKERS, PRESSERS, VISIBLE, INK, GRAY, LIGHT

PALETTES = {
    "teal_mulberry": {
        "name": "Teal & mulberry",
        "possession": "#147D80",
        "pressing": "#A4476D",
        "possession_name": "Teal",
        "pressing_name": "Mulberry",
    },
    "plum_olive": {
        "name": "Plum & olive",
        "possession": "#734C78",
        "pressing": "#87933E",
        "possession_name": "Plum",
        "pressing_name": "Olive",
    },
    "pine_rose": {
        "name": "Pine & rose",
        "possession": "#266B53",
        "pressing": "#BE6B8A",
        "possession_name": "Pine",
        "pressing_name": "Rose",
    },
    "pine_lightblue": {
        "name": "Pine & light blue",
        "possession": "#266B53",
        "pressing": "#5CAFE3",
        "pressure_edge": "#398FC4",
        "pressure_label": "#2B7EAE",
        "possession_name": "Pine",
        "pressing_name": "Light blue",
    },
    "crimson_lightblue": {
        "name": "Crimson & light blue",
        "possession": "#A52D4B",
        "pressing": "#5CAFE3",
        "pressure_edge": "#398FC4",
        "pressure_label": "#2B7EAE",
        "possession_name": "Crimson",
        "pressing_name": "Light blue",
    },
    "pine_gold": {
        "name": "Pine & gold",
        "possession": "#266B53",
        "pressing": "#D6AE45",
        "pressure_edge": "#B68A25",
        "pressure_label": "#8E6818",
        "possession_name": "Pine",
        "pressing_name": "Gold",
    },
    "navy_palevioletred": {
        "name": "Navy & pale violet red",
        "possession": "#243F63",
        "pressing": "#DB7093",
        "pressure_edge": "#BF557A",
        "pressure_label": "#A13D63",
        "possession_name": "Navy",
        "pressing_name": "Pale violet red",
    },
    "crimson_gold": {
        "name": "Crimson & gold",
        "possession": "#A52D4B",
        "pressing": "#D6AE45",
        "pressure_edge": "#B68A25",
        "pressure_label": "#8E6818",
        "possession_name": "Crimson",
        "pressing_name": "Gold",
    },
    "navy_goldenrod": {
        "name": "Navy & goldenrod",
        "possession": "#243F63",
        "pressing": "#DAA520",
        "pressure_edge": "#B78716",
        "pressure_label": "#8E6818",
        "possession_name": "Navy",
        "pressing_name": "Goldenrod",
    },
    "plum_goldenrod": {
        "name": "Plum & goldenrod",
        "possession": "#6D3A69",
        "pressing": "#DAA520",
        "pressure_edge": "#B78716",
        "pressure_label": "#8E6818",
        "possession_name": "Plum",
        "pressing_name": "Goldenrod",
    },
    "aubergine_orange": {
        "name": "Aubergine & orange",
        "possession": "#5B315E",
        "pressing": "#E58C49",
        "pressure_edge": "#C7712F",
        "pressure_label": "#A4561E",
        "possession_name": "Aubergine",
        "pressing_name": "Orange",
    },
    "petrol_coral": {
        "name": "Petrol & coral",
        "possession": "#165E66",
        "pressing": "#DF806E",
        "pressure_edge": "#BE6354",
        "pressure_label": "#9B4C40",
        "possession_name": "Petrol",
        "pressing_name": "Coral",
    },
    "burgundy_sage": {
        "name": "Burgundy & sage",
        "possession": "#792E48",
        "pressing": "#8CAC86",
        "pressure_edge": "#6B8F65",
        "pressure_label": "#4F734B",
        "possession_name": "Burgundy",
        "pressing_name": "Sage",
    },
    "slate_copper": {
        "name": "Slate & copper",
        "possession": "#3D5368",
        "pressing": "#C88A65",
        "pressure_edge": "#A76C4A",
        "pressure_label": "#89543A",
        "possession_name": "Slate",
        "pressing_name": "Copper",
    },
    "slate_warmcopper": {
        "name": "Slate & warm copper",
        "possession": "#354B5E",
        "pressing": "#CD8059",
        "pressure_edge": "#B0643F",
        "pressure_label": "#8E4F30",
        "possession_name": "Slate",
        "pressing_name": "Warm copper",
    },
    "blueslate_warmcopper": {
        "name": "Blue slate & warm copper",
        "possession": "#35546F",
        "pressing": "#CD8059",
        "pressure_edge": "#B0643F",
        "pressure_label": "#8E4F30",
        "possession_name": "Blue slate",
        "pressing_name": "Warm copper",
    },
    "slate_softcopper": {
        "name": "Slate & soft copper",
        "possession": "#4A6072",
        "pressing": "#CEA389",
        "pressure_edge": "#B58263",
        "pressure_label": "#8F6146",
        "possession_name": "Slate",
        "pressing_name": "Soft copper",
    },
    "navy_mutedgold": {
        "name": "Navy & muted gold",
        "possession": "#203A56",
        "pressing": "#C8AA5C",
        "pressure_edge": "#A68B40",
        "pressure_label": "#847030",
        "possession_name": "Navy",
        "pressing_name": "Muted gold",
    },
    "midnight_amber": {
        "name": "Midnight & amber",
        "possession": "#172F4C",
        "pressing": "#E5AC3D",
        "pressure_edge": "#C18A21",
        "pressure_label": "#946917",
        "possession_name": "Midnight",
        "pressing_name": "Amber",
    },
    "wine_powderblue": {
        "name": "Wine & powder blue",
        "possession": "#8B334F",
        "pressing": "#7FB7D4",
        "pressure_edge": "#539ABD",
        "pressure_label": "#3B7798",
        "possession_name": "Wine",
        "pressing_name": "Powder blue",
    },
    "carmine_skyblue": {
        "name": "Carmine & sky blue",
        "possession": "#B83E55",
        "pressing": "#62B5DB",
        "pressure_edge": "#3595BE",
        "pressure_label": "#25789C",
        "possession_name": "Carmine",
        "pressing_name": "Sky blue",
    },
}

PITCH = "#F5F6F5"

SHAPE = "#74777D"

FIGURE_VERSION = 12

PREVIEW_DPI = 170

PAPER_DPI = 425


def synthetic_network(variant=0):
    """Return network, shapes and checks for an invented scene indexed 0–3; positions are meters."""
    import numpy as np
    from .._core.networks import build_network, team_shapes

    # The production reader expects the provider grid. This inverse rescaling
    # makes the plotting scene's meter coordinates its exact intended input.
    # Keep the A–C scene unchanged. The other sheets illustrate different ball
    # locations and supporting formations, rather than jittering the same scene.
    scenes = [
        ATTACKERS + PRESSERS,
        [
            (36, 53),
            (9, 34),
            (16, 52),
            (14, 22),
            (25, 40),
            (44, 32),
            (35, 12),
            (38, 51),
            (29, 51),
            (28, 37),
            (43, 27),
            (24, 20),
        ],
        [
            (28, 30),
            (9, 34),
            (17, 48),
            (17, 17),
            (38, 47),
            (44, 30),
            (33, 12),
            (30, 31),
            (26, 36),
            (37, 28),
            (24, 19),
            (42, 40),
        ],
        [
            (29, 14),
            (9, 34),
            (13, 49),
            (15, 20),
            (35, 54),
            (45, 34),
            (43, 12),
            (31, 16),
            (37, 21),
            (26, 25),
            (43, 42),
            (22, 43),
        ],
    ]
    assert 0 <= variant < len(scenes)
    points = scenes[variant]
    frame = {
        "freeze_frame": [
            {
                "location": [x / 0.875, y / 0.85],
                "teammate": i < len(ATTACKERS),
                "actor": i == 0,
                "keeper": i == 1,
            }
            for i, (x, y) in enumerate(points)
        ],
        "visible_area": [z for x, y in VISIBLE for z in (x / 0.875, y / 0.85)],
    }
    event = {
        "location": frame["freeze_frame"][0]["location"],
        "team": {"id": 1},
        "possession_team": {"id": 1},
    }
    network = build_network(event, frame)
    shapes, failures = team_shapes(network)
    corrected, corrected_failures = team_shapes(network, corrected=True)
    assert not failures and not corrected_failures
    assert shapes == corrected, (
        "Preserve and report shape-mode differences before plotting."
    )
    assert np.allclose(network["xy"], points, rtol=0, atol=1e-12)
    assert not network["geometry_failures"]
    assert all(1 not in edge for edges in shapes.values() for edge in edges)
    assert all(
        network["nodes"][a]["role"] != "defender"
        and network["nodes"][b]["role"] != "defender"
        for a, b in shapes["pressed"]
    )
    assert all(
        a in network["defenders"] and b in network["defenders"]
        for a, b in shapes["pressing"]
    )
    cells_area = sum(cell.area for cell in network["areas"])
    assert abs(cells_area - network["visible"].area) < 1e-8
    checks = {
        "geometry_failures": failures,
        "legacy_equals_corrected_shapes": True,
        "coordinates_preserved": True,
        "goalkeeper_excluded_from_shapes": True,
        "shapes_separate_teams": True,
        "voronoi_area_conservation": True,
    }
    return network, shapes, checks


def render(output, palette_key, *, with_prediction=True, paper_ready=True):
    """Draw the pitch and networks in meters. Write a PNG and, in paper mode, a vector PDF."""
    output.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Arc, Polygon, Rectangle

    network, shapes, checks = synthetic_network()
    xy = network["xy"]
    palette = PALETTES[palette_key]
    possession, pressing = palette["possession"], palette["pressing"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "text.color": INK,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    base_width, base_height = (14, 6.6) if with_prediction else (11, 7.6)
    fig = plt.figure(figsize=(base_width, base_height), facecolor="white")
    panel_texts = {letter: [] for letter in "abcd"}

    def title(x, letter, words):
        """Place a panel heading at fractional x, using point-sized type; no return."""
        from .prediction_diagram import add_panel_heading

        panel_texts[letter].extend(
            add_panel_heading(fig, x * base_width, letter, words)
        )

    def pitch(rect, figure=None):
        """Return a half-pitch axis at fractional figure rectangle with coordinates in meters."""
        ax = (figure if figure is not None else fig).add_axes(rect)
        ax.set(xlim=(-3, 56), ylim=(-3, 71), aspect="equal")
        ax.axis("off")
        ax.add_patch(Rectangle((0, 0), 52.5, 68, fc=PITCH, ec=LIGHT, lw=0.8, zorder=0))
        ax.add_patch(
            Rectangle(
                (0, 13.84), 16.5, 40.32, fill=False, ec="#CAD1CC", lw=0.8, zorder=1
            )
        )
        ax.add_patch(
            Rectangle(
                (0, 24.84), 5.5, 18.32, fill=False, ec="#CAD1CC", lw=0.8, zorder=1
            )
        )
        ax.add_patch(
            Rectangle(
                (-1.7, 30.34), 1.7, 7.32, fill=False, ec="#A7B1AA", lw=0.8, zorder=1
            )
        )
        ax.add_patch(
            Arc(
                (52.5, 34),
                18.3,
                18.3,
                theta1=90,
                theta2=270,
                color="#CAD1CC",
                lw=0.8,
                zorder=1,
            )
        )
        return ax

    def players(ax, size):
        """Draw invented player roles on a meter-coordinate axis; marker size is square points."""
        ax.scatter(
            *zip(*ATTACKERS[2:]), s=size, color=possession, ec="white", lw=1, zorder=12
        )
        ax.scatter(
            *ATTACKERS[1],
            s=size * 1.1,
            marker="s",
            color=possession,
            ec="white",
            lw=1,
            zorder=12,
        )
        ax.scatter(
            *zip(*PRESSERS),
            s=size * 1.2,
            marker="^",
            color=pressing,
            ec="white",
            lw=1,
            zorder=12,
        )
        ax.scatter(
            *ATTACKERS[0],
            s=size * 2.6,
            marker="*",
            color=possession,
            ec="white",
            lw=1,
            zorder=13,
        )

    def boundary(ax, fill=False):
        """Draw the invented visible-area polygon in meter coordinates; fill is a display flag."""
        ax.add_patch(
            Polygon(
                VISIBLE,
                closed=True,
                fc="#E4ECE7" if fill else "none",
                ec="#9AA89E",
                ls=(0, (4, 3)),
                lw=0.9,
                alpha=0.7,
                zorder=2 if fill else 6,
            )
        )

    def draw_layer(ax, name, composite=False):
        """Draw computed network layer on metric coordinates; qualitative style scales have no data units."""
        if name == "space":
            for i, cell in enumerate(network["areas"]):
                color = possession if i < len(ATTACKERS) else pressing
                parts = list(cell.geoms) if hasattr(cell, "geoms") else [cell]
                for part in parts:
                    # Apply transparency to the fill only: cell boundaries must
                    # stay opaque white to reveal the Voronoi tessellation.
                    ax.add_patch(
                        Polygon(
                            list(part.exterior.coords),
                            fc=to_rgba(color, 0.18 if composite else 0.20),
                            ec="white",
                            lw=1.25 if composite else 0.9,
                            zorder=2,
                        )
                    )
        elif name == "shape":
            for side in ["pressed", "pressing"]:
                for a, b in sorted(shapes[side]):
                    ax.plot(
                        xy[[a, b], 0],
                        xy[[a, b], 1],
                        color=SHAPE,
                        lw=1.45 if composite else 1.65,
                        alpha=0.65 if composite else 0.9,
                        zorder=4,
                    )
        elif name == "pressure":
            for edge in network["pressure_edges"]:
                a, b = edge["src"], edge["dst"]
                ax.plot(
                    xy[[a, b], 0],
                    xy[[a, b], 1],
                    color=palette.get("pressure_edge", pressing),
                    lw=(0.95 + 3.4 * edge["p"]) * (1.20 if composite else 1),
                    alpha=0.94,
                    solid_capstyle="round",
                    zorder=8,
                )
        elif name == "passing":
            for edge in network["passes"]:
                a, b = network["carrier"], edge["dst"]
                opened = edge["w"] > network["parameters"].open_threshold
                ax.plot(
                    xy[[a, b], 0],
                    xy[[a, b], 1],
                    color=possession,
                    lw=(0.9 + 3.7 * edge["w"]) * (1.20 if composite else 1),
                    ls="-" if opened else (0, (3, 2)),
                    alpha=0.85,
                    solid_capstyle="round",
                    zorder=7,
                )

    if with_prediction:
        title(0.020, "a", "High-press situation")
        title(0.320, "b", "Network layers")
        c_left = 0.625 - 0.75 / base_width
        title(c_left + 0.15 / base_width, "c", "Structural Pressure Network")
        situation = pitch([0.020, 0.130, 0.290, 0.770])
        boundary(situation, fill=True)
        players(situation, 140)
        layer_columns, layer_width = (0.320, 0.460), 0.128
        # A wider slot for (c) centers its unchanged pitch below the one-line
        # heading, using the space recovered by tightening the layer grid.
        combined_rect = [c_left, 0.130, 0.350, 0.770]
    else:
        title(0.020, "a", "Network layers")
        title(0.530, "b", "Structural Pressure\nNetwork")
        layer_columns, layer_width = (0.020, 0.265), 0.185
        combined_rect = [0.530, 0.130, 0.440, 0.770]

    # Layer ordering follows the explanation, while the composite orders its
    # drawing from faint space/shape to foreground passing/pressure and players.
    # Align the field boundaries, accounting for each axis's three-meter margin.
    top_row = 0.130 + (0.770 - 0.330) * 71 / 74
    bottom_row = 0.130 + (0.770 - 0.330) * 3 / 74
    if with_prediction:
        top_row, bottom_row = 0.535, 0.165
    views = [
        (layer_columns[0], top_row, "pressure", "Pressure"),
        (layer_columns[1], top_row, "passing", "Passing"),
        (layer_columns[0], bottom_row, "space", "Player space"),
        (layer_columns[1], bottom_row, "shape", "Team shape"),
    ]
    layer_axes = []
    for x, y, layer, label in views:
        label_color = {
            "pressure": palette.get("pressure_label", pressing),
            "passing": possession,
            "space": INK,
            "shape": SHAPE,
        }[layer]
        layer_title = fig.text(
            x + layer_width / 2,
            y + (0.330 if with_prediction else 0.350),
            label,
            fontsize=15,
            weight="normal",
            ha="center",
            color=label_color,
        )
        panel_texts["b" if with_prediction else "a"].append(layer_title)
        ax = pitch([x, y, layer_width, 0.330])
        layer_axes.append(ax)
        draw_layer(ax, layer)
        if layer == "space":
            boundary(ax)
        players(ax, 60)
    c = pitch(combined_rect)
    for layer in ["space", "shape", "passing", "pressure"]:
        draw_layer(c, layer, composite=True)
    boundary(c)
    players(c, 140)
    annotated = situation if with_prediction else c
    annotated.annotate(
        "Ball carrier",
        xy=ATTACKERS[0],
        xytext=(3, 63),
        fontsize=13,
        arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1, "shrinkB": 10},
        zorder=14,
    )
    annotated.annotate(
        "Closest\npresser",
        xy=PRESSERS[0],
        xytext=(37, 43),
        fontsize=13,
        arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1, "shrinkB": 8},
        zorder=14,
    )
    annotated.text(
        6, 10, "StatsBomb 360\nobserved area", fontsize=12, color=GRAY, linespacing=1.1
    )
    attack_label = fig.text(
        0.165 if with_prediction else 0.750,
        0.126,
        "Attack →",
        fontsize=16,
        ha="center",
        color=GRAY,
    )
    panel_texts["a" if with_prediction else "b"].append(attack_label)

    role_legend = [
        Line2D(
            [],
            [],
            color="none",
            marker="*",
            markerfacecolor=possession,
            markeredgecolor="white",
            markersize=18,
            label="Carrier",
        ),
        Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor=possession,
            markeredgecolor="white",
            markersize=12,
            label="Team in possession",
        ),
        Line2D(
            [],
            [],
            color="none",
            marker="^",
            markerfacecolor=pressing,
            markeredgecolor="white",
            markersize=12,
            label="Pressing team",
        ),
        Line2D(
            [],
            [],
            color="none",
            marker="s",
            markerfacecolor=possession,
            markeredgecolor="white",
            markersize=12,
            label="Goalkeeper",
        ),
    ]
    legend_anchor = (0.55, 0.055)
    legend = fig.legend(
        handles=role_legend,
        loc="lower center",
        bbox_to_anchor=legend_anchor,
        ncol=4,
        frameon=False,
        fontsize=15.5,
        handletextpad=0.15,
        handlelength=1.3,
        columnspacing=1.3,
    )
    extension_metadata = None
    if with_prediction:
        import numpy as np
        from .prediction_diagram import add_prediction_panel

        # Keep pitch sizes and the prediction column's physical width. Recover
        # whitespace by shifting C left and starting D earlier on a shorter canvas.
        fig.canvas.draw()
        axes_before = [(ax, ax.get_position().frozen()) for ax in fig.axes]
        prediction_left = base_width - 1.40
        prediction_width = base_width * 0.20
        total_width = prediction_left + prediction_width
        ratio = base_width / total_width
        fig.set_size_inches(total_width, base_height)
        for ax, position in axes_before:
            ax.set_position(
                [
                    position.x0 * ratio,
                    position.y0,
                    position.width * ratio,
                    position.height,
                ]
            )
        for label in fig.texts:
            x, y = label.get_position()
            label.set_position((x * ratio, y))
        legend.set_bbox_to_anchor((legend_anchor[0] * ratio, legend_anchor[1]))
        fig.canvas.draw()
        for ax, before in axes_before:
            after = ax.get_position()
            assert np.allclose(
                [before.width * base_width, before.height * base_height],
                [after.width * total_width, after.height * base_height],
                rtol=0,
                atol=1e-10,
            )
        instances = [synthetic_network(variant=i) for i in range(4)]
        d_text_start = len(fig.texts)
        extension_metadata = add_prediction_panel(
            fig, instances, palette, prediction_left, prediction_width, base_height
        )
        panel_texts["d"].extend(fig.texts[d_text_start:])
        panel_axes = {"a": [situation], "b": layer_axes, "c": [c], "d": [fig.axes[-1]]}
        extension_metadata["existing_pitch_sizes_preserved"] = True
        extension_metadata["pitch_sizes_inches"] = [
            [p.width * base_width, p.height * base_height] for _, p in axes_before
        ]
        # Remove the unused bottom strip without scaling any panels or type.
        bottom_trim = 0.37
        final_height = base_height - bottom_trim
        fig.canvas.draw()
        positions = [(ax, ax.get_position().frozen()) for ax in fig.axes]
        fig.set_size_inches(total_width, final_height)
        for ax, position in positions:
            ax.set_position(
                [
                    position.x0,
                    (position.y0 * base_height - bottom_trim) / final_height,
                    position.width,
                    position.height * base_height / final_height,
                ]
            )
        for label in fig.texts:
            x, y = label.get_position()
            label.set_position((x, (y * base_height - bottom_trim) / final_height))
        legend.set_bbox_to_anchor(
            (
                legend_anchor[0] * ratio,
                (legend_anchor[1] * base_height - bottom_trim) / final_height,
            )
        )
        fig.canvas.draw()
        for ax, before in positions:
            after = ax.get_position()
            assert np.allclose(
                [before.width * total_width, before.height * base_height],
                [after.width * total_width, after.height * final_height],
                rtol=0,
                atol=1e-10,
            )
        extension_metadata["bottom_trim_inches"] = bottom_trim

        # Spend the extra 5% width entirely on equal additional gaps between
        # panels. Translate whole panels, including their headings, without
        # resizing the pitches, nodes, text or the grid inside panel B.
        expanded_width = total_width * 1.05
        extra_gap = (expanded_width - total_width) / 3
        positions = {ax: ax.get_position().frozen() for ax in fig.axes}
        fig.set_size_inches(expanded_width, final_height)
        for index, letter in enumerate("abcd"):
            shift = index * extra_gap
            for ax in panel_axes[letter]:
                position = positions[ax]
                ax.set_position(
                    [
                        (position.x0 * total_width + shift) / expanded_width,
                        position.y0,
                        position.width * total_width / expanded_width,
                        position.height,
                    ]
                )
            for label in panel_texts[letter]:
                x, y = label.get_position()
                label.set_position(((x * total_width + shift) / expanded_width, y))
        legend_drop = 0.12
        legend.set_bbox_to_anchor(
            (
                (legend_anchor[0] * base_width + extra_gap) / expanded_width,
                (legend_anchor[1] * base_height - bottom_trim - legend_drop)
                / final_height,
            )
        )
        fig.canvas.draw()
        for ax, before in positions.items():
            after = ax.get_position()
            assert np.allclose(
                [before.width * total_width, before.height * final_height],
                [after.width * expanded_width, after.height * final_height],
                rtol=0,
                atol=1e-10,
            )
        extension_metadata["panel_left_inches"] += 3 * extra_gap
        extension_metadata["width_increase_fraction"] = 0.05
        extension_metadata["extra_gap_inches"] = extra_gap
        extension_metadata["panel_offsets_inches"] = {
            letter: i * extra_gap for i, letter in enumerate("abcd")
        }
        extension_metadata["legend_downward_shift_inches"] = legend_drop
    version = FIGURE_VERSION if with_prediction else 5
    stem = f"figure1_mockup_v{version}_{palette_key}"
    png_dpi = PAPER_DPI if paper_ready else PREVIEW_DPI
    fig.savefig(output / f"{stem}.png", dpi=png_dpi, facecolor="white")
    if paper_ready:
        fig.savefig(output / f"{stem}.pdf", format="pdf", facecolor="white")
    plt.close(fig)
