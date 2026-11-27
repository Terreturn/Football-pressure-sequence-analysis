"""Draw the stack of networks and the three predicted sequence outcomes."""

from __future__ import annotations


def add_panel_heading(fig, x_inches, letter, words, gap_inches=0.14):
    """Return label/title artists with a measured horizontal gap; positions and gap are inches."""
    width = fig.get_figwidth()
    label = fig.text(
        x_inches / width,
        0.979,
        f"({letter.lower()})",
        fontsize=26,
        weight="bold",
        va="top",
        color="#202830",
    )
    # Letters have different glyph widths. Measure the label's right edge so
    # every heading has the same visible gap, rather than the same anchor offset.
    right = label.get_window_extent(fig.canvas.get_renderer()).x1 / fig.dpi
    heading = fig.text(
        (right + gap_inches) / width,
        0.972,
        words,
        fontsize=18,
        weight="normal",
        va="top",
        color="#202830",
    )
    return label, heading


def add_prediction_panel(fig, instances, palette, left, width, base_height):
    """Draw the schematic and return metadata; layout inputs are inches, network positions are meters."""
    import numpy as np
    from matplotlib.colors import to_rgba
    from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

    ink, gray, shape_color = "#202830", "#6B737B", "#74777D"
    possession, pressing = palette["possession"], palette["pressing"]
    pressure_color = palette.get("pressure_edge", pressing)
    total_width = fig.get_figwidth()
    add_panel_heading(fig, left + 0.12, "d", "Prediction")
    height = base_height * 0.77
    usable_width = width - 0.28
    ax = fig.add_axes(
        [(left + 0.14) / total_width, 0.130, usable_width / total_width, 0.770]
    )
    ax.set(xlim=(0, usable_width), ylim=(0, height), aspect="equal")
    ax.axis("off")
    center = usable_width / 2
    ax.text(
        center, 4.97, "SPN instances", ha="center", va="top", fontsize=13, color=ink
    )

    assert len(instances) == 4
    assert len({tuple(network["xy"].ravel()) for network, _, _ in instances}) == 4
    bounds = np.array([[0, 0], [52.5, 0], [52.5, 68], [0, 68]])

    # Affine projection tilts each entire network into one sheet. No vertical
    # connections are drawn: the examples are independently represented. Paint
    # lower sheets first so the top sheet occludes them, giving a view from above.
    for index, (network, shapes, checks) in enumerate(instances):
        xy = network["xy"]
        origin = np.array([0.05 * usable_width, 3.55 + 0.23 * index])
        along_x = np.array([0.68 * usable_width, -0.20])
        along_y = np.array([0.22 * usable_width, 0.48])

        def project(points):
            """Map N-by-2 meter positions to N-by-2 schematic inch coordinates for the current sheet."""
            points = np.asarray(points)
            return (
                origin + points[:, :1] / 52.5 * along_x + points[:, 1:] / 68 * along_y
            )

        z = 10 + index * 10
        corners = project(bounds)
        ax.add_patch(Polygon(corners, fc="#FAFBFB", ec="#AAB3B4", lw=1, zorder=z))
        for i, cell in enumerate(network["areas"]):
            color = (
                possession if network["nodes"][i]["role"] != "defender" else pressing
            )
            parts = list(cell.geoms) if hasattr(cell, "geoms") else [cell]
            for part in parts:
                # Repeat C's cell colors and opaque white boundaries at a
                # lighter fill and smaller stroke suited to these small sheets.
                ax.add_patch(
                    Polygon(
                        project(part.exterior.coords),
                        fc=to_rgba(color, 0.17),
                        ec="white",
                        lw=0.55,
                        zorder=z + 1,
                    )
                )
        for side in ["pressed", "pressing"]:
            for a, b in sorted(shapes[side]):
                line = project(xy[[a, b]])
                ax.plot(
                    line[:, 0],
                    line[:, 1],
                    color=shape_color,
                    lw=0.55,
                    alpha=0.65,
                    zorder=z + 2,
                )
        for edge in network["passes"]:
            line = project(xy[[network["carrier"], edge["dst"]]])
            opened = edge["w"] > network["parameters"].open_threshold
            ax.plot(
                line[:, 0],
                line[:, 1],
                color=possession,
                lw=(0.45 + 0.8 * edge["w"]) * 1.20,
                ls="-" if opened else (0, (2, 2)),
                zorder=z + 3,
            )
        for edge in network["pressure_edges"]:
            line = project(xy[[edge["src"], edge["dst"]]])
            ax.plot(
                line[:, 0],
                line[:, 1],
                color=pressure_color,
                lw=(0.5 + 0.9 * edge["p"]) * 1.20,
                zorder=z + 4,
            )
        for i, point in enumerate(project(xy)):
            defender = network["nodes"][i]["role"] == "defender"
            marker = (
                "*"
                if i == network["carrier"]
                else ("^" if defender else ("s" if i == 1 else "o"))
            )
            ax.scatter(
                *point,
                s=28 if marker == "*" else 12,
                marker=marker,
                color=pressing if defender else possession,
                ec="white",
                lw=0.35,
                zorder=z + 5,
            )

    def arrow(top, bottom):
        """Draw a vertical flow arrow between two y positions in schematic inches; return nothing."""
        ax.annotate(
            "",
            xy=(center, bottom),
            xytext=(center, top),
            arrowprops={
                "arrowstyle": "-|>",
                "color": gray,
                "lw": 1.3,
                "mutation_scale": 12,
            },
        )

    feature_shift = 0.20
    arrow(3.24, 2.99 - feature_shift)
    ax.text(
        center,
        2.84 - feature_shift,
        "Network features",
        fontsize=13,
        color=ink,
        ha="center",
        va="center",
    )
    grid_width, grid_height = 1.35, 0.32
    for row in range(4):
        for col in range(6):
            ax.add_patch(
                Rectangle(
                    (
                        center - grid_width / 2 + col * grid_width / 6,
                        2.32 - feature_shift + row * grid_height / 4,
                    ),
                    grid_width / 6,
                    grid_height / 4,
                    fc="#EEF1F2",
                    ec="#9EA8AD",
                    lw=0.5,
                )
            )
    arrow(2.23 - feature_shift, 1.73)
    ax.add_patch(
        FancyBboxPatch(
            (center - 0.82, 1.12),
            1.64,
            0.54,
            boxstyle="round,pad=0.03,rounding_size=0.06",
            fc="#F3F5F5",
            ec="#7B888C",
            lw=1.2,
        )
    )
    ax.text(center, 1.39, "XGBoost", fontsize=16, color=ink, ha="center", va="center")
    for offset, outcome in [(-0.84, "success"), (0, "neither"), (0.84, "failure")]:
        ax.annotate(
            "",
            xy=(center + offset, 0.48),
            xytext=(center + offset * 0.65, 1.06),
            arrowprops={
                "arrowstyle": "-|>",
                "color": gray,
                "lw": 1.3,
                "mutation_scale": 12,
            },
        )
        ax.text(
            center + offset,
            0.26,
            rf"$P_{{\mathrm{{{outcome}}}}}$",
            fontsize=16,
            color=ink,
            ha="center",
            va="center",
        )
    return {
        "stack_sheets": 4,
        "stack_semantics": "separate illustrative SPN instances; no inter-instance edges",
        "view": "from above; higher sheets occlude lower sheets",
        "distinct_positions": 4,
        "instance_scenarios": [
            "original situation",
            "upper flank",
            "compact center",
            "lower flank",
        ],
        "instance_positions_m": [network["xy"].tolist() for network, _, _ in instances],
        "instance_checks": [checks for _, _, checks in instances],
        "instance_pressure_link_counts": [
            len(network["pressure_edges"]) for network, _, _ in instances
        ],
        "stack_cell_fill_opacity": 0.17,
        "stack_voronoi_edge_width_points": 0.55,
        "stack_pressure_passing_width_multiplier": 1.20,
        "features": "symbolic rows, one per SPN; no empirical values",
        "features_downward_shift_inches": feature_shift,
        "model": "XGBoost gradient-boosted classifier, applied to each feature vector",
        "model_box_label": "XGBoost",
        "outputs": ["P(success)", "P(neither)", "P(failure)"],
        "output_layout": "three separate downward arrows, success/neither/failure from left to right",
        "panel_left_inches": left,
        "panel_width_inches": width,
        "model_and_outputs_shift_inches": -0.25,
    }
