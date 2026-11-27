"""Draw the club scatterplot and the distribution of within-season correlations."""

from __future__ import annotations

import numpy as np

import pandas as pd

from .typography import GRAY, LIGHT

from .label_layout import swarm_offsets

CRIMSON, LIGHT_BLUE, NAVY = "#A52D4B", "#5CAFE3", "#243F63"


def heading(fig, x, letter, title):
    """Place a bold lowercase label and a regular heading with an eight-point gap; x is a figure fraction."""
    label = fig.text(
        x, 0.922, "(" + letter + ")", fontsize=17.16, weight="bold", va="center"
    )
    fig.canvas.draw()
    edge = label.get_window_extent(fig.canvas.get_renderer()).x1 / fig.bbox.width
    fig.text(edge + 8 / 72 / fig.get_figwidth(), 0.922, title, fontsize=13, va="center")


def draw_total_panel(fig, coefficients):
    """Draw saved dimensionless value correlations by population; return display rows and layout checks."""
    from matplotlib.lines import Line2D

    rows = coefficients.loc[coefficients.measure.eq("value")].copy()
    assert len(rows) == 29 and rows.dataset.nunique() == 29
    assert rows.population.value_counts().to_dict() == {"training": 24, "outside": 5}
    bx = fig.add_axes([0.750, 0.23, 0.230, 0.64])
    bx.set(
        xlim=(-0.65, 2.15),
        ylim=(-0.22, 0.99),
        xticks=[0, 1.5],
        yticks=[0, 0.25, 0.5, 0.75],
    )
    bx.set_xticklabels(["Fitting pool\n24 seasons", "Outside\n5 seasons"])
    bx.tick_params(labelsize=10.8, length=3, pad=4)
    bx.set_ylabel(r"Spearman $\rho$ with xG difference", fontsize=11.2, labelpad=7)
    bx.grid(axis="y", color=LIGHT, linewidth=0.6)
    bx.axhline(0, color=GRAY, linewidth=0.9, linestyle=(0, (3, 3)))
    fig.canvas.draw()
    plotted = []
    for population, center in [("training", 0), ("outside", 1.5)]:
        group = rows.loc[rows.population.eq(population)].sort_values("dataset")
        offsets = swarm_offsets(bx, group.rho.to_numpy(), center)
        for row, offset in zip(group.itertuples(), offsets):
            outside = population == "outside"
            bx.scatter(
                center + offset,
                row.rho,
                marker="D" if outside else "o",
                s=26 if outside else 19,
                color=LIGHT_BLUE if outside else CRIMSON,
                edgecolor="white",
                linewidth=0.4,
                zorder=4,
            )
            plotted.append(
                {
                    "dataset": row.dataset,
                    "population": population,
                    "measure": "value",
                    "rho": row.rho,
                    "display_x": center + offset,
                }
            )
    median = float(rows.rho.median())
    # The reference summarizes all 29 seasons, not separate population medians.
    bx.plot([-0.42, 1.92], [median, median], color="black", linewidth=1.2, zorder=2)
    positive = int(rows.rho.gt(0).sum())
    assert positive == 29
    example = next(row for row in plotted if row["dataset"] == "epl_2025_2026")
    assert example["population"] == "outside"
    bx.scatter(
        example["display_x"],
        example["rho"],
        marker="D",
        s=66,
        color=LIGHT_BLUE,
        edgecolor=NAVY,
        linewidth=1.4,
        zorder=6,
    )
    handles = [
        Line2D(
            [],
            [],
            marker="D",
            linestyle="none",
            color=NAVY,
            markerfacecolor=LIGHT_BLUE,
            markeredgewidth=1.4,
            markersize=6,
            label="EPL 2025/26",
        ),
        Line2D(
            [], [], color="black", linewidth=1.2, label=f"Overall median: {median:.2f}"
        ),
    ]
    legend = bx.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.008),
        fontsize=10.8,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1,
        handlelength=1.1,
        handletextpad=0.45,
        labelspacing=0.4,
        borderaxespad=0,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer)
    assert legend_box.x0 >= bx.bbox.x0 and legend_box.x1 <= bx.bbox.x1
    assert legend_box.y0 >= bx.bbox.y0 and legend_box.y1 <= bx.bbox.y1
    # Both vertical axes span the full, identically positioned panel height.
    a_spine = fig.axes[0].spines["left"]
    b_spine = bx.spines["left"]
    a_bounds = a_spine.get_path().transformed(a_spine.get_transform()).get_extents()
    b_bounds = b_spine.get_path().transformed(b_spine.get_transform()).get_extents()
    assert np.allclose([a_bounds.y0, a_bounds.y1], [b_bounds.y0, b_bounds.y1])
    tick_boxes = [item.get_window_extent(renderer) for item in bx.get_xticklabels()]
    assert not tick_boxes[0].overlaps(tick_boxes[1])
    for row in plotted:
        point = bx.transData.transform((row["display_x"], row["rho"]))
        assert bx.bbox.contains(*point), row
        assert not legend_box.padded(3 * fig.dpi / 72).contains(*point), row
    new = pd.DataFrame(plotted)
    keys = ["dataset", "population", "measure", "rho"]
    pd.testing.assert_frame_equal(
        new[keys].sort_values(keys[:3]).reset_index(drop=True),
        rows[keys].sort_values(keys[:3]).reset_index(drop=True),
        check_exact=True,
    )
    return new, {
        "display": "total expected press value only, grouped by population",
        "positive_counts": {"value": positive},
        "medians": {"value": median},
        "highlighted_season": [example],
        "population_counts": {"training": 24, "outside": 5},
        "legend_scoped_to": "inside panel b, lower left",
        "legend_within_panel_bounds": True,
        "season_markers_covered_by_legend": 0,
        "positive_count_displayed": False,
        "vertical_axis_bounds_pixels": {
            "a": [a_bounds.y0, a_bounds.y1],
            "b": [b_bounds.y0, b_bounds.y1],
        },
        "column_labels": ["Fitting pool: 24 seasons", "Outside: 5 seasons"],
    }
