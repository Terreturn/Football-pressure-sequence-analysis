"""Draw the match-bootstrap AUC intervals and headings for Figure 2."""

from __future__ import annotations

from .typography import GRAY, INK as INK, LIGHT

FONT = 13.6

FEATURE_FONT = 12.6

PETROL, CORAL, BURGUNDY = "#165E66", "#DF806E", "#792E48"

BASE_HEIGHT, BOTTOM_TRIM, TOP_TRIM = 5.4, 0.25, 0.16

HEIGHT = BASE_HEIGHT - BOTTOM_TRIM - TOP_TRIM

PLOT_BOTTOM = (0.17 * BASE_HEIGHT - BOTTOM_TRIM) / HEIGHT

PLOT_HEIGHT = 0.71 * BASE_HEIGHT / HEIGHT

HEADER_Y = (0.925 * BASE_HEIGHT - BOTTOM_TRIM) / HEIGHT

XLABEL_Y = (0.075 * BASE_HEIGHT - BOTTOM_TRIM) / HEIGHT


def heading(fig, x, letter, title):
    """Place a panel heading at normalized x; fonts are points and all positions are figure fractions."""
    label = fig.text(
        x, HEADER_Y, "(" + letter + ")", fontsize=19.5, weight="bold", va="center"
    )
    fig.canvas.draw()
    edge = label.get_window_extent(fig.canvas.get_renderer()).x1 / fig.bbox.width
    fig.text(
        edge + 8 / 72 / fig.get_figwidth(), HEADER_Y, title, fontsize=14.8, va="center"
    )


def auc_panel(fig, rows):
    """Plot every saved AUC and interval; the displayed horizontal segments are 0.496–0.5185 and 0.65–0.80."""
    from matplotlib.lines import Line2D

    bottom, height = PLOT_BOTTOM, PLOT_HEIGHT
    left = fig.add_axes([0.140, bottom, 0.018, height])
    right = fig.add_axes([0.188, bottom, 0.122, height], sharey=left)
    for ax in (left, right):
        ax.set(ylim=(-1, 13), yticks=[])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", labelsize=FONT, length=3, pad=4)
    left.set(xlim=(0.496, 0.5185), xticks=[0.5])
    left.set_xticklabels(["0.5"])
    left.axvline(0.5, color=GRAY, linewidth=0.85, linestyle=(0, (3, 3)))
    left.annotate(
        "Chance",
        (0.5, 12.8),
        xytext=(4, -2),
        textcoords="offset points",
        fontsize=11,
        color=GRAY,
        va="top",
        ha="left",
        annotation_clip=False,
    )
    right.set(xlim=(0.65, 0.80), xticks=[0.65, 0.70, 0.75, 0.80])
    right.set_xticklabels(["0.65", "0.7", "0.75", "0.8"])
    right.grid(axis="x", color=LIGHT, linewidth=0.7)
    internal = rows.loc[rows.population.eq("internal")].iloc[0]
    right.axvline(internal.auc, color=BURGUNDY, linewidth=0.8, linestyle=(0, (3, 3)))
    for row in rows.itertuples():
        assert 0.65 < row.lower <= row.auc <= row.upper < 0.80
        right.errorbar(
            row.auc,
            row.y,
            xerr=[[row.auc - row.lower], [row.upper - row.auc]],
            fmt="s" if row.population == "internal" else "o",
            color=BURGUNDY,
            markersize=4.8,
            elinewidth=1.3,
            capsize=2.4,
            capthick=1,
            zorder=3,
        )
        label = {
            "internal": "Fitting pool",
            "epl_2025_2026": "EPL",
            "worldcup_2022": "WC 2022",
            "worldcup_2026": "WC 2026",
        }.get(row.population, row.label)
        y = bottom + height * (row.y + 1) / 14
        fig.text(0.134, y, label, fontsize=FONT, ha="right", va="center")
    for y, text in [
        (12.15, "Cross-validation"),
        (9.85, "Outside leagues: 2025/26"),
        (3.75, "Outside tournaments"),
    ]:
        fig.text(
            0.015,
            bottom + height * (y + 1) / 14,
            text,
            fontsize=FONT,
            color=GRAY,
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15},
        )
    # Two diagonal strokes mark the omitted AUC range, not a discontinuity in the estimates.
    for x in (0.158, 0.188):
        cut_height = 0.008 * BASE_HEIGHT / HEIGHT
        fig.add_artist(
            Line2D(
                [x - 0.004, x + 0.004],
                [bottom - cut_height, bottom + cut_height],
                transform=fig.transFigure,
                color="black",
                linewidth=1.1,
                clip_on=False,
            )
        )
    fig.text(0.221, XLABEL_Y, "Success AUC at onset", fontsize=FONT, ha="center")
    return {
        "left_xlim": [0.496, 0.5185],
        "right_xlim": [0.65, 0.80],
        "all_intervals_in_right_segment": True,
    }
