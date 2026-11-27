"""Draw the paper's Premier League positioning figure from club and player means in percentage points."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator, MultipleLocator
from scipy.stats import spearmanr

from .team_performance import NAVY
from .typography import LIGHT
from .label_layout import labels_without_overlap
from .club_labels import TEAM_SHORT
from .player_names import DISPLAY_NAMES
from .positioning_diagram import (
    INK,
    draw_schematic,
    heading,
    sample_geometry,
)

SCORE_LABEL = "Regain probability difference (pp)"


def ranked_players(players):
    """Select the ten highest player means (percentage points) after the season eligibility cut.

    Rank the unrounded scores. Tied players share a rank, and their provider IDs
    supply a stable display order. Every eligible pressure enters the club panel,
    including pressures by players below either individual threshold.
    """
    eligible = players.loc[
        players.pressures.ge(15) & players.season_minutes.ge(1350)
    ].copy()
    eligible["rank"] = eligible.position_quality_pp.rank(
        method="min", ascending=False
    ).astype(int)
    top = (
        eligible.sort_values(
            ["position_quality_pp", "player_id"], ascending=[False, True]
        )
        .head(10)
        .copy()
    )
    top["display_name"] = top.player_name.map(
        lambda name: DISPLAY_NAMES.get(name, name)
    )
    top["team_label"] = top.clubs.map(
        lambda names: " / ".join(
            TEAM_SHORT.get(name, name) for name in names.split("; ")
        )
    )
    return top


def club_panel(fig, clubs, interval):
    """Draw one marker per club and annotate Spearman's rho with its saved fixture-bootstrap interval.

    The horizontal axis is mean position quality in percentage points. The vertical
    axis is season points. The thin least-squares line spans the observed means.
    """
    width, height = fig.get_size_inches()
    ax = fig.add_axes([0.8 / width, 0.70 / height, 4.38 / width, 2.97 / height])
    quality = clubs.position_quality_pp.to_numpy()
    points = clubs.points.to_numpy()
    padding = max(0.01, float(np.ptp(quality)) * 0.18)
    ax.set(
        xlim=(0.18, quality.max() + padding), ylim=(14, 92), yticks=range(20, 91, 10)
    )
    ax.set_xlabel(
        "Club mean regain probability difference (pp)", fontsize=11.2, labelpad=9
    )
    ax.set_ylabel("Season points (EPL 2025/26)", fontsize=11.2, labelpad=8)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_axisbelow(True)
    ax.grid(color=LIGHT, linewidth=0.6)
    ax.tick_params(labelsize=10.8)
    fit = np.polynomial.Polynomial.fit(quality, points, 1)
    ends = np.array([quality.min(), quality.max()])
    (line,) = ax.plot(ends, fit(ends), color="#A7AFB5", lw=0.85, zorder=2)
    ax.scatter(
        quality, points, s=31, c=NAVY, edgecolor="white", linewidth=0.65, zorder=4
    )
    rho = spearmanr(quality, points).statistic
    low, high = interval
    ax.text(
        0.025,
        0.975,
        rf"$\rho = {rho:.2f}$  (95% CI ${low:.2f}$ to ${high:.2f}$)",
        transform=ax.transAxes,
        fontsize=10.8,
        va="top",
    )
    labels_without_overlap(
        ax,
        list(zip(quality, points)),
        clubs.display_name.tolist(),
        10.8,
        avoid_line=line,
        marker_clearance=3.5,
    )
    return ax


def player_panel(fig, players):
    """Draw the ranked player means as bars. Return the axis and the name labels for the spacing check."""
    width, height = fig.get_size_inches()
    ax = fig.add_axes([8.10 / width, 0.70 / height, 2.25 / width, 2.97 / height])
    scores = players.position_quality_pp.to_numpy()
    low = min(0, float(scores.min()) * 1.1)
    ax.set(xlim=(low, max(0.1, float(scores.max()) * 1.1)), ylim=(9.6, -0.6), yticks=[])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_xlabel(SCORE_LABEL, fontsize=10.5, labelpad=9)
    ax.barh(np.arange(len(players)), scores, height=0.55, color="#2A8C82")
    if low < 0:
        ax.axvline(0, color="black", lw=0.8)
    labels = []
    for index, player in enumerate(players.itertuples()):
        # Text columns sit in figure coordinates so the rank and the value align across players.
        y = (0.70 + 2.97 * (9.6 - index) / 10.2) / height
        name = f"{player.display_name} ({player.team_label})"
        labels.append(
            fig.text(7.72 / width, y, name, ha="right", va="center", fontsize=10.5)
        )
        fig.text(
            7.96 / width,
            y,
            str(player.rank),
            ha="right",
            va="center",
            fontsize=10,
            color="#68727A",
        )
        fig.text(
            10.7 / width,
            y,
            f"{player.position_quality_pp:.2f}",
            ha="right",
            va="center",
            fontsize=9,
        )
    return ax, labels


def render(output, players, clubs, interval, caption):
    """Save Figure 4 as PDF and 425 DPI PNG, with its adopted caption and the exact rows displayed in (c)."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
        }
    )
    fig = plt.figure(figsize=(11, 8.4), facecolor="white")
    xy, relative, _ = sample_geometry()
    draw_schematic(
        fig,
        xy,
        relative,
        lower_inset=True,
        tight=True,
        editorial=True,
        refined=True,
        polished=True,
    )
    for label in fig.findobj(Text):
        if label.get_text() == "Apply held-out model":
            label.set_text("Apply model")
    heading(fig, 0.35, 3.88, "(b)", "High presses and season points")
    heading(fig, 7.3, 3.88, "(c)", "Highest player scores")
    selected = ranked_players(players)
    left = club_panel(fig, clubs, interval)
    right, labels = player_panel(fig, selected)
    for ax in (left, right):
        low, high = ax.get_xlim()
        ax.set_xticks([x for x in ax.get_xticks() if low <= x <= high])
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=3, pad=4, colors="black")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label in labels:
        box = label.get_window_extent(renderer)
        if box.x0 <= left.bbox.x1 + 5 or box.x1 >= right.bbox.x0 - 7:
            raise ValueError(f"Player label overlaps a panel: {label.get_text()}")
    for label in fig.findobj(Text):
        if label.get_visible() and label.get_text():
            box = label.get_window_extent(renderer)
            if (
                box.x0 < -1
                or box.y0 < -1
                or box.x1 > fig.bbox.width + 1
                or box.y1 > fig.bbox.height + 1
            ):
                raise ValueError(f"Label is clipped: {label.get_text()}")
    stem = "figure4_broad_regain"
    fig.savefig(output / (stem + ".pdf"))
    fig.savefig(output / (stem + ".png"), dpi=425)
    plt.close(fig)
    # Retain the column names used in the paper's exported top-ten table.
    top = selected.assign(
        supported=True,
        regain_difference_100=selected.position_quality_pp,
        regain_difference_100_rank=selected["rank"],
    )
    top[
        [
            *players.columns,
            "supported",
            "regain_difference_100",
            "regain_difference_100_rank",
            "display_name",
            "team_label",
        ]
    ].to_csv(output / (stem + "_top10.csv"), index=False)
    (output / (stem + "_caption.tex")).write_text(caption.strip() + "\n")
