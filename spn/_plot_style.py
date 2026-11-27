"""Adapt the collaborator's team and player panels to a user-selected cohort."""

from contextlib import contextmanager

import matplotlib as mpl
import numpy as np

TEAM_COLOR = "#243F63"
PLAYER_COLOR = "#2A8C82"


@contextmanager
def paper_style():
    with mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        yield


def save_figure(fig, stem):
    for suffix in (".pdf", ".png"):
        fig.savefig(
            stem.with_suffix(suffix),
            facecolor="white",
            dpi=425,
            bbox_inches="tight",
            pad_inches=0.12,
        )


def player_panel(ax, table, metric, count, *, xlabel, color):
    """Rank signed means and show the number of observations behind each bar."""
    shown = table.loc[table.displayed]
    if not shown.empty:
        labels = [f"{row['name']}  (n={row[count]})" for _, row in shown.iterrows()]
        bars = ax.barh(np.arange(len(shown)), shown[metric], color=color, height=0.58)
        ax.bar_label(bars, fmt="%.3g", padding=3, fontsize=8)
        ax.margins(x=0.14)
        ax.set_yticks(np.arange(len(shown)), labels=labels)
        ax.invert_yaxis()
        ax.axvline(0, color="#77818A", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E5E8E9", linewidth=0.6)
        ax.set_xlabel(xlabel)
    else:
        message = (
            "No scored events"
            if table[count].sum() == 0
            else "No players meet the selected thresholds"
        )
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()


def labels_without_overlap(ax, points, names, line):
    """Use the collaborator's display-space collision checks, with an outside column for dense plots."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    xy = ax.transData.transform(points)
    line_path = line.get_transform().transform_path(line.get_path()) if line is not None else None
    occupied = []
    labels = []
    for point, name in zip(points, names):
        label = ax.annotate(name, point, xytext=(0, 0), textcoords="offset points", fontsize=8)
        labels.append(label)
        placed = False
        for radius in (5, 10, 16, 24, 36):
            for angle in (20, 160, -20, -160, 90, -90):
                dx, dy = radius * np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
                label.set_position((dx, dy))
                label.set_ha("left" if dx >= 0 else "right")
                label.set_va("bottom" if dy >= 0 else "top")
                box = label.get_window_extent(renderer).padded(1.5)
                if (
                    ax.bbox.contains(*box.min)
                    and ax.bbox.contains(*box.max)
                    and not any(box.overlaps(old) for old in occupied)
                    and not any(box.contains(x, y) for x, y in xy)
                    and (line_path is None or not line_path.intersects_bbox(box, filled=False))
                ):
                    occupied.append(box)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            for artist in labels:
                artist.remove()
            order = np.argsort(points[:, 1], kind="stable")
            for y, index in zip(np.linspace(0.05, 0.95, len(points)), order):
                ax.annotate(
                    names[index],
                    points[index],
                    xytext=(1.04, y),
                    textcoords="axes fraction",
                    va="center",
                    fontsize=8,
                    arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "#89939B"},
                )
            break


def team_panel(ax, table, metric, outcome, *, xlabel, ylabel):
    x, y = table[metric].to_numpy(), table[outcome].to_numpy()
    ax.set_axisbelow(True)
    ax.grid(color="#E5E8E9", linewidth=0.6)
    ax.scatter(x, y, s=33, color=TEAM_COLOR, edgecolors="white", linewidths=0.6, zorder=3)
    line = None
    if len(x) >= 2 and np.ptp(x) > 0:
        fit = np.polynomial.Polynomial.fit(x, y, 1)
        limits = np.array([x.min(), x.max()])
        (line,) = ax.plot(limits, fit(limits), color="#A7AFB5", linewidth=0.85)
    ax.margins(x=0.20, y=0.16)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    labels_without_overlap(ax, np.column_stack((x, y)), table.name.tolist(), line)
