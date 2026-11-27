"""Arrange league-season markers and position labels around the club scatterplots."""

from __future__ import annotations

import numpy as np

from .typography import INK


def labels_without_overlap(
    ax, points, names, font, *, avoid_line=None, marker_clearance=0, adjustments=None
):
    """Place all team names without overlap using display-pixel distances; return label offsets in points."""
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox, IdentityTransform

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    scale = ax.figure.dpi / 72
    pixels = ax.transData.transform(np.array(points))
    line_path = (
        avoid_line.get_transform().transform_path(avoid_line.get_path())
        if avoid_line is not None
        else None
    )
    # Prioritize the densest neighborhoods so close team pairs receive space first.
    distances = np.linalg.norm(pixels[:, None, :] - pixels[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    order = np.argsort(distances.min(axis=1), kind="stable")
    obstacles = []
    for item in ax.texts:
        if item.get_visible():
            obstacles.append(item.get_window_extent(renderer).expanded(1.03, 1.08))
    boundaries = ax.get_window_extent(renderer)
    bounds = Bbox.from_extents(
        boundaries.x0 + 2, boundaries.y0 + 2, boundaries.x1 - 2, boundaries.y1 - 2
    )
    sizes = []
    for name in names:
        probe = ax.text(0, 0, name, fontsize=font, transform=IdentityTransform())
        b = probe.get_window_extent(renderer)
        sizes.append((b.width, b.height))
        probe.remove()
    result = []
    for i in order:
        px, py = pixels[i]
        w, h = sizes[i]
        choices = []
        for radius in (5, 10, 16, 23, 32, 43, 56, 70):
            for angle in (20, 160, -20, -160, 90, -90, 45, 135, -45, -135, 0, 180):
                theta = np.deg2rad(angle)
                dx, dy = radius * np.cos(theta) * scale, radius * np.sin(theta) * scale
                ha = (
                    "left"
                    if np.cos(theta) > 0.25
                    else "right"
                    if np.cos(theta) < -0.25
                    else "center"
                )
                va = (
                    "bottom"
                    if np.sin(theta) > 0.25
                    else "top"
                    if np.sin(theta) < -0.25
                    else "center"
                )
                x0 = px + dx - (0 if ha == "left" else w if ha == "right" else w / 2)
                y0 = py + dy - (0 if va == "bottom" else h if va == "top" else h / 2)
                box = Bbox.from_bounds(x0 - 1.6, y0 - 1.6, w + 3.2, h + 3.2)
                inside = (
                    box.x0 >= bounds.x0
                    and box.x1 <= bounds.x1
                    and box.y0 >= bounds.y0
                    and box.y1 <= bounds.y1
                )
                if not inside or any(box.overlaps(old) for old in obstacles):
                    continue
                if line_path is not None and line_path.intersects_bbox(
                    box, filled=False
                ):
                    continue
                marker_box = box.padded(marker_clearance * scale)
                if any(marker_box.contains(x, y) for x, y in pixels):
                    continue
                if marker_clearance:
                    # A short unconnected label should be nearest its own point. Longer
                    # leaders should reach their point without running over another one.
                    end = np.clip(pixels[i], [box.x0, box.y0], [box.x1, box.y1])
                    others = np.delete(pixels, i, axis=0)
                    if radius <= 9:
                        closest = np.clip(others, [box.x0, box.y0], [box.x1, box.y1])
                        if np.linalg.norm(
                            others - closest, axis=1
                        ).min() < np.linalg.norm(pixels[i] - end):
                            continue
                    else:
                        delta = end - pixels[i]
                        fraction = np.clip(
                            (others - pixels[i]) @ delta / (delta @ delta), 0, 1
                        )
                        clearance = np.minimum(
                            marker_clearance * scale,
                            0.95 * np.linalg.norm(others - pixels[i], axis=1),
                        )
                        if np.any(
                            np.linalg.norm(
                                others - pixels[i] - fraction[:, None] * delta, axis=1
                            )
                            < clearance
                        ):
                            continue
                choices.append(
                    (radius, abs(dy) * 0.01, dx / scale, dy / scale, ha, va, box)
                )
            if choices:
                break
        if not choices:
            raise ValueError("No nonoverlapping placement for " + names[i])
        _, _, dx, dy, ha, va, box = sorted(choices, key=lambda v: (v[0], v[1]))[0]
        # Reserve the automatic position so an editorial nudge leaves the other labels in place.
        obstacles.append(box)
        leader = np.hypot(dx, dy) > 9
        if adjustments and names[i] in adjustments:
            dx, dy, ha, va, leader = adjustments[names[i]]
        label = ax.annotate(
            names[i],
            points[i],
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=font,
            color=INK,
            arrowprops={
                "arrowstyle": "-",
                "color": "#A7AFB5",
                "lw": 0.6,
                "shrinkA": 2.5,
                "shrinkB": 3,
            }
            if leader
            else None,
            zorder=5,
        )
        if adjustments and names[i] in adjustments and line_path is not None:
            label.update_positions(renderer)
            if line_path.intersects_bbox(
                Text.get_window_extent(label, renderer), filled=False
            ):
                label.set_bbox(
                    {
                        "boxstyle": "square,pad=0.06",
                        "facecolor": ax.get_facecolor(),
                        "edgecolor": "none",
                    }
                )
        result.append(
            {
                "team": names[i],
                "offset_points": [float(dx), float(dy)],
                "ha": ha,
                "va": va,
            }
        )
    return result


def check_team_label_layout(ax, teams):
    """Check team-name bounds and collisions against text and marker centers in display pixels."""
    from matplotlib.text import Text

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    names = set(teams.display_name)
    labels = [item for item in ax.texts if item.get_text() in names]
    boxes = [
        Text.get_window_extent(item, renderer).expanded(1.01, 1.04) for item in labels
    ]
    assert len(labels) == 20
    points = ax.transData.transform(
        teams[["value_centered", "xgd_per_match"]].to_numpy()
    )
    for i, (item, box) in enumerate(zip(labels, boxes)):
        for j in range(i):
            assert not box.overlaps(boxes[j]), (item.get_text(), labels[j].get_text())
        assert not any(box.contains(x, y) for x, y in points), item.get_text()
    return {"labels": len(labels), "text_overlaps": 0, "marker_centers_covered": 0}


def swarm_offsets(ax, values, center):
    """Return horizontal-only offsets preserving every dimensionless correlation value exactly."""
    coordinates = ax.transData.transform(
        np.column_stack([np.full(len(values), center), values])
    )
    spacing = 6.8 * ax.figure.dpi / 72
    scale = (
        ax.transData.transform([center + 1, 0])[0]
        - ax.transData.transform([center, 0])[0]
    )
    placed, offsets = [], np.zeros(len(values))
    for i in np.argsort(values, kind="stable"):
        py = coordinates[i, 1]
        for col in [0, 1, -1, 2, -2, 3, -3, 4, -4]:
            dx = col * spacing
            if all(
                (dx - oldx) ** 2 + (py - oldy) ** 2 >= spacing**2 - 1e-8
                for oldx, oldy in placed
            ):
                offsets[i] = dx / scale
                placed.append((dx, py))
                break
        else:
            raise ValueError("Swarm needs more space")
    assert max(np.abs(offsets)) < 0.46
    return offsets
