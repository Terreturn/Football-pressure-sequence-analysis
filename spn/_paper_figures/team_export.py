"""Crop and export the team figure while preserving its label extents."""

from __future__ import annotations

import numpy as np


def finish(fig, path, minimum, *, dpi=170, export_pdf=False, vertical_padding=None):
    """Save a PNG at the supplied dots/inch and optional vector PDF after typography checks; return audit metadata."""
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_boxes = []
    for item in fig.findobj(Text):
        if not item.get_visible() or not item.get_text().strip():
            continue
        box = item.get_window_extent(renderer)
        if box.width == 0 or box.height == 0:
            continue
        assert item.get_fontsize() >= minimum, item.get_text()
        assert box.x0 >= -0.5 and box.y0 >= -0.5, (item.get_text(), box)
        assert box.x1 <= fig.bbox.x1 + 0.5 and box.y1 <= fig.bbox.y1 + 0.5, (
            item.get_text(),
            box,
        )
        text_boxes.append(
            {
                "text": item.get_text(),
                "font_points": item.get_fontsize(),
                "box": list(box.bounds),
            }
        )
    export_box = None
    if vertical_padding is not None:
        # Crop only the outer vertical margins, preserving physical panel sizes and full width.
        assert vertical_padding >= 0
        tight = fig.get_tightbbox(renderer)
        bottom = max(0, np.floor((tight.y0 - vertical_padding) * dpi) / dpi)
        top = min(
            fig.get_figheight(), np.ceil((tight.y1 + vertical_padding) * dpi) / dpi
        )
        export_box = Bbox.from_extents(0, bottom, fig.get_figwidth(), top)
        assert all(
            bottom * fig.dpi <= item["box"][1]
            and item["box"][1] + item["box"][3] <= top * fig.dpi
            for item in text_boxes
        )
    if export_pdf:
        with plt.rc_context({"pdf.fonttype": 42}):
            fig.savefig(
                path.with_suffix(".pdf"), facecolor="white", bbox_inches=export_box
            )
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches=export_box)
    # Also render at 100 pixels per physical inch of a 6.5-inch-wide manuscript
    # figure. This exposes unreadable typography that a large preview can hide.
    fig.savefig(
        path.with_name(path.stem + "_paper_size.png"),
        dpi=100 * 6.5 / fig.get_figwidth(),
        facecolor="white",
        bbox_inches=export_box,
    )
    audit = {
        "size_inches": list(fig.get_size_inches()),
        "text": text_boxes,
        "minimum_font_at_6_5_inches": minimum * 6.5 / fig.get_figwidth(),
    }
    if export_box is not None:
        audit.update(
            {
                "export_box_inches": list(export_box.extents),
                "export_size_inches": [export_box.width, export_box.height],
                "vertical_padding_inches": vertical_padding,
            }
        )
    plt.close(fig)
    return audit
