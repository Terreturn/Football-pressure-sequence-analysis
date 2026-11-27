"""Export the prediction figure with its adopted margins."""

from __future__ import annotations


def cropped_export(fig, output, stem):
    """Export vector PDF and 425 DPI PNG with the same physical bounding box in inches."""
    from matplotlib import pyplot as plt

    fig.canvas.draw()
    box = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.085)
    for suffix in ["pdf", "png"]:
        fig.savefig(
            output / (stem + "." + suffix), dpi=425, bbox_inches=box, facecolor="white"
        )
    plt.close(fig)
