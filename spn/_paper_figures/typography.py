"""Set fonts, axes and colors for the prediction and team figures."""

from __future__ import annotations

INK = "#202830"

GRAY = "#717B83"

LIGHT = "#E5E9EC"


def configure_plotting():
    """Set manuscript-size typography in points and return Matplotlib's pyplot interface."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#A8B1B8",
            "axes.linewidth": 0.8,
        }
    )
    return plt
