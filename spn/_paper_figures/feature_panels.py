"""Draw the feature-contribution and log-loss panels from tabulated estimates."""

from __future__ import annotations

from .auc_panel import (
    PLOT_BOTTOM,
    PLOT_HEIGHT,
    XLABEL_Y,
    FONT,
    FEATURE_FONT,
    CORAL,
    PETROL,
    INK,
    LIGHT,
)


def shap_panel(fig, rows):
    """Draw the ten largest calibrated class-score contributions with an adaptive axis and unobstructed in-panel legend."""
    from matplotlib.patches import Patch
    from matplotlib.transforms import blended_transform_factory
    from matplotlib.ticker import MaxNLocator

    ax = fig.add_axes([0.540, PLOT_BOTTOM, 0.160, PLOT_HEIGHT])
    top = rows.iloc[:10]
    ax.set(xlim=(0, float(top.upper.max()) * 1.12), ylim=(9.7, -0.7), yticks=[])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.spines["left"].set_visible(True)
    ax.tick_params(axis="x", labelsize=FONT, length=3, pad=4)
    ax.grid(axis="x", color=LIGHT, linewidth=0.7)
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for y, row in enumerate(top.itertuples()):
        ax.barh(
            y,
            row.mean_abs_shap,
            height=0.50,
            color=CORAL if row.collective else PETROL,
            zorder=2,
        )
        ax.errorbar(
            row.mean_abs_shap,
            y,
            xerr=[[row.mean_abs_shap - row.lower], [row.upper - row.mean_abs_shap]],
            fmt="none",
            color=INK,
            elinewidth=0.9,
            capsize=2,
            zorder=3,
        )
        ax.text(
            -0.04,
            y,
            row.label,
            transform=trans,
            fontsize=FEATURE_FONT,
            ha="right",
            va="center",
        )
    legend = ax.legend(
        handles=[
            Patch(color=CORAL, label="Collective"),
            Patch(color=PETROL, label="Other"),
        ],
        loc="lower right",
        ncol=1,
        frameon=True,
        facecolor="white",
        framealpha=0.98,
        edgecolor="none",
        fontsize=FONT,
        handlelength=0.65,
        handletextpad=0.3,
        borderpad=0.1,
        borderaxespad=0.05,
        labelspacing=0.4,
    )
    for _ in range(30):
        fig.canvas.draw()
        box = legend.get_window_extent(fig.canvas.get_renderer())
        obstructed = any(
            box.y0 - 5 <= ax.transData.transform((r.upper, y))[1] <= box.y1 + 5
            and ax.transData.transform((r.upper, y))[0] >= box.x0 - 6
            for y, r in enumerate(top.itertuples())
        )
        if not obstructed:
            break
        ax.set_xlim(0, ax.get_xlim()[1] * 1.1)
    assert not obstructed
    lo, hi = ax.get_xlim()
    ticks = MaxNLocator(nbins=3).tick_values(lo, hi)
    ax.set_xticks(ticks[(ticks >= lo) & (ticks <= hi)])
    fig.text(0.620, XLABEL_Y, "Mean |SHAP value|", fontsize=FONT, ha="center")
    return {
        "xlim": list(ax.get_xlim()),
        "legend_inside_b": True,
        "legend_obscures_estimates": False,
    }


def gain_panel(fig, rows, *, tick_labels=None):
    """Plot baseline minus augmented calibrated log loss, allowing either sign and preserving paired intervals in nats per onset."""
    from matplotlib.transforms import blended_transform_factory
    from matplotlib.ticker import MaxNLocator

    ax = fig.add_axes([0.852, PLOT_BOTTOM, 0.133, PLOT_HEIGHT])
    span = max(0.005, float(rows.upper.max() - min(0, rows.lower.min())))
    lo = min(0, float(rows.lower.min()) - 0.06 * span)
    hi = max(0, float(rows.upper.max())) + 0.10 * span
    ax.set(xlim=(lo, hi), ylim=(-0.6, 10.6), yticks=[])
    if tick_labels is None:
        ticks = MaxNLocator(nbins=4).tick_values(lo, hi)
        ax.set_xticks(ticks[(ticks >= lo) & (ticks <= hi)])
    else:
        ax.set_xticks([float(label) for label in tick_labels], labels=tick_labels)
    ax.tick_params(axis="x", labelsize=FONT, length=3, pad=4)
    ax.spines["left"].set_visible(True)
    ax.grid(axis="x", color=LIGHT, linewidth=0.7)
    if lo < 0:
        ax.axvline(0, color="black", linewidth=0.8)
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for r in rows.itertuples():
        assert r.lower <= r.gain <= r.upper
        ax.errorbar(
            r.gain,
            r.y,
            xerr=[[r.gain - r.lower], [r.upper - r.gain]],
            fmt="D" if r.level == "group" else "o",
            color=CORAL,
            markersize=5 if r.level == "group" else 4,
            elinewidth=1.3,
            capsize=2.4,
            capthick=1,
            zorder=3,
        )
        label = {"Outside pooled": "Outside\npooled", "Internal": "Fitting pool"}.get(
            r.label, r.label
        )
        ax.text(
            -0.075,
            r.y,
            label,
            transform=trans,
            fontsize=FONT,
            va="center",
            ha="right",
            linespacing=1.0,
        )
    fig.text(0.910, XLABEL_Y, "Log-loss reduction", fontsize=FONT, ha="center")
    return {"xlim": [lo, hi], "paired_match_intervals": True}
