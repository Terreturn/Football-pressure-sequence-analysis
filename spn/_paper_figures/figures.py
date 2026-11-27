"""Redraw Figures 1–4 from the published plot tables; expose the same panel functions for reuse."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .network_schematic import render as draw_network_schematic
from .feature_panels import gain_panel, shap_panel
from .typography import LIGHT, configure_plotting
from .auc_panel import HEIGHT, auc_panel, heading
from .team_export import finish
from .club_offsets import EPL_LABEL_ADJUSTMENTS
from .team_performance import (
    NAVY,
    draw_total_panel,
    heading as team_heading,
)
from .label_layout import (
    check_team_label_layout,
    labels_without_overlap,
)
from .positioning_figure import render as draw_positioning
from .export import cropped_export


def figure_2(tables, output):
    """Draw onset AUC, absolute TreeSHAP contributions and the paired log-loss reductions."""
    configure_plotting()
    plt.rcParams.update({"axes.edgecolor": "black"})
    fig = plt.figure(figsize=(9.8 * 1.44, HEIGHT), facecolor="white")
    for x, letter, title in [
        (0.015, "a", "Prediction accuracy"),
        (0.370, "b", "Feature contributions"),
    ]:
        heading(fig, x, letter, title)
    auc_panel(
        fig, pd.read_csv(tables / "figure2_auc.csv", float_precision="round_trip")
    )
    shap_panel(
        fig, pd.read_csv(tables / "figure2_shap.csv", float_precision="round_trip")
    )
    heading(fig, 0.765, "c", "Collective features")
    gain_panel(
        fig,
        pd.read_csv(tables / "figure2_network_gains.csv", float_precision="round_trip"),
        tick_labels=["0.0", "0.02", "0.04"],
    )
    cropped_export(fig, output, "figure2_working")


def figure_3(tables, output):
    """Plot the Premier League club comparison and all within-season rank correlations."""
    clubs = pd.read_csv(tables / "figure3_epl_teams.csv", float_precision="round_trip")
    seasons = pd.read_csv(
        tables / "figure3_season_intervals.csv", float_precision="round_trip"
    )
    example = seasons.loc[seasons.dataset.eq("epl_2025_2026")].iloc[0]
    offsets = {
        **EPL_LABEL_ADJUSTMENTS,
        "NFO": (4, 1, "left", "center", False),
        "AVL": (16, -6, "left", "top", True),
        "BRE": (9.4, 3.42, "left", "bottom", True),
        "LEE": (3.54, -3.54, "left", "top", False),
        "CHE": (4.7, 1.71, "left", "bottom", False),
    }
    configure_plotting()
    plt.rcParams.update({"axes.edgecolor": "black"})
    fig = plt.figure(figsize=(10.14, 4.8), facecolor="white")
    team_heading(fig, 0.025, "a", "Premier League 2025/26")
    team_heading(fig, 0.690, "b", "Across league-seasons")
    ax = fig.add_axes([0.080, 0.23, 0.540, 0.64])
    limits = (-0.039, 0.059)
    if not clubs.value_centered.between(*limits).all():
        raise ValueError("A club falls outside the horizontal limits of Figure 3a")
    ax.set(
        xlim=limits,
        ylim=(-1.18, 1.24),
        xticks=[-0.02, 0, 0.02, 0.04],
        yticks=[-1, -0.5, 0, 0.5, 1],
    )
    ax.tick_params(labelsize=10.8, length=3)
    ax.grid(color=LIGHT, linewidth=0.6)
    ax.axhline(0, color="#BDC5CB", linewidth=0.8)
    ax.axvline(0, color="#BDC5CB", linewidth=0.8)
    ax.set_xlabel(
        "Expected press value (relative to league mean)", fontsize=11.2, labelpad=7
    )
    ax.set_ylabel("Expected-goal difference per match", fontsize=11.2, labelpad=7)
    ax.text(
        0.025,
        0.975,
        rf"$\rho = {example.rho:.2f}$  (95% CI {example.ci_low:.2f}–{example.ci_high:.2f})",
        transform=ax.transAxes,
        fontsize=10.8,
        va="top",
    )
    ax.scatter(
        clubs.value_centered,
        clubs.xgd_per_match,
        s=31,
        c=NAVY,
        edgecolor="white",
        linewidth=0.65,
        zorder=4,
    )
    # The line is a least-squares fit; the annotation reports the rank correlation.
    fit = np.polynomial.Polynomial.fit(
        clubs.value_centered.to_numpy(), clubs.xgd_per_match, 1
    )
    ends = np.asarray(ax.get_xlim())
    (line,) = ax.plot(ends, fit(ends), color="#A7AFB5", lw=0.85, zorder=2)
    labels_without_overlap(
        ax,
        list(zip(clubs.value_centered, clubs.xgd_per_match)),
        clubs.display_name.tolist(),
        10.8,
        avoid_line=line,
        adjustments=offsets,
    )
    check_team_label_layout(ax, clubs)
    draw_total_panel(fig, seasons)
    finish(
        fig,
        output / "figure3_working.png",
        10.8,
        dpi=425,
        export_pdf=True,
        vertical_padding=0.06,
    )


def figure_4(tables, output):
    """Draw the Premier League positioning experiment with its fixture-bootstrap interval and player cutoffs."""
    clubs = pd.read_csv(tables / "high_clubs.csv")
    players = pd.read_csv(tables / "high_players.csv")
    names = pd.read_csv(
        tables / "figure3_epl_teams.csv", usecols=["team_id", "display_name"]
    )
    clubs = clubs.merge(names, on="team_id", validate="one_to_one")
    interval = json.loads((tables / "club_interval.json").read_text())
    caption = (tables / "figure4_broad_regain_caption.tex").read_text().strip()
    draw_positioning(output, players, clubs, interval, caption)


def draw_figure(number, tables, output):
    """Render a numbered main-paper figure to a vector PDF and 425 DPI PNG in output."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # The originals were rendered in separate processes. Isolate global Matplotlib state for each figure.
    with plt.rc_context(rc=plt.rcParamsDefault):
        if number == 1:
            draw_network_schematic(
                output, "blueslate_warmcopper", with_prediction=True, paper_ready=True
            )
        else:
            {2: figure_2, 3: figure_3, 4: figure_4}[number](Path(tables), output)
