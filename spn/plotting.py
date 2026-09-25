"""Plots for the included SPN model and a user's match outputs."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from .model import SPNModel
from .output import _clean_json
from . import analysis


PLOT_STEMS = {
    "importance": "fig_current_spn_feature_importance",
    "drivers": "fig_current_spn_vt_drivers",
    "teams": "fig_current_spn_press_intensity_efficiency",
}
TABLE_NAMES = {
    "importance": "current_spn_feature_gain_importance.csv",
    "drivers": "current_spn_vt_driver_associations.csv",
    "teams": "current_spn_press_intensity_efficiency.csv",
}
FORMAT_DIRS = {"png": "PNG", "jpeg": "JPEG", "pdf": "PDF", "svg": "SVG"}


def generate_plots(
    output_dir: str | Path,
    *,
    input_source: str | Path | None = None,
    features_source: str | Path | None = None,
    model: SPNModel | None = None,
    plots: tuple[str, ...] | list[str] = ("importance", "drivers", "teams"),
    formats: tuple[str, ...] | list[str] = ("png", "jpeg", "pdf", "svg"),
    dpi: int = 300,
    terminal_weight: float = 0.5,
    bootstrap_iterations: int = 2000,
    random_state: int = 20260824,
    overwrite: bool = False,
) -> dict:
    """Generate the three supported plots using only public model/CSV exports.

    Empty or insufficient datasets are recorded as skipped, never filled with
    invented values. Malformed or mismatched input raises an explicit error.
    """
    plots = list(dict.fromkeys(plots))
    formats = list(dict.fromkeys(formats))
    if not plots or any(name not in PLOT_STEMS for name in plots):
        raise ValueError("plots must select importance, drivers and/or teams")
    if not formats or any(name not in FORMAT_DIRS for name in formats):
        raise ValueError("formats must select png, jpeg, pdf and/or svg")
    if dpi <= 0 or bootstrap_iterations < 1 or not np.isfinite(terminal_weight) or not 0 <= terminal_weight <= 1:
        raise ValueError("DPI and bootstrap iterations must be positive; terminal weight must be in [0, 1]")
    if any(name != "importance" for name in plots) and input_source is None:
        raise ValueError("drivers and teams require public prediction outputs and exported features")
    output = Path(output_dir)
    known = [output / "manifest.json"]
    for name in plots:
        known.append(output / "tables" / TABLE_NAMES[name])
        known.extend(output / folder / f"{PLOT_STEMS[name]}.{ext}" for ext, folder in FORMAT_DIRS.items())
    existing = [path for path in known if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("outputs already exist; use --overwrite or a new output directory")

    frozen_model = model or SPNModel()
    data = {}
    audit = {}
    statuses = {}
    if "importance" in plots:
        data["importance"] = analysis.gain_table(frozen_model.model, frozen_model.features)
    if any(name != "importance" for name in plots):
        try:
            predictions, features, audit = analysis.load_analysis_inputs(
                input_source, frozen_model, features_source=features_source,
            )
        except analysis.InsufficientDataError as error:
            for name in plots:
                if name != "importance":
                    statuses[name] = {"status": "skipped", "reason": str(error)}
        else:
            if "drivers" in plots:
                levels = [feature for feature in frozen_model.features if not feature.startswith("d_")]
                try:
                    drivers, _, driver_audit = analysis.driver_table(predictions, features, levels)
                except analysis.InsufficientDataError as error:
                    statuses["drivers"] = {"status": "skipped", "reason": str(error)}
                else:
                    data["drivers"] = drivers
                    audit["driver_analysis"] = driver_audit
                    audit["driver_analysis"]["undefined_correlation_features"] = drivers.loc[
                        drivers.pearson_r_vt.isna(), "feature"
                    ].tolist()
            if "teams" in plots:
                steps = analysis.build_step_values(predictions)
                sequences = analysis.build_sequence_values(steps)
                teams, _ = analysis.team_efficiency_tables(
                    sequences, [terminal_weight], baseline_weight=terminal_weight,
                )
                intervals = analysis.bootstrap_team_efficiency(
                    sequences, terminal_weight=terminal_weight,
                    iterations=bootstrap_iterations, random_state=random_state,
                )
                teams = teams.merge(intervals, on="press_team", validate="one_to_one")
                for side in ("low", "high"):
                    teams[f"relative_efficiency_ci_{side}_per_100"] = 100 * teams[f"relative_efficiency_ci_{side}"]
                data["teams"], audit["team_analysis"] = analysis.build_team_table(steps, teams, features)

    output.mkdir(parents=True, exist_ok=True)
    # Clear only requested, known outputs when explicitly replacing a report.
    # This also prevents a skipped plot from leaving a stale image behind.
    if overwrite:
        for path in existing:
            path.unlink()
    for name in plots:
        if name not in data:
            continue
        table = data[name]
        if name == "importance":
            figure = plot_gain(table)
        elif name == "drivers":
            figure = plot_drivers(table, audit["driver_analysis"]["joint_r2"])
        else:
            figure = plot_intensity_efficiency(table, audit["team_analysis"])
        paths = []
        try:
            for extension in formats:
                path = output / FORMAT_DIRS[extension] / f"{PLOT_STEMS[name]}.{extension}"
                path.parent.mkdir(parents=True, exist_ok=True)
                options = {"format": extension, "bbox_inches": "tight", "facecolor": "white", "dpi": dpi}
                if extension == "jpeg":
                    options["pil_kwargs"] = {"quality": 95, "subsampling": 0}
                figure.savefig(path, **options)
                paths.append(path)
        finally:
            plt.close(figure)
        table_path = output / "tables" / TABLE_NAMES[name]
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(table_path, index=False)
        paths.append(table_path)
        statuses[name] = {
            "status": "generated",
            "outputs": [
                {"path": str(path.relative_to(output)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in paths
            ],
        }
    manifest = _clean_json({
        "schema_version": "spn-plots/1.0",
        "model": {
            "model_id": frozen_model.config["model_id"],
            "model_version": frozen_model.config["model_version"],
            "probability_layer": frozen_model.config["probability_layer"],
            "model_artifact_sha256": frozen_model.config["artifact_sha256"],
        },
        "plots": statuses,
        "audit": audit,
        "definitions": {
            "importance": "mean split gain normalized across the model features",
            "drivers": "Pearson association of adjacent level-feature changes with forward change in p_success - p_fail; descriptive, not causal",
            "intensity": "sum of P_total over eligible anchors / team matches containing eligible sequences",
            "efficiency": "100 * (team mean complete-sequence composite - sequence-weighted input-cohort mean)",
            "terminal_weight": terminal_weight,
            "team_bootstrap": {"unit": "match within team", "iterations": bootstrap_iterations, "random_state": random_state},
            "match_denominator": "evaluated matches containing eligible sequences; zero-eligible matches are not included",
        },
    })
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest


def plot_gain(table: pd.DataFrame) -> plt.Figure:
    ordered = table.sort_values("relative_gain", ascending=True, kind="stable")
    fig, axis = plt.subplots(figsize=(11.2, 8.6))
    bars = axis.barh(
        ordered["feature"],
        ordered["relative_gain"],
        color="#7c3aed",
        height=0.72,
    )
    axis.bar_label(
        bars,
        labels=[f"{value:.1%}" for value in ordered["relative_gain"]],
        padding=4,
        fontsize=8.5,
        color="#333333",
    )
    axis.set_xlim(0, float(ordered["relative_gain"].max()) * 1.18)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.grid(axis="x", color="#d1d5db", linewidth=0.7, alpha=0.65)
    axis.set_axisbelow(True)
    axis.set_xlabel("Relative mean gain")
    axis.set_ylabel("")
    axis.set_title(
        "SPN top-19 XGBoost feature importance",
        fontsize=15,
        pad=13,
    )
    fig.subplots_adjust(left=0.31, right=0.97, top=0.92, bottom=0.09)
    return fig


def plot_drivers(
    drivers: pd.DataFrame,
    joint_r2: float,
) -> plt.Figure:
    ordered = drivers.dropna(subset=["pearson_r_vt"]).sort_values("pearson_r_vt", ascending=True, kind="stable")
    values = ordered["pearson_r_vt"].to_numpy(float)
    colours = np.where(values >= 0.0, "#2563eb", "#dc2626")

    fig, axis = plt.subplots(figsize=(11.5, 7.8))
    bars = axis.barh(
        ordered["label"], values, color=colours, height=0.72, edgecolor="none"
    )
    axis.axvline(0.0, color="#111827", linewidth=0.9)
    axis.grid(axis="x", color="#d1d5db", linewidth=0.7, alpha=0.65)
    axis.set_axisbelow(True)

    span = max(float(values.max() - values.min()), 1e-8)
    padding = 0.015 * span
    for bar, value in zip(bars, values):
        axis.text(
            value + (padding if value >= 0 else -padding),
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.3f}",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=8.5,
            color="#333333",
        )
    low = min(float(values.min()), 0.0)
    high = max(float(values.max()), 0.0)
    axis.set_xlim(low - 0.12 * span, high + 0.12 * span)
    axis.set_xlabel(r"Pearson $r$ with transition $v_t$")
    axis.set_ylabel("")
    axis.set_title(
        rf"SPN level-feature changes associated with $v_t$ "
        rf"(joint $R^2$={joint_r2:.3f})",
        fontsize=14.5,
        pad=13,
    )
    axis.legend(
        handles=[
            Patch(facecolor="#2563eb", label=r"Positive association with $v_t$"),
            Patch(facecolor="#dc2626", label=r"Negative association with $v_t$"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.35, right=0.97, top=0.91, bottom=0.10)
    return fig


def _size_for_count(table: pd.DataFrame, count: float) -> float:
    count_min = float(table["n_sequences"].min())
    count_max = float(table["n_sequences"].max())
    if count_max == count_min:
        return float(table["marker_size"].iloc[0])
    return 150.0 + 370.0 * (count - count_min) / (count_max - count_min)


def plot_intensity_efficiency(table: pd.DataFrame, audit: dict[str, object]) -> plt.Figure:
    x = table["carrier_pressure_intensity"].to_numpy(float)
    y = table["relative_efficiency_per_100_sequences"].to_numpy(float)
    low = table["relative_efficiency_ci_low_per_100"].to_numpy(float)
    high = table["relative_efficiency_ci_high_per_100"].to_numpy(float)

    fig, axis = plt.subplots(figsize=(12.2, 8.2))
    axis.axvline(
        float(audit["intensity_median"]),
        color="#6b7280",
        linestyle=(0, (5, 4)),
        linewidth=1.0,
        zorder=0,
    )
    axis.axhline(0.0, color="#111827", linewidth=1.15, zorder=0)
    # Draw the actual percentile bounds. A percentile interval need not contain
    # its point estimate, and one-match samples can differ by rounding alone.
    axis.vlines(x, low, high, colors="#94a3b8", linewidths=0.9, alpha=0.70, zorder=1)
    for bound in (low, high):
        axis.plot(x, bound, linestyle="none", marker="_", markersize=4.4,
                  markeredgewidth=0.9, color="#94a3b8", alpha=0.70, zorder=1)
    scatter = axis.scatter(
        x,
        y,
        s=table["marker_size"],
        c=table["successful_sequence_rate"],
        cmap="Blues",
        edgecolors="white",
        linewidths=1.15,
        alpha=0.92,
        zorder=2,
    )

    labels = [
        axis.text(
            x_value,
            y_value,
            team,
            fontsize=8.2,
            color="#172033",
            ha="center",
            va="center",
            zorder=3,
        )
        for x_value, y_value, team in zip(x, y, table["press_team"])
    ]
    try:
        from adjustText import adjust_text

        adjust_text(
            labels,
            x=x,
            y=y,
            ax=axis,
            expand=(1.10, 1.22),
            force_text=(0.35, 0.45),
            force_static=(0.12, 0.20),
            arrowprops={"arrowstyle": "-", "color": "#94a3b8", "lw": 0.55},
        )
    except ImportError:
        for label in labels:
            label.set_ha("left")
            label.set_position((label.get_position()[0] + 0.08, label.get_position()[1]))

    axis.grid(color="#d1d5db", linewidth=0.65, alpha=0.55)
    axis.set_axisbelow(True)
    axis.set_xlabel("Per-match carrier-pressure intensity")
    axis.set_ylabel("Relative composite efficiency per 100 sequences")
    axis.set_title(
        "SPN pressing intensity and relative composite efficiency",
        fontsize=14.5,
        pad=13,
    )

    x_span = max(float(np.ptp(x)), 1e-8)
    y_span = max(float(np.max(high) - np.min(low)), 1e-8)
    axis.set_xlim(float(np.min(x) - 0.08 * x_span), float(np.max(x) + 0.08 * x_span))
    axis.set_ylim(float(np.min(low) - 0.09 * y_span), float(np.max(high) + 0.09 * y_span))

    colorbar = fig.colorbar(scatter, ax=axis, pad=0.014, fraction=0.045)
    colorbar.set_label("Successful sequence rate")
    colorbar.ax.yaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0)
    )

    representative_counts = np.quantile(table["n_sequences"], [0.1, 0.9])
    rounding = 10 ** max(0, int(np.floor(np.log10(max(1, table["n_sequences"].max())))) - 1)
    representative_counts = np.unique(np.clip(
        np.round(representative_counts / rounding) * rounding,
        table["n_sequences"].min(), table["n_sequences"].max(),
    )).astype(int)
    size_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="#93c5fd",
            markeredgecolor="white",
            markersize=np.sqrt(_size_for_count(table, count)),
            label=f"{count:,}",
        )
        for count in representative_counts
    ]
    axis.legend(
        handles=size_handles,
        title="Evaluated sequences",
        loc="best",
        frameon=True,
        facecolor="white",
        edgecolor="#d1d5db",
        fontsize=8.3,
        title_fontsize=8.3,
    )

    fig.subplots_adjust(left=0.10, right=0.89, top=0.91, bottom=0.11)
    return fig
