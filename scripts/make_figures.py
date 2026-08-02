from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import patheffects
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ROOT / "manuscript" / "figures"

INK = "#173042"
NAVY = "#16425B"
BLUE = "#2F6690"
TEAL = "#2A9D8F"
CORAL = "#E76F51"
GOLD = "#E9C46A"
MUTED = "#687780"
LIGHT = "#EDF2F4"
GRID = "#CAD5DA"
WHITE = "#FFFFFF"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.titleweight": "bold",
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(
        OUTPUT / f"{stem}.png",
        dpi=320,
        bbox_inches="tight",
    )
    plt.close(fig)


def panel_label(
    ax: plt.Axes,
    label: str,
    x: float = -0.08,
    y: float = 1.04,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )


def clean_axis(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        ax.grid(
            axis=grid_axis,
            color=GRID,
            linewidth=0.6,
            alpha=0.7,
            zorder=0,
        )


def file_label(filename: str) -> str:
    match = re.search(r"(\d{8})_cell(\d+)", filename)
    if not match:
        return filename
    date, cell = match.groups()
    return f"{date[4:8]}-{date[0:2]}-{date[2:4]} c{cell}"


def load_summary(name: str) -> dict:
    path = ARTIFACTS / name / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def psc_kernel(time: np.ndarray, rise: float, decay: float) -> np.ndarray:
    shifted = np.maximum(time, 0.0)
    waveform = -(1.0 - np.exp(-shifted / rise)) * np.exp(
        -shifted / decay
    )
    return waveform / np.max(np.abs(waveform))


def figure_grammar() -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.7, 4.05),
        gridspec_kw={"wspace": 0.34},
    )
    time = np.linspace(0, 40, 300)
    shared = psc_kernel(time, 2.5, 15.0)
    deviations = [
        0.22 * psc_kernel(time, 1.4, 8.0),
        -0.12 * psc_kernel(time, 4.5, 20.0),
        0.10
        * (
            psc_kernel(time, 2.0, 12.0)
            - psc_kernel(time, 5.0, 18.0)
        ),
    ]
    colors = [CORAL, TEAL, BLUE]

    ax = axes[0]
    ax.plot(time, shared, color=INK, linewidth=2.3, label=r"shared $g_p(t)$")
    for index, (deviation, color) in enumerate(
        zip(deviations, colors, strict=True), start=1
    ):
        ax.plot(
            time,
            deviation,
            color=color,
            linewidth=1.7,
            label=rf"neuron {index} difference $d_{{{index},p}}(t)$",
        )
    ax.axhline(0, color=GRID, linewidth=0.8)
    ax.set(
        xlabel="Time after stimulation (ms)",
        ylabel="Normalized current",
        title="Learn single-neuron responses",
    )
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    clean_axis(ax)
    panel_label(ax, "A", x=-0.15, y=1.08)

    ax = axes[1]
    ax.axis("off")
    ax.set_title("Retain repeatable differences", pad=8)
    y_top = 0.82
    for x, label, color in [
        (0.17, "half A", BLUE),
        (0.83, "half B", TEAL),
    ]:
        circle = patches.Circle(
            (x, y_top),
            0.11,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor="none",
            alpha=0.95,
        )
        ax.add_patch(circle)
        ax.text(
            x,
            y_top,
            label,
            color=WHITE,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
        )
    arrow = patches.FancyArrowPatch(
        (0.30, y_top),
        (0.70, y_top),
        transform=ax.transAxes,
        arrowstyle="<->",
        mutation_scale=12,
        linewidth=1.6,
        color=INK,
    )
    ax.add_patch(arrow)
    ax.text(
        0.50,
        y_top + 0.08,
        "agreement",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
    )
    formula = (
        r"$\lambda_p = \left["
        r"\frac{2\sum_{s,t}A'_{spt}B'_{spt}}"
        r"{\sum_{s,t}(A'_{spt})^2+\sum_{s,t}(B'_{spt})^2}"
        r"\right]_{0}^{1}$"
    )
    ax.text(
        0.50,
        0.54,
        formula,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(
        0.50,
        0.31,
        r"$r_{s,p}(t)=g_p(t)+\lambda_p d_{s,p}(t)$",
        transform=ax.transAxes,
        ha="center",
        fontsize=12,
        fontweight="bold",
        color=NAVY,
    )
    ax.text(
        0.50,
        0.12,
        "Single-neuron trials only\nGroup trials do not tune the model",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.5,
        color=MUTED,
    )
    panel_label(ax, "B", x=-0.15, y=1.08)

    ax = axes[2]
    lam = 0.58
    responses = [shared + lam * deviation for deviation in deviations]
    prediction = np.sum(responses, axis=0)
    anonymous = 3 * shared
    observed = prediction + 0.035 * np.sin(time / 1.8) * np.exp(-time / 20)
    ax.plot(
        time,
        observed,
        color=INK,
        linewidth=2.4,
        label="recorded group response",
        zorder=4,
    )
    ax.plot(
        time,
        prediction,
        color=TEAL,
        linewidth=2.0,
        linestyle="--",
        label=r"$\sum_s r_{s,p_s}(t)$",
    )
    ax.plot(
        time,
        anonymous,
        color=CORAL,
        linewidth=1.6,
        linestyle=":",
        label="power-only baseline",
    )
    ax.axhline(0, color=GRID, linewidth=0.8)
    ax.set(
        xlabel="Time after stimulation (ms)",
        ylabel="Normalized current",
        title="Predict held-out group responses",
    )
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    clean_axis(ax)
    panel_label(ax, "C", x=-0.15, y=1.08)

    fig.subplots_adjust(top=0.76)
    fig.suptitle(
        "Model construction and prediction",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    save_figure(fig, "figure1_grammar")


def stage_box(
    ax: plt.Axes,
    x: float,
    width: float,
    color: str,
    number: str,
    title: str,
    subtitle: str,
    body: list[tuple[str, str]],
    verdict: str,
    verdict_color: str,
) -> None:
    y, height = 0.10, 0.78
    box = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor=WHITE,
        edgecolor=GRID,
        linewidth=1.2,
    )
    ax.add_patch(box)
    ax.add_patch(
        patches.Rectangle(
            (x, y + height - 0.075),
            width,
            0.075,
            transform=ax.transAxes,
            color=color,
            clip_on=False,
        )
    )
    ax.text(
        x + 0.035,
        y + height - 0.037,
        number,
        transform=ax.transAxes,
        color=WHITE,
        fontweight="bold",
        fontsize=10,
        va="center",
    )
    ax.text(
        x + 0.035,
        y + height - 0.125,
        title,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=10.5,
        va="top",
    )
    ax.text(
        x + 0.035,
        y + height - 0.19,
        subtitle,
        transform=ax.transAxes,
        fontsize=7.7,
        color=MUTED,
        va="top",
    )
    body_y = y + height - 0.30
    for label, value in body:
        ax.text(
            x + 0.035,
            body_y,
            label,
            transform=ax.transAxes,
            fontsize=7.8,
            color=MUTED,
            va="center",
        )
        ax.text(
            x + width - 0.035,
            body_y,
            value,
            transform=ax.transAxes,
            fontsize=8.2,
            fontweight="bold",
            ha="right",
            va="center",
        )
        body_y -= 0.075
    verdict_box = patches.FancyBboxPatch(
        (x + 0.035, y + 0.045),
        width - 0.07,
        0.085,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        transform=ax.transAxes,
        facecolor=verdict_color,
        edgecolor="none",
    )
    ax.add_patch(verdict_box)
    ax.text(
        x + width / 2,
        y + 0.087,
        verdict,
        transform=ax.transAxes,
        color=WHITE,
        fontsize=8.2,
        fontweight="bold",
        ha="center",
        va="center",
    )


def figure_evidence_path() -> None:
    original = load_summary("holographic_source_composition_sst")
    development = load_summary("holographic_reliability_shrinkage_sst")
    confirmation = load_summary("holographic_etoe_confirmation")
    fig, ax = plt.subplots(figsize=(11.7, 4.3))
    ax.axis("off")
    ax.set_title(
        "How the model was developed and tested",
        fontsize=14,
        fontweight="bold",
        pad=8,
    )

    xs = [0.025, 0.35, 0.675]
    width = 0.30
    stage_box(
        ax,
        xs[0],
        width,
        CORAL,
        "01",
        "Planned SST test",
        "Full neuron-specific responses",
        [
            ("Files", f"{original['files_requested']} SST-to-E"),
            (
                "Single-neuron trials",
                f"{original['trial_counts']['single_source']:,}",
            ),
            (
                "Eligible group trials",
                f"{original['trial_counts']['eligible_primary_ensemble']:,}",
            ),
            ("MSE", "q = 0.00391  pass"),
            ("Waveform rho", "q = 0.682  no gain"),
        ],
        "MSE GAIN  |  NO SHAPE GAIN",
        CORAL,
    )
    stage_box(
        ax,
        xs[1],
        width,
        GOLD,
        "02",
        "Develop rule in SST",
        "Reliability from trial halves",
        [
            ("Files", f"{development['files']} SST-to-E"),
            (
                "Eligible group trials",
                f"{development['eligible_primary_ensemble_trials']:,}",
            ),
            (
                "Mean reliability",
                f"{development['mean_power_shrinkage']['pscs_demixed']:.3f}",
            ),
            ("MSE", "q = 0.00391  pass"),
            ("Waveform rho", "q = 0.0469  pass"),
        ],
        "RULE LOCKED BEFORE NEW DATA",
        "#C08A19",
    )
    stage_box(
        ax,
        xs[2],
        width,
        TEAL,
        "03",
        "Held-out E-to-E cohort",
        "Same rule, no tuning",
        [
            ("Files", f"{confirmation['files_requested']} E-to-E"),
            (
                "Single-neuron trials",
                f"{confirmation['trial_counts']['single_source']:,}",
            ),
            (
                "Eligible group trials",
                f"{confirmation['trial_counts']['eligible_primary_ensemble']:,}",
            ),
            ("MSE", "q = 0.00195  pass"),
            ("Waveform rho", "q = 0.00879  pass"),
        ],
        "FILE-LEVEL TEST PASSED",
        TEAL,
    )

    for start, end in [(xs[0] + width, xs[1]), (xs[1] + width, xs[2])]:
        arrow = patches.FancyArrowPatch(
            (start + 0.006, 0.49),
            (end - 0.006, 0.49),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.5,
            color=INK,
        )
        ax.add_patch(arrow)

    ax.text(
        0.5,
        0.015,
        "Datasets are not pooled for significance; the E-to-E result cannot revise the original SST/PV decision.",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.3,
        color=MUTED,
    )
    save_figure(fig, "figure2_evidence_path")


def figure_primary_effects() -> None:
    path = ARTIFACTS / "holographic_etoe_confirmation"
    data = pd.read_csv(path / "file_effects.csv")
    data = data.loc[data["outcome"] == "pscs_demixed"].copy()
    data["label"] = data["file"].map(file_label)
    data = data.reset_index(drop=True)
    tests = pd.read_csv(path / "composition_tests.csv")
    tests = tests.loc[tests["outcome"] == "pscs_demixed"].set_index("metric")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.7, 4.9),
        gridspec_kw={"wspace": 0.46},
    )
    y = np.arange(len(data))

    ax = axes[0]
    values = data["mse_gain_source_vs_anonymous"].to_numpy() * 1e6
    ax.hlines(y, 0.4, values, color=GRID, linewidth=1.2, zorder=1)
    ax.scatter(
        values,
        y,
        s=38,
        color=TEAL,
        edgecolor=WHITE,
        linewidth=0.6,
        zorder=3,
    )
    ax.set_xscale("log")
    ax.set_yticks(y, data["label"])
    ax.invert_yaxis()
    mse_test = tests.loc["mse_gain_source_vs_anonymous"]
    ax.axvline(
        mse_test["mean_file_effect"] * 1e6,
        color=CORAL,
        linestyle="--",
        linewidth=1.4,
    )
    ax.set(
        xlabel=r"MSE improvement over baseline ($\times 10^{-6}$; log scale)",
        title="Every file improved in MSE",
    )
    ax.text(
        0.98,
        0.98,
        (
            f"dashed mean = {mse_test['mean_file_effect'] * 1e6:.2f}"
            f" $\\times 10^{{-6}}$\n"
            f"exact p = {mse_test['p_value']:.6f}\n"
            f"BH q = {mse_test['q_value']:.6f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    clean_axis(ax, "x")
    panel_label(ax, "A")

    ax = axes[1]
    values = data["rho_gain_source_vs_anonymous"].to_numpy() * 1e3
    colors = np.where(values >= 0, TEAL, CORAL)
    ax.hlines(y, 0, values, color=GRID, linewidth=1.2, zorder=1)
    ax.scatter(
        values,
        y,
        s=38,
        color=colors,
        edgecolor=WHITE,
        linewidth=0.6,
        zorder=3,
    )
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_yticks(y, data["label"])
    ax.invert_yaxis()
    rho_test = tests.loc["rho_gain_source_vs_anonymous"]
    ax.axvline(
        rho_test["mean_file_effect"] * 1e3,
        color=CORAL,
        linestyle="--",
        linewidth=1.4,
    )
    ax.set(
        xlabel=r"Waveform-correlation gain ($\times 10^{-3}$)",
        title="Nine of ten files improved in shape",
    )
    ax.text(
        0.98,
        0.98,
        (
            f"dashed mean = {rho_test['mean_file_effect']:.6f}\n"
            f"exact p = {rho_test['p_value']:.6f}\n"
            f"BH q = {rho_test['q_value']:.6f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    clean_axis(ax, "x")
    panel_label(ax, "B")

    fig.suptitle(
        "Prospectively held-out E-to-E cohort: results for each file",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    save_figure(fig, "figure3_primary_effects")


def figure_identity_and_ablation() -> None:
    path = ARTIFACTS / "holographic_etoe_confirmation"
    nulls = pd.read_csv(path / "source_identity_nulls.csv.gz")
    summary = pd.read_csv(path / "source_identity_null_summary.csv").set_index(
        "metric"
    )
    aggregate = (
        nulls.groupby("repetition", sort=True)[
            [
                "mse_gain_source_vs_anonymous",
                "rho_gain_source_vs_anonymous",
            ]
        ]
        .mean()
        .reset_index()
    )
    ablation = pd.read_csv(path / "selective_ablation_file_effects.csv")
    ablation["label"] = ablation["file"].map(file_label)

    fig = plt.figure(figsize=(11.5, 4.25))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.42)
    metrics = [
        ("mse_gain_source_vs_anonymous", "Shuffle test: MSE", TEAL),
        ("rho_gain_source_vs_anonymous", "Shuffle test: correlation", BLUE),
    ]
    for index, (metric, title, color) in enumerate(metrics):
        ax = fig.add_subplot(grid[0, index])
        values = aggregate[metric].to_numpy()
        z_values = (values - values.mean()) / values.std(ddof=1)
        observed_z = summary.loc[metric, "observed_z"]
        threshold = summary.loc[metric, "familywise_99_threshold_z"]
        ax.hist(
            z_values,
            bins=34,
            density=True,
            color=color,
            alpha=0.82,
            edgecolor=WHITE,
            linewidth=0.35,
        )
        ax.axvline(
            threshold,
            color=CORAL,
            linestyle="--",
            linewidth=1.5,
            label=f"familywise threshold = {threshold:.2f}",
        )
        ax.annotate(
            f"observed z = {observed_z:.1f}\n(off scale)",
            xy=(3.65, 0.80),
            xycoords=("data", "axes fraction"),
            xytext=(3.35, 0.82),
            textcoords=("data", "axes fraction"),
            arrowprops={
                "arrowstyle": "-|>",
                "color": INK,
                "linewidth": 1.3,
            },
            fontsize=8,
            fontweight="bold",
            ha="right",
        )
        ax.set(
            xlim=(-4.2, 4.2),
            xlabel="Standardized shuffled-identity effect",
            ylabel="Density" if index == 0 else None,
            title=title,
        )
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
        clean_axis(ax)
        panel_label(ax, chr(ord("A") + index))

    ax = fig.add_subplot(grid[0, 2])
    values = ablation["mean_replacement_mse_cost"].to_numpy() * 1e6
    y = np.arange(len(ablation))
    ax.hlines(y, 0.06, values, color=GRID, linewidth=1.1)
    ax.scatter(
        values,
        y,
        s=34,
        color=GOLD,
        edgecolor=INK,
        linewidth=0.45,
        zorder=3,
    )
    ax.set_xscale("log")
    ax.set_yticks(y, ablation["label"])
    ax.invert_yaxis()
    ax.set(
        xlabel=r"MSE increase after replacement ($\times 10^{-6}$)",
        title="Wrong neuron raises error",
    )
    ax.text(
        0.98,
        0.03,
        "10/10 positive\nexact p = 0.000977",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    clean_axis(ax, "x")
    panel_label(ax, "C")

    fig.suptitle(
        "Predictions require the correct neuron identity",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    save_figure(fig, "figure4_identity_ablation")


def figure_reliability() -> None:
    etoe_path = ARTIFACTS / "holographic_etoe_confirmation"
    sst_path = ARTIFACTS / "holographic_reliability_shrinkage_sst"
    etoe = pd.read_csv(etoe_path / "power_shrinkage.csv")
    sst = pd.read_csv(sst_path / "power_shrinkage.csv")
    etoe = etoe.loc[etoe["outcome"] == "pscs_demixed"].copy()
    sst = sst.loc[sst["outcome"] == "pscs_demixed"].copy()
    etoe["label"] = etoe["file"].map(file_label)

    matrix = etoe.pivot(index="label", columns="power", values="shrinkage")
    matrix = matrix.sort_index()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.65),
        gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.35},
    )

    ax = axes[0]
    image = ax.imshow(
        matrix.to_numpy(),
        aspect="auto",
        cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "reliability",
            [LIGHT, GOLD, CORAL, NAVY],
        ),
        vmin=0,
        vmax=1,
    )
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns.astype(int))
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    ax.set(
        xlabel="Stimulation power",
        title="E-to-E reliability by file and power",
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=WHITE if value > 0.58 else INK,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.042, pad=0.03)
    colorbar.set_label(r"Split-half $\lambda_p$")
    panel_label(ax, "A")

    ax = axes[1]
    for _, group in etoe.groupby("file"):
        ordered = group.sort_values("power")
        ax.plot(
            ordered["power"],
            ordered["shrinkage"],
            color=BLUE,
            linewidth=0.8,
            alpha=0.25,
        )
    cohort_styles = [
        (sst, CORAL, "SST development"),
        (etoe, TEAL, "Held-out E-to-E cohort"),
    ]
    for frame, color, label in cohort_styles:
        mean = frame.groupby("power")["shrinkage"].mean()
        ax.plot(
            mean.index,
            mean.values,
            color=color,
            linewidth=2.8,
            marker="o",
            markersize=5,
            label=label,
            path_effects=[
                patheffects.Stroke(linewidth=4.2, foreground=WHITE),
                patheffects.Normal(),
            ],
        )
    ax.set(
        xlim=(27, 63),
        ylim=(-0.04, 1.03),
        xticks=[30, 45, 60],
        xlabel="Stimulation power",
        ylabel=r"Split-half reliability $\lambda_p$",
        title="Reliability tends to rise with power",
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    clean_axis(ax, "y")
    panel_label(ax, "B")

    fig.suptitle(
        "The data decide how much neuron-specific detail to keep",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    save_figure(fig, "figure5_reliability")


def figure_reduced_models() -> None:
    path = ARTIFACTS / "holographic_feedback_analyses"
    performance = pd.read_csv(path / "file_model_performance.csv")
    labels = ["Gain", "Gain + latency", "Full waveform"]
    models = ["gain_only", "gain_latency", "full_waveform"]
    colors = [GOLD, BLUE, TEAL]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.7, 7.0),
        gridspec_kw={"wspace": 0.30, "hspace": 0.46},
    )
    panels = [
        (
            "pscs_demixed",
            "mse",
            "Demixed current: MSE",
            "MSE reduction from power-only (%)",
        ),
        (
            "pscs_demixed",
            "rho",
            "Demixed current: waveform shape",
            r"Correlation gain over power-only ($\times 10^{-3}$)",
        ),
        (
            "pscs",
            "mse",
            "Raw current: MSE",
            "MSE reduction from power-only (%)",
        ),
        (
            "pscs",
            "rho",
            "Raw current: waveform shape",
            r"Correlation gain over power-only ($\times 10^{-3}$)",
        ),
    ]
    for panel_index, (outcome, metric, title, ylabel) in enumerate(panels):
        ax = axes.flat[panel_index]
        frame = performance.loc[performance["outcome"] == outcome]
        pivot = frame.pivot(index="file", columns="model", values=metric)
        x = np.arange(len(models))
        baseline = pivot["power_only"].to_numpy()
        values = pivot[models].to_numpy()
        if metric == "mse":
            plotted = 100.0 * (baseline[:, None] - values) / baseline[:, None]
        else:
            plotted = 1000.0 * (values - baseline[:, None])
        for row in plotted:
            ax.plot(
                x,
                row,
                color=GRID,
                linewidth=0.9,
                alpha=0.85,
                zorder=1,
            )
        means = plotted.mean(axis=0)
        ax.scatter(
            np.tile(x, (len(plotted), 1)),
            plotted,
            s=12,
            color=np.tile(colors, len(plotted)),
            alpha=0.48,
            edgecolor="none",
            zorder=2,
        )
        ax.plot(x, means, color=INK, linewidth=2.0, zorder=3)
        ax.scatter(
            x,
            means,
            s=68,
            color=colors,
            edgecolor=WHITE,
            linewidth=1.0,
            zorder=4,
        )
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.set_xticks(x, labels)
        ax.set(title=title, ylabel=ylabel)
        ax.text(
            0.98,
            0.96,
            "mean: " + ", ".join(f"{value:.2f}" for value in means),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            color=MUTED,
        )
        clean_axis(ax, "y")
        panel_label(
            ax,
            chr(ord("A") + panel_index),
            x=-0.12,
            y=1.08,
        )
    fig.suptitle(
        "Source-specific strength and latency explain the predictive gain",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    save_figure(fig, "figure6_reduced_models")


def main() -> None:
    configure_style()
    figure_grammar()
    figure_evidence_path()
    figure_primary_effects()
    figure_identity_and_ablation()
    figure_reliability()
    figure_reduced_models()
    print(f"Wrote manuscript figures to {OUTPUT}")


if __name__ == "__main__":
    main()
