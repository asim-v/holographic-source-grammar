from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ROOT / "manuscript" / "tables"


def test_rows(cohort: str) -> dict[str, pd.Series]:
    frame = pd.read_csv(ARTIFACTS / cohort / "composition_tests.csv")
    frame = frame.loc[frame["outcome"] == "pscs_demixed"].set_index("metric")
    return {
        "mse": frame.loc["mse_gain_source_vs_anonymous"],
        "rho": frame.loc["rho_gain_source_vs_anonymous"],
    }


def fmt_effect(value: float, metric: str) -> str:
    if metric == "mse":
        return f"{value:.3e}"
    return f"{value:.5f}"


def write_cohort_table() -> None:
    cohorts = [
        (
            "Full response",
            "SST",
            "planned first test",
            "holographic_source_composition_sst",
        ),
        (
            "Reliability-weighted",
            "SST",
            "reliability development",
            "holographic_reliability_shrinkage_sst",
        ),
        (
            "Reliability-weighted",
            "E-to-E",
            "held-out cohort",
            "holographic_etoe_confirmation",
        ),
    ]
    lines = [
        r"\begin{tabular}{@{}lllrrrr@{}}",
        r"\toprule",
        (
            r"Model & Data & Role & MSE improvement & MSE $q$ "
            r"& Correlation improvement & Correlation $q$ \\"
        ),
        r"\midrule",
    ]
    for estimator, cohort, role, directory in cohorts:
        rows = test_rows(directory)
        mse = rows["mse"]
        rho = rows["rho"]
        lines.append(
            f"{estimator} & {cohort} & {role} & "
            f"{fmt_effect(mse['mean_file_effect'], 'mse')} & "
            f"{mse['q_value']:.4g} & "
            f"{fmt_effect(rho['mean_file_effect'], 'rho')} & "
            f"{rho['q_value']:.4g} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (OUTPUT / "cohort_results.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_confirmation_table() -> None:
    path = ARTIFACTS / "holographic_etoe_confirmation"
    nulls = pd.read_csv(path / "source_identity_null_summary.csv").set_index(
        "metric"
    )
    loo = pd.read_csv(path / "leave_one_file_out.csv")
    loo = loo.loc[loo["outcome"] == "pscs_demixed"]
    ablation = pd.read_csv(path / "selective_ablation_test.csv").iloc[0]
    rows = test_rows("holographic_etoe_confirmation")
    metrics = [
        (
            "MSE gain",
            "mse_gain_source_vs_anonymous",
            rows["mse"],
            "mse",
        ),
        (
            r"Waveform $\Delta\rho$",
            "rho_gain_source_vs_anonymous",
            rows["rho"],
            "rho",
        ),
    ]
    lines = [
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        (
            r"Measure & Mean improvement & Files improved & Exact $p$ "
            r"& Corrected $q$ & Smallest LOO \\"
        ),
        r"\midrule",
    ]
    for label, metric, row, kind in metrics:
        metric_loo = loo.loc[loo["metric"] == metric, "leave_one_out_mean"]
        lines.append(
            f"{label} & {fmt_effect(row['mean_file_effect'], kind)} & "
            f"{int(row['n_files'] if kind == 'mse' else 9)}/"
            f"{int(row['n_files'])} & "
            f"{row['p_value']:.6f} & {row['q_value']:.6f} & "
            f"{fmt_effect(metric_loo.min(), kind)} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            (
                r"MSE cost of using the wrong neuron & "
                f"{ablation['mean_file_effect']:.3e} & 10/10 & "
                f"{ablation['p_value']:.6f} & -- & -- \\\\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    (OUTPUT / "confirmation_results.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    identity_lines = [
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Measure & Observed $z$ & Familywise threshold & Familywise $p$ \\",
        r"\midrule",
        (
            f"MSE gain & "
            f"{nulls.loc['mse_gain_source_vs_anonymous', 'observed_z']:.2f} & "
            f"{nulls.loc['mse_gain_source_vs_anonymous', 'familywise_99_threshold_z']:.2f} & "
            f"{nulls.loc['mse_gain_source_vs_anonymous', 'familywise_p_value']:.3f} \\\\"
        ),
        (
            r"Waveform $\Delta\rho$ & "
            f"{nulls.loc['rho_gain_source_vs_anonymous', 'observed_z']:.2f} & "
            f"{nulls.loc['rho_gain_source_vs_anonymous', 'familywise_99_threshold_z']:.2f} & "
            f"{nulls.loc['rho_gain_source_vs_anonymous', 'familywise_p_value']:.3f} \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    (OUTPUT / "identity_null_results.tex").write_text(
        "\n".join(identity_lines),
        encoding="utf-8",
    )


def write_reduced_model_table() -> None:
    path = ARTIFACTS / "holographic_feedback_analyses"
    performance = pd.read_csv(path / "file_model_performance.csv")
    tests = pd.read_csv(path / "file_comparison_tests.csv")
    performance = performance.loc[
        performance["outcome"].eq("pscs_demixed")
    ]
    tests = tests.loc[tests["outcome"].eq("pscs_demixed")]
    labels = [
        ("Power only", "power_only", None),
        ("Source gain", "gain_only", "gain_vs_power"),
        (
            r"Source gain $+$ latency",
            "gain_latency",
            "gain_latency_vs_power",
        ),
        ("Full waveform", "full_waveform", "full_vs_power"),
    ]
    lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        (
            r"Single-neuron model & Mean MSE & MSE gain & Mean $\rho$ "
            r"& $\Delta\rho$ \\"
        ),
        r"\midrule",
    ]
    for label, model, comparison in labels:
        rows = performance.loc[performance["model"].eq(model)]
        mean_mse = rows["mse"].mean()
        mean_rho = rows["rho"].mean()
        if comparison is None:
            mse_gain = "--"
            rho_gain = "--"
        else:
            selected = tests.loc[tests["comparison"].eq(comparison)].set_index(
                "metric"
            )
            mse_gain = f"{selected.loc['mse_gain', 'mean_effect']:.3e}"
            rho_gain = f"{selected.loc['rho_gain', 'mean_effect']:.5f}"
        lines.append(
            f"{label} & {mean_mse:.3e} & {mse_gain} & "
            f"{mean_rho:.4f} & {rho_gain}" + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (OUTPUT / "reduced_model_results.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_cohort_table()
    write_confirmation_table()
    write_reduced_model_table()
    print(f"Wrote LaTeX tables to {OUTPUT}")


if __name__ == "__main__":
    main()
