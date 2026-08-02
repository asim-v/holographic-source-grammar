from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from wetware_interp.holographic_source_composition import (
    baseline_correct_response,
    benjamini_hochberg,
    build_composition_context,
    build_gain_latency_token_model,
    build_gain_only_token_model,
    build_reliability_shrunk_token_model,
    composition_metrics,
    exact_sign_flip_test,
    load_holographic_recording,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "holographic_etoe_confirmation_manifest.json"
RAW_DIR = ROOT / "data" / "raw" / "holographic_circuitmap_etoe"
OUTPUT_DIR = ROOT / "artifacts" / "holographic_feedback_analyses"
OUTCOMES = ("pscs_demixed", "pscs")
MODEL_ORDER = (
    "power_only",
    "gain_only",
    "gain_latency",
    "full_waveform",
)
COMPARISONS = (
    ("gain_vs_power", "gain_only", "power_only"),
    ("gain_latency_vs_power", "gain_latency", "power_only"),
    ("full_vs_power", "full_waveform", "power_only"),
    ("full_vs_gain", "full_waveform", "gain_only"),
    ("full_vs_gain_latency", "full_waveform", "gain_latency"),
)
EFFECT_COLUMNS = (
    "mse_gain",
    "rho_gain",
    "integrated_error_gain",
    "peak_error_gain",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run post-feedback reduced-model and clustering sensitivities."
        )
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--maximum-latency-ms", type=float, default=2.0)
    parser.add_argument("--sample-rate-hz", type=float, default=20_000.0)
    parser.add_argument("--max-files", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.maximum_latency_ms < 0 or args.sample_rate_hz <= 0:
        raise ValueError("Latency and sample-rate arguments are invalid")
    maximum_shift_samples = int(
        round(args.maximum_latency_ms * args.sample_rate_hz / 1000.0)
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    files = list(manifest["files"])
    if args.max_files is not None:
        if args.max_files < 1:
            raise ValueError("Maximum file count must be positive")
        files = files[: args.max_files]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    performance_rows: list[dict[str, object]] = []
    parameter_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, object]] = []
    for file_index, item in enumerate(files):
        filename = str(item["name"])
        path = args.raw_dir / filename
        print(f"[{file_index + 1}/{len(files)}] {filename}", flush=True)
        metadata_rows.append(_metadata_row(path))
        recording = load_holographic_recording(path)
        for outcome in OUTCOMES:
            response = baseline_correct_response(getattr(recording, outcome))
            full_result = build_reliability_shrunk_token_model(
                recording.stimulus_matrix,
                response,
                recording.targets,
                minimum_single_trials=3,
            )
            full_model = full_result.token_model
            gain_result = build_gain_only_token_model(full_model)
            latency_result = build_gain_latency_token_model(
                full_model,
                maximum_shift_samples=maximum_shift_samples,
            )
            models = {
                "gain_only": gain_result.token_model,
                "gain_latency": latency_result.token_model,
                "full_waveform": full_model,
            }
            model_metrics: dict[str, pd.DataFrame] = {}
            for model_name, model in models.items():
                context = build_composition_context(
                    recording.stimulus_matrix,
                    response,
                    model,
                )
                model_metrics[model_name] = composition_metrics(context)
                del context
            reference = model_metrics["full_waveform"]
            performance_rows.append(
                _performance_row(
                    filename,
                    outcome,
                    "power_only",
                    reference,
                    source=False,
                )
            )
            for model_name in models:
                performance_rows.append(
                    _performance_row(
                        filename,
                        outcome,
                        model_name,
                        model_metrics[model_name],
                        source=True,
                    )
                )
            for model_name, parameters in (
                ("gain_only", gain_result.parameters),
                ("gain_latency", latency_result.parameters),
            ):
                frame = parameters.copy()
                frame.insert(0, "model", model_name)
                frame.insert(0, "outcome", outcome)
                frame.insert(0, "file", filename)
                parameter_frames.append(frame)
            del (
                response,
                full_result,
                full_model,
                gain_result,
                latency_result,
                models,
                model_metrics,
                reference,
            )
            gc.collect()
        del recording
        gc.collect()

    performance = pd.DataFrame(performance_rows)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    metadata = pd.DataFrame(metadata_rows)
    comparisons = _model_comparisons(performance)
    tests = _comparison_tests(comparisons, cluster="file")
    date_effects = (
        comparisons.groupby(
            ["date_proxy", "outcome", "comparison"],
            as_index=False,
        )[[*EFFECT_COLUMNS]]
        .mean()
        .merge(
            comparisons.groupby(
                ["date_proxy", "outcome", "comparison"],
                as_index=False,
            )["file"]
            .nunique()
            .rename(columns={"file": "n_files"}),
            on=["date_proxy", "outcome", "comparison"],
        )
    )
    date_tests = _comparison_tests(
        date_effects.rename(columns={"date_proxy": "file"}),
        cluster="date_proxy",
    )

    outputs = {
        "file_model_performance.csv": performance,
        "file_model_comparisons.csv": comparisons,
        "file_comparison_tests.csv": tests,
        "date_proxy_effects.csv": date_effects,
        "date_proxy_tests.csv": date_tests,
        "reduced_model_parameters.csv.gz": parameters,
        "metadata_audit.csv": metadata,
    }
    for filename, frame in outputs.items():
        frame.to_csv(
            args.output_dir / filename,
            index=False,
            compression=(
                {"method": "gzip", "mtime": 0}
                if filename.endswith(".gz")
                else None
            ),
        )
    summary = {
        "analysis": "holographic_feedback_analyses",
        "status": "post_feedback_sensitivity",
        "files": len(files),
        "models": list(MODEL_ORDER),
        "maximum_latency_ms": args.maximum_latency_ms,
        "sample_rate_hz": args.sample_rate_hz,
        "date_proxy_warning": (
            "Recording date is a conservative session proxy, not a verified "
            "animal or slice identifier."
        ),
        "biological_hierarchy": {
            "postsynaptic_cells": "one file per labeled recorded cell",
            "slices": "not available in the public file metadata",
            "animals": "not available in the public file metadata",
        },
        "outputs": [
            {"path": filename, "rows": len(frame)}
            for filename, frame in outputs.items()
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


def _performance_row(
    filename: str,
    outcome: str,
    model: str,
    metrics: pd.DataFrame,
    *,
    source: bool,
) -> dict[str, object]:
    suffix = "source" if source else "anonymous"
    return {
        "file": filename,
        "date_proxy": filename[:8],
        "outcome": outcome,
        "model": model,
        "eligible_group_trials": len(metrics),
        "mse": float(metrics[f"mse_{suffix}"].mean()),
        "rho": float(metrics[f"rho_{suffix}"].mean()),
        "integrated_error": float(
            metrics[f"integrated_error_{suffix}"].mean()
        ),
        "peak_error": float(metrics[f"peak_error_{suffix}"].mean()),
    }


def _model_comparisons(performance: pd.DataFrame) -> pd.DataFrame:
    indexed = performance.set_index(["file", "outcome", "model"])
    rows: list[dict[str, object]] = []
    for (file_name, outcome), group in performance.groupby(
        ["file", "outcome"], sort=False
    ):
        date_proxy = str(group["date_proxy"].iloc[0])
        for comparison, candidate, reference in COMPARISONS:
            candidate_row = indexed.loc[(file_name, outcome, candidate)]
            reference_row = indexed.loc[(file_name, outcome, reference)]
            rows.append(
                {
                    "file": file_name,
                    "date_proxy": date_proxy,
                    "outcome": outcome,
                    "comparison": comparison,
                    "mse_gain": float(
                        reference_row["mse"] - candidate_row["mse"]
                    ),
                    "rho_gain": float(
                        candidate_row["rho"] - reference_row["rho"]
                    ),
                    "integrated_error_gain": float(
                        reference_row["integrated_error"]
                        - candidate_row["integrated_error"]
                    ),
                    "peak_error_gain": float(
                        reference_row["peak_error"]
                        - candidate_row["peak_error"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _comparison_tests(
    effects: pd.DataFrame,
    *,
    cluster: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (outcome, comparison), group in effects.groupby(
        ["outcome", "comparison"], sort=False
    ):
        group_rows = []
        for metric in EFFECT_COLUMNS:
            result = exact_sign_flip_test(group[metric].to_numpy())
            group_rows.append(
                {
                    "outcome": outcome,
                    "comparison": comparison,
                    "metric": metric,
                    "cluster": cluster,
                    "n_clusters": result.n_files,
                    "mean_effect": result.observed_mean,
                    "p_value": result.p_value,
                    "positive_clusters": int(np.sum(group[metric] > 0)),
                }
            )
        q_values = benjamini_hochberg(
            [row["p_value"] for row in group_rows]
        )
        for row, q_value in zip(group_rows, q_values, strict=True):
            row["q_value"] = float(q_value)
            rows.append(row)
    return pd.DataFrame(rows)


def _metadata_row(path: Path) -> dict[str, object]:
    match = re.match(r"(?P<date>\d{8})_(?P<cell>cell\d+)", path.name)
    with h5py.File(path, "r") as handle:
        embedded_mouse_id = _matlab_string(
            handle,
            handle["ExpStruct"]["mouseID"],
        )
    return {
        "file": path.name,
        "date_proxy": match.group("date") if match else "",
        "postsynaptic_cell_label": match.group("cell") if match else "",
        "embedded_mouse_id": embedded_mouse_id,
        "verified_slice_id": "",
        "verified_animal_id": "",
        "hierarchy_status": "awaiting_source-author clarification",
    }


def _matlab_string(handle: h5py.File, dataset: h5py.Dataset) -> str:
    values = dataset[()]
    if values.dtype.kind == "O":
        references = [value for value in values.flat if value]
        if not references:
            return ""
        values = handle[references[0]][()]
    return "".join(chr(int(value)) for value in values.flat if int(value))


if __name__ == "__main__":
    main()
