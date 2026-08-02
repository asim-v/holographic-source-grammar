from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from wetware_interp.holographic_source_composition import (
    baseline_correct_response,
    benjamini_hochberg,
    build_composition_context,
    build_composition_geometry,
    build_source_token_model,
    composition_metrics,
    exact_sign_flip_test,
    leave_one_out_means,
    load_holographic_recording,
    selective_token_ablation,
    signed_peak,
    source_identity_nulls,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT / "data" / "holographic_circuitmap_manifest.json"
)
DEFAULT_PROTOCOL = (
    ROOT / "docs" / "holographic_source_composition_protocol.md"
)
DEFAULT_PROTOCOL_LOCK = (
    ROOT
    / "docs"
    / "holographic_source_composition_protocol.lock.json"
)
DEFAULT_RAW_DIR = (
    ROOT / "data" / "raw" / "holographic_circuitmap"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "holographic_source_composition_sst"
)
EXPECTED_PROTOCOL_SHA256 = (
    "ef740262f89da0c1b4051b24c52b5708"
    "4c44f8bfdd7f407b7f541f59dc02d45a"
)
EXPECTED_MANIFEST_SHA256 = (
    "65864e3126d6d1b9c0f8ce11e9ff1bb"
    "d47a423fcdef4f64a54792660cadb04ab"
)
PRIMARY_OUTCOME = "pscs_demixed"
OUTCOMES = ("pscs_demixed", "pscs")
PRIMARY_METRICS = (
    "mse_gain_source_vs_anonymous",
    "rho_gain_source_vs_anonymous",
)
EXPECTED_FILE_COUNTS = {
    "sst_discovery": 9,
    "pv_confirmation": 14,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether single-source holographic response tokens "
            "compose on held-out ensemble trials."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
    )
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=DEFAULT_PROTOCOL_LOCK,
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--cohort",
        choices=tuple(EXPECTED_FILE_COUNTS),
        default="sst_discovery",
    )
    parser.add_argument(
        "--outcomes",
        choices=OUTCOMES,
        nargs="+",
        default=list(OUTCOMES),
    )
    parser.add_argument(
        "--null-repetitions",
        type=int,
        default=None,
    )
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--seed", type=int, default=20_260_730)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    (
        manifest,
        protocol_lock,
        file_records,
        input_records,
    ) = _validate_inputs(args)
    null_repetitions = (
        int(protocol_lock["source_identity_nulls"])
        if args.null_repetitions is None
        else args.null_repetitions
    )
    if null_repetitions < 1:
        raise ValueError("Null repetitions must be positive")
    if args.max_files is not None:
        if args.max_files < 1:
            raise ValueError("Maximum file count must be positive")
        file_records = file_records[: args.max_files]
    outcomes = tuple(dict.fromkeys(args.outcomes))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trial_metric_frames = []
    reliability_frames = []
    inventory_frames = []
    null_frames = []
    ablation_frames = []
    diagnostic_rows = []

    for file_index, item in enumerate(file_records):
        filename = str(item["name"])
        path = args.raw_dir / filename
        print(
            f"[{file_index + 1}/{len(file_records)}] "
            f"Reading {filename}",
            flush=True,
        )
        recording = load_holographic_recording(path)
        stimulus = recording.stimulus_matrix
        active_count = np.count_nonzero(stimulus > 0, axis=1)
        file_diagnostic = {
            "file": filename,
            "trials": len(stimulus),
            "sources": stimulus.shape[1],
            "depth_planes": int(
                len(np.unique(recording.targets[:, -1]))
            ),
            "blank_trials": int(np.sum(active_count == 0)),
            "single_source_trials": int(np.sum(active_count == 1)),
            "ensemble_trials": int(np.sum(active_count > 1)),
            "excluded_blank_nonfinite_demixed": (
                recording.excluded_blank_nonfinite_demixed
            ),
            "excluded_blank_nonfinite_raw": (
                recording.excluded_blank_nonfinite_raw
            ),
        }
        for outcome in outcomes:
            print(f"  Building {outcome} tokens", flush=True)
            response = baseline_correct_response(
                getattr(recording, outcome),
                baseline_samples=tuple(
                    int(value)
                    for value in protocol_lock.get(
                        "baseline_samples",
                        [0, 100],
                    )
                ),
                response_samples=tuple(
                    int(value)
                    for value in protocol_lock["response_samples"]
                ),
            )
            token_model = build_source_token_model(
                stimulus,
                response,
                recording.targets,
                minimum_single_trials=int(
                    protocol_lock[
                        "minimum_single_trials_per_source_power"
                    ]
                ),
            )
            context = build_composition_context(
                stimulus,
                response,
                token_model,
            )
            metrics = composition_metrics(context)
            metrics.insert(0, "outcome", outcome)
            metrics.insert(0, "file", filename)
            trial_metric_frames.append(metrics)

            reliability = token_model.reliability.copy()
            reliability.insert(0, "outcome", outcome)
            reliability.insert(0, "file", filename)
            reliability_frames.append(reliability)
            inventory_frames.append(
                _token_inventory(
                    filename,
                    outcome,
                    token_model,
                )
            )
            file_diagnostic[f"{outcome}_tokens"] = len(
                token_model.tokens
            )
            file_diagnostic[
                f"{outcome}_eligible_ensemble_trials"
            ] = len(context.trial_indices)

            if outcome == PRIMARY_OUTCOME and len(
                context.trial_indices
            ):
                geometry = build_composition_geometry(context)
                print(
                    "  Running "
                    f"{null_repetitions} source-identity nulls",
                    flush=True,
                )
                nulls = source_identity_nulls(
                    context,
                    geometry,
                    repetitions=null_repetitions,
                    seed=args.seed + file_index * 1_000_003,
                )
                nulls.insert(0, "file", filename)
                null_frames.append(nulls)
                ablation = selective_token_ablation(
                    context,
                    geometry,
                )
                ablation.insert(0, "outcome", outcome)
                ablation.insert(0, "file", filename)
                ablation_frames.append(ablation)
                file_diagnostic[
                    "selective_ablation_rows"
                ] = len(ablation)
                del geometry, nulls, ablation
            del response, token_model, context, metrics, reliability
            gc.collect()
        diagnostic_rows.append(file_diagnostic)
        del recording, stimulus
        gc.collect()

    trial_metrics = _concat(
        trial_metric_frames,
        columns=("file", "outcome"),
    )
    reliability = _concat(
        reliability_frames,
        columns=("file", "outcome"),
    )
    token_inventory = _concat(
        inventory_frames,
        columns=("file", "outcome"),
    )
    nulls = _concat(
        null_frames,
        columns=("file", "repetition", *PRIMARY_METRICS),
    )
    ablation = _concat(
        ablation_frames,
        columns=("file", "outcome", "trial_index"),
    )
    diagnostics = pd.DataFrame(diagnostic_rows)

    file_effects = _file_effects(trial_metrics)
    composition_tests = _composition_tests(file_effects)
    leave_one_file_out = _leave_one_file_out(file_effects)
    null_summary = _familywise_null_summary(
        file_effects,
        nulls,
    )
    (
        ablation_file_effects,
        ablation_test,
    ) = _ablation_summary(ablation)

    protocol_complete = (
        len(file_records) == EXPECTED_FILE_COUNTS[args.cohort]
        and null_repetitions
        == int(protocol_lock["source_identity_nulls"])
        and set(outcomes) == set(OUTCOMES)
        and args.max_files is None
    )
    gates = _evaluate_gates(
        args.cohort,
        protocol_complete,
        file_effects,
        composition_tests,
        leave_one_file_out,
        null_summary,
        ablation_file_effects,
        ablation_test,
    )

    outputs = {
        "trial_metrics.csv.gz": trial_metrics,
        "file_effects.csv": file_effects,
        "composition_tests.csv": composition_tests,
        "leave_one_file_out.csv": leave_one_file_out,
        "token_reliability.csv.gz": reliability,
        "token_inventory.csv.gz": token_inventory,
        "source_identity_nulls.csv.gz": nulls,
        "source_identity_null_summary.csv": null_summary,
        "selective_ablation.csv.gz": ablation,
        "selective_ablation_file_effects.csv": (
            ablation_file_effects
        ),
        "selective_ablation_test.csv": ablation_test,
        "file_diagnostics.csv": diagnostics,
    }
    for filename, frame in outputs.items():
        frame.to_csv(
            args.output_dir / filename,
            index=False,
            compression=(
                "gzip" if filename.endswith(".gz") else None
            ),
        )
    summary = _build_summary(
        args,
        manifest,
        protocol_lock,
        file_records,
        input_records,
        null_repetitions,
        outcomes,
        protocol_complete,
        diagnostics,
        file_effects,
        composition_tests,
        leave_one_file_out,
        null_summary,
        ablation_test,
        gates,
        outputs,
    )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


def _validate_inputs(
    args: argparse.Namespace,
) -> tuple[
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    protocol_hash = _sha256(args.protocol)
    if protocol_hash != EXPECTED_PROTOCOL_SHA256:
        raise ValueError(
            f"Protocol hash is {protocol_hash}; "
            f"expected {EXPECTED_PROTOCOL_SHA256}"
        )
    protocol_lock = json.loads(
        args.protocol_lock.read_text(encoding="utf-8")
    )
    if protocol_lock["protocol_sha256"] != protocol_hash:
        raise ValueError("Protocol lock does not match protocol")
    manifest_hash = _sha256(args.manifest)
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            f"Manifest hash is {manifest_hash}; "
            f"expected {EXPECTED_MANIFEST_SHA256}"
        )
    if protocol_lock["manifest_sha256"] != manifest_hash:
        raise ValueError("Protocol lock does not match manifest")
    if bool(protocol_lock["pv_opened"]):
        raise ValueError(
            "The prospective lock unexpectedly says PV was opened"
        )
    manifest = json.loads(
        args.manifest.read_text(encoding="utf-8")
    )
    cohort_record = manifest["cohorts"][args.cohort]
    if (
        args.cohort == "pv_confirmation"
        and not bool(cohort_record["opened"])
    ):
        raise ValueError(
            "PV confirmation remains sealed in the manifest"
        )
    file_records = [
        item
        for item in manifest["files"]
        if item["cohort"] == args.cohort
    ]
    expected_names = list(cohort_record["files"])
    if [item["name"] for item in file_records] != expected_names:
        raise ValueError("Cohort manifest order or membership changed")
    if len(file_records) != EXPECTED_FILE_COUNTS[args.cohort]:
        raise ValueError("Unexpected cohort file count")

    inputs = [
        {
            "path": _portable_path(args.protocol),
            "sha256": protocol_hash,
        },
        {
            "path": _portable_path(args.protocol_lock),
            "sha256": _sha256(args.protocol_lock),
        },
        {
            "path": _portable_path(args.manifest),
            "sha256": manifest_hash,
        },
    ]
    print("Verifying source-file checksums", flush=True)
    for item in file_records:
        path = args.raw_dir / str(item["name"])
        _verify_manifest_file(path, item)
        inputs.append(
            {
                "path": _portable_path(path),
                "size": path.stat().st_size,
                "md5": str(item["md5"]),
            }
        )
    return manifest, protocol_lock, file_records, inputs


def _verify_manifest_file(
    path: Path,
    record: dict[str, object],
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_size = int(record["size"])
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"{path.name} has {path.stat().st_size} bytes; "
            f"expected {expected_size}"
        )
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    observed = digest.hexdigest()
    if observed != str(record["md5"]):
        raise ValueError(
            f"{path.name} MD5 is {observed}; "
            f"expected {record['md5']}"
        )


def _token_inventory(
    filename: str,
    outcome: str,
    model,
) -> pd.DataFrame:
    rows = []
    for index, token in enumerate(model.tokens):
        source = int(model.token_sources[index])
        rows.append(
            {
                "file": filename,
                "outcome": outcome,
                "token_index": index,
                "source": source,
                "power": float(model.token_powers[index]),
                "depth": float(model.depth_by_source[source]),
                "n_single_trials": int(model.token_counts[index]),
                "waveform_l2": float(np.linalg.norm(token)),
                "integrated_current": float(token.sum() * 0.05),
                "peak_current": signed_peak(token),
            }
        )
    return pd.DataFrame(rows)


def _file_effects(trial_metrics: pd.DataFrame) -> pd.DataFrame:
    if trial_metrics.empty:
        return pd.DataFrame(
            columns=("file", "outcome", "eligible_trials")
        )
    metric_columns = [
        column
        for column in trial_metrics.columns
        if "_gain_" in column
    ]
    return (
        trial_metrics.groupby(
            ["file", "outcome"],
            sort=True,
            as_index=False,
        )
        .agg(
            eligible_trials=("trial_index", "size"),
            **{
                column: (column, "mean")
                for column in metric_columns
            },
        )
    )


def _composition_tests(
    file_effects: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOMES:
        group = file_effects.loc[
            file_effects["outcome"].eq(outcome)
        ]
        outcome_rows = []
        for metric in PRIMARY_METRICS:
            if group.empty:
                continue
            result = exact_sign_flip_test(group[metric])
            outcome_rows.append(
                {
                    "outcome": outcome,
                    "metric": metric,
                    "mean_file_effect": result.observed_mean,
                    "p_value": result.p_value,
                    "n_files": result.n_files,
                    "n_assignments": result.n_assignments,
                }
            )
        if outcome_rows:
            q_values = benjamini_hochberg(
                [row["p_value"] for row in outcome_rows]
            )
            for row, q_value in zip(
                outcome_rows,
                q_values,
                strict=True,
            ):
                row["q_value"] = float(q_value)
                row["passes"] = bool(
                    row["mean_file_effect"] > 0
                    and q_value < 0.05
                )
            rows.extend(outcome_rows)
    return pd.DataFrame(rows)


def _leave_one_file_out(
    file_effects: pd.DataFrame,
) -> pd.DataFrame:
    columns = (
        "outcome",
        "metric",
        "omitted_file",
        "leave_one_out_mean",
        "positive",
    )
    rows = []
    for outcome in OUTCOMES:
        group = (
            file_effects.loc[file_effects["outcome"].eq(outcome)]
            .sort_values("file")
            .reset_index(drop=True)
        )
        if len(group) < 2:
            continue
        for metric in PRIMARY_METRICS:
            estimates = leave_one_out_means(group[metric])
            for omitted, estimate in zip(
                group["file"],
                estimates,
                strict=True,
            ):
                rows.append(
                    {
                        "outcome": outcome,
                        "metric": metric,
                        "omitted_file": omitted,
                        "leave_one_out_mean": float(estimate),
                        "positive": bool(estimate > 0),
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def _familywise_null_summary(
    file_effects: pd.DataFrame,
    nulls: pd.DataFrame,
) -> pd.DataFrame:
    primary = file_effects.loc[
        file_effects["outcome"].eq(PRIMARY_OUTCOME)
    ]
    if primary.empty or nulls.empty:
        return pd.DataFrame(
            columns=(
                "metric",
                "observed_mean_file_effect",
                "null_mean",
                "null_standard_deviation",
                "observed_z",
                "familywise_99_threshold_z",
                "familywise_p_value",
                "passes",
            )
        )
    null_aggregate = (
        nulls.groupby("repetition", sort=True)[
            list(PRIMARY_METRICS)
        ]
        .mean()
        .sort_index()
    )
    observed = primary[list(PRIMARY_METRICS)].mean()
    null_mean = null_aggregate.mean()
    null_std = null_aggregate.std(ddof=1)
    if np.any(~np.isfinite(null_std)) or np.any(null_std <= 0):
        raise ValueError("Source null has zero or invalid dispersion")
    null_z = (null_aggregate - null_mean) / null_std
    maximum_null_z = null_z.max(axis=1).to_numpy()
    threshold = float(
        np.quantile(maximum_null_z, 0.99, method="higher")
    )
    rows = []
    for metric in PRIMARY_METRICS:
        observed_z = float(
            (observed[metric] - null_mean[metric])
            / null_std[metric]
        )
        familywise_p = float(
            (
                1
                + np.sum(maximum_null_z >= observed_z - 1e-15)
            )
            / (len(maximum_null_z) + 1)
        )
        rows.append(
            {
                "metric": metric,
                "observed_mean_file_effect": float(
                    observed[metric]
                ),
                "null_mean": float(null_mean[metric]),
                "null_standard_deviation": float(
                    null_std[metric]
                ),
                "null_raw_99_percentile": float(
                    np.quantile(
                        null_aggregate[metric],
                        0.99,
                        method="higher",
                    )
                ),
                "observed_z": observed_z,
                "familywise_99_threshold_z": threshold,
                "familywise_p_value": familywise_p,
                "passes": bool(observed_z > threshold),
            }
        )
    return pd.DataFrame(rows)


def _ablation_summary(
    ablation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if ablation.empty:
        return (
            pd.DataFrame(
                columns=(
                    "file",
                    "eligible_ablation_trials",
                    "mean_replacement_mse_cost",
                )
            ),
            pd.DataFrame(
                columns=(
                    "mean_file_effect",
                    "p_value",
                    "n_files",
                    "passes",
                )
            ),
        )
    trial = (
        ablation.groupby(
            ["file", "trial_index"],
            sort=True,
            as_index=False,
        )["mean_decoy_replacement_mse_cost"]
        .mean()
    )
    file_effects = (
        trial.groupby("file", sort=True, as_index=False)
        .agg(
            eligible_ablation_trials=("trial_index", "size"),
            mean_replacement_mse_cost=(
                "mean_decoy_replacement_mse_cost",
                "mean",
            ),
        )
    )
    result = exact_sign_flip_test(
        file_effects["mean_replacement_mse_cost"]
    )
    test = pd.DataFrame(
        [
            {
                "mean_file_effect": result.observed_mean,
                "p_value": result.p_value,
                "n_files": result.n_files,
                "n_assignments": result.n_assignments,
                "passes": bool(
                    result.observed_mean > 0
                    and result.p_value < 0.05
                ),
            }
        ]
    )
    return file_effects, test


def _evaluate_gates(
    cohort: str,
    protocol_complete: bool,
    file_effects: pd.DataFrame,
    composition_tests: pd.DataFrame,
    leave_one_file_out: pd.DataFrame,
    null_summary: pd.DataFrame,
    ablation_file_effects: pd.DataFrame,
    ablation_test: pd.DataFrame,
) -> dict[str, object]:
    primary_tests = composition_tests.loc[
        composition_tests["outcome"].eq(PRIMARY_OUTCOME)
    ]
    primary_metrics_pass = bool(
        len(primary_tests) == len(PRIMARY_METRICS)
        and primary_tests["passes"].all()
    )
    primary_loo = leave_one_file_out.loc[
        leave_one_file_out["outcome"].eq(PRIMARY_OUTCOME)
    ]
    leave_one_out_pass = bool(
        len(primary_loo)
        == len(PRIMARY_METRICS)
        * file_effects.loc[
            file_effects["outcome"].eq(PRIMARY_OUTCOME)
        ]["file"].nunique()
        and len(primary_loo) > 0
        and primary_loo["positive"].all()
    )
    source_null_pass = bool(
        len(null_summary) == len(PRIMARY_METRICS)
        and null_summary["passes"].all()
    )
    primary_file_count = int(
        file_effects.loc[
            file_effects["outcome"].eq(PRIMARY_OUTCOME)
        ]["file"].nunique()
    )
    ablation_pass = bool(
        len(ablation_test) == 1
        and bool(ablation_test.iloc[0]["passes"])
        and len(ablation_file_effects) == primary_file_count
    )
    minimum_files = 7 if cohort == "sst_discovery" else 1
    file_count_pass = primary_file_count >= minimum_files
    common = {
        "protocol_complete": protocol_complete,
        "primary_composition_metrics": primary_metrics_pass,
        "positive_leave_one_file_out": leave_one_out_pass,
        "familywise_source_identity_null": source_null_pass,
        "selective_token_ablation": ablation_pass,
        "usable_files": primary_file_count,
        "minimum_usable_files": minimum_files,
        "minimum_file_count": file_count_pass,
    }
    if cohort == "sst_discovery":
        common["sst_opening_gate"] = bool(
            protocol_complete
            and primary_metrics_pass
            and leave_one_out_pass
            and source_null_pass
            and ablation_pass
            and file_count_pass
        )
        common["pv_may_be_opened"] = common["sst_opening_gate"]
    else:
        common["pv_confirmation"] = bool(
            protocol_complete
            and primary_metrics_pass
            and leave_one_out_pass
            and source_null_pass
        )
    return common


def _build_summary(
    args: argparse.Namespace,
    manifest: dict[str, object],
    protocol_lock: dict[str, object],
    file_records: list[dict[str, object]],
    input_records: list[dict[str, object]],
    null_repetitions: int,
    outcomes: tuple[str, ...],
    protocol_complete: bool,
    diagnostics: pd.DataFrame,
    file_effects: pd.DataFrame,
    composition_tests: pd.DataFrame,
    leave_one_file_out: pd.DataFrame,
    null_summary: pd.DataFrame,
    ablation_test: pd.DataFrame,
    gates: dict[str, object],
    outputs: dict[str, pd.DataFrame],
) -> dict[str, object]:
    primary_effects = file_effects.loc[
        file_effects["outcome"].eq(PRIMARY_OUTCOME)
    ]
    raw_effects = file_effects.loc[
        file_effects["outcome"].eq("pscs")
    ]
    primary_loo = leave_one_file_out.loc[
        leave_one_file_out["outcome"].eq(PRIMARY_OUTCOME)
    ]
    output_records = []
    for filename, frame in outputs.items():
        path = args.output_dir / filename
        output_records.append(
            {
                "path": _portable_path(path),
                "rows": len(frame),
                "sha256": _sha256(path),
            }
        )
    return {
        "analysis": "holographic_source_composition",
        "cohort": args.cohort,
        "protocol_complete": protocol_complete,
        "protocol_sha256": protocol_lock["protocol_sha256"],
        "previous_protocol_sha256": protocol_lock.get(
            "previous_protocol_sha256"
        ),
        "manifest_sha256": protocol_lock["manifest_sha256"],
        "figshare_article_id": manifest["figshare_article_id"],
        "doi": manifest["doi"],
        "circuitmap_commit": manifest["code_commit"],
        "files_requested": len(file_records),
        "files_with_eligible_ensembles": int(
            len(primary_effects)
        ),
        "null_repetitions": null_repetitions,
        "outcomes": list(outcomes),
        "trial_counts": {
            "total": int(diagnostics["trials"].sum()),
            "single_source": int(
                diagnostics["single_source_trials"].sum()
            ),
            "ensemble": int(
                diagnostics["ensemble_trials"].sum()
            ),
            "eligible_primary_ensemble": int(
                diagnostics.get(
                    f"{PRIMARY_OUTCOME}_eligible_ensemble_trials",
                    pd.Series(dtype=int),
                ).sum()
            ),
        },
        "primary_file_mean_effects": {
            metric: (
                float(primary_effects[metric].mean())
                if len(primary_effects)
                else None
            )
            for metric in PRIMARY_METRICS
        },
        "raw_file_mean_effects": {
            metric: (
                float(raw_effects[metric].mean())
                if len(raw_effects)
                else None
            )
            for metric in PRIMARY_METRICS
        },
        "minimum_primary_leave_one_file_out": {
            metric: (
                float(
                    primary_loo.loc[
                        primary_loo["metric"].eq(metric),
                        "leave_one_out_mean",
                    ].min()
                )
                if (
                    not primary_loo.loc[
                        primary_loo["metric"].eq(metric)
                    ].empty
                )
                else None
            )
            for metric in PRIMARY_METRICS
        },
        "composition_tests": _records(composition_tests),
        "source_identity_null_tests": _records(null_summary),
        "selective_ablation_test": _records(ablation_test),
        "gates": gates,
        "interpretation": (
            "Passing supports source-specific additive composition "
            "for this perturbation cohort. It does not establish "
            "monosemantic neurons, natural-language meanings, or a "
            "complete neural language."
        ),
        "inputs": input_records,
        "outputs": output_records,
    }


def _concat(
    frames: Iterable[pd.DataFrame],
    *,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    materialized = list(frames)
    if not materialized:
        return pd.DataFrame(columns=columns)
    return pd.concat(materialized, ignore_index=True)


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


if __name__ == "__main__":
    main()
