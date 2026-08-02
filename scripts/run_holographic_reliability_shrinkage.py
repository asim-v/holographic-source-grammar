from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_holographic_source_composition as frozen
from wetware_interp.holographic_source_composition import (
    baseline_correct_response,
    build_composition_context,
    build_composition_geometry,
    build_reliability_shrunk_token_model,
    composition_metrics,
    exact_sign_flip_test,
    load_holographic_recording,
    selective_token_ablation,
    source_identity_nulls,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    ROOT
    / "docs"
    / "holographic_reliability_shrinkage_development.md"
)
DEFAULT_PROTOCOL_LOCK = (
    ROOT
    / "docs"
    / "holographic_reliability_shrinkage_development.lock.json"
)
DEFAULT_SOURCE_RESULT = (
    ROOT
    / "artifacts"
    / "holographic_source_composition_sst"
    / "summary.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "holographic_reliability_shrinkage_sst"
)
EXPECTED_PROTOCOL_SHA256 = (
    "054d4b692ffba4a2e85c65f99d1ba5e0"
    "41e27829dd1575d06ac212bbd73391a8"
)
EXPECTED_SOURCE_RESULT_SHA256 = (
    "4b499e25bbb79cc8ad34025b19e3bb6e"
    "26016d029024481c8f0e37d58732c39e"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Develop split-half reliability shrinkage on the opened "
            "SST holographic cohort."
        )
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
        "--source-result",
        type=Path,
        default=DEFAULT_SOURCE_RESULT,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=frozen.DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=frozen.DEFAULT_RAW_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--null-repetitions",
        type=int,
        default=999,
    )
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--seed", type=int, default=20_260_731)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, lock, file_records, input_records = _validate(args)
    if args.null_repetitions < 1:
        raise ValueError("Null repetitions must be positive")
    if args.max_files is not None:
        if args.max_files < 1:
            raise ValueError("Maximum file count must be positive")
        file_records = file_records[: args.max_files]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metric_frames = []
    reliability_frames = []
    shrinkage_frames = []
    null_frames = []
    ablation_frames = []
    diagnostic_rows = []
    for file_index, item in enumerate(file_records):
        filename = str(item["name"])
        print(
            f"[{file_index + 1}/{len(file_records)}] {filename}",
            flush=True,
        )
        recording = load_holographic_recording(
            args.raw_dir / filename
        )
        stimulus = recording.stimulus_matrix
        active_count = np.count_nonzero(stimulus > 0, axis=1)
        diagnostic = {
            "file": filename,
            "trials": len(stimulus),
            "single_source_trials": int(
                np.sum(active_count == 1)
            ),
            "ensemble_trials": int(np.sum(active_count > 1)),
        }
        for outcome in frozen.OUTCOMES:
            response = baseline_correct_response(
                getattr(recording, outcome)
            )
            result = build_reliability_shrunk_token_model(
                stimulus,
                response,
                recording.targets,
                minimum_single_trials=3,
            )
            model = result.token_model
            context = build_composition_context(
                stimulus,
                response,
                model,
            )
            metrics = composition_metrics(context)
            metrics.insert(0, "outcome", outcome)
            metrics.insert(0, "file", filename)
            metric_frames.append(metrics)

            reliability = model.reliability.copy()
            reliability.insert(0, "outcome", outcome)
            reliability.insert(0, "file", filename)
            reliability_frames.append(reliability)
            power_shrinkage = result.power_shrinkage.copy()
            power_shrinkage.insert(0, "outcome", outcome)
            power_shrinkage.insert(0, "file", filename)
            shrinkage_frames.append(power_shrinkage)
            diagnostic[f"{outcome}_eligible_ensembles"] = len(
                context.trial_indices
            )
            diagnostic[f"{outcome}_mean_shrinkage"] = float(
                result.power_shrinkage["shrinkage"].mean()
            )

            if outcome == frozen.PRIMARY_OUTCOME and len(
                context.trial_indices
            ):
                geometry = build_composition_geometry(context)
                nulls = source_identity_nulls(
                    context,
                    geometry,
                    repetitions=args.null_repetitions,
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
                del geometry, nulls, ablation
            del (
                response,
                result,
                model,
                context,
                metrics,
                reliability,
                power_shrinkage,
            )
            gc.collect()
        diagnostic_rows.append(diagnostic)
        del recording, stimulus
        gc.collect()

    trial_metrics = pd.concat(metric_frames, ignore_index=True)
    reliability = pd.concat(reliability_frames, ignore_index=True)
    power_shrinkage = pd.concat(
        shrinkage_frames,
        ignore_index=True,
    )
    nulls = pd.concat(null_frames, ignore_index=True)
    ablation = pd.concat(ablation_frames, ignore_index=True)
    diagnostics = pd.DataFrame(diagnostic_rows)
    file_effects = frozen._file_effects(trial_metrics)
    composition_tests = frozen._composition_tests(file_effects)
    leave_one_file_out = frozen._leave_one_file_out(file_effects)
    null_summary = frozen._familywise_null_summary(
        file_effects,
        nulls,
    )
    (
        ablation_file_effects,
        ablation_test,
    ) = frozen._ablation_summary(ablation)
    comparison_to_full = _compare_to_full(file_effects)

    protocol_complete = bool(
        args.max_files is None
        and len(file_records) == 9
        and args.null_repetitions == int(lock["source_identity_nulls"])
    )
    gates = _development_gates(
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
        "power_shrinkage.csv": power_shrinkage,
        "source_identity_nulls.csv.gz": nulls,
        "source_identity_null_summary.csv": null_summary,
        "selective_ablation.csv.gz": ablation,
        "selective_ablation_file_effects.csv": (
            ablation_file_effects
        ),
        "selective_ablation_test.csv": ablation_test,
        "comparison_to_full_tokens.csv": comparison_to_full,
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
    summary = {
        "analysis": "holographic_reliability_shrinkage",
        "status": "posthoc_sst_development",
        "protocol_complete": protocol_complete,
        "protocol_sha256": lock["protocol_sha256"],
        "source_result_sha256": lock["source_result_sha256"],
        "manifest_sha256": lock["manifest_sha256"],
        "files": len(file_records),
        "eligible_primary_ensemble_trials": int(
            trial_metrics.loc[
                trial_metrics["outcome"].eq(
                    frozen.PRIMARY_OUTCOME
                )
            ].shape[0]
        ),
        "null_repetitions": args.null_repetitions,
        "mean_power_shrinkage": {
            outcome: float(
                power_shrinkage.loc[
                    power_shrinkage["outcome"].eq(outcome),
                    "shrinkage",
                ].mean()
            )
            for outcome in frozen.OUTCOMES
        },
        "composition_tests": frozen._records(composition_tests),
        "minimum_leave_one_file_out": _minimum_loo(
            leave_one_file_out
        ),
        "source_identity_null_tests": frozen._records(
            null_summary
        ),
        "selective_ablation_test": frozen._records(
            ablation_test
        ),
        "comparison_to_full_tokens": frozen._records(
            comparison_to_full
        ),
        "gates": gates,
        "pv_may_be_opened": False,
        "interpretation": (
            "This opened-SST result is development evidence only. "
            "It cannot revise the failed original gate or support "
            "transfer without a separately locked unopened cohort."
        ),
        "inputs": input_records,
        "outputs": [
            {
                "path": _portable_path(args.output_dir / name),
                "rows": len(frame),
                "sha256": _sha256(args.output_dir / name),
            }
            for name, frame in outputs.items()
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


def _validate(
    args: argparse.Namespace,
) -> tuple[
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    protocol_hash = _sha256(args.protocol)
    source_hash = _sha256(args.source_result)
    if protocol_hash != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("Development protocol hash changed")
    if source_hash != EXPECTED_SOURCE_RESULT_SHA256:
        raise ValueError("Frozen source result hash changed")
    lock = json.loads(
        args.protocol_lock.read_text(encoding="utf-8")
    )
    if (
        lock["protocol_sha256"] != protocol_hash
        or lock["source_result_sha256"] != source_hash
    ):
        raise ValueError("Development lock is inconsistent")
    manifest_hash = _sha256(args.manifest)
    if lock["manifest_sha256"] != manifest_hash:
        raise ValueError("Manifest hash changed")
    if bool(lock["pv_opened"]) or bool(lock["etoe_opened"]):
        raise ValueError("A confirmation cohort was opened too early")
    manifest = json.loads(
        args.manifest.read_text(encoding="utf-8")
    )
    cohort = manifest["cohorts"]["sst_discovery"]
    file_records = [
        item
        for item in manifest["files"]
        if item["cohort"] == "sst_discovery"
    ]
    if (
        len(file_records) != 9
        or [item["name"] for item in file_records]
        != list(cohort["files"])
    ):
        raise ValueError("SST cohort membership changed")
    input_records = [
        {
            "path": _portable_path(args.protocol),
            "sha256": protocol_hash,
        },
        {
            "path": _portable_path(args.protocol_lock),
            "sha256": _sha256(args.protocol_lock),
        },
        {
            "path": _portable_path(args.source_result),
            "sha256": source_hash,
        },
        {
            "path": _portable_path(args.manifest),
            "sha256": manifest_hash,
        },
    ]
    print("Verifying SST checksums", flush=True)
    for item in file_records:
        path = args.raw_dir / str(item["name"])
        frozen._verify_manifest_file(path, item)
        input_records.append(
            {
                "path": _portable_path(path),
                "size": path.stat().st_size,
                "md5": item["md5"],
            }
        )
    return manifest, lock, file_records, input_records


def _compare_to_full(
    shrunk_effects: pd.DataFrame,
) -> pd.DataFrame:
    full = pd.read_csv(
        ROOT
        / "artifacts"
        / "holographic_source_composition_sst"
        / "file_effects.csv"
    )
    merged = shrunk_effects.merge(
        full,
        on=["file", "outcome"],
        suffixes=("_shrunk", "_full"),
        validate="one_to_one",
    )
    rows = []
    for outcome in frozen.OUTCOMES:
        group = merged.loc[merged["outcome"].eq(outcome)]
        for metric in frozen.PRIMARY_METRICS:
            delta = (
                group[f"{metric}_shrunk"]
                - group[f"{metric}_full"]
            )
            test = exact_sign_flip_test(delta)
            rows.append(
                {
                    "outcome": outcome,
                    "metric": metric,
                    "mean_shrunk_effect": float(
                        group[f"{metric}_shrunk"].mean()
                    ),
                    "mean_full_effect": float(
                        group[f"{metric}_full"].mean()
                    ),
                    "mean_shrunk_minus_full": (
                        test.observed_mean
                    ),
                    "p_value_shrunk_better": test.p_value,
                }
            )
    return pd.DataFrame(rows)


def _development_gates(
    protocol_complete: bool,
    file_effects: pd.DataFrame,
    composition_tests: pd.DataFrame,
    leave_one_file_out: pd.DataFrame,
    null_summary: pd.DataFrame,
    ablation_file_effects: pd.DataFrame,
    ablation_test: pd.DataFrame,
) -> dict[str, object]:
    base = frozen._evaluate_gates(
        "sst_discovery",
        protocol_complete,
        file_effects,
        composition_tests,
        leave_one_file_out,
        null_summary,
        ablation_file_effects,
        ablation_test,
    )
    all_nine = base["usable_files"] == 9
    advance = bool(
        protocol_complete
        and base["primary_composition_metrics"]
        and base["positive_leave_one_file_out"]
        and base["familywise_source_identity_null"]
        and base["selective_token_ablation"]
        and all_nine
    )
    return {
        "protocol_complete": protocol_complete,
        "primary_composition_metrics": (
            base["primary_composition_metrics"]
        ),
        "positive_leave_one_file_out": (
            base["positive_leave_one_file_out"]
        ),
        "familywise_source_identity_null": (
            base["familywise_source_identity_null"]
        ),
        "selective_token_ablation": (
            base["selective_token_ablation"]
        ),
        "all_nine_files_usable": all_nine,
        "advance_to_locked_etoe_confirmation": advance,
        "original_sst_gate_remains_failed": True,
        "pv_remains_sealed": True,
    }


def _minimum_loo(
    leave_one_file_out: pd.DataFrame,
) -> dict[str, float | None]:
    primary = leave_one_file_out.loc[
        leave_one_file_out["outcome"].eq(
            frozen.PRIMARY_OUTCOME
        )
    ]
    return {
        metric: (
            float(
                primary.loc[
                    primary["metric"].eq(metric),
                    "leave_one_out_mean",
                ].min()
            )
            if not primary.loc[
                primary["metric"].eq(metric)
            ].empty
            else None
        )
        for metric in frozen.PRIMARY_METRICS
    }


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
