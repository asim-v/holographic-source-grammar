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
    load_holographic_recording,
    selective_token_ablation,
    source_identity_nulls,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    ROOT / "docs" / "holographic_etoe_confirmation_protocol.md"
)
DEFAULT_PROTOCOL_LOCK = (
    ROOT
    / "docs"
    / "holographic_etoe_confirmation_protocol.lock.json"
)
DEFAULT_MANIFEST = (
    ROOT / "data" / "holographic_etoe_confirmation_manifest.json"
)
DEFAULT_DEVELOPMENT_RESULT = (
    ROOT
    / "artifacts"
    / "holographic_reliability_shrinkage_sst"
    / "summary.json"
)
DEFAULT_RAW_DIR = (
    ROOT / "data" / "raw" / "holographic_circuitmap_etoe"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "artifacts" / "holographic_etoe_confirmation"
)
EXPECTED_PROTOCOL_SHA256 = (
    "e8e48ec69bf0bb5fa99226e737d46f3d"
    "2e8ab21cb32c471be6863b5eacb10d83"
)
EXPECTED_MANIFEST_SHA256 = (
    "984f4e3b5187fb80420af96be4019077"
    "5a7906419fe2036f5186c54e2be5d237"
)
EXPECTED_DEVELOPMENT_SHA256 = (
    "f13b70a7407e9ceda6afe38d35486849c"
    "ed509fca9af6f60df969d15be7cd1a6"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the locked reliability-weighted E-to-E "
            "holographic confirmation."
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
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--development-result",
        type=Path,
        default=DEFAULT_DEVELOPMENT_RESULT,
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
        "--null-repetitions",
        type=int,
        default=999,
    )
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--seed", type=int, default=20_260_732)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, lock, files, inputs = _validate_inputs(args)
    if args.null_repetitions < 1:
        raise ValueError("Null repetitions must be positive")
    if args.max_files is not None:
        if args.max_files < 1:
            raise ValueError("Maximum file count must be positive")
        files = files[: args.max_files]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metric_frames = []
    reliability_frames = []
    shrinkage_frames = []
    null_frames = []
    ablation_frames = []
    diagnostic_rows = []
    for file_index, item in enumerate(files):
        filename = str(item["name"])
        print(
            f"[{file_index + 1}/{len(files)}] {filename}",
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
            "sources": stimulus.shape[1],
            "depth_planes": int(
                len(np.unique(recording.targets[:, -1]))
            ),
            "blank_trials": int(np.sum(active_count == 0)),
            "single_source_trials": int(
                np.sum(active_count == 1)
            ),
            "ensemble_trials": int(np.sum(active_count > 1)),
            "excluded_blank_nonfinite_demixed": (
                recording.excluded_blank_nonfinite_demixed
            ),
            "excluded_blank_nonfinite_raw": (
                recording.excluded_blank_nonfinite_raw
            ),
        }
        for outcome in frozen.OUTCOMES:
            response = baseline_correct_response(
                getattr(recording, outcome),
                baseline_samples=tuple(lock["baseline_samples"]),
                response_samples=tuple(lock["response_samples"]),
            )
            result = build_reliability_shrunk_token_model(
                stimulus,
                response,
                recording.targets,
                minimum_single_trials=int(
                    lock[
                        "minimum_single_trials_per_source_power"
                    ]
                ),
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
            shrinkage = result.power_shrinkage.copy()
            shrinkage.insert(0, "outcome", outcome)
            shrinkage.insert(0, "file", filename)
            shrinkage_frames.append(shrinkage)
            diagnostic[f"{outcome}_tokens"] = len(model.tokens)
            diagnostic[f"{outcome}_eligible_ensembles"] = len(
                context.trial_indices
            )
            diagnostic[f"{outcome}_mean_shrinkage"] = float(
                shrinkage["shrinkage"].mean()
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
                shrinkage,
            )
            gc.collect()
        diagnostic_rows.append(diagnostic)
        del recording, stimulus
        gc.collect()

    trial_metrics = _concat(metric_frames)
    reliability = _concat(reliability_frames)
    power_shrinkage = _concat(shrinkage_frames)
    nulls = _concat(null_frames)
    ablation = _concat(ablation_frames)
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

    protocol_complete = bool(
        args.max_files is None
        and len(files) == int(lock["expected_files"])
        and args.null_repetitions
        == int(lock["source_identity_nulls"])
    )
    gates = _confirmation_gates(
        protocol_complete,
        int(lock["minimum_usable_files"]),
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
        "analysis": "holographic_etoe_confirmation",
        "status": "prospective_confirmation",
        "protocol_complete": protocol_complete,
        "protocol_sha256": lock["protocol_sha256"],
        "manifest_sha256": lock["manifest_sha256"],
        "development_result_sha256": (
            lock["development_result_sha256"]
        ),
        "figshare_article_id": manifest["figshare_article_id"],
        "doi": manifest["doi"],
        "files_requested": len(files),
        "files_with_eligible_ensembles": int(
            file_effects.loc[
                file_effects["outcome"].eq(
                    frozen.PRIMARY_OUTCOME
                )
            ]["file"].nunique()
        ),
        "null_repetitions": args.null_repetitions,
        "trial_counts": {
            "total": int(diagnostics["trials"].sum()),
            "single_source": int(
                diagnostics["single_source_trials"].sum()
            ),
            "ensemble": int(
                diagnostics["ensemble_trials"].sum()
            ),
            "eligible_primary_ensemble": int(
                diagnostics[
                    f"{frozen.PRIMARY_OUTCOME}_eligible_ensembles"
                ].sum()
            ),
        },
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
        "gates": gates,
        "interpretation": (
            "A pass supports transfer of a reliability-weighted "
            "source-specific additive current grammar from SST to "
            "E-to-E perturbations. It does not establish "
            "monosemantic neurons or a complete neural language."
        ),
        "pv_remains_sealed": True,
        "inputs": inputs,
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


def _validate_inputs(
    args: argparse.Namespace,
) -> tuple[
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    hashes = {
        "protocol_sha256": _sha256(args.protocol),
        "manifest_sha256": _sha256(args.manifest),
        "development_result_sha256": _sha256(
            args.development_result
        ),
    }
    expected = {
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "development_result_sha256": (
            EXPECTED_DEVELOPMENT_SHA256
        ),
    }
    if hashes != expected:
        raise ValueError(
            f"Frozen confirmation inputs changed: {hashes}"
        )
    lock = json.loads(
        args.protocol_lock.read_text(encoding="utf-8")
    )
    if any(lock[key] != value for key, value in hashes.items()):
        raise ValueError("Confirmation lock is inconsistent")
    if bool(lock["pv_opened"]):
        raise ValueError("PV must remain sealed")
    manifest = json.loads(
        args.manifest.read_text(encoding="utf-8")
    )
    files = list(manifest["files"])
    if len(files) != int(lock["expected_files"]):
        raise ValueError("Unexpected E-to-E file count")
    inputs = [
        {
            "path": _portable_path(args.protocol),
            "sha256": hashes["protocol_sha256"],
        },
        {
            "path": _portable_path(args.protocol_lock),
            "sha256": _sha256(args.protocol_lock),
        },
        {
            "path": _portable_path(args.manifest),
            "sha256": hashes["manifest_sha256"],
        },
        {
            "path": _portable_path(args.development_result),
            "sha256": hashes["development_result_sha256"],
        },
    ]
    print("Verifying E-to-E checksums", flush=True)
    for item in files:
        path = args.raw_dir / str(item["name"])
        frozen._verify_manifest_file(path, item)
        inputs.append(
            {
                "path": _portable_path(path),
                "size": path.stat().st_size,
                "md5": item["md5"],
            }
        )
    return manifest, lock, files, inputs


def _confirmation_gates(
    protocol_complete: bool,
    minimum_usable_files: int,
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
    usable = int(base["usable_files"])
    file_gate = usable >= minimum_usable_files
    confirmed = bool(
        protocol_complete
        and base["primary_composition_metrics"]
        and base["positive_leave_one_file_out"]
        and base["familywise_source_identity_null"]
        and base["selective_token_ablation"]
        and file_gate
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
        "usable_files": usable,
        "minimum_usable_files": minimum_usable_files,
        "minimum_file_count": file_gate,
        "etoe_confirmation": confirmed,
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


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


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
