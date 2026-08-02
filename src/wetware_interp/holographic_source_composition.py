from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import pandas as pd


REQUIRED_DATASETS = (
    "stimulus_matrix",
    "pscs_demixed",
    "pscs",
    "targets",
    "caviar_weights_single",
)
CAVIAR_ENSEMBLE_ALIASES = (
    "caviar_weights_multi",
    "caviar_weights_ensemble",
)


@dataclass(frozen=True)
class HolographicRecording:
    filename: str
    stimulus_matrix: np.ndarray
    targets: np.ndarray
    pscs_demixed: np.ndarray
    pscs: np.ndarray
    excluded_blank_nonfinite_demixed: int = 0
    excluded_blank_nonfinite_raw: int = 0

    def __post_init__(self) -> None:
        stimulus = np.asarray(self.stimulus_matrix)
        targets = np.asarray(self.targets)
        if stimulus.ndim != 2:
            raise ValueError("Stimulus matrix must be trial by source")
        if np.any(stimulus < 0) or not np.all(np.isfinite(stimulus)):
            raise ValueError("Stimulus matrix must be finite and nonnegative")
        trial_count, source_count = stimulus.shape
        if targets.ndim != 2 or targets.shape[0] != source_count:
            raise ValueError("Targets must align with stimulus sources")
        if targets.shape[1] < 3 or not np.all(np.isfinite(targets)):
            raise ValueError("Target coordinates must be finite 3-vectors")
        for name, traces in (
            ("pscs_demixed", self.pscs_demixed),
            ("pscs", self.pscs),
        ):
            values = np.asarray(traces)
            if values.ndim != 2 or values.shape[0] != trial_count:
                raise ValueError(f"{name} must be trial by time")
            if values.shape[1] < 900:
                raise ValueError(f"{name} has fewer than 900 samples")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains nonfinite values")
        if (
            self.excluded_blank_nonfinite_demixed < 0
            or self.excluded_blank_nonfinite_raw < 0
        ):
            raise ValueError("Excluded-row counts cannot be negative")


@dataclass(frozen=True)
class SourceTokenModel:
    tokens: np.ndarray
    token_sources: np.ndarray
    token_powers: np.ndarray
    token_counts: np.ndarray
    depth_by_source: np.ndarray
    power_values: np.ndarray
    anonymous_tokens: np.ndarray
    reliability: pd.DataFrame

    def __post_init__(self) -> None:
        token_count, sample_count = self.tokens.shape
        if token_count < 1 or sample_count < 2:
            raise ValueError("At least one nonconstant-length token is required")
        for values in (
            self.token_sources,
            self.token_powers,
            self.token_counts,
        ):
            if np.asarray(values).shape != (token_count,):
                raise ValueError("Token metadata must align with tokens")
        if self.anonymous_tokens.shape != (
            len(self.power_values),
            sample_count,
        ):
            raise ValueError("Anonymous tokens must align with powers")

    @property
    def token_lookup(self) -> dict[tuple[int, float], int]:
        return {
            (int(source), float(power)): index
            for index, (source, power) in enumerate(
                zip(
                    self.token_sources,
                    self.token_powers,
                    strict=True,
                )
            )
        }

    @property
    def anonymous_lookup(self) -> dict[float, np.ndarray]:
        return {
            float(power): self.anonymous_tokens[index]
            for index, power in enumerate(self.power_values)
        }


@dataclass(frozen=True)
class CompositionContext:
    trial_indices: np.ndarray
    ensemble_sizes: np.ndarray
    response: np.ndarray
    source_prediction: np.ndarray
    anonymous_prediction: np.ndarray
    true_token_indices: np.ndarray
    active_sources: np.ndarray
    active_powers: np.ndarray
    active_mask: np.ndarray
    token_model: SourceTokenModel

    def __post_init__(self) -> None:
        trial_count, sample_count = self.response.shape
        if self.source_prediction.shape != (trial_count, sample_count):
            raise ValueError("Source predictions must align with responses")
        if self.anonymous_prediction.shape != (trial_count, sample_count):
            raise ValueError("Anonymous predictions must align with responses")
        active_shape = self.true_token_indices.shape
        if active_shape[0] != trial_count:
            raise ValueError("Active-token rows must align with responses")
        for values in (
            self.active_sources,
            self.active_powers,
            self.active_mask,
        ):
            if np.asarray(values).shape != active_shape:
                raise ValueError("Active-token metadata must align")
        for values in (self.trial_indices, self.ensemble_sizes):
            if np.asarray(values).shape != (trial_count,):
                raise ValueError("Trial metadata must align with responses")


@dataclass(frozen=True)
class CompositionGeometry:
    response_sum: np.ndarray
    response_squared_sum: np.ndarray
    response_centered_squared_sum: np.ndarray
    response_token_dot: np.ndarray
    token_sum: np.ndarray
    token_gram: np.ndarray
    anonymous_mse: np.ndarray
    anonymous_correlation: np.ndarray


@dataclass(frozen=True)
class SignFlipResult:
    observed_mean: float
    p_value: float
    n_files: int
    n_assignments: int


@dataclass(frozen=True)
class ReliabilityShrinkageResult:
    token_model: SourceTokenModel
    power_shrinkage: pd.DataFrame


@dataclass(frozen=True)
class ReducedTokenModelResult:
    token_model: SourceTokenModel
    parameters: pd.DataFrame


def load_holographic_recording(
    path: str | Path,
) -> HolographicRecording:
    source_path = Path(path)
    with h5py.File(source_path, "r") as handle:
        missing = [
            name for name in REQUIRED_DATASETS if name not in handle
        ]
        if missing:
            raise ValueError(
                "Holographic file is missing datasets: "
                + ", ".join(missing)
            )
        if not any(
            name in handle for name in CAVIAR_ENSEMBLE_ALIASES
        ):
            raise ValueError(
                "Holographic file is missing an ensemble CAVIaR "
                "weight dataset"
            )
        stimulus = np.asarray(
            handle["stimulus_matrix"],
            dtype=np.float64,
        )
        demixed = np.asarray(
            handle["pscs_demixed"],
            dtype=np.float64,
        )
        raw = np.asarray(handle["pscs"], dtype=np.float64)
        targets_stored = np.asarray(
            handle["targets"],
            dtype=np.float64,
        )
    source_count = stimulus.shape[1] if stimulus.ndim == 2 else -1
    if targets_stored.shape == (3, source_count):
        targets = targets_stored.T
    elif (
        targets_stored.ndim == 2
        and targets_stored.shape[0] == source_count
        and targets_stored.shape[1] >= 3
    ):
        targets = targets_stored
    else:
        raise ValueError("Stored target coordinates have unexpected shape")
    demixed, excluded_demixed = _sanitize_blank_nonfinite_rows(
        demixed,
        stimulus,
        name="pscs_demixed",
    )
    raw, excluded_raw = _sanitize_blank_nonfinite_rows(
        raw,
        stimulus,
        name="pscs",
    )
    return HolographicRecording(
        filename=source_path.name,
        stimulus_matrix=stimulus,
        targets=targets,
        pscs_demixed=demixed,
        pscs=raw,
        excluded_blank_nonfinite_demixed=excluded_demixed,
        excluded_blank_nonfinite_raw=excluded_raw,
    )


def _sanitize_blank_nonfinite_rows(
    traces: np.ndarray,
    stimulus_matrix: np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, int]:
    values = np.asarray(traces, dtype=np.float64)
    stimulus = np.asarray(stimulus_matrix, dtype=np.float64)
    if values.ndim != 2 or stimulus.ndim != 2:
        raise ValueError(f"{name} and stimulus must be matrices")
    if values.shape[0] != stimulus.shape[0]:
        raise ValueError(f"{name} trials do not align with stimulus")
    nonfinite_rows = np.any(~np.isfinite(values), axis=1)
    if not np.any(nonfinite_rows):
        return values, 0
    blank_rows = np.count_nonzero(stimulus > 0, axis=1) == 0
    invalid_rows = nonfinite_rows & (
        ~blank_rows | ~np.all(np.isnan(values), axis=1)
    )
    if np.any(invalid_rows):
        indices = np.flatnonzero(invalid_rows)
        raise ValueError(
            f"{name} has invalid nonfinite rows: "
            + ", ".join(str(int(index)) for index in indices[:10])
        )
    sanitized = values.copy()
    sanitized[nonfinite_rows] = 0.0
    return sanitized, int(np.sum(nonfinite_rows))


def baseline_correct_response(
    traces: np.ndarray,
    *,
    baseline_samples: tuple[int, int] = (0, 100),
    response_samples: tuple[int, int] = (120, 900),
) -> np.ndarray:
    values = np.asarray(traces, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Traces must be trial by time")
    baseline_start, baseline_stop = baseline_samples
    response_start, response_stop = response_samples
    if not (
        0 <= baseline_start < baseline_stop <= response_start
        < response_stop <= values.shape[1]
    ):
        raise ValueError("Baseline and response windows are invalid")
    baseline = values[:, baseline_start:baseline_stop].mean(
        axis=1,
        keepdims=True,
    )
    return values[:, response_start:response_stop] - baseline


def waveform_correlation(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.ndim != 1 or right.shape != left.shape:
        raise ValueError("Waveforms must be aligned vectors")
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(
        np.sqrt(np.dot(left, left) * np.dot(right, right))
    )
    if denominator <= 0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def signed_peak(waveform: np.ndarray) -> float:
    values = np.asarray(waveform, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("Peak requires a nonempty waveform")
    return float(values[int(np.argmax(np.abs(values)))])


def build_source_token_model(
    stimulus_matrix: np.ndarray,
    response: np.ndarray,
    targets: np.ndarray,
    *,
    minimum_single_trials: int = 3,
) -> SourceTokenModel:
    stimulus = np.asarray(stimulus_matrix, dtype=np.float64)
    outcomes = np.asarray(response, dtype=np.float64)
    coordinates = np.asarray(targets, dtype=np.float64)
    if stimulus.ndim != 2 or outcomes.ndim != 2:
        raise ValueError("Stimulus and response must be matrices")
    if stimulus.shape[0] != outcomes.shape[0]:
        raise ValueError("Stimulus and response trials must align")
    if coordinates.ndim != 2 or coordinates.shape[0] != stimulus.shape[1]:
        raise ValueError("Target coordinates must align with sources")
    if coordinates.shape[1] < 3:
        raise ValueError("Target coordinates require a depth column")
    if minimum_single_trials < 2:
        raise ValueError("Minimum single-trial count must be at least two")

    active_count = np.count_nonzero(stimulus > 0, axis=1)
    single_mask = active_count == 1
    token_records: list[tuple[int, float, np.ndarray, int]] = []
    reliability_rows: list[dict[str, float | int]] = []
    for source in range(stimulus.shape[1]):
        powers = np.unique(stimulus[single_mask, source])
        for power in powers[powers > 0]:
            trial_indices = np.flatnonzero(
                single_mask & (stimulus[:, source] == power)
            )
            if len(trial_indices) < minimum_single_trials:
                continue
            token = outcomes[trial_indices].mean(axis=0)
            token_records.append(
                (source, float(power), token, len(trial_indices))
            )
            first_half = outcomes[trial_indices[::2]].mean(axis=0)
            second_half = outcomes[trial_indices[1::2]].mean(axis=0)
            first_peak = signed_peak(first_half)
            second_peak = signed_peak(second_half)
            peak_denominator = abs(first_peak) + abs(second_peak)
            peak_agreement = (
                1.0
                if peak_denominator == 0
                else 1.0
                - abs(first_peak - second_peak) / peak_denominator
            )
            reliability_rows.append(
                {
                    "source": source,
                    "power": float(power),
                    "n_trials": len(trial_indices),
                    "first_half_trials": len(trial_indices[::2]),
                    "second_half_trials": len(trial_indices[1::2]),
                    "half_waveform_correlation": waveform_correlation(
                        first_half,
                        second_half,
                    ),
                    "first_half_peak": first_peak,
                    "second_half_peak": second_peak,
                    "peak_amplitude_agreement": peak_agreement,
                }
            )
    if not token_records:
        raise ValueError("No source-power token met the trial threshold")

    tokens = np.vstack([record[2] for record in token_records])
    token_sources = np.asarray(
        [record[0] for record in token_records],
        dtype=np.int64,
    )
    token_powers = np.asarray(
        [record[1] for record in token_records],
        dtype=np.float64,
    )
    token_counts = np.asarray(
        [record[3] for record in token_records],
        dtype=np.int64,
    )
    power_values = np.unique(token_powers)
    anonymous_tokens = np.vstack(
        [
            tokens[token_powers == power].mean(axis=0)
            for power in power_values
        ]
    )
    return SourceTokenModel(
        tokens=tokens,
        token_sources=token_sources,
        token_powers=token_powers,
        token_counts=token_counts,
        depth_by_source=coordinates[:, -1].copy(),
        power_values=power_values,
        anonymous_tokens=anonymous_tokens,
        reliability=pd.DataFrame(reliability_rows),
    )


def build_reliability_shrunk_token_model(
    stimulus_matrix: np.ndarray,
    response: np.ndarray,
    targets: np.ndarray,
    *,
    minimum_single_trials: int = 3,
) -> ReliabilityShrinkageResult:
    stimulus = np.asarray(stimulus_matrix, dtype=np.float64)
    outcomes = np.asarray(response, dtype=np.float64)
    full_model = build_source_token_model(
        stimulus,
        outcomes,
        targets,
        minimum_single_trials=minimum_single_trials,
    )
    active_count = np.count_nonzero(stimulus > 0, axis=1)
    single_mask = active_count == 1
    first_half = []
    second_half = []
    for source, power in zip(
        full_model.token_sources,
        full_model.token_powers,
        strict=True,
    ):
        trial_indices = np.flatnonzero(
            single_mask
            & (stimulus[:, int(source)] == float(power))
        )
        first_half.append(outcomes[trial_indices[::2]].mean(axis=0))
        second_half.append(outcomes[trial_indices[1::2]].mean(axis=0))
    first = np.vstack(first_half)
    second = np.vstack(second_half)

    shrinkage_rows = []
    shrunk_tokens = full_model.tokens.copy()
    for power_index, power in enumerate(full_model.power_values):
        selected = np.flatnonzero(full_model.token_powers == power)
        first_centered = (
            first[selected] - first[selected].mean(axis=0)
        )
        second_centered = (
            second[selected] - second[selected].mean(axis=0)
        )
        cross_signal = float(
            2.0 * np.sum(first_centered * second_centered)
        )
        observed_energy = float(
            np.sum(first_centered**2)
            + np.sum(second_centered**2)
        )
        raw_reliability = (
            cross_signal / observed_energy
            if observed_energy > 0
            else 0.0
        )
        shrinkage = float(np.clip(raw_reliability, 0.0, 1.0))
        anonymous = full_model.anonymous_tokens[power_index]
        shrunk_tokens[selected] = anonymous + shrinkage * (
            full_model.tokens[selected] - anonymous
        )
        shrinkage_rows.append(
            {
                "power": float(power),
                "n_tokens": len(selected),
                "cross_signal": cross_signal,
                "observed_energy": observed_energy,
                "raw_reliability": raw_reliability,
                "shrinkage": shrinkage,
            }
        )
    power_shrinkage = pd.DataFrame(shrinkage_rows)
    shrinkage_lookup = dict(
        zip(
            power_shrinkage["power"],
            power_shrinkage["shrinkage"],
            strict=True,
        )
    )
    reliability = full_model.reliability.copy()
    reliability["power_shrinkage"] = reliability["power"].map(
        shrinkage_lookup
    )
    model = SourceTokenModel(
        tokens=shrunk_tokens,
        token_sources=full_model.token_sources.copy(),
        token_powers=full_model.token_powers.copy(),
        token_counts=full_model.token_counts.copy(),
        depth_by_source=full_model.depth_by_source.copy(),
        power_values=full_model.power_values.copy(),
        anonymous_tokens=full_model.anonymous_tokens.copy(),
        reliability=reliability,
    )
    return ReliabilityShrinkageResult(
        token_model=model,
        power_shrinkage=power_shrinkage,
    )


def build_gain_only_token_model(
    token_model: SourceTokenModel,
) -> ReducedTokenModelResult:
    """Keep source-specific gain while sharing one waveform per power."""
    reduced_tokens = np.empty_like(token_model.tokens)
    parameter_rows: list[dict[str, float | int]] = []
    anonymous_lookup = token_model.anonymous_lookup
    for index, (source, power, token) in enumerate(
        zip(
            token_model.token_sources,
            token_model.token_powers,
            token_model.tokens,
            strict=True,
        )
    ):
        shared = anonymous_lookup[float(power)]
        denominator = float(np.dot(shared, shared))
        gain = (
            max(float(np.dot(token, shared) / denominator), 0.0)
            if denominator > 0
            else 1.0
        )
        reduced_tokens[index] = gain * shared
        parameter_rows.append(
            {
                "source": int(source),
                "power": float(power),
                "gain": gain,
                "latency_shift_samples": 0,
            }
        )
    return ReducedTokenModelResult(
        token_model=_replace_tokens(token_model, reduced_tokens),
        parameters=pd.DataFrame(parameter_rows),
    )


def build_gain_latency_token_model(
    token_model: SourceTokenModel,
    *,
    maximum_shift_samples: int = 40,
) -> ReducedTokenModelResult:
    """Fit source gain and latency to the shared waveform at each power."""
    if maximum_shift_samples < 0:
        raise ValueError("Maximum shift must be nonnegative")
    reduced_tokens = np.empty_like(token_model.tokens)
    parameter_rows: list[dict[str, float | int]] = []
    anonymous_lookup = token_model.anonymous_lookup
    shifts = sorted(
        range(-maximum_shift_samples, maximum_shift_samples + 1),
        key=lambda value: (abs(value), value),
    )
    for index, (source, power, token) in enumerate(
        zip(
            token_model.token_sources,
            token_model.token_powers,
            token_model.tokens,
            strict=True,
        )
    ):
        shared = anonymous_lookup[float(power)]
        best_error = np.inf
        best_gain = 1.0
        best_shift = 0
        best_waveform = shared
        for shift in shifts:
            shifted = shift_waveform(shared, shift)
            denominator = float(np.dot(shifted, shifted))
            gain = (
                max(float(np.dot(token, shifted) / denominator), 0.0)
                if denominator > 0
                else 1.0
            )
            waveform = gain * shifted
            error = float(np.sum((token - waveform) ** 2))
            if error < best_error:
                best_error = error
                best_gain = gain
                best_shift = shift
                best_waveform = waveform
        reduced_tokens[index] = best_waveform
        parameter_rows.append(
            {
                "source": int(source),
                "power": float(power),
                "gain": best_gain,
                "latency_shift_samples": best_shift,
            }
        )
    return ReducedTokenModelResult(
        token_model=_replace_tokens(token_model, reduced_tokens),
        parameters=pd.DataFrame(parameter_rows),
    )


def shift_waveform(waveform: np.ndarray, samples: int) -> np.ndarray:
    """Shift a waveform with zero padding; positive values delay it."""
    values = np.asarray(waveform, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Waveform must be one-dimensional")
    shifted = np.zeros_like(values)
    if samples == 0:
        shifted[:] = values
    elif 0 < samples < len(values):
        shifted[samples:] = values[:-samples]
    elif -len(values) < samples < 0:
        shifted[:samples] = values[-samples:]
    return shifted


def _replace_tokens(
    token_model: SourceTokenModel,
    tokens: np.ndarray,
) -> SourceTokenModel:
    return SourceTokenModel(
        tokens=np.asarray(tokens, dtype=np.float64),
        token_sources=token_model.token_sources.copy(),
        token_powers=token_model.token_powers.copy(),
        token_counts=token_model.token_counts.copy(),
        depth_by_source=token_model.depth_by_source.copy(),
        power_values=token_model.power_values.copy(),
        anonymous_tokens=token_model.anonymous_tokens.copy(),
        reliability=token_model.reliability.copy(),
    )


def build_composition_context(
    stimulus_matrix: np.ndarray,
    response: np.ndarray,
    token_model: SourceTokenModel,
) -> CompositionContext:
    stimulus = np.asarray(stimulus_matrix, dtype=np.float64)
    outcomes = np.asarray(response, dtype=np.float64)
    if stimulus.ndim != 2 or outcomes.ndim != 2:
        raise ValueError("Stimulus and response must be matrices")
    if stimulus.shape[0] != outcomes.shape[0]:
        raise ValueError("Stimulus and response trials must align")
    if stimulus.shape[1] != len(token_model.depth_by_source):
        raise ValueError("Stimulus sources do not align with token model")

    token_lookup = token_model.token_lookup
    anonymous_lookup = token_model.anonymous_lookup
    eligible: list[
        tuple[int, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    active_count = np.count_nonzero(stimulus > 0, axis=1)
    for trial_index in np.flatnonzero(active_count > 1):
        sources = np.flatnonzero(stimulus[trial_index] > 0)
        powers = stimulus[trial_index, sources]
        keys = [
            (int(source), float(power))
            for source, power in zip(sources, powers, strict=True)
        ]
        if not all(key in token_lookup for key in keys):
            continue
        indices = np.asarray(
            [token_lookup[key] for key in keys],
            dtype=np.int64,
        )
        eligible.append(
            (
                int(trial_index),
                sources.astype(np.int64),
                powers.astype(np.float64),
                indices,
            )
        )

    sample_count = outcomes.shape[1]
    if not eligible:
        empty_float = np.empty((0, 0), dtype=np.float64)
        empty_int = np.empty((0, 0), dtype=np.int64)
        return CompositionContext(
            trial_indices=np.empty(0, dtype=np.int64),
            ensemble_sizes=np.empty(0, dtype=np.int64),
            response=np.empty((0, sample_count), dtype=np.float64),
            source_prediction=np.empty(
                (0, sample_count),
                dtype=np.float64,
            ),
            anonymous_prediction=np.empty(
                (0, sample_count),
                dtype=np.float64,
            ),
            true_token_indices=empty_int,
            active_sources=empty_int.copy(),
            active_powers=empty_float,
            active_mask=np.empty((0, 0), dtype=bool),
            token_model=token_model,
        )

    trial_indices = np.asarray(
        [record[0] for record in eligible],
        dtype=np.int64,
    )
    ensemble_sizes = np.asarray(
        [len(record[1]) for record in eligible],
        dtype=np.int64,
    )
    max_sources = int(ensemble_sizes.max())
    shape = (len(eligible), max_sources)
    true_token_indices = np.zeros(shape, dtype=np.int64)
    active_sources = np.zeros(shape, dtype=np.int64)
    active_powers = np.full(shape, np.nan, dtype=np.float64)
    active_mask = np.zeros(shape, dtype=bool)
    source_prediction = np.zeros(
        (len(eligible), sample_count),
        dtype=np.float64,
    )
    anonymous_prediction = np.zeros_like(source_prediction)
    for row, (_, sources, powers, token_indices) in enumerate(eligible):
        count = len(sources)
        true_token_indices[row, :count] = token_indices
        active_sources[row, :count] = sources
        active_powers[row, :count] = powers
        active_mask[row, :count] = True
        source_prediction[row] = token_model.tokens[
            token_indices
        ].sum(axis=0)
        anonymous_prediction[row] = np.vstack(
            [anonymous_lookup[float(power)] for power in powers]
        ).sum(axis=0)
    return CompositionContext(
        trial_indices=trial_indices,
        ensemble_sizes=ensemble_sizes,
        response=outcomes[trial_indices],
        source_prediction=source_prediction,
        anonymous_prediction=anonymous_prediction,
        true_token_indices=true_token_indices,
        active_sources=active_sources,
        active_powers=active_powers,
        active_mask=active_mask,
        token_model=token_model,
    )


def composition_metrics(
    context: CompositionContext,
    *,
    sample_period_ms: float = 0.05,
) -> pd.DataFrame:
    if sample_period_ms <= 0:
        raise ValueError("Sample period must be positive")
    columns = (
        "trial_index",
        "ensemble_size",
        "mse_zero",
        "mse_anonymous",
        "mse_source",
        "mse_gain_source_vs_zero",
        "mse_gain_source_vs_anonymous",
        "rho_anonymous",
        "rho_source",
        "rho_gain_source_vs_anonymous",
        "integrated_error_zero",
        "integrated_error_anonymous",
        "integrated_error_source",
        "integrated_error_gain_source_vs_anonymous",
        "peak_error_zero",
        "peak_error_anonymous",
        "peak_error_source",
        "peak_error_gain_source_vs_anonymous",
    )
    if not len(context.trial_indices):
        return pd.DataFrame(columns=columns)

    response = context.response
    source = context.source_prediction
    anonymous = context.anonymous_prediction
    zero = np.zeros_like(response)
    mse_zero = np.mean(response**2, axis=1)
    mse_anonymous = np.mean((response - anonymous) ** 2, axis=1)
    mse_source = np.mean((response - source) ** 2, axis=1)
    rho_anonymous = rowwise_correlation(response, anonymous)
    rho_source = rowwise_correlation(response, source)

    response_integral = response.sum(axis=1) * sample_period_ms
    anonymous_integral = anonymous.sum(axis=1) * sample_period_ms
    source_integral = source.sum(axis=1) * sample_period_ms
    zero_integral = zero.sum(axis=1) * sample_period_ms
    integrated_zero = np.abs(response_integral - zero_integral)
    integrated_anonymous = np.abs(
        response_integral - anonymous_integral
    )
    integrated_source = np.abs(response_integral - source_integral)

    response_peak = rowwise_signed_peak(response)
    anonymous_peak = rowwise_signed_peak(anonymous)
    source_peak = rowwise_signed_peak(source)
    zero_peak = rowwise_signed_peak(zero)
    peak_zero = np.abs(response_peak - zero_peak)
    peak_anonymous = np.abs(response_peak - anonymous_peak)
    peak_source = np.abs(response_peak - source_peak)

    return pd.DataFrame(
        {
            "trial_index": context.trial_indices,
            "ensemble_size": context.ensemble_sizes,
            "mse_zero": mse_zero,
            "mse_anonymous": mse_anonymous,
            "mse_source": mse_source,
            "mse_gain_source_vs_zero": mse_zero - mse_source,
            "mse_gain_source_vs_anonymous": (
                mse_anonymous - mse_source
            ),
            "rho_anonymous": rho_anonymous,
            "rho_source": rho_source,
            "rho_gain_source_vs_anonymous": (
                rho_source - rho_anonymous
            ),
            "integrated_error_zero": integrated_zero,
            "integrated_error_anonymous": integrated_anonymous,
            "integrated_error_source": integrated_source,
            "integrated_error_gain_source_vs_anonymous": (
                integrated_anonymous - integrated_source
            ),
            "peak_error_zero": peak_zero,
            "peak_error_anonymous": peak_anonymous,
            "peak_error_source": peak_source,
            "peak_error_gain_source_vs_anonymous": (
                peak_anonymous - peak_source
            ),
        },
        columns=columns,
    )


def rowwise_correlation(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.ndim != 2 or right.shape != left.shape:
        raise ValueError("Waveform matrices must align")
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(
        np.sum(left_centered**2, axis=1)
        * np.sum(right_centered**2, axis=1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )


def rowwise_signed_peak(waveforms: np.ndarray) -> np.ndarray:
    values = np.asarray(waveforms, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Waveforms must be a matrix")
    if values.shape[1] < 1:
        raise ValueError("Waveforms cannot have zero samples")
    indices = np.argmax(np.abs(values), axis=1)
    return values[np.arange(len(values)), indices]


def build_composition_geometry(
    context: CompositionContext,
) -> CompositionGeometry:
    response = context.response
    tokens = context.token_model.tokens
    response_sum = response.sum(axis=1)
    response_squared_sum = np.sum(response**2, axis=1)
    response_centered_squared_sum = np.maximum(
        response_squared_sum
        - response_sum**2 / response.shape[1],
        0.0,
    )
    return CompositionGeometry(
        response_sum=response_sum,
        response_squared_sum=response_squared_sum,
        response_centered_squared_sum=(
            response_centered_squared_sum
        ),
        response_token_dot=response @ tokens.T,
        token_sum=tokens.sum(axis=1),
        token_gram=tokens @ tokens.T,
        anonymous_mse=np.mean(
            (response - context.anonymous_prediction) ** 2,
            axis=1,
        ),
        anonymous_correlation=rowwise_correlation(
            response,
            context.anonymous_prediction,
        ),
    )


def selected_token_metrics(
    context: CompositionContext,
    geometry: CompositionGeometry,
    selected_token_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(selected_token_indices, dtype=np.int64)
    if selected.shape != context.true_token_indices.shape:
        raise ValueError("Selected token indices must align with trials")
    if np.any(
        (selected[context.active_mask] < 0)
        | (
            selected[context.active_mask]
            >= len(context.token_model.tokens)
        )
    ):
        raise ValueError("Selected token index is outside the token library")
    safe = np.where(context.active_mask, selected, 0)
    response_prediction_dot = np.sum(
        np.where(
            context.active_mask,
            np.take_along_axis(
                geometry.response_token_dot,
                safe,
                axis=1,
            ),
            0.0,
        ),
        axis=1,
    )
    prediction_sum = np.sum(
        np.where(
            context.active_mask,
            geometry.token_sum[safe],
            0.0,
        ),
        axis=1,
    )
    pair_mask = (
        context.active_mask[:, :, None]
        & context.active_mask[:, None, :]
    )
    prediction_squared_sum = np.sum(
        np.where(
            pair_mask,
            geometry.token_gram[
                safe[:, :, None],
                safe[:, None, :],
            ],
            0.0,
        ),
        axis=(1, 2),
    )
    sample_count = context.response.shape[1]
    mse = np.maximum(
        (
            geometry.response_squared_sum
            - 2.0 * response_prediction_dot
            + prediction_squared_sum
        )
        / sample_count,
        0.0,
    )
    centered_dot = (
        response_prediction_dot
        - geometry.response_sum * prediction_sum / sample_count
    )
    prediction_centered_squared_sum = np.maximum(
        prediction_squared_sum
        - prediction_sum**2 / sample_count,
        0.0,
    )
    denominator = np.sqrt(
        geometry.response_centered_squared_sum
        * prediction_centered_squared_sum
    )
    correlation = np.divide(
        centered_dot,
        denominator,
        out=np.zeros_like(centered_dot),
        where=denominator > 0,
    )
    return mse, correlation


def source_identity_nulls(
    context: CompositionContext,
    geometry: CompositionGeometry,
    *,
    repetitions: int = 999,
    seed: int = 0,
) -> pd.DataFrame:
    if repetitions < 1:
        raise ValueError("Null repetitions must be positive")
    if not len(context.trial_indices):
        return pd.DataFrame(
            columns=(
                "repetition",
                "mse_gain_source_vs_anonymous",
                "rho_gain_source_vs_anonymous",
            )
        )
    groups: dict[tuple[float, float], list[int]] = {}
    model = context.token_model
    for token_index, (source, power) in enumerate(
        zip(
            model.token_sources,
            model.token_powers,
            strict=True,
        )
    ):
        key = (
            float(model.depth_by_source[int(source)]),
            float(power),
        )
        groups.setdefault(key, []).append(token_index)

    rng = np.random.default_rng(seed)
    rows = []
    for repetition in range(repetitions):
        remap = np.arange(len(model.tokens), dtype=np.int64)
        for token_indices in groups.values():
            indices = np.asarray(token_indices, dtype=np.int64)
            remap[indices] = rng.permutation(indices)
        selected = remap[context.true_token_indices]
        mse, correlation = selected_token_metrics(
            context,
            geometry,
            selected,
        )
        rows.append(
            {
                "repetition": repetition,
                "mse_gain_source_vs_anonymous": float(
                    np.mean(geometry.anonymous_mse - mse)
                ),
                "rho_gain_source_vs_anonymous": float(
                    np.mean(
                        correlation
                        - geometry.anonymous_correlation
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def selective_token_ablation(
    context: CompositionContext,
    geometry: CompositionGeometry,
) -> pd.DataFrame:
    columns = (
        "trial_index",
        "ensemble_size",
        "source",
        "power",
        "depth",
        "true_token_index",
        "n_decoys",
        "true_removal_mse_cost",
        "mean_decoy_replacement_mse_cost",
        "median_decoy_replacement_mse_cost",
    )
    if not len(context.trial_indices):
        return pd.DataFrame(columns=columns)

    model = context.token_model
    groups: dict[tuple[float, float], np.ndarray] = {}
    for token_index, (source, power) in enumerate(
        zip(
            model.token_sources,
            model.token_powers,
            strict=True,
        )
    ):
        key = (
            float(model.depth_by_source[int(source)]),
            float(power),
        )
        groups.setdefault(key, []).append(token_index)
    groups = {
        key: np.asarray(indices, dtype=np.int64)
        for key, indices in groups.items()
    }

    sample_count = context.response.shape[1]
    rows: list[dict[str, float | int]] = []
    for row in range(len(context.trial_indices)):
        active = context.active_mask[row]
        selected = context.true_token_indices[row, active]
        active_sources = set(
            int(value) for value in context.active_sources[row, active]
        )
        prediction_token_dot = geometry.token_gram[
            selected
        ].sum(axis=0)
        response_token_dot = geometry.response_token_dot[row]
        for position, true_index in enumerate(selected):
            source = int(model.token_sources[true_index])
            power = float(model.token_powers[true_index])
            depth = float(model.depth_by_source[source])
            candidates = groups[(depth, power)]
            decoys = np.asarray(
                [
                    index
                    for index in candidates
                    if int(model.token_sources[index])
                    not in active_sources
                ],
                dtype=np.int64,
            )
            if not len(decoys):
                continue
            residual_dot_true = (
                response_token_dot[true_index]
                - prediction_token_dot[true_index]
            )
            true_removal_cost = (
                2.0 * residual_dot_true
                + geometry.token_gram[true_index, true_index]
            ) / sample_count

            residual_dot_delta = (
                response_token_dot[decoys]
                - response_token_dot[true_index]
                - prediction_token_dot[decoys]
                + prediction_token_dot[true_index]
            )
            delta_squared = (
                geometry.token_gram[decoys, decoys]
                + geometry.token_gram[true_index, true_index]
                - 2.0 * geometry.token_gram[decoys, true_index]
            )
            replacement_cost = (
                -2.0 * residual_dot_delta + delta_squared
            ) / sample_count
            rows.append(
                {
                    "trial_index": int(context.trial_indices[row]),
                    "ensemble_size": int(
                        context.ensemble_sizes[row]
                    ),
                    "source": source,
                    "power": power,
                    "depth": depth,
                    "true_token_index": int(true_index),
                    "n_decoys": len(decoys),
                    "true_removal_mse_cost": float(
                        true_removal_cost
                    ),
                    "mean_decoy_replacement_mse_cost": float(
                        replacement_cost.mean()
                    ),
                    "median_decoy_replacement_mse_cost": float(
                        np.median(replacement_cost)
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def exact_sign_flip_test(
    values: Sequence[float],
) -> SignFlipResult:
    effects = np.asarray(values, dtype=np.float64)
    if effects.ndim != 1 or not len(effects):
        raise ValueError("Sign-flip effects must be a nonempty vector")
    if len(effects) > 20:
        raise ValueError("Exact sign-flip supports at most 20 files")
    if not np.all(np.isfinite(effects)):
        raise ValueError("Sign-flip effects must be finite")
    signs = np.asarray(
        list(product((-1.0, 1.0), repeat=len(effects))),
        dtype=np.float64,
    )
    null = (signs * effects[None, :]).mean(axis=1)
    observed = float(effects.mean())
    return SignFlipResult(
        observed_mean=observed,
        p_value=float(np.mean(null >= observed - 1e-15)),
        n_files=len(effects),
        n_assignments=len(null),
    )


def benjamini_hochberg(
    p_values: Sequence[float],
) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("P-values must be a vector")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("P-values must lie in [0, 1]")
    order = np.argsort(values)
    ranked = (
        values[order]
        * len(values)
        / np.arange(1, len(values) + 1)
    )
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def leave_one_out_means(
    effects: Sequence[float],
) -> np.ndarray:
    values = np.asarray(effects, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("Leave-one-out requires at least two effects")
    return (values.sum() - values) / (len(values) - 1)
