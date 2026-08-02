from __future__ import annotations

import h5py
import numpy as np
import pytest

from wetware_interp.holographic_source_composition import (
    baseline_correct_response,
    benjamini_hochberg,
    build_composition_context,
    build_composition_geometry,
    build_gain_latency_token_model,
    build_gain_only_token_model,
    build_reliability_shrunk_token_model,
    build_source_token_model,
    composition_metrics,
    exact_sign_flip_test,
    leave_one_out_means,
    load_holographic_recording,
    selected_token_metrics,
    selective_token_ablation,
    source_identity_nulls,
    shift_waveform,
)


def _synthetic_composition():
    token_a = np.asarray([1.0, 0.0, 0.0, 0.0])
    token_b = np.asarray([0.0, 1.0, 0.0, 0.0])
    token_c = np.asarray([0.0, 0.0, 3.0, 0.0])
    tokens = (token_a, token_b, token_c)
    stimulus_rows = []
    response_rows = []
    for source, token in enumerate(tokens):
        for _ in range(3):
            stimulus = np.zeros(3)
            stimulus[source] = 30.0
            stimulus_rows.append(stimulus)
            response_rows.append(token)
    stimulus_rows.extend(
        [
            np.asarray([30.0, 30.0, 0.0]),
            np.asarray([30.0, 30.0, 0.0]),
        ]
    )
    response_rows.extend([token_a + token_b, token_a + token_b])
    stimulus = np.vstack(stimulus_rows)
    response = np.vstack(response_rows)
    targets = np.asarray(
        [
            [0.0, 0.0, 25.0],
            [1.0, 0.0, 25.0],
            [2.0, 0.0, 25.0],
        ]
    )
    model = build_source_token_model(
        stimulus,
        response,
        targets,
        minimum_single_trials=3,
    )
    context = build_composition_context(
        stimulus,
        response,
        model,
    )
    return stimulus, response, model, context


def test_loader_reads_trial_first_hdf5_orientation(tmp_path) -> None:
    path = tmp_path / "toy.mat"
    stimulus = np.asarray(
        [
            [30.0, 0.0],
            [0.0, 30.0],
            [30.0, 30.0],
        ]
    )
    traces = np.arange(2700, dtype=float).reshape(3, 900)
    targets = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [25.0, 50.0],
        ]
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("stimulus_matrix", data=stimulus)
        handle.create_dataset("pscs_demixed", data=traces)
        handle.create_dataset("pscs", data=traces + 1.0)
        handle.create_dataset("targets", data=targets)
        handle.create_dataset("caviar_weights_single", data=np.ones(2))
        handle.create_dataset(
            "caviar_weights_ensemble",
            data=np.ones(2),
        )

    recording = load_holographic_recording(path)

    assert recording.stimulus_matrix.shape == (3, 2)
    assert recording.pscs_demixed.shape == (3, 900)
    assert recording.targets.shape == (2, 3)
    assert np.allclose(recording.targets[:, -1], [25.0, 50.0])


def test_loader_excludes_only_all_nan_blank_trace(tmp_path) -> None:
    path = tmp_path / "blank_nan.mat"
    stimulus = np.asarray(
        [[30.0, 0.0], [0.0, 0.0]],
        dtype=float,
    )
    traces = np.zeros((2, 900), dtype=float)
    traces[1] = np.nan
    with h5py.File(path, "w") as handle:
        handle.create_dataset("stimulus_matrix", data=stimulus)
        handle.create_dataset("pscs_demixed", data=traces)
        handle.create_dataset("pscs", data=np.zeros_like(traces))
        handle.create_dataset(
            "targets",
            data=np.zeros((3, 2), dtype=float),
        )
        handle.create_dataset("caviar_weights_single", data=np.ones(2))
        handle.create_dataset("caviar_weights_multi", data=np.ones(2))

    recording = load_holographic_recording(path)

    assert recording.excluded_blank_nonfinite_demixed == 1
    assert np.allclose(recording.pscs_demixed[1], 0.0)


def test_loader_rejects_nonfinite_stimulated_trace(tmp_path) -> None:
    path = tmp_path / "stimulated_nan.mat"
    stimulus = np.asarray([[30.0], [0.0]], dtype=float)
    traces = np.zeros((2, 900), dtype=float)
    traces[0] = np.nan
    with h5py.File(path, "w") as handle:
        handle.create_dataset("stimulus_matrix", data=stimulus)
        handle.create_dataset("pscs_demixed", data=traces)
        handle.create_dataset("pscs", data=np.zeros_like(traces))
        handle.create_dataset(
            "targets",
            data=np.zeros((3, 1), dtype=float),
        )
        handle.create_dataset("caviar_weights_single", data=np.ones(1))
        handle.create_dataset("caviar_weights_multi", data=np.ones(1))

    with pytest.raises(ValueError, match="invalid nonfinite rows"):
        load_holographic_recording(path)


def test_baseline_correction_uses_frozen_windows() -> None:
    traces = np.zeros((2, 900), dtype=float)
    traces[0, :100] = 2.0
    traces[0, 120:] = 5.0
    traces[1, :100] = -1.0
    traces[1, 120:] = -4.0

    response = baseline_correct_response(traces)

    assert response.shape == (2, 780)
    assert np.allclose(response[0], 3.0)
    assert np.allclose(response[1], -3.0)


def test_anonymous_token_equal_weights_source_means() -> None:
    stimulus = np.zeros((11, 2), dtype=float)
    response = np.zeros((11, 3), dtype=float)
    stimulus[:3, 0] = 30.0
    response[:3] = 1.0
    stimulus[3:11, 1] = 30.0
    response[3:11] = 5.0
    targets = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )

    model = build_source_token_model(
        stimulus,
        response,
        targets,
        minimum_single_trials=3,
    )

    assert np.allclose(model.tokens[0], 1.0)
    assert np.allclose(model.tokens[1], 5.0)
    assert np.allclose(model.anonymous_tokens[0], 3.0)


def test_split_half_shrinkage_retains_reproducible_source_signal() -> None:
    stimulus, response, full_model, _ = _synthetic_composition()
    targets = np.asarray(
        [
            [0.0, 0.0, 25.0],
            [1.0, 0.0, 25.0],
            [2.0, 0.0, 25.0],
        ]
    )

    result = build_reliability_shrunk_token_model(
        stimulus,
        response,
        targets,
        minimum_single_trials=3,
    )

    assert np.allclose(result.power_shrinkage["shrinkage"], 1.0)
    assert np.allclose(result.token_model.tokens, full_model.tokens)
    assert np.allclose(
        result.token_model.anonymous_tokens,
        full_model.anonymous_tokens,
    )


def test_split_half_shrinkage_removes_antireproducible_signal() -> None:
    stimulus = np.zeros((8, 2), dtype=float)
    response = np.zeros((8, 3), dtype=float)
    for trial in range(4):
        stimulus[trial, 0] = 30.0
        response[trial, 0] = 1.0 if trial % 2 == 0 else -1.0
        stimulus[trial + 4, 1] = 30.0
        response[trial + 4, 0] = (
            -1.0 if trial % 2 == 0 else 1.0
        )
    targets = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )

    result = build_reliability_shrunk_token_model(
        stimulus,
        response,
        targets,
        minimum_single_trials=3,
    )

    assert np.allclose(result.power_shrinkage["shrinkage"], 0.0)
    assert np.allclose(
        result.token_model.tokens,
        result.token_model.anonymous_tokens[0],
    )


def test_source_tokens_predict_unseen_ensemble_without_fitting() -> None:
    _, _, _, context = _synthetic_composition()

    metrics = composition_metrics(context)

    assert len(metrics) == 2
    assert np.allclose(metrics["mse_source"], 0.0)
    assert np.all(metrics["mse_gain_source_vs_anonymous"] > 0)
    assert np.all(metrics["rho_gain_source_vs_anonymous"] > 0)


def test_gain_only_model_keeps_shared_shape_and_source_amplitude() -> None:
    _, _, model, _ = _synthetic_composition()

    result = build_gain_only_token_model(model)

    shared = model.anonymous_tokens[0]
    for token, gain in zip(
        result.token_model.tokens,
        result.parameters["gain"],
        strict=True,
    ):
        assert np.allclose(token, gain * shared)


def test_gain_only_model_clips_opposite_sign_projection() -> None:
    _, _, model, _ = _synthetic_composition()
    opposite_tokens = model.tokens.copy()
    opposite_tokens[0] = -model.anonymous_tokens[0]
    opposite_model = type(model)(
        tokens=opposite_tokens,
        token_sources=model.token_sources,
        token_powers=model.token_powers,
        token_counts=model.token_counts,
        depth_by_source=model.depth_by_source,
        power_values=model.power_values,
        anonymous_tokens=model.anonymous_tokens,
        reliability=model.reliability,
    )

    result = build_gain_only_token_model(opposite_model)

    assert result.parameters.loc[0, "gain"] == 0.0
    assert np.allclose(result.token_model.tokens[0], 0.0)


def test_gain_latency_model_recovers_known_delay() -> None:
    _, _, model, _ = _synthetic_composition()
    shared = model.anonymous_tokens[0]
    delayed = shift_waveform(shared, 1) * 2.0
    delayed_tokens = np.vstack(
        [delayed, model.tokens[1], model.tokens[2]]
    )
    delayed_model = type(model)(
        tokens=delayed_tokens,
        token_sources=model.token_sources,
        token_powers=model.token_powers,
        token_counts=model.token_counts,
        depth_by_source=model.depth_by_source,
        power_values=model.power_values,
        anonymous_tokens=model.anonymous_tokens,
        reliability=model.reliability,
    )

    result = build_gain_latency_token_model(
        delayed_model,
        maximum_shift_samples=2,
    )

    assert result.parameters.loc[0, "latency_shift_samples"] == 1
    assert np.isclose(result.parameters.loc[0, "gain"], 2.0)
    assert np.allclose(result.token_model.tokens[0], delayed)


def test_fast_geometry_matches_direct_waveform_metrics() -> None:
    _, _, _, context = _synthetic_composition()
    geometry = build_composition_geometry(context)

    mse, correlation = selected_token_metrics(
        context,
        geometry,
        context.true_token_indices,
    )
    direct = composition_metrics(context)

    assert np.allclose(mse, direct["mse_source"])
    assert np.allclose(correlation, direct["rho_source"])


def test_depth_power_null_is_deterministic() -> None:
    _, _, _, context = _synthetic_composition()
    geometry = build_composition_geometry(context)

    first = source_identity_nulls(
        context,
        geometry,
        repetitions=17,
        seed=11,
    )
    second = source_identity_nulls(
        context,
        geometry,
        repetitions=17,
        seed=11,
    )

    assert first.equals(second)
    assert len(first) == 17
    assert np.isfinite(
        first[
            [
                "mse_gain_source_vs_anonymous",
                "rho_gain_source_vs_anonymous",
            ]
        ].to_numpy()
    ).all()


def test_matched_decoy_replacement_detects_source_identity() -> None:
    _, _, _, context = _synthetic_composition()
    geometry = build_composition_geometry(context)

    ablation = selective_token_ablation(context, geometry)

    assert len(ablation) == 4
    assert (ablation["n_decoys"] == 1).all()
    assert (
        ablation["mean_decoy_replacement_mse_cost"] > 0
    ).all()


def test_exact_sign_flip_bh_and_leave_one_out() -> None:
    result = exact_sign_flip_test([1.0, 2.0, 3.0])
    adjusted = benjamini_hochberg([result.p_value, 0.5])
    leave_one_out = leave_one_out_means([1.0, 2.0, 3.0])

    assert result.observed_mean == 2.0
    assert result.p_value == 0.125
    assert np.allclose(adjusted, [0.25, 0.5])
    assert np.allclose(leave_one_out, [2.5, 2.0, 1.5])
