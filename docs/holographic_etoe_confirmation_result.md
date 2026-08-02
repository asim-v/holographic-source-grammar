# E-to-E confirmation of a reliability-weighted source grammar

> Historical locked-gate report. Reduced-model, raw-current, and
> biological-unit sensitivities added after external feedback are reported in
> `docs/holographic_feedback_analyses.md`. Those analyses narrow the strongest
> current claim to source-specific response strength and latency.

## Decision

The prospective E-to-E confirmation passed every locked gate.

Together with the opened-SST development result, this supports a scoped causal
grammar:

`message(s,p,t) = shared(p,t) +
                  reliability(p) * source_deviation(s,p,t)`

`ensemble_response(t) = sum of active source messages`

The rule was learned only from single-source interventions and predicted
previously unused multi-source interventions.

## Frozen analysis

- Protocol SHA256:
  `e8e48ec69bf0bb5fa99226e737d46f3d2e8ab21cb32c471be6863b5eacb10d83`
- Manifest SHA256:
  `984f4e3b5187fb80420af96be40190775a7906419fe2036f5186c54e2be5d237`
- Ten checksum-verified `EtoE` files.
- 204,054 total trials.
- 129,613 single-source trials estimated tokens and reliability.
- 74,435 ensemble trials evaluated composition and fit no parameter.
- All ten files and all ensemble trials were eligible.
- 999 depth-and-power-preserving source permutations per file.

One file used the unused schema alias `caviar_weights_ensemble` rather than
`caviar_weights_multi`. This was recorded before any response value was read.

## Primary confirmation

| Comparison | Mean file effect | Exact p | BH q | Files positive |
|---|---:|---:|---:|---:|
| Source vs anonymous, MSE | 0.00002120 | 0.000977 | 0.001953 | 10/10 |
| Source vs anonymous, waveform rho | 0.003973 | 0.008789 | 0.008789 | 9/10 |

The minimum leave-one-file-out effects were `0.00000562` for MSE and
`0.002800` for waveform correlation.

The mean demixed reliability coefficient was `0.4791`. Thus, in E-to-E data,
roughly half of the raw source-specific deviation was retained by a symmetric
split-half estimator learned without ensemble responses.

## Identity and necessity

Both primary effects exceeded the familywise identity-null threshold:

| Metric | Observed z | Familywise threshold z | Familywise p |
|---|---:|---:|---:|
| MSE gain | 29.47 | 2.93 | 0.001 |
| Waveform-rho gain | 35.93 | 2.93 | 0.001 |

Replacing an active token with a nonstimulated token at the same depth and
power increased MSE in all ten files. The mean file effect was `0.00000457`
with exact `p = 0.000977`.

## Robustness and limit

Integrated-current and peak-current errors improved in 10/10 files; a
post-hoc three-supporting-outcome correction gives `q = 0.00146` for both.
Raw-current MSE also passed, but raw waveform correlation did not
(`q = 0.803`). Demixing is therefore important for the temporal-shape claim.

The demixed source predictor had mean waveform correlation `0.4109`, compared
with `0.4070` for the anonymous predictor. It did not consistently beat an
all-zero waveform by file-level MSE: only 4/10 files were positive and the
exact test was `p = 0.456`. Small or absent currents make zero a strong
squared-error predictor in many preparations.

The confirmed claim is therefore incremental and source-specific: knowing
which neurons were stimulated improves the predicted current beyond exact
power, depth, ensemble size, and a shared waveform. This is not yet a complete
digital twin of total current.

## Mechanistic claim boundary

The evidence is:

- **predictive:** single-source tokens predict held-out ensemble waveforms;
- **selective:** true source identity beats depth-power permutations and
  matched nonstimulated source substitutions;
- **transferable:** the same frozen reliability rule passes in inhibitory SST
  development data and a prospectively locked excitatory-source cohort;
- **perturbational:** source neurons and their combinations were directly
  stimulated holographically.

This warrants the scoped statement:

> In these cortical circuit-mapping preparations, a stimulated neuron
> contributes a reproducible source-specific current deviation around a
> power-conditioned shared waveform, and those deviations compose
> approximately additively across simultaneous interventions.

It does not establish that a neuron is monosemantic, that this current token is
its meaning during natural behavior, or that neural computation is generally
additive.

## Next falsification

The current lexicon memorizes each exact source-power pair. The next compactness
test must hold out that pair and ask whether a source message learned at other
powers predicts its contribution at the unseen power. Passing would show a
power-invariant source word rather than a lookup table. Failure would bound the
grammar to exact intervention intensity.
