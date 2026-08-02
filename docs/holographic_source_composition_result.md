# SST holographic source-composition result

## Decision

The locked SST opening gate failed. The sealed PV cohort remains unopened and
must not be used to rescue the hypothesis.

The failure is narrow: full source-and-power waveforms carry selective
amplitude information, but they do not improve waveform shape correlation over
an anonymous power-matched template.

## Frozen analysis

- Protocol SHA256:
  `ef740262f89da0c1b4051b24c52b57084c44f8bfdd7f407b7f541f59dc02d45a`
- Manifest SHA256:
  `65864e3126d6d1b9c0f8ce11e9ff1bbd47a423fcdef4f64a54792660cadb04ab`
- Nine checksum-verified `SSTtoE` files.
- 46,578 total trials, including 26,803 single-source trials.
- 19,217 of 19,769 ensemble trials had every exact source-power token.
- Tokens were estimated only from single-source trials.
- Ensemble trials fit no coefficient, gain, latency, or nonlinear parameter.
- 999 depth-and-power-preserving source-identity permutations per file.

## Primary tests

| Comparison | Mean file effect | Exact p | BH q | Files positive | Decision |
|---|---:|---:|---:|---:|---|
| Source vs anonymous, MSE | 0.0008777 | 0.001953 | 0.003906 | 9/9 | Pass |
| Source vs anonymous, waveform rho | -0.002628 | 0.6816 | 0.6816 | 6/9 | Fail |

The minimum leave-one-file-out MSE gain was positive (`0.0005465`), whereas
the minimum waveform-correlation gain was negative (`-0.005129`).

## Identity and necessity

Both observed primary statistics exceeded the standardized familywise 99th
percentile of the source-identity null:

| Metric | Observed z | Familywise threshold z | Familywise p |
|---|---:|---:|---:|
| MSE gain | 11.94 | 3.20 | 0.001 |
| Waveform-rho gain | 19.37 | 3.20 | 0.001 |

The negative absolute rho gain still beats shuffled source identities because
shuffling is substantially worse (`null mean = -0.02973`). Thus true identity
contains waveform information, but the anonymous average is a better
shape-denoising estimator.

Replacing an active token with a nonstimulated token at the same depth and
power increased MSE in all nine files. The mean file effect was `0.0002416`
with exact `p = 0.001953`.

## Supporting outcomes

Source-specific addition improved integrated-current error in 9/9 files
(mean `0.3166`) and peak-current error in 9/9 files (mean `0.01718`). A
post-hoc three-outcome correction gives `q = 0.00293` for both. Raw-current
robustness did not pass: MSE gain was not significant and waveform correlation
was worse.

## Mechanistic boundary

The supported statement is:

> Under SST holographic stimulation, source identity contributes a selective
> and additively useful current-amplitude term beyond stimulation power and
> depth.

The unsupported statement is:

> A neuron's complete single-source waveform is a stable compositional token.

This does not establish a monosemantic neuron or a complete neural language.

## Exploratory diagnostic

Across the nine files, median split-half token waveform reliability tracked
the file-level waveform-correlation gain with Spearman `rho = 0.957`. The two
largest negative shape effects occurred in files whose median split-half
correlation was zero. This association was examined after the frozen result
and is hypothesis-generating.

The next development test will use split-half reliability estimated only from
single-source trials to shrink noisy source deviations toward the anonymous
power template. No ensemble response may select the shrinkage. A successful
SST development result must be confirmed in an unopened non-PV cohort under a
new lock; the failed PV opening contract remains binding.
