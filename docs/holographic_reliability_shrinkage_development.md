# Locked development screen: reliability-shrunk source tokens

Status: prospective with respect to the shrinkage analysis, but explicitly
post-hoc with respect to the opened SST ensemble outcomes.

This screen cannot change the failed SST gate in the original holographic
source-composition protocol and cannot authorize opening PV.

## Motivation

The frozen full-waveform analysis showed:

- source-specific MSE, integral, peak, identity-null, and ablation effects;
- no source-specific waveform-correlation advantage over an anonymous
  power-matched token;
- an exploratory association between split-half token reliability and
  waveform-correlation gain.

The resulting mechanistic hypothesis is that source identity contributes a
reproducible low-dimensional amplitude deviation around a shared temporal
response, while unreproducible source-waveform detail is measurement noise.

## Data boundary

- The same nine checksum-pinned SST files and time windows as the original
  protocol.
- Single-source trials estimate every token and shrinkage coefficient.
- Ensemble trials are prediction-only and fit no value.
- The PV cohort remains sealed regardless of this result.
- The unopened ten-file `EtoE` cohort is not downloaded or inspected during
  this development screen.

## Split-half reliability shrinkage

For each file, outcome, and exact stimulation power:

1. Form alternating-half source tokens `A_s` and `B_s`.
2. Center each half independently over source identity:
   `A'_s = A_s - mean_s(A_s)` and
   `B'_s = B_s - mean_s(B_s)`.
3. Estimate the symmetric reliability ratio

   `lambda = 2 sum_s,t(A'_s B'_s) /
             [sum_s,t(A'_s^2) + sum_s,t(B'_s^2)]`.

4. Clip `lambda` to `[0, 1]`.
5. Form the final full-data token

   `K_shrunk(s,p) = K_anonymous(p) +
                    lambda_p [K_full(s,p) - K_anonymous(p)]`.

There is no candidate grid, ensemble-based selection, source filter, latency
shift, gain fit, or nonlinear ensemble parameter.

## Development screen

Apply the same eligible trials, anonymous baseline, metrics, exact file-level
sign-flip tests, BH correction, leave-one-file-out checks, 999 depth-power
source permutations, and matched-decoy substitution as the original protocol.

The shrinkage hypothesis advances to a separately locked `EtoE` confirmation
only if:

1. both demixed MSE and waveform-correlation gains over anonymous are positive
   with `q < 0.05`;
2. every leave-one-file-out aggregate is positive for both metrics;
3. both observed metrics exceed the standardized familywise 99th-percentile
   identity-null threshold;
4. matched-decoy replacement cost is positive with exact `p < 0.05`;
5. all nine SST files have eligible ensemble trials.

Raw-current analysis is supporting. Passing this screen is development
evidence only.

## Interpretation boundary

A pass would nominate a causal code of reliability-weighted source deviations
around a shared power-specific temporal kernel. It would not establish
cross-cell-type transfer, monosemantic neurons, or a complete neural language.
