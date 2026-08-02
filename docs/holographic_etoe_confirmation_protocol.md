# Locked confirmation: reliability-weighted E-to-E source grammar

Status: prospective confirmation on an unopened cohort.

No `EtoE` file was downloaded, opened, or inspected before this lock. The
cohort was inventoried only from the Figshare article metadata.

The first parser smoke, before any response value or prediction statistic was
read, found one schema alias: nine files contain `caviar_weights_multi`, while
`12142022_cell1` contains the equivalent unused field
`caviar_weights_ensemble`. Either exact name is accepted. This amendment does
not alter any array used by the analysis.

## Question

Does the reliability-weighted source grammar developed in SST perturbations
transfer, without method tuning, to excitatory-source holographic
perturbations?

The candidate grammar is:

`ensemble current = sum_s [shared power token + reliable source deviation]`

Every term is estimated from single-source trials. Ensemble responses are
prediction-only.

## Frozen provenance

- Figshare article `25641435`, version 1.
- DOI `10.6084/m9.figshare.25641435.v1`.
- Code reference `marcustriplett/circuitmap`, commit
  `75934895f8ef02b8045ab1a8eee592a062c2489e`.
- Ten-file `EtoE` manifest:
  `data/holographic_etoe_confirmation_manifest.json`
  (`sha256=984f4e3b5187fb80420af96be40190775a7906419fe2036f5186c54e2be5d237`).
- Frozen SST development result:
  `artifacts/holographic_reliability_shrinkage_sst/summary.json`
  (`sha256=204523174a72fc3ff076e74b48815661299fd1a2285b89d6faacd8de0d2f8a93`).

The original PV cohort remains sealed. This confirmation cannot alter the
failed full-token SST/PV opening decision.

## Input and trial contract

Use the same parser and checks as the SST analysis:

- `stimulus_matrix`: trial by source;
- `pscs_demixed` and `pscs`: trial by 900 samples;
- `targets`: source coordinates;
- `caviar_weights_single` and either `caviar_weights_multi` or
  `caviar_weights_ensemble` must exist, but are not predictors, filters, or
  labels;
- stimulation values must be nonnegative and exact powers are not binned;
- an all-`NaN` response row is permitted only for an exactly blank stimulus
  row; any nonfinite stimulated or partially nonfinite row aborts.

Single-source, ensemble, and blank trials are defined by respectively one,
more than one, and zero positive stimulus entries. Blank trials are excluded.

Subtract the trialwise mean of samples 0--99. Analyze samples 120--899
(`1--40 ms`) at `20 kHz`. The primary outcome is `pscs_demixed`; raw `pscs`
is supporting.

## Frozen token estimator

A source-power token requires at least three single-source trials. Within each
source and exact power, deterministically split trials by alternating order.

For each file, outcome, and power:

1. form alternating-half source tokens `A_s` and `B_s`;
2. center each half independently over source identity;
3. calculate

   `lambda = 2 sum(A'_s B'_s) /
             [sum(A'_s^2) + sum(B'_s^2)]`;

4. clip `lambda` to `[0, 1]`;
5. estimate full single-source means and construct

   `K(s,p) = K_anonymous(p) +
             lambda_p [K_full(s,p) - K_anonymous(p)]`.

The anonymous token is the equal-weight mean of eligible source-token means at
the exact power. There is no candidate grid, ensemble-derived coefficient,
latency shift, gain, saturation term, source filter, or rank selection.

An ensemble trial is eligible only if every active source has an exact-power
token.

## Frozen predictors and metrics

- `anonymous_additive`: sum exact-power anonymous tokens.
- `reliability_source_additive`: sum exact source-power shrunk tokens.
- `zero`: all zeros after baseline correction, supporting only.

Primary paired improvements are:

- anonymous MSE minus reliability-source MSE;
- reliability-source waveform Pearson correlation minus anonymous
  correlation.

Pearson correlation is zero if either waveform is constant. Supporting metrics
are MSE versus zero, integrated-current absolute error, and signed
maximum-absolute-excursion peak error, using the same definitions as SST.

Average paired trial improvements within file first. Across files, run exact
one-sided sign-flip tests and BH-correct the two primary metrics. Both effects
must be positive with `q < 0.05`. Every leave-one-file-out aggregate must be
positive for both.

## Source identity and necessity

Run 999 deterministic identity-null repetitions. Independently at each exact
power, permute source-token identities within imaging-depth plane. Preserve
power, depth, ensemble size, token spectrum, and postsynaptic file.

Aggregate each repetition equally over files. Standardize each metric by its
own null mean and standard deviation, form the repetition-wise maximum across
MSE and correlation, and require both observed z scores to exceed its
familywise 99th percentile.

For selective necessity, replace each active token with every eligible
nonstimulated token at the same depth and exact power. Average replacement MSE
cost over decoys, active sources, trials, then files. Require a positive exact
file-level sign-flip result with `p < 0.05`.

## Confirmation gate

The E-to-E grammar confirms only if all conditions hold:

1. both corrected primary metrics pass;
2. every leave-one-file-out primary aggregate is positive;
3. both familywise source-identity null gates pass;
4. selective matched-decoy replacement passes;
5. at least eight of ten files have eligible ensemble trials.

Raw-current failure cannot overturn a demixed confirmation and cannot rescue a
demixed failure. SST and E-to-E effects are reported separately and are never
pooled for significance.

## Interpretation boundary

Passing supports transfer of a reliability-weighted, source-specific additive
current grammar from inhibitory SST to excitatory source perturbations.
Failure rejects that transfer claim but does not erase the within-SST
amplitude evidence.

Neither result establishes monosemantic neurons, natural-language semantics,
or a complete neural language.
