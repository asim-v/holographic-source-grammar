# Locked protocol: composition of single-neuron source tokens

Status: prospective SST analysis with a sealed PV confirmation cohort.

One SST file was opened only to verify the HDF5 schema, array orientation,
trial counts, powers, and checksum. No PSC waveform, trial-level response, or
single-to-ensemble prediction statistic was examined before this lock.

## Prospective implementation clarification

The following operational definitions were added before any response waveform
array was read. The previous protocol hash is retained in the lock file.

- The anonymous token at an exact power is the equal-weight mean of eligible
  source-level token means, rather than a trial-count-weighted mean.
- Waveform Pearson correlation is zero when either waveform has zero variance.
- Integrated current is the response-window sample sum multiplied by
  `0.05 ms`. Peak current is the signed sample with the largest absolute
  excursion; the reported peak error is the absolute difference between those
  signed extrema.
- Source permutations are performed independently at each exact power among
  eligible sources in the same imaging-depth plane.
- Because MSE and correlation have different units, their familywise maximum
  null is formed after standardizing each null distribution by its own mean
  and standard deviation. Each standardized observed statistic must exceed
  the 99th percentile of the repetition-wise maximum standardized null.
- Selective ablation is operationalized as replacing one active source token
  with each eligible nonstimulated token from the same depth and exact power.
  The cost is replacement MSE minus the full true-token prediction MSE,
  averaged first over decoys, then active tokens, then trials within file.

The first real-file parser smoke test found that six SST files encode their
single blank `pscs_demixed` row as all `NaN`. This was established using only
finiteness masks and stimulus counts, before any response value or composition
statistic was examined. An all-`NaN` trace is therefore accepted only when its
stimulus row is exactly blank and is then excluded under the frozen blank-trial
rule. Any partially nonfinite trace or any nonfinite stimulated trial aborts
the analysis.

## Question

Do response tokens learned exclusively from single-neuron holographic
stimulation compose to predict unseen multi-neuron ensemble responses?

This is an independent mammalian test of the causal-grammar hypothesis
suggested by the worm atlas: source identity contributes a direct
connectivity-linked token, while population messages can be composed across
simultaneously active sources.

## Frozen data boundary

- Figshare article `25641435`, version 1.
- DOI `10.6084/m9.figshare.25641435.v1`.
- Manifest:
  `data/holographic_circuitmap_manifest.json`
  (`sha256=65864e3126d6d1b9c0f8ce11e9ff1bbd47a423fcdef4f64a54792660cadb04ab`).
- Analysis code reference:
  `https://github.com/marcustriplett/circuitmap`, commit
  `75934895f8ef02b8045ab1a8eee592a062c2489e`.
- Discovery cohort: all nine `SSTtoE` files.
- Confirmation cohort: all fourteen `PVtoE` files. These files remain
  unopened and undownloaded until every SST opening gate below passes.

The `EtoE` and `EtoPV` files are outside this protocol.

## Input contract

Each MATLAB v7.3 file must contain:

- `stimulus_matrix`, trial by presynaptic source;
- `pscs_demixed`, trial by 900 time samples;
- `pscs`, the corresponding raw current;
- `targets`, source coordinates;
- `caviar_weights_single` and `caviar_weights_multi`.

The stored arrays are read in HDF5 orientation. There are 20 samples per
millisecond, stimulation occurs at sample 100, and the trace spans
`-5` to `40 ms`.

The primary response is `pscs_demixed`. Raw `pscs` is a robustness outcome.
CAVIaR weights are not predictors, filters, labels, or selection variables in
the primary analysis.

## Trial partition

For each postsynaptic-cell file:

1. A single-source trial has exactly one positive entry in
   `stimulus_matrix`.
2. An ensemble trial has more than one positive entry.
3. Blank trials are excluded.
4. Single-source trials are used only to construct and validate source tokens.
5. Ensemble trials are used only for composition evaluation and never fit a
   token, gain, saturation parameter, time window, or source filter.

The baseline is the mean from samples 0--99 (`-5` to `0 ms`). Subtract it
trialwise. The frozen response window is samples 120--899 (`1` to `40 ms`);
the first post-stimulus millisecond is excluded.

## Source tokens

For every source and exact stimulation power, estimate a waveform token as
the arithmetic mean of all baseline-corrected single-source trials. A token
requires at least three single-source trials at that power.

As an internal reliability audit, split trials deterministically by their
within-source/power order into alternating halves. Report half-token waveform
correlation and peak-amplitude agreement. This audit does not select tokens.

An ensemble trial is eligible only when every stimulated source has an
eligible single-source token at the exact delivered power.

## Frozen predictors

For each eligible ensemble trial:

- `zero`: an all-zero response after baseline correction;
- `anonymous_additive`: sum the mean single-source token for each delivered
  power, averaging eligible source-token means equally over source identity
  within the file;
- `source_additive`: sum the exact source-and-power tokens for all stimulated
  sources.

No coefficient is fitted on ensemble responses. In particular, there is no
ensemble-derived gain, intercept, saturation, latency shift, or kernel.

## Primary metrics

Calculate per-trial:

- response-window MSE;
- waveform Pearson correlation;
- integrated-current absolute error;
- peak-current absolute error.

The primary comparisons are `source_additive` versus
`anonymous_additive` for MSE and waveform correlation. MSE versus `zero`,
integrated-current error, and peak error are supporting.

For each file, average paired trial improvements first. Across the nine SST
files, use exact one-sided sign-flip tests. Benjamini-Hochberg correction is
applied across the two primary metrics. Both must have positive file-mean
improvement and `q < 0.05`.

Every leave-one-file-out aggregate must retain positive MSE and correlation
improvement.

## Source-identity null

Run 999 deterministic null repetitions. Within each file and imaging-depth
plane, permute source-token identities while retaining the exact stimulation
power. Apply the same permutation to every ensemble trial in that repetition.
Do not refit or use ensemble responses.

For MSE and waveform correlation, the observed mean file-level improvement
must exceed the 99th percentile of the maximum null distribution across both
metrics. This controls source identity while preserving power, ensemble size,
depth, token spectrum, and postsynaptic cell.

## Selective token ablation

For every eligible ensemble trial, replace each true source token in turn
with every eligible nonstimulated source token from the same depth and exact
power. Compare each replacement prediction with the full prediction containing
the true token.

The true-minus-decoy ablation cost must be positive under an exact file-level
sign-flip test. This is a necessity test for source-specific composition, not
just total token energy.

## SST opening gate

The sealed PV cohort may be opened only if all conditions hold:

1. both primary composition metrics pass corrected file-level tests;
2. every leave-one-file-out aggregate is positive;
3. both source-identity null tests pass the familywise 99th-percentile gate;
4. selective token ablation passes;
5. at least seven of nine SST files contain eligible ensemble trials.

Raw-current robustness is reported but is not required to open PV.

## PV confirmation

If the SST gate passes, download and checksum all fourteen PV files. Apply the
identical parser, time window, minimum token count, predictors, metrics,
permutations, and ablation test without tuning.

PV confirms cross-cell-type composition only if both primary metrics and the
source-identity familywise null pass at `q < 0.05`, with positive
leave-one-file-out aggregates. SST and PV effect sizes are reported
separately; pooling them cannot rescue either cohort.

## Interpretation boundary

- Passing SST alone supports source-specific additive composition in SST
  perturbations.
- Passing SST and sealed PV supports a transferable direct-source grammar
  across two inhibitory cell classes.
- Failure of additive composition does not imply that source tokens are
  absent; it may indicate saturation, stochastic recruitment, or nonlinear
  postsynaptic integration, which require a separately locked follow-up.
- No result establishes monosemantic neurons, natural-language meanings, or a
  complete neural language.
