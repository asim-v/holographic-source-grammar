# Post-feedback model and metadata sensitivities

Status: post-feedback analysis on the opened E-to-E cohort. These comparisons
do not alter the historical locked gate.

## Questions

1. Does source identity contribute more than a scalar response strength?
2. Does a source-specific latency account for the temporal gain?
3. Which effects remain in raw current?
4. Can biological independence be established from public metadata?

All reduced-model parameters are learned from single-neuron trials. Group
responses are used only for evaluation.

## Model hierarchy

```text
power only:         shared(power, time)
source gain:        gain(source, power) * shared(power, time)
gain + latency:     gain(source, power) * shifted_shared(power, time)
full waveform:      shared + reliability * source_difference
```

Gain is a nonnegative least-squares projection onto the shared waveform. The
latency model searches integer shifts from -2 to +2 ms and refits nonnegative
gain at each shift.

## Demixed-current results

| Candidate vs reference | MSE gain | Positive files | Exact p | Correlation gain | Positive files | Exact p |
|---|---:|---:|---:|---:|---:|---:|
| Gain vs power | 2.051e-5 | 10/10 | 0.000977 | 0 | 0/10 | 1.000 |
| Gain + latency vs power | 2.119e-5 | 10/10 | 0.000977 | 0.001901 | 9/10 | 0.001953 |
| Full waveform vs power | 2.120e-5 | 10/10 | 0.000977 | 0.003973 | 9/10 | 0.008789 |
| Full waveform vs gain | 6.890e-7 | 10/10 | 0.000977 | 0.003973 | 9/10 | 0.008789 |
| Full waveform vs gain + latency | 6.270e-9 | 8/10 | 0.495 | 0.002072 | 9/10 | 0.104 |

Source gain retains 96.7% of the full model's mean MSE improvement. Gain plus
latency retains nearly all MSE improvement and part of the correlation gain.
The remaining full-waveform advantage over gain plus latency is not conclusive
at the file level.

## Raw-current results

All three source-specific models improve raw-current MSE over power only in
10/10 files. Gain alone does not improve mean raw waveform correlation. Gain
plus latency does improve raw correlation in 10/10 files (mean gain 0.00777,
exact `p=0.000977`). The unrestricted full waveform's mean correlation effect
relative to power only is -0.00650 (`p=0.803`). Direct physical recordings
therefore support source-specific strength and latency, but not an advantage
for unrestricted waveform detail.

## Biological-unit audit

The files identify ten recorded postsynaptic cells on three dates. Their
embedded `ExpStruct.mouseID` values are cell labels, not animal identifiers,
and no slice identifier is available. The file-to-slice-to-animal hierarchy
therefore remains unresolved.

When file effects are first averaged within acquisition date, the full-model
MSE and correlation effects remain positive on all three dates. With three
date groups, the smallest possible one-sided exact p-value is 0.125. Date is a
conservative session proxy, not a claimed biological identifier.

## Interpretation

The strongest supported statement is that source-specific response strength
and latency learned during individual stimulation predict held-out group
currents. Unrestricted temporal detail beyond those two parameters and
biological-level replication require further tests.

## Reproduction

```bash
python scripts/run_holographic_feedback_analyses.py
python scripts/make_figures.py
python scripts/make_tables.py
```

Compact outputs are under `artifacts/holographic_feedback_analyses/`.
