# Claim boundary

## Supported

In the analyzed cortical-slice recordings, source-specific response strength
and latency learned from single-neuron stimulation improve prediction of
held-out group current beyond exact stimulation power, imaging depth, group
size, and a shared power-conditioned waveform.

The effect is predictive because it is estimated only from single-source
interventions and evaluated on held-out multi-source interventions. It is
selective because true identity beats depth-and-power preserving permutations
and matched nonstimulated-source substitutions. The reliability-weighted rule
was developed in an SST cohort and tested without tuning in a prospectively
locked E-to-E cohort. The reduced hierarchy below was added after feedback.

A source-gain-only model retains 96.7% of the full model's demixed MSE gain.
Gain plus a source-specific latency shift retains nearly all MSE gain and part
of the demixed waveform-correlation gain. In raw current, gain plus latency
improves both MSE and waveform correlation in all ten files (mean correlation
gain 0.00777, exact p = 0.000977).

## Not supported

- A complete digital twin of the total current waveform.
- Independence of the ten files at the slice or animal level. That hierarchy
  is absent from the public metadata.
- An unrestricted source-specific waveform advantage beyond gain plus latency.
- A source response that generalizes across power; exact source-power pairs
  are currently estimated separately.
- Monosemantic neurons or one-neuron/one-concept correspondence.
- Semantic content during natural behavior.
- Universal additive neural computation.
- PV-to-E confirmation. The original PV cohort remains sealed because the
  first SST protocol failed its waveform-correlation opening gate.

## Strongest falsification next

Obtain the file-to-slice-to-animal mapping and repeat inference at the highest
independent biological level. Then preregister the fixed model hierarchy
power-only -> gain -> gain plus latency -> full waveform in a new cohort.
