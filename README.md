# Single-Neuron Response Strength and Latency

This repository contains the manuscript, compact derived data, frozen
protocols, and analysis code for:

> **Single-neuron response strength and latency predict held-out group
> currents**
>
> Javier Emilio Bazan Sanchez  
> Faculty of Sciences, UNAM  
> bazan@ciencias.unam.mx

The analysis asks whether the response to a group of stimulated neurons can
be predicted from what each neuron does alone.

## Dataset and prediction task

The public recordings come from cortical slices. Each file contains one
patched postsynaptic cell, 214--687 candidate stimulated neurons across four
or five imaging depths, a trial-by-neuron stimulation matrix, and a 45-ms
postsynaptic-current trace for every trial in raw and NWD-demixed form.
Single-neuron trials are used for learning; group trials are held out for
evaluation.

The ten E-to-E files contain 129,613 single-neuron trials and 74,435 group
trials. Their filenames identify ten recorded cells on three dates. Public
metadata do not identify animal or slice membership, so file-level inference
is accompanied by a conservative three-date sensitivity analysis.

## Main result

Each source-power response is decomposed as

```text
response(source, power, time)
  = shared(power, time)
  + reliability(power) * source_deviation(source, power, time)
```

The model adds the learned responses without fitting the group trials. A
first version kept every neuron-specific difference. It improved MSE but not
waveform shape in SST data, which motivated the reliability rule. The
revised model keeps only differences that repeat across two halves of the
single-neuron trials. That rule was locked before opening the E-to-E files.

In ten E-to-E files, all 74,435 ensemble trials were eligible:

| Primary comparison | Mean file effect | Exact p | BH q | Positive files |
|---|---:|---:|---:|---:|
| Neuron-specific vs power-only, MSE gain | 2.12e-5 | 0.000977 | 0.001953 | 10/10 |
| Neuron-specific vs power-only, waveform-correlation gain | 0.003973 | 0.008789 | 0.008789 | 9/10 |

Every leave-one-file-out aggregate remained positive. Depth-and-power
preserving identity permutations and matched nonstimulated-source
substitutions also passed their frozen gates.

A post-feedback model comparison shows what carries the improvement. A
source-specific scalar gain retains 96.7% of the demixed MSE gain. Adding a
source-specific latency shift retains nearly all of it. The unrestricted
waveform does not conclusively outperform gain plus latency. In raw current,
gain plus latency improves both MSE and waveform correlation in all ten files;
the unrestricted waveform improves MSE but not mean correlation.

Together, the tests show that source-specific response strength and latency
learned from single-neuron stimulation add predictive information beyond
laser power, imaging depth, response size, and group size.

## Repository map

- `manuscript/`: LaTeX source, bibliography, figures, tables, and compiled PDF.
- `scripts/make_figures.py`: regenerates every manuscript figure from tables.
- `scripts/make_tables.py`: regenerates LaTeX result tables.
- `src/wetware_interp/`: parser, token estimators, predictors, nulls, and tests.
- `scripts/run_*.py`: frozen SST and E-to-E analyses.
- `scripts/run_holographic_feedback_analyses.py`: reduced-model, raw-current,
  metadata, and date-cluster sensitivities added after external feedback.
- `artifacts/`: compact derived tables needed to audit claims and figures.
- `data/`: checksum-pinned Figshare manifests; raw recordings are not tracked.
- `docs/`: protocols, lock records, results, claim boundary, and data guide.
- `docs/PROVENANCE_SANITIZATION.md`: audit record for portable path metadata.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python scripts/make_figures.py
.venv/Scripts/python scripts/make_tables.py
.venv/Scripts/python -m pytest
```

On macOS or Linux, replace `.venv/Scripts/python` with `.venv/bin/python`.
Build the paper with Tectonic:

```bash
python scripts/build_paper.py
```

The complete raw-data reproduction downloads about 9.7 GiB for the analyzed
SST and E-to-E cohorts. See `docs/DATA.md` and `docs/REPRODUCIBILITY.md`.

## Provenance

The recordings come from Gajowa and Triplett's CC BY 4.0
[Figshare dataset](https://doi.org/10.6084/m9.figshare.25641435.v1), associated
with Triplett et al., *Nature Neuroscience* (2025),
doi:10.1038/s41593-025-02053-7. The upstream `circuitmap` repository is pinned
at commit `75934895f8ef02b8045ab1a8eee592a062c2489e`.

Code in this repository is MIT licensed. Manuscript, figures, and derived
tables are CC BY 4.0; see `LICENSE-DATA.md`.
