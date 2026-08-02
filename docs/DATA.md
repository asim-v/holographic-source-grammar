# Data guide

## Public source

Gajowa, M. and Triplett, M. A. (2025), *Neural circuit mapping using
two-photon holographic optogenetics*, Figshare, version 1,
doi:10.6084/m9.figshare.25641435.v1, CC BY 4.0.

The article contains 19.43 GB across PV-to-E, SST-to-E, E-to-E, and E-to-PV
experiments. This study analyzed:

| Cohort | Files | Approximate download | Role |
|---|---:|---:|---|
| SST-to-E | 9 | 2.01 GiB | Original test and reliability development |
| E-to-E | 10 | 7.64 GiB | Prospectively held-out file-level test |
| PV-to-E | 14 | not downloaded | Sealed after the original SST gate failed |

## What is versioned here

The `artifacts/` directories include file-level effects, exact tests,
leave-one-file-out results, identity-null draws, power-level shrinkage,
token-reliability audits, diagnostics, and JSON summaries. These compact
tables are sufficient to regenerate the paper figures and audit every reported
aggregate.

The large per-trial metric and exhaustive selective-ablation tables are
recomputable but omitted from Git. Their filenames and checksums remain in the
frozen summaries.

## What an E-to-E file contains

Each file represents one patched postsynaptic cell and contains:

- a trial-by-candidate `stimulus_matrix` with exact delivered powers;
- raw (`pscs`) and NWD-demixed (`pscs_demixed`) current traces;
- candidate-neuron coordinates and imaging depth in `targets`;
- 900 samples per trace at 20 kHz, or 45 ms;
- single-neuron, group, and occasional blank trials.

Across ten files there are 4,724 candidate-neuron entries, 129,613
single-neuron trials, 74,435 group trials, and six excluded blank trials.
Files identify ten recorded cells on three acquisition dates. The embedded
`ExpStruct.mouseID` values repeat labels such as `cell3_emx_map`; they do not
identify animals. Slice identifiers are also unavailable. See
`artifacts/holographic_feedback_analyses/metadata_audit.csv`.

## Downloading raw recordings

Download and verify SST files:

```bash
python scripts/download_holographic_circuitmap.py --cohort sst_discovery
```

Download and verify the frozen E-to-E cohort:

```bash
python scripts/download_holographic_etoe_confirmation.py --unlock-etoe
```

The `--unlock-etoe` flag acknowledges that the historical protocol lock was
written before opening the cohort; it does not alter any analysis setting.
Both scripts use pinned URLs, byte sizes, and checksums. Raw `.mat` files are
ignored by Git. Review the download size before running.

## Data ethics

This repository reanalyzes already-public animal electrophysiology data and
introduces no new animal experiments. The source publication reports the
original animal-care approvals and experimental procedures.
