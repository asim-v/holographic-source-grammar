# Reproducibility

## Lightweight reproduction

This path uses the checked-in derived tables and takes seconds:

```bash
python -m pip install -e ".[dev]"
python scripts/make_figures.py
python scripts/make_tables.py
python -m pytest
tectonic --keep-logs --keep-intermediates manuscript/main.tex
```

## Full analysis

After downloading the checksum-pinned recordings:

```bash
python scripts/run_holographic_source_composition.py
python scripts/run_holographic_reliability_shrinkage.py
python scripts/run_holographic_etoe_confirmation.py
python scripts/run_holographic_feedback_analyses.py
```

The runners write deterministic CSV/JSON artifacts. They use:

- samples 0--99 for trialwise baseline;
- samples 120--899 as the frozen 1--40 ms response window;
- at least three single-source trials per exact source-power token;
- exact one-sided file-level sign-flip tests;
- Benjamini-Hochberg correction across the two primary metrics;
- 999 deterministic depth-and-power preserving identity permutations;
- matched nonstimulated-source substitutions at the same depth and power.

No ensemble response estimates a token, reliability coefficient, gain,
latency, saturation parameter, source filter, or rank.

The final runner is explicitly post-feedback. It compares the frozen full
waveform against source-gain and source-gain-plus-latency reductions, repeats
the comparison on raw current, audits embedded biological identifiers, and
adds a conservative acquisition-date sensitivity.

## Historical order

1. Full source-power tokens were prospectively tested in nine SST files.
2. MSE passed, waveform correlation failed, and PV stayed sealed.
3. Split-half reliability shrinkage was developed post hoc in opened SST data.
4. Its method was frozen before any E-to-E file was downloaded or inspected.
5. The same rule passed every locked E-to-E file-level gate.
6. External feedback motivated reduced-model and biological-unit
   sensitivities on the already-open E-to-E cohort.

The protocols and SHA-256 lock records under `docs/` preserve this order.
