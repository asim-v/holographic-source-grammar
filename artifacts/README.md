# Derived analysis artifacts

These directories preserve the three inferential stages:

- `holographic_source_composition_sst`: prospective full-token SST test;
- `holographic_reliability_shrinkage_sst`: post-hoc SST development;
- `holographic_etoe_confirmation`: prospective E-to-E confirmation.
- `holographic_feedback_analyses`: post-feedback reduced-model, raw-current,
  metadata, and date-cluster sensitivities.

Checked-in files include JSON summaries, file-level paired effects, exact
tests, leave-one-file-out results, diagnostics, power-level shrinkage,
token-reliability audits, structured identity-null draws, and file-level
matched-decoy effects. They are sufficient to regenerate every manuscript
figure and table.

Two large deterministic outputs are intentionally omitted from ordinary Git:

- `trial_metrics.csv.gz`;
- `selective_ablation.csv.gz`.

The frozen `summary.json` files retain their expected filenames and SHA-256
digests. Running the corresponding `scripts/run_*.py` command from the raw
recordings regenerates them.

All tables are derived from the CC BY 4.0 source recordings identified in
`LICENSE-DATA.md`.

The frozen JSON summaries retain absolute paths from the original execution
environment because those files participate in the historical SHA-256 chain.
The paths are provenance strings only; public reproduction uses repository
relative defaults and does not require the original directory layout.
