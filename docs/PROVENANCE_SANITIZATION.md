# Provenance path sanitization

On 2026-08-02, absolute local paths in the three tracked analysis summaries
were replaced with repository-relative paths. No measurements, model outputs,
statistical values, row counts, dataset checksums, figures, or manuscript
claims changed.

Because the development and confirmation protocols pin upstream summaries by
SHA-256, this byte-level metadata change required updated public hashes:

| Artifact | Original SHA-256 | Public sanitized SHA-256 |
|---|---|---|
| `artifacts/holographic_source_composition_sst/summary.json` | `006ea9149a76a99d7d5ea56702465251901ea0114f1c97a26f4cadd69d2b4700` | `4b499e25bbb79cc8ad34025b19e3bb6e26016d029024481c8f0e37d58732c39e` |
| `artifacts/holographic_reliability_shrinkage_sst/summary.json` | `204523174a72fc3ff076e74b48815661299fd1a2285b89d6faacd8de0d2f8a93` | `f13b70a7407e9ceda6afe38d35486849ced509fca9af6f60df969d15be7cd1a6` |

The lock records retain the original hashes alongside the sanitized hashes.
Analysis scripts now emit portable repository-relative paths so reruns do not
expose workstation-specific directory names.
