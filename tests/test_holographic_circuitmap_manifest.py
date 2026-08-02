from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.download_holographic_circuitmap import (
    EXPECTED_COUNTS,
    MANIFEST_PATH,
    validate_manifest,
)


def test_holographic_manifest_is_pinned_and_partitioned() -> None:
    manifest = json.loads(
        Path(MANIFEST_PATH).read_text(encoding="utf-8")
    )

    validate_manifest(manifest)

    assert len(manifest["files"]) == sum(EXPECTED_COUNTS.values())
    assert manifest["cohorts"]["sst_discovery"]["opened"] is True
    assert manifest["cohorts"]["pv_confirmation"]["opened"] is False
    assert {
        item["cohort"] for item in manifest["files"]
    } == set(EXPECTED_COUNTS)


def test_holographic_manifest_rejects_cohort_drift() -> None:
    manifest = json.loads(
        Path(MANIFEST_PATH).read_text(encoding="utf-8")
    )
    corrupted = copy.deepcopy(manifest)
    corrupted["cohorts"]["sst_discovery"]["files"].pop()

    with pytest.raises(ValueError, match="inconsistent"):
        validate_manifest(corrupted)
