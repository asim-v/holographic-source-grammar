from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "data" / "holographic_etoe_confirmation_manifest.json"
)
PROTOCOL = (
    ROOT / "docs" / "holographic_etoe_confirmation_protocol.md"
)
LOCK = (
    ROOT
    / "docs"
    / "holographic_etoe_confirmation_protocol.lock.json"
)
SCRIPT = (
    ROOT / "scripts" / "download_holographic_etoe_confirmation.py"
)


def _load_downloader():
    spec = importlib.util.spec_from_file_location(
        "download_holographic_etoe_confirmation",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_etoe_manifest_is_complete_and_sealed() -> None:
    module = _load_downloader()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    module.validate_manifest(manifest)

    assert manifest["file_count"] == 10
    assert manifest["total_size_bytes"] == 8_200_077_180
    assert manifest["opened_at_manifest_lock"] is False
    assert all(
        item["md5"] == item["md5"].lower()
        for item in manifest["files"]
    )


def test_etoe_lock_pins_protocol_manifest_and_development() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    development = (
        ROOT
        / "artifacts"
        / "holographic_reliability_shrinkage_sst"
        / "summary.json"
    )

    assert lock["protocol_sha256"] == _sha256(PROTOCOL)
    assert lock["manifest_sha256"] == _sha256(MANIFEST)
    assert lock["development_result_sha256"] == _sha256(
        development
    )
    assert lock["etoe_opened"] is False
    assert lock["pv_opened"] is False
