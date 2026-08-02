from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from wetware_interp.download import USER_AGENT


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT / "data" / "holographic_etoe_confirmation_manifest.json"
)
DEFAULT_PROTOCOL = (
    ROOT / "docs" / "holographic_etoe_confirmation_protocol.md"
)
DEFAULT_PROTOCOL_LOCK = (
    ROOT
    / "docs"
    / "holographic_etoe_confirmation_protocol.lock.json"
)
DEFAULT_DEVELOPMENT_RESULT = (
    ROOT
    / "artifacts"
    / "holographic_reliability_shrinkage_sst"
    / "summary.json"
)
DEFAULT_RAW_DIR = (
    ROOT / "data" / "raw" / "holographic_circuitmap_etoe"
)
EXPECTED_PROTOCOL_SHA256 = (
    "e8e48ec69bf0bb5fa99226e737d46f3d"
    "2e8ab21cb32c471be6863b5eacb10d83"
)
EXPECTED_MANIFEST_SHA256 = (
    "984f4e3b5187fb80420af96be4019077"
    "5a7906419fe2036f5186c54e2be5d237"
)
EXPECTED_DEVELOPMENT_SHA256 = (
    "f13b70a7407e9ceda6afe38d35486849c"
    "ed509fca9af6f60df969d15be7cd1a6"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the protocol-locked E-to-E holographic "
            "confirmation cohort."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
    )
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=DEFAULT_PROTOCOL_LOCK,
    )
    parser.add_argument(
        "--development-result",
        type=Path,
        default=DEFAULT_DEVELOPMENT_RESULT,
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--unlock-etoe",
        action="store_true",
        help="Acknowledge the prospective E-to-E confirmation lock.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("Worker count must be positive")
    if not args.verify_only and not args.unlock_etoe:
        raise ValueError(
            "E-to-E is sealed; pass --unlock-etoe only after "
            "the confirmation protocol is locked"
        )
    manifest, files = validate_frozen_inputs(args)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(args.workers, len(files))
    ) as executor:
        futures = {
            executor.submit(
                materialize,
                item,
                args.raw_dir,
                verify_only=args.verify_only,
            ): item
            for item in files
        }
        for index, future in enumerate(
            concurrent.futures.as_completed(futures),
            start=1,
        ):
            item = futures[future]
            try:
                status = future.result()
                print(
                    f"[{index}/{len(files)}] {status}: "
                    f"{item['name']}",
                    flush=True,
                )
            except Exception as error:
                failures.append((item["name"], str(error)))
                print(
                    f"[{index}/{len(files)}] FAILED: "
                    f"{item['name']}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
    if failures:
        return 1
    print(
        f"Verified {len(files)} E-to-E files "
        f"({int(manifest['total_size_bytes']) / 1024**3:.2f} GiB).",
        flush=True,
    )
    return 0


def validate_frozen_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    hashes = {
        "protocol_sha256": file_sha256(args.protocol),
        "manifest_sha256": file_sha256(args.manifest),
        "development_result_sha256": file_sha256(
            args.development_result
        ),
    }
    expected = {
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "development_result_sha256": (
            EXPECTED_DEVELOPMENT_SHA256
        ),
    }
    if hashes != expected:
        raise ValueError(
            f"Frozen input hash mismatch: {hashes} != {expected}"
        )
    lock = json.loads(
        args.protocol_lock.read_text(encoding="utf-8")
    )
    if any(lock[key] != value for key, value in hashes.items()):
        raise ValueError("Confirmation lock is inconsistent")
    if bool(lock["etoe_opened"]) or bool(lock["pv_opened"]):
        raise ValueError(
            "Confirmation lock unexpectedly records an opened cohort"
        )
    manifest = json.loads(
        args.manifest.read_text(encoding="utf-8")
    )
    validate_manifest(manifest)
    return manifest, list(manifest["files"])


def validate_manifest(manifest: dict[str, object]) -> None:
    if int(manifest["figshare_article_id"]) != 25641435:
        raise ValueError("Unexpected Figshare article")
    if int(manifest["figshare_version"]) != 1:
        raise ValueError("Unexpected Figshare version")
    if manifest["cohort"] != "etoe_confirmation":
        raise ValueError("Unexpected confirmation cohort")
    if bool(manifest["opened_at_manifest_lock"]):
        raise ValueError("E-to-E was not sealed at manifest lock")
    files = list(manifest["files"])
    if len(files) != 10 or int(manifest["file_count"]) != 10:
        raise ValueError("E-to-E requires exactly ten files")
    names = [str(item["name"]) for item in files]
    if len(names) != len(set(names)):
        raise ValueError("E-to-E file names are not unique")
    if not all("_EtoE_" in name for name in names):
        raise ValueError("Manifest contains a non-E-to-E file")
    if sum(int(item["size"]) for item in files) != int(
        manifest["total_size_bytes"]
    ):
        raise ValueError("Manifest total size is inconsistent")
    for item in files:
        if (
            int(item["size"]) < 1
            or len(str(item["md5"])) != 32
            or str(item["download_url"])
            != f"https://ndownloader.figshare.com/files/{item['id']}"
        ):
            raise ValueError(
                f"Invalid E-to-E record: {item['name']}"
            )


def materialize(
    item: dict[str, object],
    raw_dir: Path,
    *,
    verify_only: bool,
) -> str:
    target = raw_dir / str(item["name"])
    if target.exists():
        verify_file(target, item)
        return "verified"
    if verify_only:
        raise FileNotFoundError(target)
    partial = target.with_suffix(target.suffix + ".part")
    expected_size = int(item["size"])
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise ValueError(
            f"{partial.name} exceeds its expected final size"
        )
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(
        str(item["download_url"]),
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        status = int(getattr(response, "status", 200))
        if offset and status != 206:
            offset = 0
        mode = "ab" if offset else "wb"
        with partial.open(mode) as stream:
            while chunk := response.read(4 * 1024 * 1024):
                stream.write(chunk)
    verify_file(partial, item)
    partial.replace(target)
    return "downloaded"


def verify_file(
    path: Path,
    item: dict[str, object],
) -> None:
    expected_size = int(item["size"])
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"{path.name} has {path.stat().st_size} bytes; "
            f"expected {expected_size}"
        )
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != str(item["md5"]):
        raise ValueError(f"{path.name} failed MD5 verification")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
