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
MANIFEST_PATH = (
    ROOT / "data" / "holographic_circuitmap_manifest.json"
)
RAW_DIR = ROOT / "data" / "raw" / "holographic_circuitmap"
EXPECTED_ARTICLE_ID = 25641435
EXPECTED_VERSION = 1
EXPECTED_COUNTS = {
    "sst_discovery": 9,
    "pv_confirmation": 14,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download checksum-pinned holographic circuit-mapping "
            "files from Figshare."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
    )
    parser.add_argument(
        "--cohort",
        choices=tuple(EXPECTED_COUNTS),
        default="sst_discovery",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--unlock-pv",
        action="store_true",
        help=(
            "Acknowledge that the sealed PV confirmation cohort "
            "may be downloaded."
        ),
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.cohort == "pv_confirmation" and not args.unlock_pv:
        parser.error(
            "PV is sealed; pass --unlock-pv only after the SST gate"
        )

    manifest = json.loads(
        args.manifest.read_text(encoding="utf-8")
    )
    validate_manifest(manifest)
    files = [
        item
        for item in manifest["files"]
        if item["cohort"] == args.cohort
    ]
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(args.workers, len(files))
    ) as executor:
        futures = {
            executor.submit(
                _materialize,
                item,
                args.raw_dir,
                force=args.force,
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
                failures.append((str(item["name"]), str(error)))
                print(
                    f"[{index}/{len(files)}] FAILED: "
                    f"{item['name']}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
    if failures:
        return 1
    print(
        f"Verified {len(files)} files "
        f"({sum(int(item['size']) for item in files) / 1024**3:.2f} GiB)."
    )
    return 0


def validate_manifest(manifest: dict[str, object]) -> None:
    if int(manifest["figshare_article_id"]) != EXPECTED_ARTICLE_ID:
        raise ValueError("Unexpected Figshare article")
    if int(manifest["figshare_version"]) != EXPECTED_VERSION:
        raise ValueError("Unexpected Figshare version")
    files = manifest["files"]
    names = [str(item["name"]) for item in files]
    if len(names) != len(set(names)):
        raise ValueError("Manifest file names are not unique")
    for cohort, expected_count in EXPECTED_COUNTS.items():
        cohort_record = manifest["cohorts"][cohort]
        cohort_files = [
            item for item in files if item["cohort"] == cohort
        ]
        if len(cohort_files) != expected_count:
            raise ValueError(
                f"{cohort} has {len(cohort_files)} files; "
                f"expected {expected_count}"
            )
        listed = set(cohort_record["files"])
        observed = {str(item["name"]) for item in cohort_files}
        if listed != observed:
            raise ValueError(f"{cohort} file list is inconsistent")
    for item in files:
        if (
            int(item["size"]) < 1
            or len(str(item["md5"])) != 32
            or not str(item["download_url"]).startswith("https://")
        ):
            raise ValueError(
                f"Invalid manifest record: {item['name']}"
            )


def _materialize(
    item: dict[str, object],
    raw_dir: Path,
    *,
    force: bool,
    verify_only: bool,
) -> str:
    target = raw_dir / str(item["name"])
    if target.exists() and not force:
        _verify_file(target, item)
        return "verified"
    if verify_only:
        raise FileNotFoundError(target)
    partial = target.with_suffix(target.suffix + ".part")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(
        str(item["download_url"]),
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request) as response:
            with partial.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        _verify_file(partial, item)
        partial.replace(target)
    finally:
        if partial.exists():
            partial.unlink()
    return "downloaded"


def _verify_file(
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
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    observed = digest.hexdigest()
    if observed != str(item["md5"]):
        raise ValueError(
            f"{path.name} MD5 is {observed}; "
            f"expected {item['md5']}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
