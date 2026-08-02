from __future__ import annotations

import concurrent.futures
import hashlib
import urllib.request
from pathlib import Path


USER_AGENT = "holographic-source-grammar/0.1 (public research dataset)"


def checksum(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_size(url: str) -> int:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request) as response:
        return int(response.headers["Content-Length"])


def download_part(
    url: str,
    part_path: Path,
    start: int,
    end: int,
) -> Path:
    expected_size = end - start + 1
    if part_path.exists() and part_path.stat().st_size == expected_size:
        return part_path
    if part_path.exists() and part_path.stat().st_size > expected_size:
        part_path.unlink()

    for _ in range(10):
        existing_size = part_path.stat().st_size if part_path.exists() else 0
        if existing_size == expected_size:
            return part_path

        request_start = start + existing_size
        request = urllib.request.Request(
            url,
            headers={
                "Range": f"bytes={request_start}-{end}",
                "User-Agent": USER_AGENT,
            },
        )
        with urllib.request.urlopen(request) as response, part_path.open(
            "ab"
        ) as output:
            if response.status != 206:
                raise RuntimeError(
                    f"Server ignored range {request_start}-{end}: "
                    f"HTTP {response.status}"
                )
            while chunk := response.read(1024 * 1024):
                output.write(chunk)

    actual_size = part_path.stat().st_size if part_path.exists() else 0
    if actual_size != expected_size:
        raise RuntimeError(
            f"Part {part_path.name} has {actual_size} bytes after retries; "
            f"expected {expected_size}"
        )
    return part_path


def download_ranges(url: str, target: Path, connections: int = 8) -> None:
    if connections < 1:
        raise ValueError("connections must be at least 1")

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    parts_dir = target.with_suffix(target.suffix + ".parts")
    parts_dir.mkdir(exist_ok=True)

    total = remote_size(url)
    chunk_size = (total + connections - 1) // connections
    ranges = []
    for index in range(connections):
        start = index * chunk_size
        if start >= total:
            break
        end = min(start + chunk_size - 1, total - 1)
        ranges.append((index, start, end))

    print(
        f"Downloading {total / 1024**2:.1f} MiB in "
        f"{len(ranges)} verified ranges..."
    )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(ranges)
    ) as executor:
        futures = {
            executor.submit(
                download_part,
                url,
                parts_dir / f"part-{index:03d}",
                start,
                end,
            ): index
            for index, start, end in ranges
        }
        for future in concurrent.futures.as_completed(futures):
            part_path = future.result()
            print(f"Completed {part_path.name}")

    with partial.open("wb") as output:
        for index, _, _ in ranges:
            part_path = parts_dir / f"part-{index:03d}"
            with part_path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)

    if partial.stat().st_size != total:
        raise RuntimeError(
            f"Assembled file has {partial.stat().st_size} bytes; expected {total}"
        )
    partial.replace(target)
    for part_path in parts_dir.iterdir():
        part_path.unlink()
    parts_dir.rmdir()
