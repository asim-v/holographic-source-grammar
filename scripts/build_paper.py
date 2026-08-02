from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"


def run(command: list[str], cwd: Path = ROOT) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    run([sys.executable, "scripts/make_figures.py"])
    run([sys.executable, "scripts/make_tables.py"])

    tectonic = os.environ.get("TECTONIC") or shutil.which("tectonic")
    if not tectonic:
        raise SystemExit(
            "Tectonic was not found. Install it from "
            "https://tectonic-typesetting.github.io/ and rerun."
        )
    run(
        [
            tectonic,
            "--keep-logs",
            "--keep-intermediates",
            "main.tex",
        ],
        cwd=MANUSCRIPT,
    )
    print(f"Built {MANUSCRIPT / 'main.pdf'}")


if __name__ == "__main__":
    main()

