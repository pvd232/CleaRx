#!/usr/bin/env python3
"""Install pinned remote dependencies and start the commit-bound smoke worker."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TRANSFORMERS_COMMIT = "7d9754a05193eb79b1d86aa744b622b8068008cd"
REPOSITORY_URL = "https://github.com/pvd232/CleaRx.git"
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def run(command: list[str], *, cwd: Path | None = None) -> None:
    """Run one bootstrap command and propagate its failure."""
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    """Install dependencies, clone the requested commit, and replace this process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    if _GIT_COMMIT.fullmatch(args.git_commit) is None:
        raise ValueError("--git-commit must be a full lowercase commit")

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            f"git+https://github.com/huggingface/transformers.git@{TRANSFORMERS_COMMIT}",
            "accelerate>=1.10,<2",
            "huggingface-hub>=0.34,<2",
            "librosa>=0.11,<1",
            "safetensors>=0.6,<1",
            "soundfile>=0.13,<1",
        ]
    )
    temporary = Path(tempfile.mkdtemp(prefix="clearx-audio-memory-smoke-"))
    checkout = temporary / "CleaRx"
    run(["git", "clone", "--quiet", REPOSITORY_URL, str(checkout)])
    run(["git", "checkout", "--quiet", args.git_commit], cwd=checkout)
    worker = checkout / "experiments" / "audio_memory_smoke_worker.py"
    if not worker.is_file():
        raise RuntimeError("requested commit does not contain the smoke worker")
    os.chdir(checkout)
    os.execv(
        sys.executable,
        [sys.executable, str(worker), "--git-commit", args.git_commit],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
