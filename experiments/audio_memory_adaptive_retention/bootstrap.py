"""Install pinned dependencies and run one immutable retention pilot checkout."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
TRANSFORMERS_COMMIT = "7d9754a05193eb79b1d86aa744b622b8068008cd"


def run(*argv: str, cwd: Path | None = None) -> None:
    """Run one bootstrap command and stop at its first nonzero exit."""
    subprocess.run(argv, cwd=cwd, check=True)


def main() -> int:
    """Prepare a clean Colab checkout and start its A100 worker."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--git-commit", required=True)
    parser.add_argument(
        "--checkout", type=Path, default=Path("/content/clearx-retention")
    )
    args = parser.parse_args()
    if _GIT_COMMIT.fullmatch(args.git_commit) is None:
        raise ValueError("--git-commit must be a full lowercase commit")

    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        f"git+https://github.com/huggingface/transformers.git@{TRANSFORMERS_COMMIT}",
        "accelerate>=1.10,<2",
        "librosa>=0.11,<1",
        "qwen-omni-utils>=0.0.8,<0.1",
        "soundfile>=0.13,<1",
    )
    if args.checkout.exists():
        shutil.rmtree(args.checkout)
    run(
        "git",
        "clone",
        "--quiet",
        "https://github.com/pvd232/CleaRx.git",
        str(args.checkout),
    )
    run("git", "checkout", "--quiet", "--detach", args.git_commit, cwd=args.checkout)
    observed = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.checkout, text=True
    ).strip()
    if observed != args.git_commit:
        raise RuntimeError("bootstrap checkout differs from the requested commit")
    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--editable",
        ".",
        cwd=args.checkout,
    )
    run(
        sys.executable,
        "experiments/audio_memory_adaptive_retention/run_experiment.py",
        "--experiment-commit",
        args.git_commit,
        cwd=args.checkout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
