"""Install pinned dependencies and execute one immutable VIPER checkout."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
TRANSFORMERS_COMMIT = "7d9754a05193eb79b1d86aa744b622b8068008cd"
VIPER_COMMIT = "43a939a7abb2412cf5da7a0ddd5d01f67d1d84ab"


def run(*argv: str, cwd: Path | None = None) -> None:
    """Run one bootstrap command and stop at its first nonzero exit."""
    subprocess.run(argv, cwd=cwd, check=True)


def validate_checkout_path(path: Path) -> Path:
    """Accept one absent, direct child of /content with the experiment prefix."""
    resolved = path.resolve()
    if resolved.parent != Path("/content"):
        raise ValueError("--checkout must be a direct child of /content")
    if not resolved.name.startswith("clearx-crossed-retention-"):
        raise ValueError("--checkout must use the clearx-crossed-retention- prefix")
    if resolved.exists():
        raise FileExistsError("--checkout already exists; choose a fresh path")
    return resolved


def main() -> int:
    """Prepare a fresh Colab checkout and start the governed experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    args = parser.parse_args()
    if _GIT_COMMIT.fullmatch(args.git_commit) is None:
        raise ValueError("--git-commit must be a full lowercase commit")
    checkout = validate_checkout_path(args.checkout)
    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        f"git+https://github.com/huggingface/transformers.git@{TRANSFORMERS_COMMIT}",
        f"git+https://github.com/pvd232/viper.git@{VIPER_COMMIT}",
        "accelerate>=1.10,<2",
        "librosa>=0.11,<1",
        "qwen-omni-utils>=0.0.8,<0.1",
        "soundfile>=0.13,<1",
    )
    run(
        "git",
        "clone",
        "--quiet",
        "https://github.com/pvd232/CleaRx.git",
        str(checkout),
    )
    run("git", "checkout", "--quiet", "--detach", args.git_commit, cwd=checkout)
    observed = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
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
        cwd=checkout,
    )
    run(
        sys.executable,
        "experiments/audio_memory_crossed_retention/run_viper.py",
        "--expected-commit",
        args.git_commit,
        cwd=checkout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
