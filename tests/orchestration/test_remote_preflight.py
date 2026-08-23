"""Exercise the A100 wrapper's local safety boundary without launching Colab."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from conftest import initialize_git


def test_a100_wrapper_rejects_dirty_tree_before_colab(repo_copy: Path, tmp_path: Path) -> None:
    """Stop remote execution before invoking a configured Colab binary."""
    commit = initialize_git(repo_copy)
    marker = tmp_path / "colab-called"
    fake_colab = tmp_path / "colab"
    fake_colab.write_text(f"#!/usr/bin/env bash\ntouch '{marker}'\n", encoding="utf-8")
    fake_colab.chmod(0o755)
    (repo_copy / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    environment = os.environ | {"COLAB_BIN": str(fake_colab)}
    completed = subprocess.run(
        [
            "bash", "scripts/remote/a100_run.sh", "--packet", "P0-003B", "--commit", commit,
            "--script", "scripts/remote/verify_remote_environment.sh", "--dry-run"
        ],
        cwd=repo_copy,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "dirty Git tree" in completed.stderr
    assert not marker.exists()


def test_a100_fetch_rejects_path_escape(repo_copy: Path, tmp_path: Path) -> None:
    """Reject a traversal segment before invoking the Colab downloader."""
    initialize_git(repo_copy)
    marker = tmp_path / "colab-called"
    fake_colab = tmp_path / "colab"
    fake_colab.write_text(f"#!/usr/bin/env bash\ntouch '{marker}'\n", encoding="utf-8")
    fake_colab.chmod(0o755)
    environment = os.environ | {
        "COLAB_BIN": str(fake_colab),
        "LOCAL_PATH": "runs/P0-003B/../../outside.txt",
    }
    completed = subprocess.run(
        ["bash", "scripts/remote/a100_fetch.sh", "--session", "test", "--remote", "fixture.bin"],
        cwd=repo_copy,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "traversal segments" in completed.stderr
    assert not marker.exists()
