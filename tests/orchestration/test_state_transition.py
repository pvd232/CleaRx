"""Exercise lifecycle completion's dependency on a verified result receipt."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from conftest import initialize_git


def test_completion_without_result_is_rejected(repo_copy: Path) -> None:
    """Refuse validating-to-complete without an explicit result record."""
    initialize_git(repo_copy)
    environment = os.environ | {"PATH": f"/Users/machina/miniconda3/envs/clearx/bin:{os.environ['PATH']}"}
    completed = subprocess.run(
        [
            "/Users/machina/miniconda3/envs/clearx/bin/python",
            "scripts/set_packet_state.py",
            "BOOTSTRAP-001",
            "validating",
            "--actor",
            "test",
            "--reason",
            "test validating transition",
        ],
        cwd=repo_copy,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "running -> validating" in completed.stdout
    completed = subprocess.run(
        [
            "/Users/machina/miniconda3/envs/clearx/bin/python",
            "scripts/set_packet_state.py",
            "BOOTSTRAP-001",
            "complete",
            "--actor",
            "test",
            "--reason",
            "attempt without evidence",
        ],
        cwd=repo_copy,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "requires --result" in completed.stdout
