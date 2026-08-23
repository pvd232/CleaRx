"""Exercise lifecycle completion's dependency on a verified result receipt."""

from __future__ import annotations

import os
import json
import subprocess
from pathlib import Path

from conftest import initialize_git


def test_completion_without_result_is_rejected(repo_copy: Path) -> None:
    """Refuse validating-to-complete without an explicit result record."""
    state_path = repo_copy / "orchestration/state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    bootstrap = state["packets"]["BOOTSTRAP-001"]
    bootstrap["history"] = bootstrap["history"][:3]
    bootstrap["status"] = "running"
    for packet_id, entry in state["packets"].items():
        if packet_id != "BOOTSTRAP-001":
            entry["history"] = entry["history"][:1]
            entry["status"] = "planned"
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
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
