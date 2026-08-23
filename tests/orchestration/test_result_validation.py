"""Exercise result bindings to packet and artifact bytes at one Git commit."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from conftest import initialize_git, load_yaml
from orchestration_lib import ContractError, validate_result_file


def committed_bytes(repo: Path, commit: str, path: str) -> bytes:
    """Read one test artifact from the repository snapshot."""
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=repo, check=True, capture_output=True
    ).stdout


def make_result(repo: Path, commit: str) -> Path:
    """Build a complete Bootstrap result bound to the initial test commit."""
    packet_path = repo / "orchestration/work_packets/BOOTSTRAP-001.yaml"
    packet = load_yaml(packet_path)
    artifacts = []
    for path in packet["outputs"]:
        content = committed_bytes(repo, commit, path)
        artifacts.append(
            {"path": path, "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}
        )
    result = {
        "schema_version": "1.0.0",
        "packet_id": "BOOTSTRAP-001",
        "packet_sha256": hashlib.sha256(committed_bytes(repo, commit, "orchestration/work_packets/BOOTSTRAP-001.yaml")).hexdigest(),
        "git_commit": commit,
        "run_id": "bootstrap-test",
        "environment": {"kind": "local_m1"},
        "commands": [{"command": "python scripts/validate_bootstrap.py", "exit_code": 0}],
        "artifacts": artifacts,
        "required_checks": {name: True for name in packet["acceptance"]["required_checks"]},
        "outcome": "complete",
        "deviations": [],
        "started_at": "2026-08-23T00:00:00Z",
        "finished_at": "2026-08-23T00:01:00Z"
    }
    result_path = repo / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result_path


def test_result_uses_recorded_commit_artifacts(repo_copy: Path) -> None:
    """Accept committed artifact bytes even after mutable state advances."""
    commit = initialize_git(repo_copy)
    result_path = make_result(repo_copy, commit)
    state_path = repo_copy / "orchestration/state.json"
    state_path.write_text(state_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    result = validate_result_file(repo_copy, result_path)
    assert result["git_commit"] == commit


def test_result_artifact_digest_mismatch_is_rejected(repo_copy: Path) -> None:
    """Reject a result whose artifact hash differs from its Git snapshot."""
    commit = initialize_git(repo_copy)
    result_path = make_result(repo_copy, commit)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["artifacts"][0]["sha256"] = "0" * 64
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match="artifact identity mismatch"):
        validate_result_file(repo_copy, result_path)


def test_failed_required_check_cannot_complete(repo_copy: Path) -> None:
    """Reject a complete result with one false required check."""
    commit = initialize_git(repo_copy)
    result_path = make_result(repo_copy, commit)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    first_check = next(iter(result["required_checks"]))
    result["required_checks"][first_check] = False
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match="does not pass every required check"):
        validate_result_file(repo_copy, result_path)
