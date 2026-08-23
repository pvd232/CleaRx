"""Shared isolated-repository fixtures for orchestration contract tests."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml


SOURCE_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = SOURCE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))


@pytest.fixture
def repo_copy(tmp_path: Path) -> Path:
    """Copy the current repository content without its Git object database."""
    destination = tmp_path / "repo"
    shutil.copytree(
        SOURCE_ROOT,
        destination,
        ignore=shutil.ignore_patterns(".git", ".pytest_cache", "__pycache__"),
    )
    return destination


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one mapping-valued YAML test record."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    """Write deterministic block-style YAML for a mutated test record."""
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def sha256(path: Path) -> str:
    """Hash exact test-file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rebind_packet(repo: Path, packet_id: str) -> None:
    """Propagate a mutated packet digest through plan and lifecycle state."""
    packet_path = repo / "orchestration" / "work_packets" / f"{packet_id}.yaml"
    packet_digest = sha256(packet_path)
    plan_path = repo / "orchestration" / "plan.yaml"
    plan = load_yaml(plan_path)
    for node in plan["nodes"]:
        if node["id"] == packet_id:
            node["packet_sha256"] = packet_digest
            break
    write_yaml(plan_path, plan)
    state_path = repo / "orchestration" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["plan_sha256"] = sha256(plan_path)
    entry = state["packets"][packet_id]
    entry["packet_sha256"] = packet_digest
    for event in entry["history"]:
        event["packet_sha256"] = packet_digest
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def initialize_git(repo: Path) -> str:
    """Create one deterministic local commit and return its object ID."""
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "CleaRx Tests"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "tests@clearx.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "add", "--all"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=repo, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
