"""Exercise fixture-manifest schema and exact artifact identity checks."""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import initialize_git
from orchestration_lib import ContractError
from validate_fixtures import validate_manifest


def rewrite_descriptor(repo: Path, mutate: object) -> Path:
    """Mutate the strict descriptor and rebind its envelope artifact identity."""
    manifest_path = repo / "tests/fixtures/manifests/fixture_contract_examples/bootstrap_smoke.json"
    descriptor_path = repo / "tests/fixtures/native/bootstrap_smoke.descriptor.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    mutate(descriptor)
    descriptor_path.write_text(json.dumps(descriptor, indent=2) + "\n", encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    descriptor_artifact = next(item for item in manifest["artifacts"] if item["dtype"].endswith("fixture-descriptor+json"))
    descriptor_artifact["sha256"] = hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
    descriptor_artifact["shape"] = [descriptor_path.stat().st_size]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def test_fixture_contract_example_is_valid(repo_copy: Path) -> None:
    """Accept the checked-in fixture contract example."""
    manifest = repo_copy / "tests/fixtures/manifests/fixture_contract_examples/bootstrap_smoke.json"
    validate_manifest(repo_copy, manifest)


def test_fixture_checksum_mismatch_is_rejected(repo_copy: Path) -> None:
    """Reject a manifest that does not identify the referenced bytes."""
    manifest_path = repo_copy / "tests/fixtures/manifests/fixture_contract_examples/bootstrap_smoke.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match="artifact identity mismatch"):
        validate_manifest(repo_copy, manifest_path)


def test_fixture_missing_provenance_is_rejected(repo_copy: Path) -> None:
    """Reject a descriptor that drops the tensor-manifest identity join."""
    manifest_path = rewrite_descriptor(repo_copy, lambda value: value["provenance"].pop("tensor_manifest_sha256"))
    with pytest.raises(ContractError, match="tensor_manifest_sha256"):
        validate_manifest(repo_copy, manifest_path)


def test_fixture_tolerance_profile_drift_is_rejected(repo_copy: Path) -> None:
    """Reject a named profile whose numeric acceptance limit changed."""
    manifest_path = rewrite_descriptor(repo_copy, lambda value: value["comparison"].__setitem__("atol", 0.5))
    with pytest.raises(ContractError, match="tolerance profile changed"):
        validate_manifest(repo_copy, manifest_path)


def test_fixture_descriptor_artifact_join_is_rejected(repo_copy: Path) -> None:
    """Reject descriptor metadata that differs from the envelope artifact."""
    manifest_path = rewrite_descriptor(repo_copy, lambda value: value["outputs"][0].__setitem__("dtype", "float32"))
    with pytest.raises(ContractError, match="descriptor artifacts do not match"):
        validate_manifest(repo_copy, manifest_path)


def test_fixture_source_model_drift_is_rejected(repo_copy: Path) -> None:
    """Reject source-model provenance that differs from the frozen upstream lock."""
    manifest_path = rewrite_descriptor(
        repo_copy,
        lambda value: value["provenance"].__setitem__("source_model_commit", "a" * 40),
    )
    with pytest.raises(ContractError, match="source-model identity mismatch"):
        validate_manifest(repo_copy, manifest_path)


def test_fixture_sampling_join_is_rejected(repo_copy: Path) -> None:
    """Reject envelope sampling that differs from descriptor execution."""
    manifest_path = repo_copy / "tests/fixtures/manifests/fixture_contract_examples/bootstrap_smoke.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sampling"]["top_k"] = 50
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match="sampling does not match"):
        validate_manifest(repo_copy, manifest_path)


def test_contract_only_invokes_p0_fixture_harness(repo_copy: Path) -> None:
    """Reject contract-only acceptance when the P0 harness rejects its definition."""
    harness = repo_copy / "tools/reference/fixture_harness.py"
    harness.parent.mkdir(parents=True, exist_ok=True)
    harness.write_text("raise SystemExit(7)\n", encoding="utf-8")
    initialize_git(repo_copy)
    completed = subprocess.run(
        [sys.executable, "scripts/validate_fixtures.py", "--contract-only"],
        cwd=repo_copy,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "fixture harness contract validation failed" in completed.stdout
