"""Exercise fixture-manifest schema and exact artifact identity checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestration_lib import ContractError
from validate_fixtures import validate_manifest


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
