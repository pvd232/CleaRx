#!/usr/bin/env python3
"""Validate fixture envelopes, strict descriptors, artifact bytes, and contracts."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from fixture_contract import validate_fixture_envelope
from orchestration_lib import ContractError, load_record, load_schema, repository_root, safe_repo_path, sha256_file, validate_schema


def validate_manifest(root: Path, manifest_path: Path) -> None:
    """Validate one fixture envelope through its strict descriptor."""
    validate_fixture_envelope(root, manifest_path)


def required_fixture_pairs(root: Path) -> set[tuple[str, str]]:
    """Load the boundary-case cross product frozen by P0-003A."""
    sys.path.insert(0, str(root / "tools" / "reference"))
    from fixture_harness import required_model_fixture_pairs, validate_contract

    validate_contract()
    return required_model_fixture_pairs()


def validate_reference_index(
    root: Path,
    index_path: Path,
    expected_pairs: Iterable[tuple[str, str]] | None = None,
) -> int:
    """Validate the generated index, every envelope, and complete contract coverage."""
    index = load_record(index_path)
    validate_schema(index, load_schema(root, "fixture_index.schema.json"), str(index_path))
    expected = set(expected_pairs) if expected_pairs is not None else required_fixture_pairs(root)
    actual: set[tuple[str, str]] = set()
    manifest_paths: set[str] = set()
    descriptor_ids: set[str] = set()
    for entry in index["fixtures"]:
        pair = (entry["boundary_id"], entry["case_id"])
        if pair in actual:
            raise ContractError(f"fixture index repeats boundary-case pair: {pair}")
        actual.add(pair)
        relative_path = entry["manifest_path"]
        if not relative_path.startswith("tests/fixtures/reference/") or relative_path == "tests/fixtures/reference/manifest.json":
            raise ContractError(f"fixture index manifest leaves reference root: {relative_path}")
        if relative_path in manifest_paths:
            raise ContractError(f"fixture index repeats manifest path: {relative_path}")
        manifest_paths.add(relative_path)
        manifest_path = safe_repo_path(root, relative_path)
        if not manifest_path.is_file() or sha256_file(manifest_path) != entry["manifest_sha256"]:
            raise ContractError(f"fixture index manifest identity mismatch: {relative_path}")
        descriptor = validate_fixture_envelope(root, manifest_path)
        if descriptor["fixture_kind"] != "model_boundary":
            raise ContractError(f"generated fixture has wrong kind: {relative_path}")
        if descriptor["boundary"]["id"] != entry["boundary_id"] or descriptor["case"]["id"] != entry["case_id"]:
            raise ContractError(f"fixture index boundary-case join mismatch: {relative_path}")
        if descriptor["provenance"]["generator_commit"] != index["generator_commit"]:
            raise ContractError(f"fixture index generator commit mismatch: {relative_path}")
        if descriptor["descriptor_id"] in descriptor_ids:
            raise ContractError(f"fixture index repeats descriptor ID: {descriptor['descriptor_id']}")
        descriptor_ids.add(descriptor["descriptor_id"])
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ContractError(f"fixture index coverage mismatch: missing={missing[:5]} extra={extra[:5]}")
    return len(actual)


def main() -> int:
    """Validate contract examples or all generated reference manifests."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    root = repository_root()
    examples = sorted((root / "tests" / "fixtures" / "manifests" / "fixture_contract_examples").glob("*.json"))
    try:
        for manifest in examples:
            validate_manifest(root, manifest)
        harness = root / "tools" / "reference" / "fixture_harness.py"
        if args.contract_only and harness.is_file():
            completed = subprocess.run([sys.executable, str(harness), "--contract-only"], cwd=root)
            if completed.returncode:
                raise ContractError("fixture harness contract validation failed")
        generated_count = 0
        if not args.contract_only:
            index_path = root / "tests" / "fixtures" / "reference" / "manifest.json"
            if not index_path.is_file():
                raise ContractError("generated reference fixture index is missing")
            generated_count = validate_reference_index(root, index_path)
    except (ContractError, OSError, ValueError) as error:
        print(f"INVALID: {error}")
        return 1
    print(f"VALID: {len(examples)} contract examples, {generated_count} generated fixtures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
