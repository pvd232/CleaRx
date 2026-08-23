#!/usr/bin/env python3
"""Validate CleaRx fixture manifests and every artifact checksum they declare."""

import argparse
from pathlib import Path

from orchestration_lib import ContractError, load_record, load_schema, repository_root, safe_repo_path, sha256_file, validate_schema


def validate_manifest(root: Path, manifest_path: Path) -> None:
    """Validate one fixture manifest and its referenced artifact bytes."""
    manifest = load_record(manifest_path)
    validate_schema(manifest, load_schema(root, "fixture.schema.json"), str(manifest_path))
    for artifact in manifest["artifacts"]:
        path = safe_repo_path(root, artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise ContractError(f"fixture artifact identity mismatch: {artifact['path']}")


def main() -> int:
    """Validate contract examples or all generated reference manifests."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    root = repository_root()
    base = root / "tests" / "fixtures" / "manifests"
    pattern = "fixture_contract_examples/*.json" if args.contract_only else "**/*.json"
    manifests = sorted(base.glob(pattern))
    try:
        for manifest in manifests:
            validate_manifest(root, manifest)
    except (ContractError, OSError, ValueError) as error:
        print(f"INVALID: {error}")
        return 1
    print(f"VALID: {len(manifests)} fixture manifests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
