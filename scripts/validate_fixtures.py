#!/usr/bin/env python3
"""Validate fixture envelopes, strict descriptors, artifact bytes, and contracts."""

import argparse
import subprocess
import sys
from pathlib import Path

from fixture_contract import validate_fixture_envelope
from orchestration_lib import ContractError, repository_root


def validate_manifest(root: Path, manifest_path: Path) -> None:
    """Validate one fixture envelope through its strict descriptor."""
    validate_fixture_envelope(root, manifest_path)


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
        harness = root / "tools" / "reference" / "fixture_harness.py"
        if args.contract_only and harness.is_file():
            completed = subprocess.run([sys.executable, str(harness), "--contract-only"], cwd=root)
            if completed.returncode:
                raise ContractError("fixture harness contract validation failed")
    except (ContractError, OSError, ValueError) as error:
        print(f"INVALID: {error}")
        return 1
    print(f"VALID: {len(manifests)} fixture manifests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
