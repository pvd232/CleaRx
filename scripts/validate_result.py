#!/usr/bin/env python3
"""Validate one packet result against its schema, commit, packet, and artifacts."""

import argparse
from pathlib import Path

from orchestration_lib import ContractError, repository_root, validate_result_file


def main() -> int:
    """Validate the result path supplied on the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    root = repository_root()
    try:
        result = validate_result_file(root, (root / args.result).resolve())
    except (ContractError, OSError, ValueError) as error:
        print(f"INVALID: {error}")
        return 1
    print(f"VALID: {result['packet_id']} run {result['run_id']} outcome={result['outcome']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
