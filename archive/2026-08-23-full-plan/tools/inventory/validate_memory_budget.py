#!/usr/bin/env python3
"""Validate a manifest-derived, phase-specific L4 memory budget."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.contract_validation import ContractViolation, load_json, require, require_commit, require_keys, require_sha256  # noqa: E402


PHASES = {"listening", "voxzip_concurrent_listening", "voxzip_sequential_endpoint", "thinker_prefill", "talker_startup", "talker_sustained_output", "barge_in", "context_rebuild"}
CATEGORIES = {"weights", "kv", "workspace", "staging", "runtime", "fragmentation"}


def validate_memory_budget(path: Path) -> dict[str, Any]:
    """Validate tensor reconciliation, units, phases, ranges, and reserve."""
    record = load_json(path)
    require_keys(record, {"schema_version", "budget_id", "source_commit", "tensor_manifest_sha256", "host_profile_sha256", "units", "hard_ceiling_bytes", "reserve_bytes", "tensor_bytes", "tensor_categories", "phases"}, "memory budget")
    require(record["schema_version"] == "1.0.0", "memory budget: unsupported schema version")
    require_commit(record["source_commit"], "memory budget.source_commit")
    require_sha256(record["tensor_manifest_sha256"], "memory budget.tensor_manifest_sha256")
    require_sha256(record["host_profile_sha256"], "memory budget.host_profile_sha256")
    require_keys(record["units"], {"bytes", "decimal_gb", "binary_gib"}, "memory budget.units")
    require(record["units"] == {"bytes": 1, "decimal_gb": 1000000000, "binary_gib": 1073741824}, "memory budget: unit definitions changed")
    require(record["reserve_bytes"] > 0, "memory budget: reserve is absent")
    require(set(record["tensor_categories"]) >= {"thinker", "aut", "talker", "mtp", "code2wav"}, "memory budget: tensor categories are incomplete")
    require(sum(record["tensor_categories"].values()) == record["tensor_bytes"], "memory budget: tensor sum does not reconcile")
    require(set(record["phases"]) == PHASES, "memory budget: phase set is incomplete")
    for phase, values in record["phases"].items():
        require_keys(values, {"expected_bytes", "range_min_bytes", "range_max_bytes", "measured_bytes", "categories"}, f"memory phase {phase}")
        require(set(values["categories"]) == CATEGORIES, f"memory phase {phase}: categories are incomplete")
        require(values["range_min_bytes"] <= values["expected_bytes"] <= values["range_max_bytes"], f"memory phase {phase}: expected value leaves range")
        require(values["range_max_bytes"] + record["reserve_bytes"] <= record["hard_ceiling_bytes"], f"memory phase {phase}: range plus reserve exceeds ceiling")
    return record


def main() -> int:
    """Validate one memory-budget JSON file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        record = validate_memory_budget(args.path)
    except (ContractViolation, OSError, ValueError) as error:
        print(f"INVALID: {error}")
        return 1
    print(f"VALID: {record['budget_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
