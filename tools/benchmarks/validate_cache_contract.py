#!/usr/bin/env python3
"""Validate the design-only expert-cache and allocator benchmark contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.contract_validation import (  # noqa: E402
    ContractViolation,
    load_json,
    require,
    require_keys,
    validate_benchmark_schema,
)


TRACE_FIELDS = {"layer", "token_position", "selected_experts", "routing_weights", "timestamp_ns"}
ALLOCATOR_EVENTS = {"allocate", "release", "transfer_start", "transfer_end", "compute_start", "compute_end"}
OVERLAP_METRICS = {"h2d_duration_ns", "compute_duration_ns", "overlap_duration_ns", "unhidden_transfer_ns"}


def validate_cache_contract(root: Path, path: Path) -> dict[str, Any]:
    """Validate trace, transfer, allocator, and overlap definitions."""
    record = load_json(path)
    validate_benchmark_schema(root, record, str(path))
    workload = record["workload"]
    require_keys(workload, {"claim_level", "trace_schema", "transfer_cases", "allocator_events", "overlap_measurements", "representative_shape_source"}, "cache workload")
    require(workload["claim_level"] == "design_only", "cache workload: claim level must remain design_only")
    require(set(workload["trace_schema"]["required_fields"]) == TRACE_FIELDS, "cache workload: router trace fields are incomplete")
    require(isinstance(workload["transfer_cases"], list) and len(workload["transfer_cases"]) >= 3, "cache workload: pinned transfer cases are incomplete")
    require(set(workload["allocator_events"]) == ALLOCATOR_EVENTS, "cache workload: allocator events are incomplete")
    require(set(workload["overlap_measurements"]) == OVERLAP_METRICS, "cache workload: overlap measurements are incomplete")
    require("tensor_manifest.json" in workload["representative_shape_source"], "cache workload: representative shape source is not manifest-bound")
    require(OVERLAP_METRICS <= set(record["metrics"]), "cache contract: benchmark metrics omit overlap measurements")
    return record


def main() -> int:
    """Validate the canonical expert-cache benchmark contract."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path, default=Path("tests/fixtures/manifests/expert_cache_contract.json"))
    args = parser.parse_args()
    try:
        record = validate_cache_contract(ROOT, args.path)
    except (ContractViolation, OSError, ValueError) as error:
        print(f"INVALID: {error}")
        return 1
    print(f"VALID: {record['benchmark_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
