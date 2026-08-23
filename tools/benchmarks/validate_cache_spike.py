#!/usr/bin/env python3
"""Validate manifest-derived cache transfers and allocator simulation evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.contract_validation import ContractViolation, load_json, require, require_commit, require_keys, require_sha256  # noqa: E402


def validate_cache_spike(path: Path) -> dict[str, Any]:
    """Validate identities, representative shapes, raw repetitions, and limits."""
    record = load_json(path)
    require_keys(record, {"schema_version", "spike_id", "source_commit", "host_profile_sha256", "tensor_manifest_sha256", "memory_budget_sha256", "contract_sha256", "slot_pool_target_bytes", "representative_shapes", "transfer_measurements", "allocator_simulation", "limitations"}, "cache spike")
    require(record["schema_version"] == "1.0.0", "cache spike: unsupported schema version")
    require_commit(record["source_commit"], "cache spike.source_commit")
    for field in ("host_profile_sha256", "tensor_manifest_sha256", "memory_budget_sha256", "contract_sha256"):
        require_sha256(record[field], f"cache spike.{field}")
    require(record["slot_pool_target_bytes"] == 1610612736, "cache spike: slot-pool target changed")
    require(isinstance(record["representative_shapes"], list) and record["representative_shapes"], "cache spike: representative shapes are absent")
    require(all(shape.get("source") == "tensor_manifest.json" and shape.get("bytes", 0) > 0 for shape in record["representative_shapes"]), "cache spike: shape is not manifest-derived")
    measurements = record["transfer_measurements"]
    require(isinstance(measurements, list) and len(measurements) >= 30, "cache spike: fewer than 30 transfer repetitions")
    required_measurements = {"case_id", "payload_bytes", "h2d_duration_ns", "compute_duration_ns", "overlap_duration_ns", "unhidden_transfer_ns", "repetition_index"}
    require(all(required_measurements <= set(row) for row in measurements), "cache spike: raw transfer measurement is incomplete")
    require_keys(record["allocator_simulation"], {"events_valid", "arena_peak_bytes", "schedules"}, "cache spike.allocator_simulation")
    require(record["allocator_simulation"]["events_valid"] is True, "cache spike: allocator events are invalid")
    require(isinstance(record["limitations"], list) and record["limitations"], "cache spike: limitations are absent")
    return record


def main() -> int:
    """Validate one expert-cache spike JSON file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        record = validate_cache_spike(args.path)
    except (ContractViolation, OSError, ValueError) as error:
        print(f"INVALID: {error}")
        return 1
    print(f"VALID: {record['spike_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
