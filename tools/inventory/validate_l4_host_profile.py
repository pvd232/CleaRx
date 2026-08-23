#!/usr/bin/env python3
"""Validate the certified L4 host profile and measurement obligations."""

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
    require_commit,
    require_keys,
    require_timestamp,
)


REQUIRED_PHASES = {
    "listening",
    "voxzip_concurrent_listening",
    "voxzip_sequential_endpoint",
    "thinker_prefill",
    "talker_startup",
    "talker_sustained_output",
    "barge_in",
    "context_rebuild",
}
REQUIRED_METRICS = {"phase_live_sets", "allocator_peak", "process_rss"}


def validate_l4_profile(path: Path) -> dict[str, Any]:
    """Validate one captured host identity and its future measurement contract."""
    record = load_json(path)
    require_keys(
        record,
        {"schema_version", "profile_id", "captured_at", "source_commit", "gce", "os", "cpu", "memory", "gpu", "pcie", "storage", "software", "thermal_power", "measurement_contract"},
        "L4 profile",
    )
    require(record["schema_version"] == "1.0.0", "L4 profile: unsupported schema version")
    require_timestamp(record["captured_at"], "L4 profile.captured_at")
    require_commit(record["source_commit"], "L4 profile.source_commit")
    require_keys(record["gce"], {"project", "instance", "zone", "machine_type", "provisioning_model", "boot_disk_gib"}, "L4 profile.gce")
    require(record["gce"]["machine_type"] == "g2-standard-12", "L4 profile: unexpected machine type")
    require(record["gce"]["provisioning_model"] == "SPOT", "L4 profile: provisioning model is not SPOT")
    require_keys(record["gpu"], {"name", "memory_total_mib", "driver_version", "uuid"}, "L4 profile.gpu")
    require(record["gpu"]["name"] == "NVIDIA L4", "L4 profile: GPU is not NVIDIA L4")
    require(record["gpu"]["memory_total_mib"] >= 22000, "L4 profile: GPU memory is below the L4 contract")
    require_keys(record["pcie"], {"bus_id", "generation_current", "generation_max", "width_current", "width_max", "numa_node"}, "L4 profile.pcie")
    require(record["pcie"]["width_current"] > 0 and record["pcie"]["width_max"] > 0, "L4 profile: PCIe width is absent")
    require_keys(record["storage"], {"device", "filesystem", "size_bytes", "available_bytes", "gce_disk_type"}, "L4 profile.storage")
    require(record["storage"]["size_bytes"] > record["storage"]["available_bytes"] > 0, "L4 profile: storage byte counts are invalid")
    require_keys(record["software"], {"kernel", "cuda", "nvcc", "compiler"}, "L4 profile.software")
    require_keys(record["thermal_power"], {"power_limit_w", "temperature_c", "persistence_mode", "compute_mode"}, "L4 profile.thermal_power")
    contract = record["measurement_contract"]
    require_keys(contract, {"phases", "metrics", "sampling_interval_ms"}, "L4 profile.measurement_contract")
    require(set(contract["phases"]) == REQUIRED_PHASES, "L4 profile: phase measurement set is incomplete")
    require(set(contract["metrics"]) == REQUIRED_METRICS, "L4 profile: measurement metric set is incomplete")
    require(contract["sampling_interval_ms"] > 0, "L4 profile: sampling interval must be positive")
    return record


def main() -> int:
    """Validate one L4 host profile supplied on the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        record = validate_l4_profile(args.path)
    except (ContractViolation, OSError, ValueError) as error:
        print(f"INVALID: {error}")
        return 1
    print(f"VALID: {record['profile_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
