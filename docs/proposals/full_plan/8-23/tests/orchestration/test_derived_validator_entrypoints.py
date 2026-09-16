"""Exercise derived Phase 0 validators with complete and severed records."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tools.benchmarks.validate_cache_spike import validate_cache_spike
from tools.contract_validation import ContractViolation
from tools.inventory.validate_abi_contract import OWNERSHIP_FIELDS, REQUIRED_HEADINGS, REQUIRED_TYPES, validate_abi_contract
from tools.inventory.validate_deployment_acceptance import ARTIFACT_FIELDS, file_sha256, validate_deployment_acceptance
from tools.inventory.validate_memory_budget import CATEGORIES, PHASES, validate_memory_budget


REPO = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: object) -> None:
    """Write one deterministic test JSON file."""
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_memory_budget_requires_every_phase(tmp_path: Path) -> None:
    """Accept a reconciled budget and reject one missing live phase."""
    path = tmp_path / "budget.json"
    phase = {"expected_bytes": 10, "range_min_bytes": 9, "range_max_bytes": 11, "measured_bytes": None, "categories": {category: 0 for category in CATEGORIES}}
    record = {"schema_version": "1.0.0", "budget_id": "b", "source_commit": "a" * 40, "tensor_manifest_sha256": "b" * 64, "host_profile_sha256": "c" * 64, "units": {"bytes": 1, "decimal_gb": 1000000000, "binary_gib": 1073741824}, "hard_ceiling_bytes": 100, "reserve_bytes": 10, "tensor_bytes": 3, "tensor_categories": {"thinker": 1, "aut": 1, "talker": 1, "mtp": 0, "code2wav": 0}, "phases": {name: dict(phase) for name in PHASES}}
    write_json(path, record)
    validate_memory_budget(path)
    record["phases"].pop(next(iter(PHASES)))
    write_json(path, record)
    with pytest.raises(ContractViolation, match="phase set"):
        validate_memory_budget(path)


def test_cache_spike_requires_raw_repetitions(tmp_path: Path) -> None:
    """Accept 30 raw transfers and reject an undersized result set."""
    path = tmp_path / "spike.json"
    row = {"case_id": "c", "payload_bytes": 1, "h2d_duration_ns": 1, "compute_duration_ns": 1, "overlap_duration_ns": 0, "unhidden_transfer_ns": 1, "repetition_index": 0}
    record = {"schema_version": "1.0.0", "spike_id": "s", "source_commit": "a" * 40, "host_profile_sha256": "b" * 64, "tensor_manifest_sha256": "c" * 64, "memory_budget_sha256": "d" * 64, "contract_sha256": "e" * 64, "slot_pool_target_bytes": 1610612736, "representative_shapes": [{"source": "tensor_manifest.json", "bytes": 1}], "transfer_measurements": [{**row, "repetition_index": index} for index in range(30)], "allocator_simulation": {"events_valid": True, "arena_peak_bytes": 1, "schedules": ["aut"]}, "limitations": ["microbenchmark only"]}
    write_json(path, record)
    validate_cache_spike(path)
    record["transfer_measurements"] = record["transfer_measurements"][:29]
    write_json(path, record)
    with pytest.raises(ContractViolation, match="fewer than 30"):
        validate_cache_spike(path)


def test_abi_contract_requires_cancellation_handshake(tmp_path: Path) -> None:
    """Accept required ABI vocabulary and reject a severed cancellation join."""
    path = tmp_path / "abi.md"
    text = "# abi_v0\n" + "\n".join(sorted(REQUIRED_HEADINGS)) + "\n" + " ".join(f"`{name}`" for name in sorted(REQUIRED_TYPES | OWNERSHIP_FIELDS)) + "\nclearx.metadata.version reject diagnostic cancel_requested cancel_acknowledged\n"
    path.write_text(text, encoding="utf-8")
    validate_abi_contract(path)
    path.write_text(text.replace("cancel_acknowledged", ""), encoding="utf-8")
    with pytest.raises(ContractViolation, match="cancellation handshake"):
        validate_abi_contract(path)


def test_deployment_acceptance_binds_exact_artifact_bytes(tmp_path: Path) -> None:
    """Accept exact deployment inputs and reject one severed artifact hash."""
    for relative_path in ARTIFACT_FIELDS.values():
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative_path, destination)
    host = json.loads((tmp_path / ARTIFACT_FIELDS["host_profile"]).read_text(encoding="utf-8"))
    thresholds = json.loads((tmp_path / ARTIFACT_FIELDS["thresholds"]).read_text(encoding="utf-8"))
    record = {
        "schema_version": "1.0.0",
        "acceptance_id": "test-deployment-acceptance",
        "source_commit": "a" * 40,
        "artifacts": {
            field: {"path": relative_path, "sha256": file_sha256(tmp_path / relative_path)}
            for field, relative_path in ARTIFACT_FIELDS.items()
        },
        "certified_host": {
            "profile_id": host["profile_id"],
            "gpu_uuid": host["gpu"]["uuid"],
            "machine_type": host["gce"]["machine_type"],
            "zone": host["gce"]["zone"],
        },
        "workloads": ["correctness-greedy", "release-quality", "l4-performance"],
        "gates": {
            gate: {
                "metric": binding["metric"],
                "threshold_path": f"thresholds.{gate}",
                "corpus_slices": ["synthetic", "speech"],
                "sampling_ids": [binding.get("sampling_id", "release-quality")],
                "workload_id": "release-quality",
            }
            for gate, binding in thresholds["thresholds"].items()
            if gate in {"correctness", "semantic", "acoustic", "material_improvement", "latency"}
        },
        "alternate_hosts": [{"profile_id": "a100-colab", "classification": "exploratory_only"}],
    }
    path = tmp_path / "deployment_acceptance.json"
    write_json(path, record)
    validate_deployment_acceptance(tmp_path, path)
    record["artifacts"]["corpus"]["sha256"] = "0" * 64
    write_json(path, record)
    with pytest.raises(ContractViolation, match="corpus hash mismatch"):
        validate_deployment_acceptance(tmp_path, path)
