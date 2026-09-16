"""Exercise the Bootstrap-002 Phase 0 validator entry points."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.benchmarks.validate_cache_contract import validate_cache_contract
from tools.contract_validation import ContractViolation
from tools.inventory.validate_evaluation_contract import validate_evaluation_contract
from tools.inventory.validate_l4_host_profile import REQUIRED_METRICS, REQUIRED_PHASES, validate_l4_profile
from tools.inventory.validate_upstream_lock import OFFICIAL_REPOSITORIES, validate_upstream_lock


ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: object) -> None:
    """Write one deterministic JSON test record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def copy_schema(tmp_path: Path) -> Path:
    """Create a test root containing the frozen benchmark schema."""
    root = tmp_path / "repo"
    destination = root / "orchestration/schemas/benchmark.schema.json"
    destination.parent.mkdir(parents=True)
    destination.write_bytes((ROOT / "orchestration/schemas/benchmark.schema.json").read_bytes())
    return root


def test_upstream_lock_accepts_complete_offline_contract(tmp_path: Path) -> None:
    """Accept authoritative repositories whose symbols occur in the gap matrix."""
    root = tmp_path / "repo"
    (root / "docs/architecture").mkdir(parents=True)
    repositories = []
    matrix_parts = []
    for repository_id, url in OFFICIAL_REPOSITORIES.items():
        symbol = f"{repository_id}_symbol"
        path = f"source/{repository_id}.cpp"
        repositories.append({"id": repository_id, "url": url, "commit": "a" * 40, "commit_time": "2026-08-23T00:00:00Z", "role": "test", "evidence_symbols": [{"path": path, "symbol": symbol}]})
        matrix_parts.append(f"{'a' * 40} {path} {symbol}")
    write_json(root / "upstream.lock", {"schema_version": "1.0.0", "frozen_at": "2026-08-23T00:00:00Z", "repositories": repositories, "file_identities": [{"repository": "qwen-model", "path": "config.json", "sha256": "b" * 64}], "toolchains": {"local_m1": {"compiler": "clang"}, "l4": {"compiler": "gcc", "cuda": "12.9", "driver": "580"}}})
    (root / "docs/architecture/runtime_gap_matrix.md").write_text("\n".join(matrix_parts), encoding="utf-8")
    validate_upstream_lock(root, resolve=False)
    repositories[0]["evidence_symbols"][0]["symbol"] = "missing_symbol"
    write_json(root / "upstream.lock", load := {**json.loads((root / "upstream.lock").read_text()), "repositories": repositories})
    assert load["repositories"]
    with pytest.raises(ContractViolation, match="source symbol is absent"):
        validate_upstream_lock(root, resolve=False)


def test_l4_profile_requires_every_measurement_phase(tmp_path: Path) -> None:
    """Accept a complete L4 profile and reject a missing phase."""
    path = tmp_path / "profile.json"
    profile = {"schema_version": "1.0.0", "profile_id": "l4-test", "captured_at": "2026-08-23T00:00:00Z", "source_commit": "a" * 40, "gce": {"project": "p", "instance": "i", "zone": "z", "machine_type": "g2-standard-12", "provisioning_model": "SPOT", "boot_disk_gib": 375}, "os": {"name": "Ubuntu"}, "cpu": {"model": "x"}, "memory": {"total_bytes": 1}, "gpu": {"name": "NVIDIA L4", "memory_total_mib": 23034, "driver_version": "580", "uuid": "GPU-x"}, "pcie": {"bus_id": "0000:00:00.0", "generation_current": 4, "generation_max": 4, "width_current": 16, "width_max": 16, "numa_node": 0}, "storage": {"device": "/dev/x", "filesystem": "ext4", "size_bytes": 10, "available_bytes": 9, "gce_disk_type": "pd-ssd"}, "software": {"kernel": "k", "cuda": "12.9", "nvcc": "12.9.41", "compiler": "gcc"}, "thermal_power": {"power_limit_w": 72, "temperature_c": 50, "persistence_mode": False, "compute_mode": "Default"}, "measurement_contract": {"phases": sorted(REQUIRED_PHASES), "metrics": sorted(REQUIRED_METRICS), "sampling_interval_ms": 100}}
    write_json(path, profile)
    validate_l4_profile(path)
    profile["measurement_contract"]["phases"].pop()
    write_json(path, profile)
    with pytest.raises(ContractViolation, match="phase measurement set"):
        validate_l4_profile(path)


def test_cache_contract_requires_allocator_events(tmp_path: Path) -> None:
    """Accept the design contract and reject an incomplete event vocabulary."""
    root = copy_schema(tmp_path)
    path = root / "cache.json"
    workload = {"claim_level": "design_only", "trace_schema": {"required_fields": ["layer", "token_position", "selected_experts", "routing_weights", "timestamp_ns"]}, "transfer_cases": [{}, {}, {}], "allocator_events": ["allocate", "release", "transfer_start", "transfer_end", "compute_start", "compute_end"], "overlap_measurements": ["h2d_duration_ns", "compute_duration_ns", "overlap_duration_ns", "unhidden_transfer_ns"], "representative_shape_source": "tensor_manifest.json"}
    record = {"schema_version": "1.0.0", "benchmark_id": "cache", "host_profile": "pending", "workload": workload, "metrics": workload["overlap_measurements"], "repetitions": 10, "warmups": 2}
    write_json(path, record)
    validate_cache_contract(root, path)
    workload["allocator_events"].pop()
    write_json(path, record)
    with pytest.raises(ContractViolation, match="allocator events"):
        validate_cache_contract(root, path)


def test_evaluation_contract_requires_failure_attribution(tmp_path: Path) -> None:
    """Accept paired evaluation records and reject missing attribution fields."""
    root = copy_schema(tmp_path)
    paired = ["item_id", "seed", "sampling_id"]
    corpus = {"schema_version": "1.0.0", "corpus_id": "test", "version": "1", "license": "test", "items": [{"id": "i", "stratum": "speech", "prompt": "p", "audio": "a.wav", "audio_sha256": "a" * 64, "license": "test"}], "sampling": {"id": "s"}, "seeds": [1], "paired_keys": paired}
    threshold_fields = {name: {"metric": name} for name in ("correctness", "semantic", "acoustic", "material_improvement", "latency")}
    threshold_fields["confidence"] = {"method": "paired bootstrap", "level": 0.95, "decision_rule": "upper bound"}
    threshold_fields["failure_attribution"] = {"required_fields": ["first_divergent_module", "failure_class", "corpus_item_id", "seed", "host_profile"]}
    thresholds = {"schema_version": "1.0.0", "benchmark_id": "evaluation", "host_profile": "pending", "workload": {"corpus": "test"}, "metrics": ["correctness"], "repetitions": 3, "warmups": 1, "thresholds": threshold_fields, "paired_keys": paired}
    write_json(root / "evaluation_corpus_manifest.json", corpus)
    write_json(root / "benchmark_thresholds.json", thresholds)
    validate_evaluation_contract(root)
    threshold_fields["failure_attribution"]["required_fields"].pop()
    write_json(root / "benchmark_thresholds.json", thresholds)
    with pytest.raises(ContractViolation, match="failure attribution"):
        validate_evaluation_contract(root)
