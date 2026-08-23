"""Exercise valid and invalid plan, packet, ownership, and state contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import load_yaml, rebind_packet, write_yaml
from orchestration_lib import ContractError, validate_all


def test_current_package_is_valid(repo_copy: Path) -> None:
    """Accept the unmodified Bootstrap orchestration package."""
    plan, state, packets = validate_all(repo_copy)
    assert len(plan["nodes"]) == len(state["packets"]) == len(packets)
    assert {"BOOTSTRAP-001", "BOOTSTRAP-002", "P0-001", "P0-004"} <= set(packets)


def test_stale_packet_digest_is_rejected(repo_copy: Path) -> None:
    """Reject packet bytes changed without rebinding the plan."""
    packet = repo_copy / "orchestration" / "work_packets" / "P0-001.yaml"
    packet.write_text(packet.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match="packet digest mismatch"):
        validate_all(repo_copy)


def test_repository_escape_is_rejected(repo_copy: Path) -> None:
    """Reject a packet write scope that escapes the repository."""
    packet_path = repo_copy / "orchestration" / "work_packets" / "P0-001.yaml"
    packet = load_yaml(packet_path)
    packet["execution"]["write_scope"] = ["../outside"]
    write_yaml(packet_path, packet)
    rebind_packet(repo_copy, "P0-001")
    with pytest.raises(ContractError, match="does not match|path leaves repository"):
        validate_all(repo_copy)


def test_same_wave_scope_overlap_is_rejected(repo_copy: Path) -> None:
    """Reject concurrent-wave packets that own the same path."""
    packet_path = repo_copy / "orchestration" / "work_packets" / "P0-002.yaml"
    packet = load_yaml(packet_path)
    packet["execution"]["write_scope"] = ["upstream.lock"]
    write_yaml(packet_path, packet)
    rebind_packet(repo_copy, "P0-002")
    with pytest.raises(ContractError, match="overlapping scopes"):
        validate_all(repo_copy)


def test_discontinuous_state_history_is_rejected(repo_copy: Path) -> None:
    """Reject a lifecycle event whose prior state is not the preceding event."""
    state_path = repo_copy / "orchestration" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["packets"]["BOOTSTRAP-001"]["history"][-1]["previous_state"] = "planned"
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match="history is discontinuous"):
        validate_all(repo_copy)
