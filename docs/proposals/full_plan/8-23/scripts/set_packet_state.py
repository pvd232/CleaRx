#!/usr/bin/env python3
"""Apply one authorized lifecycle transition to orchestration/state.json."""

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime

from orchestration_lib import (
    ALLOWED_TRANSITIONS,
    ContractError,
    repository_root,
    safe_repo_path,
    validate_all,
    validate_result_file,
)


def main() -> int:
    """Validate and atomically append the requested packet transition."""
    parser = argparse.ArgumentParser()
    parser.add_argument("packet_id")
    parser.add_argument("next_state")
    parser.add_argument("--actor", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--result", help="Repository-relative result.json required for a complete transition.")
    args = parser.parse_args()
    root = repository_root()
    try:
        plan, state, _ = validate_all(root)
        node_by_id = {node["id"]: node for node in plan["nodes"]}
        if args.packet_id not in state["packets"]:
            raise ContractError(f"unknown packet: {args.packet_id}")
        entry = state["packets"][args.packet_id]
        previous = entry["status"]
        if args.next_state not in ALLOWED_TRANSITIONS[previous]:
            raise ContractError(f"invalid transition: {previous} -> {args.next_state}")
        if args.next_state in {"ready", "running", "validating", "complete"}:
            incomplete = [dep for dep in node_by_id[args.packet_id]["depends_on"] if state["packets"][dep]["status"] != "complete"]
            if incomplete:
                raise ContractError(f"dependencies incomplete: {incomplete}")
        if args.next_state == "complete":
            if not args.result:
                raise ContractError("complete transition requires --result")
            result_path = safe_repo_path(root, args.result)
            result = validate_result_file(root, result_path)
            if result["packet_id"] != args.packet_id or result["outcome"] != "complete":
                raise ContractError("completion result does not authorize this packet transition")
        entry["history"].append(
            {
                "packet_sha256": entry["packet_sha256"],
                "previous_state": previous,
                "next_state": args.next_state,
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "actor": args.actor,
                "reason": args.reason,
            }
        )
        entry["status"] = args.next_state
        state_path = root / "orchestration" / "state.json"
        descriptor, temporary = tempfile.mkstemp(prefix="state.", suffix=".json", dir=state_path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, state_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    except (ContractError, OSError, ValueError) as error:
        print(f"INVALID: {error}")
        return 1
    print(f"UPDATED: {args.packet_id} {previous} -> {args.next_state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
