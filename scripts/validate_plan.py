#!/usr/bin/env python3
"""Validate the canonical CleaRx plan, packets, state, DAG, and ownership."""

from orchestration_lib import ContractError, repository_root, validate_all


def main() -> int:
    """Run static orchestration validation and return a shell status."""
    try:
        plan, state, _ = validate_all(repository_root())
    except (ContractError, OSError, ValueError) as error:
        print(f"INVALID: {error}")
        return 1
    statuses: dict[str, int] = {}
    for entry in state["packets"].values():
        statuses[entry["status"]] = statuses.get(entry["status"], 0) + 1
    print(f"VALID: {len(plan['nodes'])} packets; statuses={statuses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
