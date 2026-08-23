#!/usr/bin/env python3
"""Preflight and optionally run one ready packet's declared acceptance commands."""

import argparse
import subprocess

from orchestration_lib import ContractError, repository_root, validate_all


def main() -> int:
    """Require a clean bound packet before executing its acceptance commands."""
    parser = argparse.ArgumentParser()
    parser.add_argument("packet_id")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = repository_root()
    try:
        _, state, packets = validate_all(root)
        if args.packet_id not in packets:
            raise ContractError(f"unknown packet: {args.packet_id}")
        if state["packets"][args.packet_id]["status"] != "ready":
            raise ContractError(f"packet is not ready: {args.packet_id}")
        if subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True).stdout:
            raise ContractError("packet execution requires a clean Git tree")
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
        print(f"READY: {args.packet_id} commit={commit} digest={state['packets'][args.packet_id]['packet_sha256']}")
        if not args.execute:
            return 0
        for command in packets[args.packet_id]["acceptance"]["commands"]:
            completed = subprocess.run(command, cwd=root, shell=True)
            if completed.returncode:
                return completed.returncode
    except (ContractError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"INVALID: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
