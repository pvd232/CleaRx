#!/usr/bin/env python3
"""Run the complete local acceptance gate for Bootstrap-001."""

import subprocess
import sys

from orchestration_lib import ContractError, repository_root, validate_all


def main() -> int:
    """Validate contracts, shell syntax, and the orchestration test suite."""
    root = repository_root()
    try:
        plan, _, _ = validate_all(root)
        for script in sorted((root / "scripts" / "remote").glob("*.sh")):
            subprocess.run(["bash", "-n", str(script)], cwd=root, check=True)
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests/orchestration"],
            cwd=root,
            check=True,
        )
    except (ContractError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"BOOTSTRAP INVALID: {error}")
        return 1
    print(f"BOOTSTRAP VALID: {len(plan['nodes'])} packets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
