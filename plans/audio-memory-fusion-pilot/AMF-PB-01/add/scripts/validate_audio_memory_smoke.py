#!/usr/bin/env python3
"""Validate one retained audio-memory smoke result against its source commit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.research.audio_memory_fusion import validate_smoke_result


def main() -> int:
    """Load one result file and reject any contract mismatch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    value = json.loads(args.result.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("smoke result must be a JSON object")
    validate_smoke_result(value, args.git_commit)
    print(f"VALID: {args.result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
