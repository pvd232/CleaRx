#!/usr/bin/env python3
"""Generate the deterministic UTF-8 payload used by the fixture contract smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    """Write the exact 25-byte smoke payload."""
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_bytes(b"clearx bootstrap fixture\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
