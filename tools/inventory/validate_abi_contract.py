#!/usr/bin/env python3
"""Validate required types, ownership fields, cancellation, and metadata in abi_v0."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.contract_validation import ContractViolation, require  # noqa: E402


REQUIRED_HEADINGS = {"## Version", "## Typed boundaries", "## Buffer ownership", "## Synchronization", "## Errors and cancellation", "## GGUF metadata rejection"}
REQUIRED_TYPES = {"PcmInput", "AutFrames", "ThinkerConditioning", "TalkerRequest", "TalkerResult", "ExpertLoad", "MtpFrame", "CodecFrame", "Code2WavChunk", "SchedulerEvent", "ContextRebuild"}
OWNERSHIP_FIELDS = {"owner", "lifetime", "device", "alignment", "stream", "ready_event", "mutable", "cancellation"}


def validate_abi_contract(path: Path) -> str:
    """Validate the frozen ABI document's mechanically required vocabulary."""
    text = path.read_text(encoding="utf-8")
    for heading in REQUIRED_HEADINGS:
        require(heading in text, f"ABI contract: missing heading {heading}")
    for type_name in REQUIRED_TYPES:
        require(f"`{type_name}`" in text, f"ABI contract: missing type {type_name}")
    for field in OWNERSHIP_FIELDS:
        require(f"`{field}`" in text, f"ABI contract: missing ownership field {field}")
    require("abi_v0" in text and "clearx.metadata.version" in text, "ABI contract: metadata version is absent")
    require("cancel_requested" in text and "cancel_acknowledged" in text, "ABI contract: cancellation handshake is incomplete")
    require("reject" in text.lower() and "diagnostic" in text.lower(), "ABI contract: loader rejection diagnostic is absent")
    return text


def main() -> int:
    """Validate one ABI contract Markdown file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        validate_abi_contract(args.path)
    except (ContractViolation, OSError, ValueError) as error:
        print(f"INVALID: {error}")
        return 1
    print(f"VALID: {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
