#!/usr/bin/env python3
"""Validate authoritative upstream identities and cited runtime-gap symbols."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.contract_validation import (  # noqa: E402
    ContractViolation,
    load_json,
    require,
    require_commit,
    require_keys,
    require_sha256,
    require_timestamp,
)


OFFICIAL_REPOSITORIES = {
    "llama.cpp": "https://github.com/ggml-org/llama.cpp.git",
    "whisper.cpp": "https://github.com/ggml-org/whisper.cpp.git",
    "qwen3-omni": "https://github.com/QwenLM/Qwen3-Omni.git",
    "transformers": "https://github.com/huggingface/transformers.git",
    "qwen-model": "https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "qwen-gguf": "https://huggingface.co/ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF",
}


def commit_resolves(url: str, commit: str) -> bool:
    """Fetch one commit into a temporary empty repository."""
    with tempfile.TemporaryDirectory(prefix="clearx-lock-") as directory:
        subprocess.run(["git", "init", "-q"], cwd=directory, check=True)
        completed = subprocess.run(
            ["git", "fetch", "-q", "--depth=1", url, commit],
            cwd=directory,
            capture_output=True,
        )
        return completed.returncode == 0


def validate_upstream_lock(root: Path, resolve: bool = True) -> dict[str, Any]:
    """Validate the lock, optional remote commits, and gap-matrix citations."""
    record = load_json(root / "upstream.lock")
    require_keys(record, {"schema_version", "frozen_at", "repositories", "file_identities", "toolchains"}, "upstream.lock")
    require(record["schema_version"] == "1.0.0", "upstream.lock: unsupported schema version")
    require_timestamp(record["frozen_at"], "upstream.lock.frozen_at")
    repositories = record["repositories"]
    require(isinstance(repositories, list), "upstream.lock.repositories must be an array")
    by_id = {repository.get("id"): repository for repository in repositories}
    require(set(by_id) == set(OFFICIAL_REPOSITORIES), "upstream.lock: repository set is incomplete")
    matrix = (root / "docs/architecture/runtime_gap_matrix.md").read_text(encoding="utf-8")
    for repository_id, official_url in OFFICIAL_REPOSITORIES.items():
        repository = by_id[repository_id]
        require_keys(repository, {"id", "url", "commit", "commit_time", "role", "evidence_symbols"}, repository_id)
        require(repository["url"] == official_url, f"{repository_id}: source is not authoritative")
        require_commit(repository["commit"], f"{repository_id}.commit")
        require_timestamp(repository["commit_time"], f"{repository_id}.commit_time")
        symbols = repository["evidence_symbols"]
        require(isinstance(symbols, list) and symbols, f"{repository_id}: evidence_symbols is empty")
        for symbol in symbols:
            require_keys(symbol, {"path", "symbol"}, f"{repository_id}.evidence_symbol")
            require(repository["commit"] in matrix, f"{repository_id}: commit is absent from gap matrix")
            require(symbol["path"] in matrix and symbol["symbol"] in matrix, f"{repository_id}: source symbol is absent from gap matrix")
        if resolve:
            require(commit_resolves(official_url, repository["commit"]), f"{repository_id}: commit does not resolve")
    identities = record["file_identities"]
    require(isinstance(identities, list) and identities, "upstream.lock.file_identities is empty")
    for identity in identities:
        require_keys(identity, {"repository", "path", "sha256"}, "file identity")
        require(identity["repository"] in by_id, "file identity references an unknown repository")
        require_sha256(identity["sha256"], f"{identity['repository']}:{identity['path']}")
    toolchains = record["toolchains"]
    require_keys(toolchains, {"local_m1", "l4"}, "upstream.lock.toolchains")
    require("compiler" in toolchains["local_m1"], "local toolchain compiler is absent")
    require_keys(toolchains["l4"], {"compiler", "cuda", "driver"}, "l4 toolchain")
    return record


def main() -> int:
    """Validate the canonical upstream lock and print its repository count."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Skip remote commit fetches.")
    args = parser.parse_args()
    try:
        record = validate_upstream_lock(ROOT, resolve=not args.offline)
    except (ContractViolation, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"INVALID: {error}")
        return 1
    print(f"VALID: {len(record['repositories'])} authoritative repositories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
