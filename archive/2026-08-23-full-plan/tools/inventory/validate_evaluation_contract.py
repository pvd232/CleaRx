#!/usr/bin/env python3
"""Validate the frozen evaluation corpus and benchmark threshold contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.contract_validation import (  # noqa: E402
    ContractViolation,
    load_json,
    require,
    require_keys,
    require_sha256,
    validate_benchmark_schema,
)


REQUIRED_ATTRIBUTION = {"first_divergent_module", "failure_class", "corpus_item_id", "seed", "host_profile"}


def validate_evaluation_contract(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate corpus identities, paired keys, margins, and failure attribution."""
    corpus = load_json(root / "evaluation_corpus_manifest.json")
    thresholds = load_json(root / "benchmark_thresholds.json")
    require_keys(corpus, {"schema_version", "corpus_id", "version", "license", "items", "sampling", "seeds", "paired_keys"}, "evaluation corpus")
    require(corpus["schema_version"] == "1.0.0", "evaluation corpus: unsupported schema version")
    require(isinstance(corpus["items"], list) and corpus["items"], "evaluation corpus: no items")
    for item in corpus["items"]:
        require_keys(item, {"id", "stratum", "prompt", "audio", "audio_sha256", "license"}, "evaluation item")
        require_sha256(item["audio_sha256"], f"evaluation item {item['id']}")
    require(isinstance(corpus["seeds"], list) and corpus["seeds"], "evaluation corpus: seeds are absent")
    require(set(corpus["paired_keys"]) >= {"item_id", "seed", "sampling_id"}, "evaluation corpus: paired keys are incomplete")
    validate_benchmark_schema(root, thresholds, "benchmark_thresholds.json")
    require_keys(thresholds["thresholds"], {"correctness", "semantic", "acoustic", "material_improvement", "latency", "confidence", "failure_attribution"}, "benchmark thresholds")
    require(set(thresholds["thresholds"]["failure_attribution"]["required_fields"]) == REQUIRED_ATTRIBUTION, "benchmark thresholds: failure attribution is incomplete")
    for category in ("correctness", "semantic", "acoustic", "material_improvement", "latency"):
        require("metric" in thresholds["thresholds"][category], f"benchmark thresholds: {category} metric is absent")
    require_keys(thresholds["thresholds"]["confidence"], {"method", "level", "decision_rule"}, "benchmark confidence")
    require(set(thresholds["paired_keys"]) == set(corpus["paired_keys"]), "evaluation contract: paired keys disagree")
    return corpus, thresholds


def main() -> int:
    """Validate the canonical evaluation corpus and threshold files."""
    argparse.ArgumentParser().parse_args()
    try:
        corpus, thresholds = validate_evaluation_contract(ROOT)
    except (ContractViolation, OSError, ValueError) as error:
        print(f"INVALID: {error}")
        return 1
    print(f"VALID: {corpus['corpus_id']} / {thresholds['benchmark_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
