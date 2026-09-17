"""Counterexamples for the crossed equal-budget retention pilot."""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from clearx.audio_memory.crossed_retention import (
    ALPHA,
    AUDIO_ONLY_FEATURE_NAMES,
    AUDIO_QUALITIES,
    EXPECTED_COMPUTE_CAPABILITY,
    EXPECTED_DEVICE_NAME,
    EXPERIMENT_ID,
    MEMORY_QUALITIES,
    MODEL_COMMIT,
    POSITION_COUNTS,
    RAVDESS_AUDIO_SHA256,
    SCHEMA_VERSION,
    TEST_ACTORS,
    TRAIN_ACTORS,
    TRANSFORMERS_COMMIT,
    VIPER_COMMIT,
    assemble_candidate_embeddings,
    crossed_retention_payload,
    frozen_item_specs,
    result_identity,
    validate_result,
)
from clearx.audio_memory.gate import FEATURE_NAMES

EXPERIMENT_COMMIT = "a" * 40
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = REPOSITORY_ROOT / "experiments/audio_memory_crossed_retention/bootstrap.py"


def measured_items() -> list[dict[str, Any]]:
    """Build a deterministic 72-item fixture with the production identities."""
    items: list[dict[str, Any]] = []
    for frozen in frozen_item_specs():
        actor = int(frozen["actor_id"])
        statement = int(frozen["statement_id"])
        memory = MEMORY_QUALITIES.index(frozen["memory_quality"])
        audio = AUDIO_QUALITIES.index(frozen["audio_quality"])
        agreement = (2 - memory) * (1.0 - 0.25 * audio)
        benefit = 0.55 + 0.7 * agreement + 0.12 * statement + 0.01 * actor
        base = 4.0 + 0.08 * actor + 0.15 * memory + 0.1 * audio
        transcript_positions = 3
        audio_positions = 12
        full_positions = 30
        gate_values = (
            agreement,
            0.05 * (memory + 1),
            0.2 * agreement,
            0.03 * (audio + 1),
            1.2 - 0.2 * audio,
            0.1 + 0.05 * audio,
            0.4 + 0.25 * audio,
            audio_positions / transcript_positions,
        )
        audio_values = (
            1.2 - 0.2 * audio + 0.01 * actor,
            0.1 + 0.05 * audio,
            0.4 + 0.25 * audio + 0.01 * statement,
        )
        conditions = {}
        for positions in POSITION_COUNTS:
            fraction = positions / 4
            conditions[f"positions_{positions}"] = {
                "acoustic_positions": positions,
                "alpha": ALPHA,
                "input_positions": (
                    full_positions
                    - audio_positions
                    - 2
                    + transcript_positions
                    + positions
                ),
                "transcript_positions": transcript_positions,
                "teacher_kl": base - benefit * fraction,
                "teacher_top1_agreement": positions == 4,
            }
        items.append(
            {
                **frozen,
                "audio_sha256": RAVDESS_AUDIO_SHA256[frozen["filename"]],
                "audio_feature_positions": audio_positions,
                "transcript_token_positions": transcript_positions,
                "full_input_positions": full_positions,
                "gate_features": dict(zip(FEATURE_NAMES, gate_values, strict=True)),
                "audio_only_features": dict(
                    zip(AUDIO_ONLY_FEATURE_NAMES, audio_values, strict=True)
                ),
                "conditions": conditions,
            }
        )
    return items


def completed_result() -> dict[str, Any]:
    """Return one internally consistent synthetic completed result."""
    items = measured_items()
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "source": {"experiment_commit": EXPERIMENT_COMMIT},
        "identity": result_identity(),
        "environment": {
            "device": EXPECTED_DEVICE_NAME,
            "compute_capability": list(EXPECTED_COMPUTE_CAPABILITY),
        },
        "observations": {"item_count": len(items)},
        "retention": crossed_retention_payload(items),
        "items": items,
    }


def test_frozen_protocol_is_complete_crossed_product() -> None:
    """The item protocol is the declared complete cross and frozen revisions."""
    items = frozen_item_specs()
    assert len(items) == 72
    assert TRAIN_ACTORS == ("01", "02", "03", "04")
    assert TEST_ACTORS == ("05", "06")
    assert MODEL_COMMIT == "26291f793822fb6be9555850f06dfe95f2d7e695"
    assert TRANSFORMERS_COMMIT == "7d9754a05193eb79b1d86aa744b622b8068008cd"
    assert VIPER_COMMIT == "43a939a7abb2412cf5da7a0ddd5d01f67d1d84ab"
    bootstrap_source = BOOTSTRAP.read_text(encoding="utf-8")
    assert f'TRANSFORMERS_COMMIT = "{TRANSFORMERS_COMMIT}"' in bootstrap_source
    assert f'VIPER_COMMIT = "{VIPER_COMMIT}"' in bootstrap_source
    counts = Counter((item["actor_id"], item["statement_id"]) for item in items)
    assert set(counts.values()) == {6}
    cells = {
        (item["memory_quality"], item["audio_quality"])
        for item in items
        if item["actor_id"] == "01" and item["statement_id"] == "01"
    }
    assert cells == {
        (memory, audio) for memory in MEMORY_QUALITIES for audio in AUDIO_QUALITIES
    }


def test_payload_rejects_missing_crossed_cell_and_wrong_audio_hash() -> None:
    """A missing cell or substituted recording cannot enter policy fitting."""
    with pytest.raises(ValueError, match="72-item crossed product"):
        crossed_retention_payload(measured_items()[:-1])
    substituted = measured_items()
    substituted[0]["audio_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="audio_sha256"):
        crossed_retention_payload(substituted)


def test_candidate_assembly_preserves_transcript_embeddings() -> None:
    """The shared assembler retains transcript bytes and appends the chosen rows."""
    prefix = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    transcript = np.arange(9, dtype=np.float32).reshape(1, 3, 3) + 100
    suffix = np.arange(3, dtype=np.float32).reshape(1, 1, 3) + 200
    for positions in POSITION_COUNTS:
        acoustic = np.full((1, positions, 3), 17, dtype=np.float32)
        candidate = assemble_candidate_embeddings(
            prefix,
            transcript,
            acoustic,
            suffix,
            join=lambda blocks: np.concatenate(blocks, axis=1),
        )
        assert candidate.shape == (1, 6 + positions, 3)
        assert candidate[:, 2:5].tobytes() == transcript.tobytes()
        assert np.array_equal(candidate[:, 5 : 5 + positions], acoustic)


def test_candidate_assembly_rejects_incompatible_parts() -> None:
    """The assembler rejects different hidden dimensions and unbatched input."""
    valid = np.zeros((1, 1, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="share batch and hidden"):
        assemble_candidate_embeddings(
            valid,
            np.zeros((1, 1, 4), dtype=np.float32),
            valid,
            valid,
            join=lambda blocks: np.concatenate(blocks, axis=1),
        )
    with pytest.raises(ValueError, match="batch, positions, hidden"):
        assemble_candidate_embeddings(
            valid,
            np.zeros((1, 3), dtype=np.float32),
            valid,
            valid,
            join=lambda blocks: np.concatenate(blocks, axis=1),
        )


def test_crossed_payload_compares_equal_budget_policies() -> None:
    """Every held-out policy spends 48 positions and reports clustered effects."""
    payload = crossed_retention_payload(measured_items())
    evaluation = payload["evaluation"]
    assert evaluation["test_item_count"] == 24
    assert evaluation["test_recording_count"] == 4
    assert {
        value["acoustic_positions"] for value in evaluation["budgets"].values()
    } == {48}
    assert (
        len(
            {
                value["aggregate_input_positions"]
                for value in evaluation["budgets"].values()
            }
        )
        == 1
    )
    for contrast in evaluation["clustered_contrasts"].values():
        assert contrast["cluster_count"] == 4
        assert contrast["resample_count"] == 256
        assert len(contrast["resampled_means"]) == 256
        assert set(contrast["actor_effects"]) == set(TEST_ACTORS)
    grouped = Counter(
        item_id.rsplit("-", 2)[0] + "-" + item_id.rsplit("-", 2)[2]
        for item_id in payload["selected"]["adaptive_audio_only"]
    )
    assert set(grouped.values()) == {3}


def test_learned_allocations_ignore_held_out_labels() -> None:
    """Changing held-out KL values cannot change either learned allocation."""
    original = crossed_retention_payload(measured_items())
    changed_items = measured_items()
    for index, item in enumerate(changed_items):
        if item["split"] == "test":
            item["conditions"]["positions_4"]["teacher_kl"] += 0.01 * (index + 1)
    changed = crossed_retention_payload(changed_items)
    assert (
        changed["selected"]["adaptive_agreement"]
        == original["selected"]["adaptive_agreement"]
    )
    assert (
        changed["selected"]["adaptive_audio_only"]
        == original["selected"]["adaptive_audio_only"]
    )
    assert (
        changed["predictions"]["measured_benefit"]
        != original["predictions"]["measured_benefit"]
    )


def test_result_validation_binds_commit_hardware_and_recomputed_payload() -> None:
    """Validation accepts only the independent commit and observed A100 identity."""
    report = validate_result(
        completed_result(), expected_experiment_commit=EXPERIMENT_COMMIT
    )
    assert report["verified"] is True
    assert report["experiment_commit"] == EXPERIMENT_COMMIT
    assert report["item_count"] == 72
    assert report["maximum_numeric_delta"] <= 1e-8


@pytest.mark.parametrize("tamper", ["commit", "device", "capability", "metric"])
def test_result_validation_rejects_provenance_and_metric_tampering(
    tamper: str,
) -> None:
    """Independent verification rejects forged provenance and derived values."""
    result = completed_result()
    if tamper == "commit":
        result["source"]["experiment_commit"] = "b" * 40
    elif tamper == "device":
        result["environment"]["device"] = "NVIDIA L4"
    elif tamper == "capability":
        result["environment"]["compute_capability"] = [8, 9]
    else:
        result["retention"]["evaluation"]["agreement_minus_fixed"] += 1.0
    with pytest.raises(ValueError):
        validate_result(result, expected_experiment_commit=EXPERIMENT_COMMIT)


def test_bootstrap_rejects_broad_or_existing_checkout(tmp_path: Path) -> None:
    """The Colab bootstrap never deletes or reuses a caller-selected directory."""
    outside_content = subprocess.run(
        [
            sys.executable,
            str(BOOTSTRAP),
            "--git-commit",
            EXPERIMENT_COMMIT,
            "--checkout",
            str(tmp_path / "clearx-crossed-retention-run"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert outside_content.returncode != 0
    assert "direct child of /content" in outside_content.stderr
