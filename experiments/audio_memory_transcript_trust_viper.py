"""Verify the transcript-trust gate through one immutable VIPER run."""

from __future__ import annotations

import json
import math
from typing import Any

from viper import execution
from viper.authoring import experiment, input, plan, replicate, stage, variant
from viper.config import DiagnosticConfig
from viper.outputs import StageOutputs, output
from viper.references import GitFileRef
from viper.repository import read_source
from viper.runtime import LocalEnvSpec, observe_python_env
from viper.stages import StageContext, diagnostic

from tools.research.audio_memory_gate import gate_payload, load_json

RESULT_PATH = "runs/audio-memory-transcript-trust/20260916T101500Z-a100/result.json"


def maximum_delta(expected: Any, observed: Any) -> float:
    """Return the largest finite numeric difference across matching structures."""
    if isinstance(expected, dict) and isinstance(observed, dict):
        if set(expected) != set(observed):
            raise ValueError("gate mappings contain different keys")
        return max(
            (maximum_delta(expected[key], observed[key]) for key in expected),
            default=0.0,
        )
    if isinstance(expected, list) and isinstance(observed, list):
        if len(expected) != len(observed):
            raise ValueError("gate sequences contain different lengths")
        return max(
            (
                maximum_delta(expected_value, observed_value)
                for expected_value, observed_value in zip(
                    expected,
                    observed,
                    strict=True,
                )
            ),
            default=0.0,
        )
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        delta = abs(float(expected) - float(observed))
        if not math.isfinite(delta):
            raise ValueError("gate comparison produced a non-finite difference")
        return delta
    if expected != observed:
        raise ValueError("gate values differ")
    return 0.0


@diagnostic(config=DiagnosticConfig)
def verify_gate(context: StageContext[DiagnosticConfig]) -> None:
    """Verify the ASR stop rule or recompute a completed gate fit."""
    result = load_json(context.inputs["a100_result"])
    status = result.get("status")
    if status == "stopped_after_preflight":
        preflight = result.get("preflight")
        if not isinstance(preflight, dict):
            raise TypeError("stopped A100 result must contain its preflight")
        observations = preflight.get("items")
        if not isinstance(observations, list) or len(observations) != 2:
            raise ValueError("preflight must contain two observations")
        correct = sum(
            observation.get("expected") == observation.get("predicted")
            for observation in observations
            if isinstance(observation, dict)
        )
        minimum_correct = int(preflight.get("minimum_correct", -1))
        expected_pass = correct >= minimum_correct
        if (
            int(preflight.get("correct", -1)) != correct
            or bool(preflight.get("passed")) != expected_pass
            or expected_pass
        ):
            raise ValueError("recorded preflight does not justify stopping")
        report = {
            "schema_version": "1.0.0",
            "verified": True,
            "status": status,
            "a100_source_commit": result["source"]["experiment_commit"],
            "preflight_correct": correct,
            "stop_rule_verified": True,
        }
        destination = context.outputs["report"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    if status != "completed":
        raise ValueError("A100 gate result has an unknown status")
    items = result.get("items")
    if not isinstance(items, list):
        raise TypeError("A100 gate result must contain item observations")
    expected = gate_payload(items)
    observed = result.get("gate")
    if not isinstance(observed, dict):
        raise TypeError("A100 gate result must contain the fitted gate")
    delta = maximum_delta(expected, observed)
    if delta > 1e-8:
        raise ValueError(f"recomputed gate differs by {delta}")

    report = {
        "schema_version": "1.0.0",
        "verified": True,
        "a100_source_commit": result["source"]["experiment_commit"],
        "item_count": len(items),
        "training_item_count": expected["model"]["training_item_count"],
        "test_item_count": len(expected["test_predictions"]),
        "selected_ridge": expected["model"]["ridge"],
        "fixed_alpha_selected_on_train": expected["fixed_alpha_selected_on_train"],
        "maximum_numeric_delta": delta,
        "directional_success": result["learned_evaluation"]["directional_success"],
        "predicted_alpha_orders_memory_quality": result["learned_evaluation"][
            "predicted_alpha_orders_memory_quality"
        ],
        "oracle_alpha_orders_memory_quality": result["learned_evaluation"][
            "oracle_alpha_orders_memory_quality"
        ],
    }
    destination = context.outputs["report"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


verification = stage(
    verify_gate,
    stage_id="verify_gate",
    inputs=(input("a100_result", path=RESULT_PATH, data_role="eval"),),
    outputs=StageOutputs(
        report=output(
            path="verification.json",
            loader=load_json,
            data_role="eval",
        )
    ),
)
study = experiment(
    experiment_id="audio_memory_transcript_trust_verification",
    variants=(
        variant(
            "ravdess_transcript_corruption",
            stages=(verification,),
            estimator=verification.outputs["report"],
        ),
    ),
    replicates=(replicate(seed=7),),
)


def main() -> None:
    """Run the VIPER verification and print its durable result locations."""
    source = read_source()
    environment = LocalEnvSpec(
        lockfile=GitFileRef(
            repository=source.repository,
            commit=source.commit,
            path="environment.yml",
        ),
        python_env=observe_python_env(),
    )
    draft = plan(experiment=study, source=source, env=environment)
    resolved = execution.run(draft)
    report_path = (
        resolved.path.parent / "artifacts/verify_gate/report/verification.json"
    )
    print(
        "CLEARX_VIPER_TRANSCRIPT_TRUST_VERIFICATION="
        + json.dumps(
            {
                "status": resolved.status,
                "result_path": str(resolved.path),
                "report_path": str(report_path),
                "report": load_json(report_path),
            },
            allow_nan=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
