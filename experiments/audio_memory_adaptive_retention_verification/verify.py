"""Recompute one retained adaptive-retention result through VIPER."""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
from pathlib import Path

from viper import execution
from viper.authoring import experiment, input, plan, replicate, stage, variant
from viper.config import DiagnosticConfig
from viper.outputs import StageOutputs, output
from viper.references import GitFileRef
from viper.repository import read_source
from viper.runtime import LocalEnvSpec, observe_python_env
from viper.stages import StageContext, diagnostic

from clearx.audio_memory.gate import load_json
from clearx.audio_memory.retention import validate_result


@diagnostic(config=DiagnosticConfig)
def verify_retention(context: StageContext[DiagnosticConfig]) -> None:
    """Reload the A100 result and recompute its model, allocation, and metrics."""
    result = load_json(context.inputs["a100_result"])
    report = {
        "schema_version": "1.0.0",
        **validate_result(result),
    }
    destination = context.outputs["report"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_study(result_path: str):  # type: ignore[no-untyped-def]
    """Build the immutable diagnostic around one repository-relative result."""
    verification = stage(
        verify_retention,
        stage_id="verify_retention",
        inputs=(input("a100_result", path=result_path, data_role="eval"),),
        outputs=StageOutputs(
            report=output(
                path="verification.json",
                loader=load_json,
                data_role="eval",
            )
        ),
    )
    return experiment(
        experiment_id="audio_memory_adaptive_retention_verification",
        variants=(
            variant(
                "ravdess_equal_budget",
                stages=(verification,),
                estimator=verification.outputs["report"],
            ),
        ),
        replicates=(replicate(seed=7),),
    )


def main() -> None:
    """Run VIPER verification and print its durable result locations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result_path = args.result.as_posix()
    if args.result.is_absolute() or not args.result.is_file():
        raise ValueError("--result must name an existing repository-relative file")

    source = read_source()
    environment = LocalEnvSpec(
        lockfile=GitFileRef(
            repository=source.repository,
            commit=source.commit,
            path="environment.yml",
        ),
        python_env=observe_python_env(),
    )
    draft = plan(experiment=build_study(result_path), source=source, env=environment)
    resolved = execution.run(draft)
    report_path = (
        resolved.path.parent / "artifacts/verify_retention/report/verification.json"
    )
    print(
        "CLEARX_VIPER_ADAPTIVE_RETENTION_VERIFICATION="
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
