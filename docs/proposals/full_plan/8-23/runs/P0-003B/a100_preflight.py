#!/usr/bin/env python3
"""Verify the packet-bound A100 runtime before downloading model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


TRANSFORMERS_COMMIT = "7d9754a05193eb79b1d86aa744b622b8068008cd"
MODEL_REPOSITORY = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_COMMIT = "26291f793822fb6be9555850f06dfe95f2d7e695"
MODEL_CONFIG_SHA256 = "eab5093d47807aaf894119506b238b2b1cee70d08456e894fee9a012d88f2e0d"


def run(command: list[str], cwd: Path | None = None) -> None:
    """Run one preflight command and propagate its failure."""
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    """Validate source identity, package import, model access, and A100 capacity."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-id", required=True)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    if args.packet_id != "P0-003B" or len(args.git_commit) != 40:
        raise ValueError("preflight requires packet P0-003B and a full Git commit")

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            f"git+https://github.com/huggingface/transformers.git@{TRANSFORMERS_COMMIT}",
            "accelerate>=1.10,<2",
            "huggingface_hub>=0.34,<2",
            "safetensors>=0.6,<1",
        ]
    )

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import Qwen3OmniMoeForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    properties = torch.cuda.get_device_properties(0)
    if "A100" not in properties.name:
        raise RuntimeError(f"requested A100, received {properties.name}")

    with tempfile.TemporaryDirectory(prefix="clearx-preflight-") as temporary:
        checkout = Path(temporary) / "CleaRx"
        run(["git", "clone", "--quiet", "https://github.com/pvd232/CleaRx.git", str(checkout)])
        run(["git", "checkout", "--quiet", args.git_commit], cwd=checkout)
        packet = checkout / "orchestration/work_packets/P0-003B.yaml"
        if not packet.is_file():
            raise RuntimeError("packet is absent from the requested Git commit")
        config_path = Path(
            hf_hub_download(repo_id=MODEL_REPOSITORY, filename="config.json", revision=MODEL_COMMIT)
        )
        config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
        if config_sha256 != MODEL_CONFIG_SHA256:
            raise RuntimeError("model configuration identity mismatch")
        config = json.loads(config_path.read_text(encoding="utf-8"))

    disk = shutil.disk_usage("/content")
    result = {
        "packet_id": args.packet_id,
        "git_commit": args.git_commit,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": properties.name,
        "device_total_bytes": properties.total_memory,
        "disk_free_bytes": disk.free,
        "transformers_commit": TRANSFORMERS_COMMIT,
        "reference_class": Qwen3OmniMoeForConditionalGeneration.__name__,
        "model_repository": MODEL_REPOSITORY,
        "model_commit": MODEL_COMMIT,
        "model_config_sha256": config_sha256,
        "talker_layers": config["talker_config"]["text_config"]["num_hidden_layers"],
        "residual_codebooks": config["talker_config"]["code_predictor_config"]["num_code_groups"] - 1,
        "code2wav_quantizers": config["code2wav_config"]["num_quantizers"],
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
