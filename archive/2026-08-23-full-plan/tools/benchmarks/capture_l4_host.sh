#!/usr/bin/env bash
set -euo pipefail

# Run on the certified L4 host and emit one JSON profile to standard output.
: "${SOURCE_COMMIT:?Set SOURCE_COMMIT to the committed capture-tool revision.}"
: "${GCE_PROJECT:?Set GCE_PROJECT.}"
: "${GCE_INSTANCE:?Set GCE_INSTANCE.}"
: "${GCE_ZONE:?Set GCE_ZONE.}"
: "${GCE_MACHINE_TYPE:?Set GCE_MACHINE_TYPE.}"
: "${GCE_PROVISIONING_MODEL:?Set GCE_PROVISIONING_MODEL.}"
: "${GCE_BOOT_DISK_GIB:?Set GCE_BOOT_DISK_GIB.}"
: "${GCE_DISK_TYPE:?Set GCE_DISK_TYPE.}"

python3 - <<'PY'
from __future__ import annotations

import json
import os
import platform
import re
import resource
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(*command: str) -> str:
    """Return stripped stdout for one successful host inspection command."""
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def read_os_release() -> dict[str, str]:
    """Parse the host's operating-system identity file."""
    values: dict[str, str] = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value.strip('"')
    return values


def meminfo_bytes(name: str) -> int:
    """Read one kB-valued Linux memory counter as bytes."""
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{name}:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError(f"missing /proc/meminfo field: {name}")


gpu_fields = [field.strip() for field in run(
    "nvidia-smi",
    "--query-gpu=name,uuid,memory.total,driver_version,pci.bus_id,pstate,power.limit,temperature.gpu,persistence_mode,compute_mode",
    "--format=csv,noheader,nounits",
).split(",")]
pcie_fields = [int(field.strip()) for field in run(
    "nvidia-smi",
    "--query-gpu=pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max",
    "--format=csv,noheader,nounits",
).split(",")]
bus_id = gpu_fields[4].lower().replace("00000000:", "0000:")
numa_path = Path("/sys/bus/pci/devices") / bus_id / "numa_node"
lscpu = json.loads(run("lscpu", "-J"))["lscpu"]
cpu_values = {entry["field"].rstrip(":"): entry["data"] for entry in lscpu}
filesystem_source, filesystem_type = run("findmnt", "-no", "SOURCE,FSTYPE", "/").split()
storage = os.statvfs("/")
nvcc_output = run("/usr/local/cuda/bin/nvcc", "--version")
nvcc_match = re.search(r"V([0-9.]+)", nvcc_output)
if nvcc_match is None:
    raise RuntimeError("nvcc version was not present in compiler output")
os_release = read_os_release()
memlock_soft, memlock_hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)

profile = {
    "schema_version": "1.0.0",
    "profile_id": "clearx-mantra-g2-l4-v1",
    "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "source_commit": os.environ["SOURCE_COMMIT"],
    "gce": {
        "project": os.environ["GCE_PROJECT"],
        "instance": os.environ["GCE_INSTANCE"],
        "zone": os.environ["GCE_ZONE"],
        "machine_type": os.environ["GCE_MACHINE_TYPE"],
        "provisioning_model": os.environ["GCE_PROVISIONING_MODEL"],
        "boot_disk_gib": int(os.environ["GCE_BOOT_DISK_GIB"]),
    },
    "os": {
        "pretty_name": os_release["PRETTY_NAME"],
        "id": os_release["ID"],
        "version_id": os_release["VERSION_ID"],
    },
    "cpu": {
        "architecture": cpu_values["Architecture"],
        "logical_cpus": int(cpu_values["CPU(s)"]),
        "model": cpu_values["Model name"],
        "threads_per_core": int(cpu_values["Thread(s) per core"]),
        "cores_per_socket": int(cpu_values["Core(s) per socket"]),
        "sockets": int(cpu_values["Socket(s)"]),
        "numa_nodes": int(cpu_values["NUMA node(s)"]),
    },
    "memory": {
        "total_bytes": meminfo_bytes("MemTotal"),
        "available_bytes_at_capture": meminfo_bytes("MemAvailable"),
        "memlock_soft_bytes": memlock_soft,
        "memlock_hard_bytes": memlock_hard,
    },
    "gpu": {
        "name": gpu_fields[0],
        "uuid": gpu_fields[1],
        "memory_total_mib": int(gpu_fields[2]),
        "driver_version": gpu_fields[3],
        "performance_state": gpu_fields[5],
    },
    "pcie": {
        "bus_id": bus_id,
        "generation_current": pcie_fields[0],
        "generation_max": pcie_fields[1],
        "width_current": pcie_fields[2],
        "width_max": pcie_fields[3],
        "numa_node": int(numa_path.read_text(encoding="utf-8").strip()),
    },
    "storage": {
        "device": filesystem_source,
        "filesystem": filesystem_type,
        "size_bytes": storage.f_blocks * storage.f_frsize,
        "available_bytes": storage.f_bavail * storage.f_frsize,
        "gce_disk_type": os.environ["GCE_DISK_TYPE"],
    },
    "software": {
        "kernel": platform.release(),
        "cuda": "12.9",
        "nvcc": nvcc_match.group(1),
        "compiler": f"GCC {run('gcc', '-dumpfullversion', '-dumpversion')}",
    },
    "thermal_power": {
        "power_limit_w": float(gpu_fields[6]),
        "temperature_c": int(gpu_fields[7]),
        "persistence_mode": gpu_fields[8] == "Enabled",
        "compute_mode": gpu_fields[9],
    },
    "measurement_contract": {
        "phases": [
            "listening",
            "voxzip_concurrent_listening",
            "voxzip_sequential_endpoint",
            "thinker_prefill",
            "talker_startup",
            "talker_sustained_output",
            "barge_in",
            "context_rebuild",
        ],
        "metrics": ["phase_live_sets", "allocator_peak", "process_rss"],
        "sampling_interval_ms": 100,
    },
    "sources": {
        "gpu": "nvidia-smi query-gpu",
        "pcie": "nvidia-smi PCIe link query plus sysfs NUMA node",
        "cpu": "lscpu -J",
        "memory": "/proc/meminfo plus RLIMIT_MEMLOCK",
        "storage": "findmnt plus statvfs",
        "software": "/etc/os-release, uname, nvcc, and gcc",
    },
}
json.dump(profile, sys.stdout, indent=2)
print()
PY
