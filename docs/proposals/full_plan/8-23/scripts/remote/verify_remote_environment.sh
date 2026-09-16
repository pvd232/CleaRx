#!/usr/bin/env bash
set -euo pipefail

# Emit stable host/tool facts consumed by a remote result manifest.
printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
printf 'python=%s\n' "$(python --version 2>&1)"
printf 'kernel=%s\n' "$(uname -srv)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi
