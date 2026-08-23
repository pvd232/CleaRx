#!/usr/bin/env bash
set -euo pipefail

# Download one declared Colab artifact into a packet run directory.
if [[ $# -ne 4 || "$1" != "--session" || "$3" != "--remote" ]]; then
  echo "Usage: $0 --session NAME --remote PATH" >&2
  exit 2
fi

SESSION="$2"
REMOTE_PATH="$4"
LOCAL_PATH="${LOCAL_PATH:?Set LOCAL_PATH to runs/<packet>/<run>/...}"
COLAB_BIN="${COLAB_BIN:-colab}"
REPO_ROOT=$(git rev-parse --show-toplevel)

case "$LOCAL_PATH" in
  runs/*) ;;
  *) echo "LOCAL_PATH must stay below runs/." >&2; exit 1 ;;
esac
case "/$LOCAL_PATH/" in
  *"/../"*|*"/./"*) echo "LOCAL_PATH must not contain traversal segments." >&2; exit 1 ;;
esac

mkdir -p "$REPO_ROOT/$(dirname "$LOCAL_PATH")"
"$COLAB_BIN" --auth=adc download --session "$SESSION" "$REMOTE_PATH" "$REPO_ROOT/$LOCAL_PATH"
