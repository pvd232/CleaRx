#!/usr/bin/env bash
set -euo pipefail

# Launch one packet-bound script on an ephemeral Colab A100 after local checks.
PACKET_ID=""
EXPECTED_COMMIT=""
REMOTE_SCRIPT=""
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"
DRY_RUN=false
COLAB_BIN="${COLAB_BIN:-colab}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --packet) PACKET_ID="$2"; shift 2 ;;
    --commit) EXPECTED_COMMIT="$2"; shift 2 ;;
    --script) REMOTE_SCRIPT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$PACKET_ID" || -z "$EXPECTED_COMMIT" || -z "$REMOTE_SCRIPT" ]]; then
  echo "Usage: $0 --packet ID --commit SHA --script FILE [--dry-run]" >&2
  exit 2
fi
if [[ ! "$PACKET_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ || ! "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Packet ID or commit has an invalid format." >&2
  exit 2
fi
case "$REMOTE_SCRIPT" in
  /*) echo "Remote script must be a repository-relative path without traversal." >&2; exit 2 ;;
esac
case "/$REMOTE_SCRIPT/" in
  *"/../"*|*"/./"*) echo "Remote script must be a repository-relative path without traversal." >&2; exit 2 ;;
esac

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing remote execution from a dirty Git tree." >&2
  exit 1
fi
if [[ "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
  echo "Current commit does not match --commit." >&2
  exit 1
fi
if [[ ! -f "orchestration/work_packets/${PACKET_ID}.yaml" || ! -f "$REMOTE_SCRIPT" ]]; then
  echo "Packet or remote script is missing." >&2
  exit 1
fi
if ! git ls-files --error-unmatch "$REMOTE_SCRIPT" >/dev/null 2>&1; then
  echo "Remote script must be tracked by Git." >&2
  exit 1
fi

python scripts/run_packet.py "$PACKET_ID" >/dev/null
"$COLAB_BIN" --help >/dev/null
"$COLAB_BIN" run --help >/dev/null

if [[ "$DRY_RUN" == true ]]; then
  echo "A100 preflight valid: packet=$PACKET_ID commit=$EXPECTED_COMMIT script=$REMOTE_SCRIPT"
  exit 0
fi

exec "$COLAB_BIN" --auth=adc run --gpu A100 --timeout "$TIMEOUT_SECONDS" \
  "$REMOTE_SCRIPT" --packet-id "$PACKET_ID" --git-commit "$EXPECTED_COMMIT"
