#!/usr/bin/env bash
# cluster/ray-stop.sh — stop Ray cluster containers on this node
#
# Run on the head node and on each worker node.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LITELLM_PID_FILE="$PROJECT_DIR/.litellm.pid"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── LiteLLM (head node only) ──────────────────────────────────────────────────

if [[ -f "$LITELLM_PID_FILE" ]]; then
    PID="$(cat "$LITELLM_PID_FILE")"
    if kill -0 "$PID" 2>/dev/null; then
        log "Stopping LiteLLM (pid $PID) ..."
        kill "$PID"
    else
        log "LiteLLM process not running."
    fi
    rm -f "$LITELLM_PID_FILE"
fi

# ── Ray containers ────────────────────────────────────────────────────────────

for name in vllm-ray-head vllm-ray-worker; do
    if docker ps --filter "name=$name" --format '{{.Names}}' | grep -q "^${name}$"; then
        log "Stopping $name ..."
        docker stop "$name"
        docker rm "$name"
    else
        log "$name not running."
    fi
done

log "Done."
