#!/usr/bin/env bash
# stop.sh — stop LiteLLM proxy and vLLM container

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="vllm-server"
LITELLM_PID_FILE="$SCRIPT_DIR/.litellm.pid"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── LiteLLM ──────────────────────────────────────────────────────────────────

if [[ -f "$LITELLM_PID_FILE" ]]; then
    PID="$(cat "$LITELLM_PID_FILE")"
    if kill -0 "$PID" 2>/dev/null; then
        log "Stopping LiteLLM (pid $PID) ..."
        kill "$PID"
    else
        log "LiteLLM process not running."
    fi
    rm -f "$LITELLM_PID_FILE"
else
    log "No LiteLLM pid file found."
fi

# ── vLLM container ────────────────────────────────────────────────────────────

if docker ps --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log "Stopping vLLM container ..."
    docker stop "$CONTAINER_NAME"
    docker rm "$CONTAINER_NAME"
else
    log "vLLM container not running."
fi

log "Done."
