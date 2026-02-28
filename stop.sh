#!/usr/bin/env bash
# stop.sh — stop LiteLLM proxy and vLLM container
#
# Usage:
#   ./stop.sh           — stop services; keep container so cached kernels survive restart
#   ./stop.sh --clean   — stop and remove container (needed when switching models)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="vllm-server"
LITELLM_PID_FILE="$SCRIPT_DIR/.litellm.pid"
CLEAN="${1:-}"

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
else
    log "vLLM container not running."
fi

if [[ "$CLEAN" == "--clean" ]]; then
    docker rm "$CONTAINER_NAME" 2>/dev/null && log "Container removed." || true
else
    log "Container stopped (not removed). Run './stop.sh --clean' to remove it."
    log "Use --clean when switching models to force a fresh container."
fi

log "Done."
