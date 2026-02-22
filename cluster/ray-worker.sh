#!/usr/bin/env bash
# cluster/ray-worker.sh — join an existing Ray cluster as a worker node
#
# Run this on each WORKER node AFTER cluster/ray-head.sh is running on the head.
# Workers contribute their GPU to the shared tensor-parallel vLLM instance.
# Workers do NOT need the model files downloaded locally.
#
# Usage:
#   ./cluster/ray-worker.sh <head-ip>
#
# Example:
#   ./cluster/ray-worker.sh 192.168.0.10

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_IMAGE="avarok/dgx-vllm-nvfp4-kernel:v22"
CONTAINER_NAME="vllm-ray-worker"
RAY_PORT=6379
HEAD_IP="${1:?Usage: $0 <head-ip>}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ── join Ray cluster ──────────────────────────────────────────────────────────

if docker ps --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log "Ray worker container already running."
else
    docker rm "$CONTAINER_NAME" 2>/dev/null || true

    log "Pulling vLLM image ..."
    docker pull "$VLLM_IMAGE"
    log "Image ready."

    log "Joining Ray cluster at ${HEAD_IP}:${RAY_PORT} ..."
    docker run -d \
        --gpus all \
        --net host \
        --ipc host \
        --name "$CONTAINER_NAME" \
        -e HEAD_IP="$HEAD_IP" \
        "$VLLM_IMAGE" \
        ray-worker

    log "Worker joined cluster."
fi

log "Watch head node for loading progress: docker logs vllm-ray-head --follow"
log "Stop this worker: ./cluster/ray-stop.sh"
