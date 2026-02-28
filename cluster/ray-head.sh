#!/usr/bin/env bash
# cluster/ray-head.sh — start a multi-node vLLM instance via Ray (HEAD node)
#
# Use this when a model is too large to fit on a single Spark and must be
# sharded across multiple GPUs using tensor parallelism.
#
# For models that fit on one Spark (e.g. Qwen3-32B-NVFP4 at ~18 GB),
# use the load-balancing approach instead (cluster/lb-worker.sh /
# cluster/lb-proxy.sh) — it's simpler and gives better throughput.
#
# Cluster setup order:
#   1. Run this script on the HEAD node first.
#   2. Run cluster/ray-worker.sh on each WORKER node, passing the head IP.
#   3. vLLM will start once all workers have joined.
#
# Usage:
#   ./cluster/ray-head.sh <num_nodes>
#
# Example (3-node cluster):
#   ./cluster/ray-head.sh 3

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck source=../model.conf
source "$PROJECT_DIR/model.conf"

VLLM_IMAGE="avarok/dgx-vllm-nvfp4-kernel:v23"
CONTAINER_NAME="vllm-ray-head"
VLLM_PORT=8000
RAY_PORT=6379
MODEL_DIR="$PROJECT_DIR/models/$MODEL_DIR_NAME"
NUM_NODES="${1:?Usage: $0 <num_nodes>}"
# Each Spark has 1 GPU; total tensor-parallel size = number of nodes
TOTAL_TP="$NUM_NODES"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

wait_for_http() {
    local url="$1" label="$2" timeout="${3:-900}"
    log "Waiting for $label ..."
    local elapsed=0
    until curl -sf "$url" > /dev/null 2>&1; do
        sleep 5; elapsed=$((elapsed + 5))
        [[ $elapsed -ge $timeout ]] && die "$label did not become ready within ${timeout}s"
        [[ $((elapsed % 60)) -eq 0 ]] && log "  still waiting ($elapsed s) — waiting for $NUM_NODES nodes to join..."
    done
    log "$label is ready."
}

# ── prereqs ───────────────────────────────────────────────────────────────────

[[ -f "$MODEL_DIR/config.json" ]] || die "Model not found at $MODEL_DIR. Run: ./download-model.sh"

# ── Ray head + vLLM container ─────────────────────────────────────────────────

if docker ps --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log "Ray head container already running."
else
    docker rm "$CONTAINER_NAME" 2>/dev/null || true

    log "Pulling vLLM image ..."
    docker pull "$VLLM_IMAGE"
    log "Image ready."

    LOCAL_IP=$(hostname -I | awk '{print $1}')
    log "Starting Ray head node + vLLM (tp=$TOTAL_TP across $NUM_NODES nodes) ..."
    log "Ray address: ${LOCAL_IP}:${RAY_PORT}"

    # The official NVIDIA vLLM container (26.01) lacks proper NVFP4 kernel support
    # for the DGX Spark's GB10 GPU (SM121). Avarok's patched container fixes this.
    # See: https://blog.avarok.net/we-unlocked-nvfp4-on-dgx-spark-and-its-20-faster-than-awq-72b0f3e58b83
    extra_env_args=()
    for kv in $EXTRA_DOCKER_ENVS; do extra_env_args+=(-e "$kv"); done

    docker run -d \
        --gpus all \
        --net host \
        --ipc host \
        --name "$CONTAINER_NAME" \
        -v "$MODEL_DIR:/model" \
        -v "$PROJECT_DIR/.cache/vllm:/root/.cache/vllm" \
        -e MODEL=/model \
        -e PORT="$VLLM_PORT" \
        -e GPU_MEMORY_UTIL=0.85 \
        -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
        -e MAX_NUM_SEQS=128 \
        -e HEAD_IP="$LOCAL_IP" \
        -e TENSOR_PARALLEL_SIZE="$TOTAL_TP" \
        -e VLLM_DEEP_GEMM_WARMUP=skip \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        -e VLLM_EXTRA_ARGS="--served-model-name $SERVED_MODEL_NAME --quantization $QUANTIZATION --enable-prefix-caching --attention-backend flashinfer --enable-auto-tool-choice --tool-call-parser $TOOL_CALL_PARSER${EXTRA_VLLM_FLAGS:+ $EXTRA_VLLM_FLAGS}" \
        "${extra_env_args[@]+"${extra_env_args[@]}"}" \
        "$VLLM_IMAGE" \
        serve
    #
    # MAX_MODEL_LEN — set to the model's full context. Claude Code requests up to
    #   32K output tokens, so anything less than input+32K will cause vLLM to
    #   reject the request with a 400 error. vLLM only allocates as much KV cache
    #   as GPU memory allows; requests exceeding available KV cache are queued,
    #   not rejected.
    #
    # Startup optimization notes:
    #   -v .cache/vllm:/root/.cache/vllm — persists torch compile cache across
    #       container restarts. First cold start is slow; subsequent restarts skip
    #       recompilation entirely.
    #   VLLM_DEEP_GEMM_WARMUP=skip — skips DeepGEMM JIT warmup (saves minutes;
    #       first-token latency may spike briefly on initial requests).

    log "Head container started. Loading pre-quantized NVFP4 model (allow 15 min) ..."
fi

LOCAL_IP=$(hostname -I | awk '{print $1}')
log ""
log "Now run on each of the $((NUM_NODES - 1)) worker node(s):"
log "  ./cluster/ray-worker.sh ${LOCAL_IP}"
log ""
log "vLLM will begin loading once all $NUM_NODES nodes have joined."
log "Watch progress: docker logs $CONTAINER_NAME --follow"

wait_for_http "http://localhost:${VLLM_PORT}/health" "vLLM ($NUM_NODES-node cluster)" 3600

# ── LiteLLM ──────────────────────────────────────────────────────────────────

LITELLM="$PROJECT_DIR/.venv/bin/litellm"
LITELLM_PID_FILE="$PROJECT_DIR/.litellm.pid"
LITELLM_PORT=4000

[[ -x "$LITELLM" ]] || die "litellm not found. Run: uv sync --project $PROJECT_DIR"

if [[ -f "$LITELLM_PID_FILE" ]] && kill -0 "$(cat "$LITELLM_PID_FILE")" 2>/dev/null; then
    log "LiteLLM already running (pid $(cat "$LITELLM_PID_FILE"))."
else
    log "Starting LiteLLM proxy on port ${LITELLM_PORT} ..."
    "$LITELLM" \
        --config "$PROJECT_DIR/litellm-config.yaml" \
        --port "$LITELLM_PORT" \
        >> "$PROJECT_DIR/litellm.log" 2>&1 &
    echo $! > "$LITELLM_PID_FILE"
    log "LiteLLM started (pid $!), logging to litellm.log"
fi

# ── done ─────────────────────────────────────────────────────────────────────

cat <<EOF

$NUM_NODES-node Ray cluster ready.

Claude Code (this machine):
  source "$PROJECT_DIR/use-local.sh" && claude

Claude Code (another machine):
  export ANTHROPIC_BASE_URL=http://${LOCAL_IP}:${LITELLM_PORT}
  export ANTHROPIC_AUTH_TOKEN=none
  claude

Stop:
  ./cluster/ray-stop.sh          # on head node
  ./cluster/ray-stop.sh          # on each worker node
EOF
