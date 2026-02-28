#!/usr/bin/env bash
# cluster/lb-proxy.sh — start LiteLLM proxy load-balancing across multiple vLLM workers
#
# Run this on ONE node after all workers are up via cluster/lb-worker.sh.
# LiteLLM round-robins requests across all worker IPs.
#
# Usage:
#   ./cluster/lb-proxy.sh <ip1> [ip2] [ip3] ...
#
# Example (3-node cluster):
#   ./cluster/lb-proxy.sh 192.168.0.10 192.168.0.11 192.168.0.12

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LITELLM_PORT=4000
VLLM_PORT=8000
LITELLM="$PROJECT_DIR/.venv/bin/litellm"
LITELLM_PID_FILE="$PROJECT_DIR/.litellm.pid"
CONFIG_FILE="$PROJECT_DIR/.cluster-lb-litellm-config.yaml"
MODEL_NAME="Qwen/Qwen3-Coder-Next-NVFP4"
SERVED_NAME="claude-sonnet-4-6"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

wait_for_http() {
    local url="$1" label="$2" timeout="${3:-30}"
    log "Waiting for $label ..."
    local elapsed=0
    until curl -sf "$url" > /dev/null 2>&1; do
        sleep 2; elapsed=$((elapsed + 2))
        [[ $elapsed -ge $timeout ]] && die "$label did not become ready within ${timeout}s"
    done
    log "$label is ready."
}

# ── prereqs ───────────────────────────────────────────────────────────────────

[[ $# -ge 1 ]] || die "Usage: $0 <ip1> [ip2] ..."
[[ -x "$LITELLM" ]] || die "litellm not found. Run: uv sync --project $PROJECT_DIR"

# ── generate config dynamically from provided IPs ─────────────────────────────

cat > "$CONFIG_FILE" <<YAML
model_list:
YAML

for ip in "$@"; do
    cat >> "$CONFIG_FILE" <<YAML
  - model_name: $SERVED_NAME
    litellm_params:
      model: openai/$MODEL_NAME
      api_base: http://${ip}:${VLLM_PORT}/v1
      api_key: "none"
YAML
done

cat >> "$CONFIG_FILE" <<YAML
litellm_settings:
  drop_params: true
  ignore_invalid_params: true
general_settings: {}
YAML

log "Generated config with $# worker(s):"
for ip in "$@"; do log "  http://${ip}:${VLLM_PORT}/v1"; done

# ── LiteLLM ──────────────────────────────────────────────────────────────────

if [[ -f "$LITELLM_PID_FILE" ]] && kill -0 "$(cat "$LITELLM_PID_FILE")" 2>/dev/null; then
    log "LiteLLM already running (pid $(cat "$LITELLM_PID_FILE")). Stop it first with ./stop.sh"
    exit 1
fi

"$LITELLM" \
    --config "$CONFIG_FILE" \
    --port "$LITELLM_PORT" \
    >> "$PROJECT_DIR/litellm.log" 2>&1 &
echo $! > "$LITELLM_PID_FILE"
log "LiteLLM started (pid $!), logging to litellm.log"

wait_for_http "http://localhost:${LITELLM_PORT}/health" "LiteLLM" 30

# ── done ─────────────────────────────────────────────────────────────────────

LOCAL_IP=$(hostname -I | awk '{print $1}')
cat <<EOF

LiteLLM proxy running on port ${LITELLM_PORT}, load-balancing $# worker(s).

Claude Code (this machine):
  source "$PROJECT_DIR/use-local.sh" && claude

Claude Code (another machine):
  export ANTHROPIC_BASE_URL=http://${LOCAL_IP}:${LITELLM_PORT}
  export ANTHROPIC_AUTH_TOKEN=none
  claude

Stop proxy:
  ./stop.sh

Stop workers:
  Run ./stop.sh on each Spark worker node.
EOF
