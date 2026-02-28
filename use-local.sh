#!/usr/bin/env bash
# use-local.sh — configure Claude Code to use local vLLM via LiteLLM
#
# Usage (source, don't execute):
#   source ~/Documents/vllm/use-local.sh
#   claude
#
# To undo:
#   source ~/Documents/vllm/use-local.sh --reset

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=model.conf
source "$SCRIPT_DIR/model.conf"

LITELLM_HOST="${LITELLM_HOST:-localhost}"
LITELLM_PORT="${LITELLM_PORT:-4000}"

if [[ "${1:-}" == "--reset" ]]; then
    unset ANTHROPIC_BASE_URL
    unset ANTHROPIC_AUTH_TOKEN
    echo "Claude Code restored to default Anthropic API."
else
    export ANTHROPIC_BASE_URL="http://${LITELLM_HOST}:${LITELLM_PORT}"
    export ANTHROPIC_AUTH_TOKEN="none"
    echo "Claude Code configured:"
    echo "  ANTHROPIC_BASE_URL=${ANTHROPIC_BASE_URL}"
    echo "  Model: $MODEL_DIR_NAME via vLLM (prefix caching enabled)"
    echo ""
    echo "Run: claude"
    echo "Undo: source use-local.sh --reset"
fi
