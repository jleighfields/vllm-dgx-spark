#!/usr/bin/env bash
# download-model.sh — download Qwen3-Coder-Next-NVFP4 (~40 GB) to models/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="Cirrascale/Qwen3-Coder-Next-NVFP4"
MODEL_DIR="$SCRIPT_DIR/models/Qwen3-Coder-Next-NVFP4"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

mkdir -p "$MODEL_DIR"
log "Downloading $MODEL (~40 GB) to $MODEL_DIR ..."
log "This will take a while depending on your connection."

~/.local/bin/uv run --project "$SCRIPT_DIR" python -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', local_dir='$MODEL_DIR')
"

log "Download complete: $MODEL_DIR"
