#!/usr/bin/env bash
# download-model.sh — download the model configured in model.conf to models/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=model.conf
source "$SCRIPT_DIR/model.conf"

MODEL_DIR="$SCRIPT_DIR/models/$MODEL_DIR_NAME"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

mkdir -p "$MODEL_DIR"
log "Downloading $MODEL_REPO to $MODEL_DIR ..."
log "This will take a while depending on the model size and your connection."

~/.local/bin/uv run --project "$SCRIPT_DIR" python -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_REPO', local_dir='$MODEL_DIR')
"

log "Download complete: $MODEL_DIR"
