#!/usr/bin/env bash
# cleanup-models.sh — delete unused model directories and HF hub cache duplicates
#
# Lists all model directories under models/ and HuggingFace hub-cached repos
# under ~/.cache/huggingface/hub/, marks the currently-active model (from
# model.conf), and interactively prompts before deleting each non-active item.
#
# The active model is never offered for deletion. Default answer is No, so
# pressing Enter on every prompt is safe.
#
# Usage: ./cleanup-models.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=model.conf
source "$SCRIPT_DIR/model.conf"

ACTIVE="${MODEL_DIR_NAME:?MODEL_DIR_NAME not set in model.conf}"
MODELS_DIR="$SCRIPT_DIR/models"
HUB_DIR="${HOME}/.cache/huggingface/hub"

confirm() {
    local prompt="$1"
    local ans
    read -r -p "$prompt [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]]
}

human_size() { du -sh "$1" 2>/dev/null | awk '{print $1}'; }

echo "Active model (from model.conf): $ACTIVE"
echo

# ── models/ directory ─────────────────────────────────────────────────────────
echo "=== Project model dirs ($MODELS_DIR) ==="
if [[ ! -d "$MODELS_DIR" ]]; then
    echo "  (no models/ directory)"
else
    shopt -s nullglob
    for dir in "$MODELS_DIR"/*/; do
        name=$(basename "$dir")
        size=$(human_size "$dir")
        if [[ "$name" == "$ACTIVE" ]]; then
            printf "  KEEP   %-50s %s  [active]\n" "$name" "$size"
        else
            printf "  unused %-50s %s\n" "$name" "$size"
        fi
    done
    shopt -u nullglob
fi
echo

shopt -s nullglob
for dir in "$MODELS_DIR"/*/; do
    name=$(basename "$dir")
    [[ "$name" == "$ACTIVE" ]] && continue
    size=$(human_size "$dir")
    if confirm "Delete models/$name ($size)?"; then
        rm -rf "$dir"
        echo "  deleted models/$name"
    else
        echo "  kept models/$name"
    fi
done
shopt -u nullglob

# ── HuggingFace hub cache ─────────────────────────────────────────────────────
echo
echo "=== HuggingFace hub cache ($HUB_DIR) ==="
if [[ ! -d "$HUB_DIR" ]]; then
    echo "  (no hub cache)"
else
    total=$(human_size "$HUB_DIR")
    echo "  Total cache size: $total"
    shopt -s nullglob
    for repo in "$HUB_DIR"/models--*/; do
        rname=$(basename "$repo")
        rsize=$(human_size "$repo")
        printf "  %-60s %s\n" "$rname" "$rsize"
    done
    shopt -u nullglob
    echo
    echo "  These are redundant copies left behind by snapshot_download. The"
    echo "  files served by vLLM live in models/<name>/ and are independent —"
    echo "  deleting the hub cache won't break anything currently deployed."
    if confirm "Delete entire hub cache?"; then
        rm -rf "$HUB_DIR"
        echo "  hub cache cleared"
    else
        echo "  hub cache kept"
    fi
fi

echo
echo "Done."
