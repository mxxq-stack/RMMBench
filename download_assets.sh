#!/bin/bash
# Download RMMBench-robocasa-scenes assets and extract into RMMBench/assets
# Env vars: HF_ENDPOINT (default https://hf-mirror.com), KEEP_ZIP=1 to keep zips
set -euo pipefail

REPO="mxxq/RMMBench-robocasa-scenes"
FILES=("obj.zip" "scene.zip")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS_DIR="${ASSETS_DIR:-$SCRIPT_DIR/RMMBench/assets}"
BASE="${HF_ENDPOINT:-https://hf-mirror.com}/datasets/${REPO}/resolve/main"

echo ">> assets dir: $ASSETS_DIR"
echo ">> source: $BASE"
mkdir -p "$ASSETS_DIR"

for f in "${FILES[@]}"; do
    echo ">> downloading $f ..."
    curl -L -C - --retry 5 --retry-delay 5 -o "$ASSETS_DIR/$f" "$BASE/$f"

    echo ">> extracting $f ..."
    unzip -q -o "$ASSETS_DIR/$f" -d "$ASSETS_DIR"

    if [[ "${KEEP_ZIP:-0}" != "1" ]]; then
        rm -f "$ASSETS_DIR/$f"
    fi
done

ls -d "$ASSETS_DIR/obj" "$ASSETS_DIR/scenes_fixtures" >/dev/null \
    || { echo "!! expected obj/ or scenes_fixtures/ not found" >&2; exit 1; }
echo ">> done"
