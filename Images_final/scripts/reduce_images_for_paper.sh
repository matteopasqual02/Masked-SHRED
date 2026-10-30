#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"
uv run python Images_final/scripts/reduce_image_dpi.py \
    Images_final \
    Images_final/Images_low \
    --ratio 4 \
    --dpi 300
uv run python Images_final/scripts/reduce_image_dpi.py \
    Images_final \
    Images_final/Images_low \
    --ratio 8 \
    --dpi 300 \
    --name-contains reconstruction_grid
