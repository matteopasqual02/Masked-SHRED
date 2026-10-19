#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"
uv run python Images/scripts/reduce_image_dpi.py \
    Images \
    Images/Images_low \
    --ratio 6 \
    --dpi 300
uv run python Images/scripts/reduce_image_dpi.py \
    Images \
    Images/Images_low \
    --ratio 12 \
    --dpi 300 \
    --name-contains reconstruction_grid
